"""Source-backed hypotheses for failed weak-output/criterion pairs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from . import jev_questions
from .diagnosis import split_sentences
from .evaluation.calibration import DecisionPolicy, runtime_question_identity
from .gateway import Gateway, ProviderError
from .jev import (
    ChoiceDecision,
    JevResponseError,
    NoulDecision,
    batch_decision_payload,
    parse_decision,
)
from .models import Tier
from .runner import PanelResult

ATTRIBUTION_KIND_OPTIONS = jev_questions.FAILURE_ATTRIBUTION_KINDS
MAX_ATTRIBUTION_REQUEST_BYTES = 96_000


@dataclass(frozen=True, slots=True)
class AttributionBudget:
    pair_cap: int
    dollar_cap: float

    @classmethod
    def for_tier(
        cls,
        tier: Tier,
        *,
        pair_cap: int | None = None,
        dollar_cap: float | None = None,
    ) -> AttributionBudget:
        defaults = tier.budget
        budget = cls(
            defaults.attribution_pairs if pair_cap is None else pair_cap,
            defaults.attribution_dollars if dollar_cap is None else dollar_cap,
        )
        if (
            budget.pair_cap < 0
            or not math.isfinite(budget.dollar_cap)
            or budget.dollar_cap < 0
        ):
            raise ValueError("attribution budgets must be non-negative")
        return budget


def _priced_rates(gateway: Gateway, model: str) -> tuple[float, float] | None:
    loader = getattr(gateway, "list_models", None)
    if not callable(loader):
        return None
    try:
        info = loader().get(model)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if info is None:
        return None
    rates = (info.input_cost_per_token, info.output_cost_per_token)
    if any(rate is None or not math.isfinite(rate) or rate < 0 for rate in rates):
        return None
    return float(rates[0]), float(rates[1])


def _retry_multiplier(gateway: Gateway) -> int:
    config = getattr(gateway, "config", None)
    if config is None:
        config = getattr(getattr(gateway, "gateway", None), "config", None)
    retries = getattr(config, "max_retries", 0)
    return max(1, retries + 1) if isinstance(retries, int) else 1


def _role_cost(gateway: Gateway) -> float | None:
    report = gateway.usage_report()
    costs = report.get("cost_by_role")
    if not isinstance(costs, Mapping):
        return None
    value = costs.get("judge_attribution", 0.0)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return max(0.0, float(value))
    return None


def _parsed(raw: Any) -> ChoiceDecision | NoulDecision | None:
    try:
        decision = parse_decision(raw)
    except JevResponseError:
        return None
    return decision if isinstance(decision, (ChoiceDecision, NoulDecision)) else None


def _policy_gate(
    *,
    request: Mapping[str, Any],
    raw: Any,
    decision: ChoiceDecision | NoulDecision | None,
    snapshot: str | None,
    decision_policy: DecisionPolicy | None,
    name: str,
) -> tuple[bool, float, dict[str, Any]]:
    threshold = 0.8
    if decision is None:
        return False, threshold, {"reason": "missing_or_malformed_answer"}
    if not isinstance(snapshot, str) or not snapshot:
        return False, threshold, {"reason": "missing_answering_snapshot"}
    if decision_policy is None:
        return True, threshold, {"disposition": "provisional_default"}
    question_id = f"failure-attribution:{name}"
    identity = runtime_question_identity(
        question_id,
        request,
        family="failure_attribution",
        rubric_version="issue-54-v1",
        snapshot=snapshot,
        policy_version=decision_policy.policy_version,
    )
    identity = replace(
        identity,
        event_mapping={"selected_correctness": True}
        if isinstance(decision, ChoiceDecision)
        else {"polarity": "positive"},
    )
    policy = decision_policy.apply(
        question_id=question_id,
        identity=identity,
        decision=decision,
        raw_answer=raw,
        snapshot=snapshot,
    )
    if policy.threshold is not None:
        threshold = policy.threshold
    return (
        policy.is_legacy or policy.may_gate,
        threshold,
        {"identity": identity.to_dict(), "policy": asdict(policy)},
    )


def _requests(
    run: PanelResult, test: Mapping[str, Any], pair_id: str
) -> list[dict[str, Any]]:
    sentences = split_sentences(run.prompt)
    state = {
        "candidate_prompt": run.prompt,
        "candidate_sentences": [
            {"id": sentence.id, "text": sentence.text} for sentence in sentences
        ],
        "output": run.output,
        "criterion": str(test.get("question", "")),
        "candidate_id": run.candidate_id,
        "model": run.model,
        "sample": run.sample,
        "test_id": str(test.get("id", "")),
    }
    return [
        {
            "key": f"failure-attribution:{pair_id}:pointer",
            "model": "jev",
            "type": "choice",
            "state": state,
            "question": jev_questions.FAILURE_ATTRIBUTION_POINTER_QUESTION,
            "options": [sentence.id for sentence in sentences] + ["none"],
            "question_schema": {"protocol": "issue-54-v1", "role": "pointer"},
        },
        {
            "key": f"failure-attribution:{pair_id}:kind",
            "model": "jev",
            "type": "choice",
            "state": state,
            "question": jev_questions.FAILURE_ATTRIBUTION_KIND_QUESTION,
            "options": ATTRIBUTION_KIND_OPTIONS,
            "question_schema": {"protocol": "issue-54-v1", "role": "kind"},
        },
        {
            "key": f"failure-attribution:{pair_id}:attributable",
            "model": "jev",
            "type": "noul",
            "state": state,
            "question": jev_questions.FAILURE_ATTRIBUTION_NOUL_QUESTION,
            "question_schema": {"protocol": "issue-54-v1", "role": "attributable"},
        },
    ]


def attribute_failed_pairs(
    panel: Sequence[PanelResult],
    tests: Sequence[Mapping[str, Any]],
    pair_outcomes: Sequence[Mapping[str, Any]],
    candidate_ids: Collection[str],
    gateway: Gateway,
    *,
    judge_model: str,
    run_id: str,
    budget: AttributionBudget,
    decision_policy: DecisionPolicy | None = None,
) -> tuple[dict[str, tuple[dict[str, Any], ...]], dict[str, Any]]:
    """Inspect bounded failed pairs; only supported records enter writer summaries."""
    eligible = sorted(
        (
            outcome
            for outcome in pair_outcomes
            if outcome.get("status") == "failed"
            and outcome.get("candidate_id") in candidate_ids
        ),
        key=lambda item: (
            str(item.get("candidate_id")),
            str(item.get("model")),
            int(item.get("sample", 0)),
            str(item.get("test_id")),
            int(item.get("output_index", 0)),
        ),
    )
    by_candidate: dict[str, list[dict[str, Any]]] = {item: [] for item in candidate_ids}
    records: list[dict[str, Any]] = []
    rates = (
        _priced_rates(gateway, judge_model) if budget.pair_cap and eligible else None
    )
    retry_multiplier = _retry_multiplier(gateway)
    round_cost_before = _role_cost(gateway)
    spent_reserved = 0.0
    requested = 0
    for outcome in eligible:
        output_index = int(outcome["output_index"])
        test_index = int(outcome["test_index"])
        run = panel[output_index]
        test = tests[test_index]
        prompt_digest = hashlib.sha256(run.prompt.encode("utf-8")).hexdigest()
        pair_id = f"{output_index:04d}:{test_index:04d}"
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "candidate_id": run.candidate_id,
            "prompt_digest": prompt_digest,
            "model": run.model,
            "sample": run.sample,
            "test_id": str(test.get("id", f"t{test_index}")),
            "pass_probability": float(outcome["pass_probability"]),
            "policy_version": (
                decision_policy.policy_version
                if decision_policy is not None
                else "default-0.8"
            ),
            "status": "skipped",
            "reason": "pair_budget_exhausted",
        }
        questions = _requests(run, test, pair_id)
        for question in questions:
            question["model"] = judge_model
        _, envelope = batch_decision_payload(questions, model=judge_model)
        input_bytes = len(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))
        reservation = (
            max(0.00001, 2.0 * (math.ceil(input_bytes / 4) * rates[0] + 128 * rates[1]))
            * retry_multiplier
            if rates is not None
            else None
        )
        if requested >= budget.pair_cap:
            pass
        elif rates is None:
            record["reason"] = "missing_trustworthy_pricing"
        elif input_bytes > MAX_ATTRIBUTION_REQUEST_BYTES:
            record["reason"] = "request_exceeds_provider_limit"
        elif reservation is None or spent_reserved + reservation > budget.dollar_cap:
            record["reason"] = "dollar_budget_exhausted"
        else:
            requested += 1
            spent_reserved += reservation
            record["status"] = "unresolved"
            record["reason"] = "answer_not_decisive"
            record["questions"] = questions
            record["reservation_usd"] = reservation
            log_start = len(gateway.decision_log)
            cost_before = _role_cost(gateway)
            try:
                answers = gateway.decide_batch(
                    questions, role="judge_attribution", run_id=run_id
                )
            except ProviderError as exc:
                record["reason"] = f"provider_{exc.kind}"
            else:
                record["answers"] = answers
                if len(answers) != len(questions):
                    record["reason"] = "incomplete_answer"
                else:
                    decisions = [_parsed(answer) for answer in answers]
                    logged = gateway.decision_log[log_start:]
                    snapshots = [
                        item.get("answered_by") if isinstance(item, Mapping) else None
                        for item in logged
                    ]
                    if len(snapshots) != len(questions):
                        record["reason"] = "missing_answering_snapshot"
                    else:
                        policies = [
                            _policy_gate(
                                request=question,
                                raw=answer,
                                decision=decision,
                                snapshot=snapshot,
                                decision_policy=decision_policy,
                                name=name,
                            )
                            for question, answer, decision, snapshot, name in zip(
                                questions,
                                answers,
                                decisions,
                                snapshots,
                                ("pointer", "kind", "attributable"),
                                strict=True,
                            )
                        ]
                        record["policies"] = [item[2] for item in policies]
                        record["snapshots"] = snapshots
                        pointer, kind, attributable = decisions
                        valid_ids = {
                            sentence.id for sentence in split_sentences(run.prompt)
                        }
                        supported = (
                            isinstance(pointer, ChoiceDecision)
                            and isinstance(kind, ChoiceDecision)
                            and isinstance(attributable, NoulDecision)
                            and pointer.selected in valid_ids
                            and kind.selected in ATTRIBUTION_KIND_OPTIONS
                            and kind.selected != "unknown"
                            and pointer.confidence >= policies[0][1]
                            and kind.confidence >= policies[1][1]
                            and attributable.probability >= policies[2][1]
                            and all(item[0] for item in policies)
                        )
                        if supported:
                            sentence = next(
                                sentence
                                for sentence in split_sentences(run.prompt)
                                if sentence.id == pointer.selected
                            )
                            record.update(
                                {
                                    "status": "supported",
                                    "reason": "source_backed_hypothesis",
                                    "sentence_id": sentence.id,
                                    "sentence_text": sentence.text,
                                    "kind": kind.selected,
                                    "attributable_probability": attributable.probability,
                                    "pointer_confidence": pointer.confidence,
                                    "kind_confidence": kind.confidence,
                                }
                            )
                        else:
                            record["reason"] = "low_confidence_or_unsupported_source"
            cost_after = _role_cost(gateway)
            if cost_before is not None and cost_after is not None:
                actual = max(0.0, cost_after - cost_before)
                record["cost_usd_measured"] = actual
                spent_reserved += max(0.0, actual - reservation)
        records.append(record)
        by_candidate[run.candidate_id].append(record)
    round_cost_after = _role_cost(gateway)
    return (
        {candidate_id: tuple(items) for candidate_id, items in by_candidate.items()},
        {
            "protocol": "issue-54-v1",
            "eligible_pair_count": len(eligible),
            "requested_pair_count": requested,
            "attributed_count": sum(item["status"] == "supported" for item in records),
            "unresolved_count": sum(item["status"] == "unresolved" for item in records),
            "skipped_count": sum(item["status"] == "skipped" for item in records),
            "reserved_cost_usd": spent_reserved,
            "measured_cost_usd": (
                max(0.0, round_cost_after - round_cost_before)
                if round_cost_after is not None and round_cost_before is not None
                else None
            ),
            "pair_cap": budget.pair_cap,
            "dollar_cap": budget.dollar_cap,
            "pairs": records,
        },
    )
