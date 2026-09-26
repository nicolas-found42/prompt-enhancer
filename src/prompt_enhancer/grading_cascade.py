"""Bounded confirmation of uncertain weak-panel grades.

The Gateway keeps raw answers. This module owns the caller-specific decision
table and records unresolved evidence without turning uncertainty into failure.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from . import jev_questions
from .diagnosis import split_sentences
from .evaluation.calibration import DecisionPolicy, runtime_question_identity
from .gateway import Gateway, ProviderError, completion_text, writer_messages
from .jev import JevResponseError, NoulDecision, parse_decision
from .runner import PanelResult


@dataclass(frozen=True, slots=True)
class CascadeBudget:
    pair_cap: int
    dollar_cap: float
    judge_reservation_usd: float = 0.001

    @classmethod
    def for_tier(
        cls,
        tier: str,
        *,
        pair_cap: int | None = None,
        dollar_cap: float | None = None,
        judge_reservation_usd: float = 0.001,
    ) -> CascadeBudget:
        from .models import Tier

        defaults = Tier.parse(tier).budget
        resolved = cls(
            pair_cap=defaults.grading_confirmation_pairs
            if pair_cap is None
            else pair_cap,
            dollar_cap=defaults.grading_cascade_dollars
            if dollar_cap is None
            else dollar_cap,
            judge_reservation_usd=judge_reservation_usd,
        )
        if (
            resolved.pair_cap < 0
            or not math.isfinite(resolved.dollar_cap)
            or resolved.dollar_cap < 0
            or not math.isfinite(resolved.judge_reservation_usd)
            or resolved.judge_reservation_usd < 0
        ):
            raise ValueError("grading cascade budgets must be non-negative")
        return resolved


def _probability(answer: Any) -> float | None:
    try:
        decision = parse_decision(answer)
    except JevResponseError:
        return None
    return decision.probability if isinstance(decision, NoulDecision) else None


def _source_spans(text: str) -> list[dict[str, Any]]:
    return [
        {
            "id": sentence.id,
            "start": sentence.start,
            "end": sentence.end,
            "text": sentence.text,
        }
        for sentence in split_sentences(text)
    ]


def _exact_check(criterion: str, output: str) -> dict[str, Any] | None:
    lowered = criterion.casefold()
    if "valid json" in lowered:
        try:
            json.loads(output)
        except json.JSONDecodeError:
            passed = False
        else:
            passed = True
        return {"kind": "valid_json", "passed": passed}
    word_limit = re.search(r"\bat (least|most) (\d+) words\b", lowered)
    if word_limit is not None:
        bound = int(word_limit.group(2))
        count = len(re.findall(r"\b\w+\b", output))
        return {
            "kind": "word_count",
            "operator": word_limit.group(1),
            "bound": bound,
            "observed": count,
            "passed": count >= bound
            if word_limit.group(1) == "least"
            else count <= bound,
        }
    return None


def _requires_exact_check(criterion: str) -> bool:
    lowered = criterion.casefold()
    numeric_constraint = re.search(
        r"\b\d+\s+(?:citations?|items?|words?|sentences?|characters?|steps?|examples?|sources?)\b",
        lowered,
    )
    return numeric_constraint is not None or any(
        word in lowered
        for word in ("execute", "compile", "syntax", "unit test", "run code")
    )


def _verdict(
    probabilities: Mapping[str, float | None],
    cutoffs: Mapping[str, float] | None = None,
) -> str:
    sufficient = probabilities.get("sufficient")
    meets = probabilities.get("meets")
    violation = probabilities.get("violation")
    if sufficient is None or meets is None or violation is None:
        return "unresolved"
    high = cutoffs or {name: 0.8 for name in ("sufficient", "meets", "violation")}
    epsilon = 1e-12
    if (
        sufficient >= high["sufficient"] - epsilon
        and meets >= high["meets"] - epsilon
        and violation <= 1.0 - high["violation"] + epsilon
    ):
        return "confirmed_pass"
    if (
        sufficient >= high["sufficient"] - epsilon
        and meets <= 1.0 - high["meets"] + epsilon
        and violation >= high["violation"] - epsilon
    ):
        return "confirmed_fail"
    return "unresolved"


def _catalog_rates(gateway: Gateway, model: str) -> tuple[float, float] | None:
    loader = getattr(gateway, "list_models", None)
    if not callable(loader):
        return None
    try:
        snapshot = loader()
        info = snapshot.get(model)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if info is None:
        return None
    input_rate = info.input_cost_per_token
    output_rate = info.output_cost_per_token
    if (
        input_rate is None
        or output_rate is None
        or not math.isfinite(input_rate)
        or not math.isfinite(output_rate)
        or input_rate < 0
        or output_rate < 0
    ):
        return None
    return input_rate, output_rate


def _reserve_cost(
    state: Any, rates: tuple[float, float], *, output_tokens: int
) -> float:
    input_tokens = math.ceil(
        len(json.dumps(state, ensure_ascii=False).encode("utf-8")) / 4
    )
    return max(0.00001, 2.0 * (input_tokens * rates[0] + output_tokens * rates[1]))


def _role_cost(gateway: Gateway, role: str) -> float | None:
    report = gateway.usage_report()
    costs = report.get("cost_by_role")
    if not isinstance(costs, Mapping):
        return None
    value = costs.get(role, 0.0)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return max(0.0, float(value))
    return None


def _retry_multiplier(gateway: Gateway) -> int:
    config = getattr(gateway, "config", None)
    if config is None:
        config = getattr(getattr(gateway, "gateway", None), "config", None)
    retries = getattr(config, "max_retries", 0)
    return max(1, retries + 1) if isinstance(retries, int) else 1


def _confirmation_policy(
    requests: Sequence[Mapping[str, Any]],
    gateway: Gateway,
    decision_policy: DecisionPolicy | None,
    *,
    log_start: int,
) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    cutoffs = {name: 0.8 for name in jev_questions.GRADING_CONFIRM_QUESTIONS}
    logged = gateway.decision_log[log_start:]
    if len(logged) != len(requests):
        return None, []
    evidence: list[dict[str, Any]] = []
    for request, entry in zip(requests, logged, strict=True):
        name = str(request["key"]).rsplit(":", 1)[-1]
        snapshot = entry.get("answered_by")
        if not isinstance(snapshot, str) or not snapshot:
            return None, evidence
        if decision_policy is None:
            evidence.append(
                {
                    "question_id": f"grade-confirm:{name}",
                    "snapshot": snapshot,
                    "policy": "provisional_default",
                }
            )
            continue
        question_id = f"grade-confirm:{name}"
        identity = runtime_question_identity(
            question_id,
            request,
            family="grading_confirmation",
            rubric_version="issue-49-v1",
            snapshot=snapshot,
            policy_version=decision_policy.policy_version,
        )
        policy = decision_policy.resolve(
            question_id, identity=identity, snapshot=snapshot
        )
        evidence.append(
            {
                "question_id": question_id,
                "identity": identity.to_dict(),
                "policy": asdict(policy),
            }
        )
        if policy.disposition in {"ranker", "abstain"}:
            return None, evidence
        if policy.may_gate and policy.threshold is not None:
            cutoffs[name] = policy.threshold
    return cutoffs, evidence


def _strong_evidence(
    raw: Any, run: PanelResult, criterion: str
) -> tuple[dict[str, str] | None, str]:
    try:
        value = json.loads(completion_text(raw))
    except (ValueError, TypeError):
        return None, "invalid_evidence_schema"
    if not isinstance(value, Mapping):
        return None, "invalid_evidence_schema"
    verdict = value.get("suggested_verdict")
    prompt_quote = value.get("prompt_quote")
    output_quote = value.get("output_quote")
    rationale = value.get("rationale")
    if (
        verdict not in {"pass", "fail"}
        or not isinstance(prompt_quote, str)
        or not isinstance(output_quote, str)
        or not isinstance(rationale, str)
        or not 0 < len(prompt_quote) <= 500
        or not 0 < len(output_quote) <= 500
        or len(rationale) > 500
    ):
        return None, "invalid_evidence_schema"
    if prompt_quote not in run.prompt or output_quote not in run.output:
        return None, "invalid_evidence_quote"
    exact = _exact_check(criterion, run.output)
    if exact is not None and (verdict == "pass") is not exact["passed"]:
        return None, "deterministic_evidence_conflict"
    if exact is None and _requires_exact_check(criterion):
        return None, "deterministic_check_unavailable"
    return {
        "suggested_verdict": verdict,
        "prompt_quote": prompt_quote,
        "output_quote": output_quote,
        "rationale": rationale,
    }, "valid_evidence"


def _verification_requests(
    *,
    pair_id: str,
    judge_model: str,
    state: Mapping[str, Any],
    criterion: str,
    suggested_verdict: str,
) -> list[dict[str, Any]]:
    return [
        {
            "key": f"grade-verify:{pair_id}:{name}",
            "model": judge_model,
            "type": "noul",
            "state": state,
            "question": question,
            "question_schema": {
                "criterion": criterion,
                "suggested_verdict": suggested_verdict,
            },
        }
        for name, question in jev_questions.GRADING_VERIFY_QUESTIONS.items()
    ]


def _verification_policy(
    requests: Sequence[Mapping[str, Any]],
    gateway: Gateway,
    decision_policy: DecisionPolicy | None,
) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    if decision_policy is None:
        return None, []
    cutoffs: dict[str, float] = {}
    evidence: list[dict[str, Any]] = []
    for request in requests:
        name = str(request["key"]).rsplit(":", 1)[-1]
        question_id = f"grade-verify:{name}"
        identity = runtime_question_identity(
            question_id,
            request,
            family="grading_verification",
            rubric_version="issue-49-v1",
            snapshot=gateway.jev_model,
            policy_version=decision_policy.policy_version,
        )
        policy = decision_policy.resolve(
            question_id, identity=identity, snapshot=gateway.jev_model
        )
        evidence.append(
            {
                "question_id": question_id,
                "identity": identity.to_dict(),
                "policy": asdict(policy),
            }
        )
        if not policy.may_gate or policy.threshold is None or policy.predicate:
            return None, evidence
        cutoffs[name] = policy.threshold
    return cutoffs, evidence


def resolve_uncertain_grades(
    panel: Sequence[PanelResult],
    tests: Sequence[Mapping[str, Any]],
    pair_scores: Mapping[tuple[int, int], float],
    gateway: Gateway,
    *,
    judge_model: str,
    run_id: str,
    budget: CascadeBudget,
    screen_statuses: Mapping[int, str],
    strong_model: str,
    decision_policy: DecisionPolicy | None = None,
    ineligible_pairs: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    uncertainty_bands: Mapping[tuple[int, int], tuple[float, float]] | None = None,
) -> tuple[
    dict[tuple[int, int], float], set[int], list[dict[str, Any]], dict[str, Any]
]:
    """Confirm only borderline pairs, ranked by distance to the pass boundary."""
    eligible = sorted(
        (
            (output_index, test_index)
            for output_index, test_index in pair_scores
            if (
                (uncertainty_bands or {}).get((output_index, test_index), (0.3, 0.7))[0]
                <= pair_scores[(output_index, test_index)]
                <= (uncertainty_bands or {}).get(
                    (output_index, test_index), (0.3, 0.7)
                )[1]
            )
            and screen_statuses.get(output_index) != "steering_detected"
            and (output_index, test_index) not in (ineligible_pairs or {})
        ),
        key=lambda pair: (
            abs(pair_scores[pair] - 0.5),
            f"{pair[0]:04d}:{pair[1]:04d}",
        ),
    )
    overrides: dict[tuple[int, int], float] = {}
    unresolved_outputs: set[int] = set()
    evidence: list[dict[str, Any]] = []
    for (output_index, test_index), calibration in (ineligible_pairs or {}).items():
        if screen_statuses.get(output_index) == "steering_detected":
            continue
        run = panel[output_index]
        unresolved_outputs.add(output_index)
        evidence.append(
            {
                "pair_id": f"{output_index:04d}:{test_index:04d}",
                "candidate_id": run.candidate_id,
                "model": run.model,
                "sample": run.sample,
                "test_id": str(tests[test_index].get("id", f"t{test_index}")),
                "initial_pass_probability": pair_scores[(output_index, test_index)],
                "stage": "initial",
                "status": "unresolved",
                "reason": "calibration_not_gate_capable",
                "calibration": dict(calibration),
            }
        )
    spent_reserved = 0.0
    measured_costs: dict[str, float] = {}
    retry_multiplier = _retry_multiplier(gateway)
    confirmations = 0
    escalations = 0
    verifications = 0
    for pair in eligible:
        output_index, test_index = pair
        run = panel[output_index]
        test = tests[test_index]
        pair_id = f"{output_index:04d}:{test_index:04d}"
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "candidate_id": run.candidate_id,
            "model": run.model,
            "sample": run.sample,
            "test_id": str(test.get("id", f"t{test_index}")),
            "initial_pass_probability": pair_scores[pair],
            "stage": "initial",
            "status": "unresolved",
            "reason": "confirmation_unavailable",
        }
        state = {
            "prompt": run.prompt,
            "output": run.output,
            "criterion": test.get("question", ""),
            "prompt_spans": _source_spans(run.prompt),
            "output_spans": _source_spans(run.output),
            "exact_check": _exact_check(str(test.get("question", "")), run.output),
        }
        judge_rates = (
            _catalog_rates(gateway, judge_model)
            if confirmations < budget.pair_cap
            and screen_statuses.get(output_index) != "screen_unresolved"
            else None
        )
        judge_reservation = max(
            budget.judge_reservation_usd,
            _reserve_cost(state, judge_rates, output_tokens=128)
            if judge_rates is not None
            else 0.0,
        )
        if screen_statuses.get(output_index) == "screen_unresolved":
            record["reason"] = "output_screen_unresolved"
        elif confirmations >= budget.pair_cap:
            record["reason"] = "pair_budget_exhausted"
        elif spent_reserved + judge_reservation * retry_multiplier > budget.dollar_cap:
            record["reason"] = "dollar_budget_exhausted"
        else:
            requests = [
                {
                    "key": f"grade-confirm:{pair_id}:{name}",
                    "model": judge_model,
                    "type": "noul",
                    "state": state,
                    "question": question,
                    "question_schema": {
                        "criterion": str(test.get("question", "")),
                        "test_id": str(test.get("id", f"t{test_index}")),
                    },
                }
                for name, question in jev_questions.GRADING_CONFIRM_QUESTIONS.items()
            ]
            confirmations += 1
            confirmation_reservation = judge_reservation * retry_multiplier
            spent_reserved += confirmation_reservation
            record["confirmation_reservation_usd"] = confirmation_reservation
            record["stage"] = "confirmation"
            record["questions"] = requests
            log_start = len(gateway.decision_log)
            cost_before = _role_cost(gateway, "judge_confirmation")
            try:
                answers = gateway.decide_batch(
                    requests, role="judge_confirmation", run_id=run_id
                )
            except ProviderError as exc:
                record["reason"] = f"confirmation_provider_{exc.kind}"
            else:
                record["answers"] = answers
                if len(answers) != len(requests):
                    record["reason"] = "confirmation_incomplete"
                else:
                    probabilities = {
                        name: _probability(answer)
                        for name, answer in zip(
                            jev_questions.GRADING_CONFIRM_QUESTIONS,
                            answers,
                            strict=True,
                        )
                    }
                    record["probabilities"] = probabilities
                    cutoffs, policies = _confirmation_policy(
                        requests, gateway, decision_policy, log_start=log_start
                    )
                    record["confirmation_policies"] = policies
                    if cutoffs is None:
                        record["reason"] = "confirmation_calibration_not_gate_capable"
                    else:
                        status = _verdict(probabilities, cutoffs)
                        exact = state["exact_check"]
                        deterministic_unavailable = (
                            exact is None
                            and _requires_exact_check(str(test.get("question", "")))
                        )
                        if deterministic_unavailable:
                            status = "unresolved"
                        if (
                            exact is not None
                            and status != "unresolved"
                            and (status == "confirmed_pass") is not exact["passed"]
                        ):
                            status = "unresolved"
                            record["deterministic_conflict"] = exact
                        record["confirmation_status"] = status
                        record["status"] = status
                        record["reason"] = (
                            "decisive_jev_confirmation"
                            if status != "unresolved"
                            else "deterministic_check_unavailable"
                            if deterministic_unavailable
                            else "confirmation_not_decisive"
                        )
                        if status != "unresolved":
                            overrides[pair] = 1.0 if status == "confirmed_pass" else 0.0
            cost_after = _role_cost(gateway, "judge_confirmation")
            if cost_before is not None and cost_after is not None:
                actual = max(0.0, cost_after - cost_before)
                measured_costs["judge_confirmation"] = (
                    measured_costs.get("judge_confirmation", 0.0) + actual
                )
                spent_reserved += max(0.0, actual - confirmation_reservation)
                record["confirmation_cost_usd_measured"] = actual
            if record["reason"] == "confirmation_not_decisive":
                rates = _catalog_rates(gateway, strong_model)
                if rates is None:
                    record["reason"] = "missing_trustworthy_fallback_pricing"
                else:
                    evidence_state = {
                        "prompt": run.prompt,
                        "output": run.output,
                        "criterion": str(test.get("question", "")),
                        "prompt_spans": _source_spans(run.prompt),
                        "output_spans": _source_spans(run.output),
                        "exact_check": state["exact_check"],
                    }
                    strong_reservation = _reserve_cost(
                        evidence_state, rates, output_tokens=512
                    )
                    combined_reservation = (
                        strong_reservation + judge_reservation
                    ) * retry_multiplier
                    if spent_reserved + combined_reservation > budget.dollar_cap:
                        record["reason"] = "fallback_dollar_budget_exhausted"
                    else:
                        spent_reserved += combined_reservation
                        escalations += 1
                        record["stage"] = "escalation"
                        record["escalation_reservation_usd"] = strong_reservation
                        strong_cost_before = _role_cost(gateway, "judge_escalation")
                        try:
                            raw_strong = gateway.chat(
                                strong_model,
                                writer_messages(
                                    "Return JSON only with suggested_verdict (pass or fail), "
                                    "prompt_quote, output_quote, and rationale. Quotes must "
                                    "be exact substrings of the provided prompt and output. "
                                    "Provide evidence only; Jev makes the final decision.",
                                    evidence_state,
                                ),
                                role="judge_escalation",
                                run_id=run_id,
                                max_tokens=512,
                            )
                        except ProviderError as exc:
                            record["reason"] = f"escalation_provider_{exc.kind}"
                        else:
                            record["strong_answer"] = raw_strong
                            proposed, reason = _strong_evidence(
                                raw_strong, run, str(test.get("question", ""))
                            )
                            record["reason"] = reason
                            if proposed is not None:
                                verification_state = {
                                    **evidence_state,
                                    "proposed_evidence": proposed,
                                }
                                verification_requests = _verification_requests(
                                    pair_id=pair_id,
                                    judge_model=judge_model,
                                    state=verification_state,
                                    criterion=str(test.get("question", "")),
                                    suggested_verdict=proposed["suggested_verdict"],
                                )
                                cutoffs, policies = _verification_policy(
                                    verification_requests, gateway, decision_policy
                                )
                                record["verification_policies"] = policies
                                if cutoffs is None:
                                    record["reason"] = "missing_gate_calibration"
                                else:
                                    record["stage"] = "verification"
                                    record["verification_questions"] = (
                                        verification_requests
                                    )
                                    verifications += 1
                                    verify_log_start = len(gateway.decision_log)
                                    verify_cost_before = _role_cost(
                                        gateway, "judge_verification"
                                    )
                                    try:
                                        verification_answers = gateway.decide_batch(
                                            verification_requests,
                                            role="judge_verification",
                                            run_id=run_id,
                                        )
                                    except ProviderError as exc:
                                        record["reason"] = (
                                            f"verification_provider_{exc.kind}"
                                        )
                                    else:
                                        record["verification_answers"] = (
                                            verification_answers
                                        )
                                        if len(verification_answers) != len(
                                            verification_requests
                                        ):
                                            record["reason"] = "verification_incomplete"
                                        else:
                                            verified_probabilities = {
                                                name: _probability(answer)
                                                for name, answer in zip(
                                                    jev_questions.GRADING_VERIFY_QUESTIONS,
                                                    verification_answers,
                                                    strict=True,
                                                )
                                            }
                                            record["verification_probabilities"] = (
                                                verified_probabilities
                                            )
                                            support = verified_probabilities["support"]
                                            verdict = _verdict(
                                                verified_probabilities, cutoffs
                                            )
                                            proposed_status = (
                                                "confirmed_pass"
                                                if proposed["suggested_verdict"]
                                                == "pass"
                                                else "confirmed_fail"
                                            )
                                            if (
                                                support is not None
                                                and support >= cutoffs["support"]
                                                and verdict == proposed_status
                                            ):
                                                observed = gateway.decision_log[
                                                    verify_log_start:
                                                ]
                                                if len(observed) == len(
                                                    verification_requests
                                                ) and all(
                                                    entry.get("answered_by")
                                                    == gateway.jev_model
                                                    for entry in observed
                                                ):
                                                    record["status"] = verdict
                                                    record["reason"] = (
                                                        "verified_fallback_evidence"
                                                    )
                                                    overrides[pair] = (
                                                        1.0
                                                        if verdict == "confirmed_pass"
                                                        else 0.0
                                                    )
                                                else:
                                                    record["reason"] = (
                                                        "verification_snapshot_mismatch"
                                                    )
                                            else:
                                                record["reason"] = (
                                                    "verification_not_decisive_or_consistent"
                                                )
                                    verify_cost_after = _role_cost(
                                        gateway, "judge_verification"
                                    )
                                    if (
                                        verify_cost_before is not None
                                        and verify_cost_after is not None
                                    ):
                                        actual = max(
                                            0.0, verify_cost_after - verify_cost_before
                                        )
                                        measured_costs["judge_verification"] = (
                                            measured_costs.get(
                                                "judge_verification", 0.0
                                            )
                                            + actual
                                        )
                                        spent_reserved += max(
                                            0.0,
                                            actual
                                            - judge_reservation * retry_multiplier,
                                        )
                                        record["verification_cost_usd_measured"] = (
                                            actual
                                        )
                        strong_cost_after = _role_cost(gateway, "judge_escalation")
                        if (
                            strong_cost_before is not None
                            and strong_cost_after is not None
                        ):
                            actual = max(0.0, strong_cost_after - strong_cost_before)
                            measured_costs["judge_escalation"] = (
                                measured_costs.get("judge_escalation", 0.0) + actual
                            )
                            spent_reserved += max(
                                0.0, actual - strong_reservation * retry_multiplier
                            )
                            record["escalation_cost_usd_measured"] = actual
        if record["status"] == "unresolved":
            unresolved_outputs.add(output_index)
        evidence.append(record)
    report = {
        "protocol": "issue-49-grading-cascade-v1",
        "eligible_pair_count": len(eligible),
        "confirmation_count": confirmations,
        "confirmed_pass_count": sum(
            item["status"] == "confirmed_pass" for item in evidence
        ),
        "confirmed_fail_count": sum(
            item["status"] == "confirmed_fail" for item in evidence
        ),
        "unresolved_count": sum(item["status"] == "unresolved" for item in evidence),
        "escalation_count": escalations,
        "verification_count": verifications,
        "reserved_cost_usd": spent_reserved,
        "measured_cost_by_role_usd": measured_costs,
        "retry_reservation_multiplier": retry_multiplier,
        "pair_cap": budget.pair_cap,
        "dollar_cap": budget.dollar_cap,
        "pairs": evidence,
    }
    return overrides, unresolved_outputs, evidence, report
