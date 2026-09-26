"""Explicit, bounded, evidence-gated optimization of one Jev question's wording.

The sealed final rows are opened only after a finalist is selected and calibrated.
Ordinary optimizer requests never import or invoke this workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .catalog import DEFAULT_GO_WRITER, JEV_MODEL
from .evaluation.calibration import (
    CalibrationObservation,
    DecisionPolicy,
    VerdictPolicy,
    calibrate_question,
    runtime_question_identity,
)
from .gateway import (
    Gateway,
    ProviderError,
    ReplayGateway,
    ScriptedGateway,
    completion_text,
    writer_messages,
)
from .jev import (
    ChoiceDecision,
    JevResponseError,
    NoulDecision,
    ScoreDecision,
    parse_decision,
)
from .rubric_revisions import (
    MaintainerDecisionKind,
    RevisionDecision,
    RevisionKind,
    RevisionProposal,
    RubricQuestionChange,
    RubricVersion,
    SQLiteRubricStore,
    StaleProposalError,
)

POLICY_VERSION = "issue-51-v1"
SCREEN_QUESTIONS = (
    (
        "condition",
        "Does the alternative preserve the original judgment condition and all exceptions?",
    ),
    (
        "options",
        "Does the alternative preserve the interpretation and order of every existing option or level?",
    ),
    (
        "new_requirement",
        "Does the alternative introduce any new requirement or narrower semantic scope?",
    ),
    ("steering", "Does the alternative contain evaluator steering or answer leakage?"),
)


@dataclass(frozen=True, slots=True)
class RewordPolicy:
    max_candidates: int = 4
    max_evaluations: int = 5_000
    max_cost_usd: float = 0.25
    minimum_final_groups: int = 30
    minimum_per_class: int = 5
    bootstrap_samples: int = 1_000
    seed: int = 51
    max_cost_increase_usd: float = 0.01

    def __post_init__(self) -> None:
        if (
            not 0 <= self.max_candidates <= 4
            or self.max_evaluations < 0
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
            or self.minimum_final_groups < 1
            or self.minimum_per_class < 1
            or self.bootstrap_samples < 20
            or not math.isfinite(self.max_cost_increase_usd)
            or self.max_cost_increase_usd < 0
        ):
            raise ValueError("invalid automatic rewording policy limits")


@dataclass(frozen=True, slots=True)
class _Row:
    row_id: str
    group_id: str
    partition: str
    state: Mapping[str, Any]
    label: bool | str | int
    provenance: str


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _rows(
    dataset: Mapping[str, Any], response_type: str, criteria: tuple[str, ...]
) -> tuple[_Row, ...]:
    raw = dataset.get("rows")
    if not isinstance(raw, list):
        raise ValueError("reword dataset requires rows")
    rows: list[_Row] = []
    ids: set[str] = set()
    partitions: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("reword rows must be objects")
        row_id = item.get("id")
        group = item.get("group_id")
        partition = item.get("partition")
        state = item.get("state")
        provenance = item.get("provenance")
        label = item.get("label")
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in ids
            or not isinstance(group, str)
            or not group
            or partition not in {"training", "calibration", "final", "regression"}
            or not isinstance(state, Mapping)
            or provenance
            not in {"human", "source", "deterministic_synthetic", "weak", "delegated"}
        ):
            raise ValueError(
                "invalid reword row identity, partition, state, or provenance"
            )
        if group in partitions and partitions[group] != partition:
            raise ValueError("source group crosses reword partitions")
        if response_type == "noul":
            if not isinstance(label, bool):
                raise ValueError("Noul labels must be booleans")
        elif response_type == "choice":
            if not isinstance(label, str) or label not in criteria:
                raise ValueError("Choice labels must name an existing option")
        elif response_type == "score":
            if (
                not isinstance(label, int)
                or isinstance(label, bool)
                or not 0 <= label < len(criteria)
            ):
                raise ValueError("Score labels must index an existing level")
        else:
            raise ValueError("unsupported reword question shape")
        ids.add(row_id)
        partitions[group] = partition
        rows.append(_Row(row_id, group, partition, dict(state), label, str(provenance)))
    return tuple(rows)


def _shape(dataset: Mapping[str, Any], response_type: str) -> tuple[str, ...]:
    raw = dataset.get("criteria", ())
    if response_type == "noul":
        if raw not in ((), []):
            raise ValueError("Noul rewording cannot change criteria")
        return ()
    if (
        not isinstance(raw, list)
        or len(raw) < 2
        or any(not isinstance(item, str) or not item for item in raw)
    ):
        raise ValueError("Choice and Score rewording require ordered criteria")
    if len(set(raw)) != len(raw):
        raise ValueError("criteria must be unique")
    return tuple(raw)


def _probabilities(
    raw: Any, response_type: str, criteria: tuple[str, ...]
) -> tuple[float, ...] | None:
    try:
        decision = parse_decision(raw)
    except JevResponseError:
        return None
    if response_type == "noul" and isinstance(decision, NoulDecision):
        return (1 - decision.probability, decision.probability)
    if response_type == "choice" and isinstance(decision, ChoiceDecision):
        probabilities = decision.probabilities
        if set(probabilities) == set(criteria):
            return tuple(probabilities[item] for item in criteria)
    if response_type == "score" and isinstance(decision, ScoreDecision):
        probabilities = decision.probabilities
        if set(probabilities) == set(str(index) for index in range(len(criteria))):
            return tuple(probabilities[str(index)] for index in range(len(criteria)))
    return None


def _loss(
    probabilities: tuple[float, ...],
    label: bool | str | int,
    response_type: str,
    criteria: tuple[str, ...],
) -> float:
    if response_type == "noul":
        return (probabilities[1] - float(label)) ** 2
    index = criteria.index(label) if response_type == "choice" else int(label)
    if response_type == "choice":
        return sum(
            (probability - float(position == index)) ** 2
            for position, probability in enumerate(probabilities)
        ) / len(probabilities)
    return sum(
        (sum(probabilities[level:]) - float(index >= level)) ** 2
        for level in range(1, len(criteria))
    ) / (len(criteria) - 1)


def _classification(
    probabilities: tuple[float, ...],
    response_type: str,
    threshold: float,
    criteria: tuple[str, ...],
) -> bool | str | int:
    if response_type == "noul":
        return probabilities[1] >= threshold
    selected = max(
        range(len(probabilities)), key=lambda index: (probabilities[index], -index)
    )
    return criteria[selected] if response_type == "choice" else selected


def _precision_recall(
    rows: Sequence[_Row],
    predictions: Mapping[str, tuple[float, ...]],
    response_type: str,
    criteria: tuple[str, ...],
    threshold: float,
) -> tuple[float, float]:
    labels: tuple[bool | str | int, ...] = (
        (False, True)
        if response_type == "noul"
        else criteria
        if response_type == "choice"
        else tuple(range(len(criteria)))
    )
    precision: list[float] = []
    recall: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            predicted = _classification(
                predictions[row.row_id], response_type, threshold, criteria
            )
            tp += predicted == label and row.label == label
            fp += predicted == label and row.label != label
            fn += predicted != label and row.label == label
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
    return sum(precision) / len(labels), sum(recall) / len(labels)


def _bootstrap_lower(
    group_improvements: Mapping[str, float], seed: int, samples: int
) -> float:
    values = [group_improvements[key] for key in sorted(group_improvements)]
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)
    )
    return estimates[max(0, int(0.05 * samples) - 1)]


def _stability_floor(
    repeats: object,
    rows: Sequence[_Row],
    predictions: Mapping[str, Mapping[str, tuple[float, ...]]],
    texts: tuple[str, str],
    *,
    response_type: str,
    criteria: tuple[str, ...],
    missing_when: str,
    snapshot: str,
) -> tuple[float, str]:
    if repeats is None:
        return 0.0, "unavailable_no_paired_repeats"
    if not isinstance(repeats, Mapping):
        raise RuntimeError("paired repeat evidence must be an object")
    floors: list[float] = []
    for text in texts:
        by_row = repeats.get(_digest(text))
        if not isinstance(by_row, Mapping):
            raise RuntimeError("paired repeat evidence is incomplete")
        deltas: list[float] = []
        for row in rows:
            entry = by_row.get(row.row_id)
            if not isinstance(entry, Mapping) or entry.get("snapshot") != snapshot:
                raise RuntimeError("paired repeat snapshot or row is missing")
            probabilities = _probabilities(
                entry.get("raw_answer"), response_type, criteria
            )
            if probabilities is None:
                raise RuntimeError("paired repeat answer is malformed")
            if response_type == "noul" and missing_when == "no":
                probabilities = (probabilities[1], probabilities[0])
            baseline_loss = _loss(
                predictions[text][row.row_id], row.label, response_type, criteria
            )
            repeat_loss = _loss(probabilities, row.label, response_type, criteria)
            deltas.append(abs(baseline_loss - repeat_loss))
        floors.append(sum(deltas) / len(deltas))
    return max(floors), "paired_repeated_predictions"


class _Budget:
    def __init__(self, gateway: Gateway, policy: RewordPolicy) -> None:
        self.gateway = gateway
        self.policy = policy
        self.evaluations = 0
        self.reserved_usd = 0.0
        self.offline = isinstance(gateway, (ScriptedGateway, ReplayGateway))
        self.catalog = None
        self.initial_measured = self._role_cost()
        if not self.offline:
            loader = getattr(gateway, "list_models", None)
            try:
                self.catalog = loader() if callable(loader) else None
            except (RuntimeError, ProviderError, TypeError, ValueError):
                self.catalog = None

    def _role_cost(self) -> float:
        roles = self.gateway.usage_report().get("cost_by_role", {})
        if not isinstance(roles, Mapping):
            return 0.0
        return sum(
            float(roles.get(name, 0.0))
            for name in ("writer_reword", "judge_reword_screen", "judge_reword_eval")
            if isinstance(roles.get(name, 0.0), (int, float))
        )

    def measured(self) -> float:
        return max(0.0, self._role_cost() - self.initial_measured)

    def estimate(self, model: str, payload: Any) -> float:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode()) > 96_000:
            raise RuntimeError("request size limit reached")
        cost = 0.0
        if not self.offline:
            info = self.catalog.get(model) if self.catalog is not None else None
            if (
                info is None
                or info.input_cost_per_token is None
                or info.output_cost_per_token is None
                or not math.isfinite(info.input_cost_per_token)
                or not math.isfinite(info.output_cost_per_token)
                or info.input_cost_per_token < 0
                or info.output_cost_per_token < 0
            ):
                raise RuntimeError("missing trustworthy pricing")
            config = getattr(self.gateway, "config", None)
            if config is None:
                config = getattr(getattr(self.gateway, "gateway", None), "config", None)
            retries = getattr(config, "max_retries", 0)
            multiplier = max(1, retries + 1) if isinstance(retries, int) else 1
            cost = (
                2
                * multiplier
                * (
                    math.ceil(len(encoded.encode()) / 4) * info.input_cost_per_token
                    + 256 * info.output_cost_per_token
                )
            )
        return cost

    def reserve(self, model: str, payload: Any, *, evaluation_count: int = 0) -> float:
        if self.evaluations + evaluation_count > self.policy.max_evaluations:
            raise RuntimeError("metric evaluation budget exhausted")
        if self.measured() > self.policy.max_cost_usd:
            raise RuntimeError("measured dollar budget exhausted")
        cost = self.estimate(model, payload)
        if self.reserved_usd + cost > self.policy.max_cost_usd:
            raise RuntimeError("dollar budget exhausted")
        self.reserved_usd += cost
        self.evaluations += evaluation_count
        return cost


def optimize_reword(
    store: SQLiteRubricStore,
    gateway: Gateway,
    question_id: str,
    dataset: Mapping[str, Any],
    *,
    attempt_id: str,
    policy: RewordPolicy | None = None,
    writer_model: str = DEFAULT_GO_WRITER,
    judge_model: str = JEV_MODEL,
    decision_policy: DecisionPolicy | None = None,
) -> dict[str, Any]:
    """Propose once, screen, select on training, calibrate, then open final labels."""
    prior = store.get_reword_attempt(attempt_id)
    if prior is not None:
        return dict(prior)
    policy = policy or RewordPolicy()
    rubric = store.active_rubric()
    question = rubric.question(question_id)
    if question is None:
        raise ValueError("reword target must be an active question")
    criteria = _shape(dataset, question.response_type)
    rows = _rows(dataset, question.response_type, criteria)
    partitions = {
        name: tuple(row for row in rows if row.partition == name)
        for name in ("training", "calibration", "final", "regression")
    }
    final_digest = _digest(
        [asdict(row) for row in sorted(partitions["final"], key=lambda row: row.row_id)]
    )
    final_groups = tuple(sorted({row.group_id for row in partitions["final"]}))
    report: dict[str, Any] = {
        "attempt_id": attempt_id,
        "actor": "automatic_policy",
        "policy_version": POLICY_VERSION,
        "base_version_id": rubric.version_id,
        "question_id": question_id,
        "question_type": question.response_type,
        "answering_snapshot": gateway.jev_model,
        "holdout_digest": final_digest,
        "partitions": {
            name: {
                "rows": len(group),
                "groups": len({row.group_id for row in group}),
                "provenance": dict(Counter(row.provenance for row in group)),
            }
            for name, group in partitions.items()
        },
        "gates": {},
        "status": "hold",
    }
    consumed = False
    adopted: RubricVersion | None = None
    automated_decision: RevisionDecision | None = None

    def finish(reason: str) -> dict[str, Any]:
        report["reason"] = reason
        report["budget"] = {
            "metric_evaluations": budget.evaluations,
            "reserved_usd": budget.reserved_usd,
            "measured_usd": budget.measured(),
        }
        if adopted is not None and budget.measured() > policy.max_cost_usd:
            report["status"] = "hold"
            report["reason"] = "measured dollar budget exhausted"
            report.pop("adopted_version_id", None)
            adopted_rubric = None
            persisted_decision = None
        else:
            adopted_rubric = adopted
            persisted_decision = automated_decision
        store.record_reword_attempt(
            attempt_id,
            final_digest,
            report,
            holdout_groups=final_groups,
            adopted_rubric=adopted_rubric,
            automated_decision=persisted_decision,
            consume_holdout=consumed,
        )
        return report

    budget = _Budget(gateway, policy)
    if gateway.jev_model != JEV_MODEL:
        return finish("answering snapshot is not the verified pin")
    if store.holdout_consumed(final_digest, final_groups):
        return finish("validation budget exhausted")
    if (
        not partitions["training"]
        or not partitions["calibration"]
        or not partitions["final"]
    ):
        return finish("training, calibration, and sealed final rows are required")
    if any(row.provenance in {"weak", "delegated"} for row in partitions["final"]):
        return finish("insufficient independent evidence")
    training_state = {
        "question": asdict(question),
        "criteria": list(criteria),
        "training_examples": [asdict(row) for row in partitions["training"]],
    }
    instructions = (
        "Reword this Jev question without changing its judgment, exceptions, option identities, "
        'option order, or scope. Return JSON only as {"alternatives":["..."]} with at most four alternatives. '
        "Use only the supplied training examples."
    )
    try:
        budget.reserve(writer_model, training_state)
        raw_proposal = gateway.chat(
            writer_model,
            writer_messages(instructions, training_state),
            role="writer_reword",
        )
        proposed = json.loads(completion_text(raw_proposal))
        alternatives = (
            proposed.get("alternatives") if isinstance(proposed, Mapping) else None
        )
        if (
            not isinstance(alternatives, list)
            or len(alternatives) > policy.max_candidates
        ):
            return finish("invalid or excessive writer alternatives")
        texts = [
            item.strip()
            for item in alternatives
            if isinstance(item, str) and item.strip()
        ]
        if (
            len(texts) != len(alternatives)
            or len(set(texts)) != len(texts)
            or question.text in texts
        ):
            return finish("invalid or duplicate writer alternatives")
        report["proposal"] = {"raw": raw_proposal, "alternatives": texts}
        screened: list[str] = []
        screen_records: list[dict[str, Any]] = []
        report["semantic_screen"] = screen_records
        explicitly_drifted = 0
        for text in texts:
            state = {
                "original": question.text,
                "alternative": text,
                "criteria": list(criteria),
                "training_contrasts": [
                    asdict(row) for row in partitions["training"][:12]
                ],
            }
            requests = [
                {
                    "key": f"reword-screen:{_digest(text)[:12]}:{key}",
                    "model": judge_model,
                    "type": "noul",
                    "state": state,
                    "question": wording,
                }
                for key, wording in SCREEN_QUESTIONS
            ]
            budget.reserve(judge_model, requests)
            answers = gateway.decide_batch(requests, role="judge_reword_screen")
            parsed = [_probabilities(item, "noul", ()) for item in answers]
            values = [item[1] if item is not None else None for item in parsed]
            snapshot = [
                item.get("answered_by")
                for item in gateway.decision_log[-len(requests) :]
            ]
            gates: list[bool] = []
            policy_evidence: list[dict[str, Any]] = []
            for (name, _), request, raw, probability, served in zip(
                SCREEN_QUESTIONS, requests, answers, values, snapshot, strict=True
            ):
                preserving = name in {"condition", "options"}
                safe_probability = (
                    probability
                    if preserving
                    else 1 - probability
                    if probability is not None
                    else None
                )
                threshold = 0.9
                allowed = True
                resolved = None
                if (
                    decision_policy is not None
                    and probability is not None
                    and served == gateway.jev_model
                ):
                    identity = runtime_question_identity(
                        f"reword_screen:{name}",
                        request,
                        family="reword_screen",
                        rubric_version=POLICY_VERSION,
                        snapshot=served,
                        policy_version=decision_policy.policy_version,
                    )
                    identity = replace(
                        identity,
                        event_mapping={
                            "polarity": "positive" if preserving else "negative"
                        },
                    )
                    resolved = decision_policy.apply(
                        question_id=f"reword_screen:{name}",
                        identity=identity,
                        decision=parse_decision(raw),
                        raw_answer=raw,
                        snapshot=served,
                    )
                    allowed = resolved.is_legacy or resolved.may_gate
                    if resolved.threshold is not None:
                        threshold = resolved.threshold
                gates.append(
                    allowed
                    and safe_probability is not None
                    and safe_probability >= threshold
                    and served == gateway.jev_model
                )
                policy_evidence.append(
                    {
                        "name": name,
                        "safe_probability": safe_probability,
                        "threshold": threshold,
                        "allowed": allowed,
                        "policy": asdict(resolved) if resolved is not None else None,
                    }
                )
            passed = len(gates) == 4 and all(gates)
            explicit_drift = len(values) == 4 and any(
                value is not None and (value <= 0.1 if index < 2 else value >= 0.9)
                for index, value in enumerate(values)
            )
            explicitly_drifted += explicit_drift
            screen_records.append(
                {
                    "text": text,
                    "raw_answers": answers,
                    "probabilities": values,
                    "snapshots": snapshot,
                    "policy": policy_evidence,
                    "passed": passed,
                    "explicit_drift": explicit_drift,
                }
            )
            if passed:
                screened.append(text)
        report["semantic_screen"] = screen_records
        report["gates"]["semantic_equivalence"] = bool(screened)
        if not screened:
            report["status"] = (
                "reject" if texts and explicitly_drifted == len(texts) else "hold"
            )
            return finish("semantic screening held or rejected all alternatives")
        cache: dict[str, tuple[tuple[float, ...], Any, str]] = {}
        raw_evaluations: list[dict[str, Any]] = []
        report["evaluations"] = raw_evaluations
        evaluation_costs: dict[str, float] = defaultdict(float)

        def evaluate(
            text: str, selected_rows: Sequence[_Row]
        ) -> dict[str, tuple[float, ...]]:
            def event_probabilities(raw: Any) -> tuple[float, ...] | None:
                probabilities = _probabilities(raw, question.response_type, criteria)
                if (
                    probabilities is not None
                    and question.response_type == "noul"
                    and question.missing_when == "no"
                ):
                    return (probabilities[1], probabilities[0])
                return probabilities

            result: dict[str, tuple[float, ...]] = {}
            for row in selected_rows:
                key = _digest(
                    (
                        text,
                        row.state,
                        question.response_type,
                        criteria,
                        gateway.jev_model,
                        POLICY_VERSION,
                        question.question_version + 1,
                    )
                )
                request = {
                    "key": f"reword-eval:{key[:16]}",
                    "model": judge_model,
                    "type": question.response_type,
                    "state": dict(row.state),
                    "question": text,
                    "question_schema": {
                        "version": question.question_version + 1,
                        "policy": POLICY_VERSION,
                    },
                }
                if question.response_type == "choice":
                    request["options"] = list(criteria)
                elif question.response_type == "score":
                    request["levels"] = list(criteria)
                evaluation_costs[text] += budget.estimate(judge_model, request)
                cache_hit = key in cache
                if key not in cache:
                    persisted = store.get_reword_evaluation(key)
                    if persisted is not None:
                        raw = persisted.get("raw_answer")
                        snapshot = persisted.get("snapshot")
                        probabilities = event_probabilities(raw)
                        if snapshot != gateway.jev_model or probabilities is None:
                            raise RuntimeError(
                                "cached evaluation has mismatched provenance"
                            )
                        cache[key] = (probabilities, raw, snapshot)
                        cache_hit = True
                    else:
                        budget.reserve(judge_model, request, evaluation_count=1)
                        raw = gateway.decide(request, role="judge_reword_eval")
                        snapshot = gateway.decision_log[-1].get("answered_by")
                        probabilities = event_probabilities(raw)
                        if snapshot != gateway.jev_model or probabilities is None:
                            raise RuntimeError(
                                "missing, malformed, or mismatched answering snapshot"
                            )
                        cache[key] = (probabilities, raw, snapshot)
                        store.cache_reword_evaluation(
                            key,
                            {"raw_answer": raw, "snapshot": snapshot},
                        )
                probabilities, raw, snapshot = cache[key]
                raw_evaluations.append(
                    {
                        "row_id": row.row_id,
                        "text_digest": _digest(text),
                        "request": request,
                        "raw_answer": raw,
                        "snapshot": snapshot,
                        "cache_hit": cache_hit,
                    }
                )
                result[row.row_id] = probabilities
            return result

        training_predictions = {
            text: evaluate(text, partitions["training"])
            for text in (question.text, *screened)
        }
        # Group folds are fixed before the final set is opened. The held-out fold
        # receives no fitted threshold or label information from another fold.
        groups = sorted({row.group_id for row in partitions["training"]})
        folds = {
            group: int(_digest((policy.seed, group))[:8], 16) % 5 for group in groups
        }
        training_loss = {
            text: sum(
                sum(
                    _loss(
                        predictions[row.row_id],
                        row.label,
                        question.response_type,
                        criteria,
                    )
                    for row in partitions["training"]
                    if folds[row.group_id] == fold
                )
                for fold in range(5)
            )
            / len(partitions["training"])
            for text, predictions in training_predictions.items()
        }
        finalist = min(
            training_loss,
            key=lambda text: (
                training_loss[text],
                text != question.text,
                len(text),
                _digest(text),
            ),
        )
        report["training"] = {
            "group_folds": folds,
            "brier": {_digest(text): value for text, value in training_loss.items()},
            "finalist_digest": _digest(finalist),
        }
        if finalist == question.text:
            return finish("unchanged baseline selected on training")
        for text in (question.text, finalist):
            evaluate(text, partitions["calibration"])
        if question.response_type != "noul":
            return finish(
                "nonbinary calibration mapping is unavailable for automatic adoption"
            )
        future_version_id = f"rubric-reword-{attempt_id}"
        identity = runtime_question_identity(
            f"rubric:{question_id}",
            {"type": "noul", "query": finalist},
            family="rubric",
            rubric_version=future_version_id,
            snapshot=gateway.jev_model,
            policy_version=POLICY_VERSION,
        )
        identity = replace(
            identity,
            event_mapping={
                "polarity": "positive" if question.missing_when == "yes" else "negative"
            },
        )
        raw_by_row = {
            item["row_id"]: item["raw_answer"]
            for item in raw_evaluations
            if item["text_digest"] == _digest(finalist)
        }
        observations = [
            CalibrationObservation(
                event_id=f"reword:{attempt_id}:{row.row_id}",
                source_group=row.group_id,
                example_id=row.row_id,
                identity=identity,
                label=row.label,
                raw_answer=raw_by_row[row.row_id],
                provenance=row.provenance,
                answering_snapshot=gateway.jev_model,
                state=row.state,
            )
            for row in partitions["calibration"]
        ]
        verdict_policy = VerdictPolicy(
            policy_version=POLICY_VERSION,
            require_control=False,
            require_repeats=False,
            require_brier_better_than_control=False,
        )
        calibrated = calibrate_question(
            identity,
            observations,
            verdict_policy=verdict_policy,
            seed=policy.seed,
            bootstrap_seed=policy.seed,
            bootstrap_resamples=100,
            name=f"reword:{question_id}:{attempt_id}",
        )
        report["calibration"] = {
            **calibrated.to_dict(),
            "verdict_policy": verdict_policy.to_dict(),
        }
        report["gates"]["calibrated_gate"] = calibrated.verdict in {
            "gate",
            "gate-above-confidence",
        }
        threshold = calibrated.threshold
        if not report["gates"]["calibrated_gate"] or threshold is None:
            return finish("calibration did not approve a gate")
        final_rows = partitions["final"]
        classes = Counter(row.label for row in final_rows)
        support = len(
            {row.group_id for row in final_rows}
        ) >= policy.minimum_final_groups and all(
            classes[label] >= policy.minimum_per_class for label in (False, True)
        )
        report["gates"]["independent_final_support"] = support
        if not support:
            return finish("insufficient independent final support")
        consumed = True
        final_predictions = {
            text: evaluate(text, final_rows) for text in (question.text, finalist)
        }
        group_deltas: dict[str, list[float]] = defaultdict(list)
        baseline_losses: list[float] = []
        candidate_losses: list[float] = []
        for row in final_rows:
            base_loss = _loss(
                final_predictions[question.text][row.row_id],
                row.label,
                question.response_type,
                criteria,
            )
            candidate_loss = _loss(
                final_predictions[finalist][row.row_id],
                row.label,
                question.response_type,
                criteria,
            )
            baseline_losses.append(base_loss)
            candidate_losses.append(candidate_loss)
            group_deltas[row.group_id].append(base_loss - candidate_loss)
        lower = _bootstrap_lower(
            {
                group: sum(values) / len(values)
                for group, values in group_deltas.items()
            },
            policy.seed,
            policy.bootstrap_samples,
        )
        base_precision, base_recall = _precision_recall(
            final_rows,
            final_predictions[question.text],
            question.response_type,
            criteria,
            question.threshold,
        )
        candidate_precision, candidate_recall = _precision_recall(
            final_rows,
            final_predictions[finalist],
            question.response_type,
            criteria,
            threshold or 0.5,
        )
        regression_predictions = {
            text: evaluate(text, partitions["regression"])
            for text in (question.text, finalist)
        }
        additional_regressions = [
            row.row_id
            for row in partitions["regression"]
            if _classification(
                regression_predictions[question.text][row.row_id],
                question.response_type,
                question.threshold,
                criteria,
            )
            == row.label
            and _classification(
                regression_predictions[finalist][row.row_id],
                question.response_type,
                threshold or 0.5,
                criteria,
            )
            != row.label
        ]
        noise_floor, noise_provenance = _stability_floor(
            dataset.get("repeat_answers"),
            final_rows,
            final_predictions,
            (question.text, finalist),
            response_type=question.response_type,
            criteria=criteria,
            missing_when=question.missing_when,
            snapshot=gateway.jev_model,
        )
        report["final_validation"] = {
            "lower_bound_improvement_brier": lower,
            "baseline_brier": sum(baseline_losses) / len(baseline_losses),
            "candidate_brier": sum(candidate_losses) / len(candidate_losses),
            "noise_floor_brier": noise_floor,
            "noise_floor_provenance": noise_provenance,
            "baseline_precision": base_precision,
            "baseline_recall": base_recall,
            "candidate_precision": candidate_precision,
            "candidate_recall": candidate_recall,
            "additional_regression_ids": additional_regressions,
            "group_count": len(group_deltas),
            "label_provenance": dict(Counter(row.provenance for row in final_rows)),
            "evaluation_cost_increase_usd": evaluation_costs[finalist]
            - evaluation_costs[question.text],
        }
        report["gates"].update(
            {
                "paired_brier_improvement": lower > max(0.01, noise_floor),
                "stability_evidence": noise_provenance == "paired_repeated_predictions",
                "precision": candidate_precision >= base_precision,
                "recall": candidate_recall >= base_recall,
                "regression_fixtures": not additional_regressions,
                "cost": evaluation_costs[finalist] - evaluation_costs[question.text]
                <= policy.max_cost_increase_usd,
            }
        )
        report["evaluations"] = raw_evaluations
        if not all(report["gates"].values()):
            return finish("final validation gates did not all pass")
        changed = replace(
            question,
            text=finalist,
            threshold=threshold or 0.5,
            question_version=question.question_version + 1,
            calibration_snapshot=gateway.jev_model,
            calibration_policy_version=POLICY_VERSION,
            calibration_artifact=calibrated.artifact().to_dict(),
        )
        change = RubricQuestionChange(
            RevisionKind.REWORD,
            question_id,
            question,
            changed,
            "automatic held-out rewording",
        )
        adopted = rubric.apply(
            change, version_id=f"rubric-reword-{attempt_id}", created_at="automatic"
        )
        report["status"] = "adopted"
        report["adopted_version_id"] = adopted.version_id
        report["baseline_hash"] = _digest(asdict(question))
        report["candidate_hash"] = _digest(asdict(changed))
        proposal = RevisionProposal(
            proposal_id=f"reword-proposal-{attempt_id}",
            workflow_id=f"reword-workflow-{attempt_id}",
            base_rubric_version_id=rubric.version_id,
            change=change,
            evidence=(),
            evaluation=None,
            created_at="automatic",
        )
        automated_decision = RevisionDecision(
            decision_id=f"reword-decision-{attempt_id}",
            proposal_id=proposal.proposal_id,
            maintainer="automatic_policy",
            decision=MaintainerDecisionKind.ADOPT,
            rationale="all automatic adoption gates passed",
            proposal=proposal,
            resulting_rubric=adopted,
            decided_at="automatic",
            actor_type="automatic",
            automatic_evidence=report,
        )
        return finish("all automatic adoption gates passed")
    except StaleProposalError:
        adopted = None
        automated_decision = None
        report["status"] = "hold"
        report.pop("adopted_version_id", None)
        return finish("active rubric changed after evaluation")
    except sqlite3.IntegrityError:
        prior = store.get_reword_attempt(attempt_id)
        if prior is not None:
            return dict(prior)
        adopted = None
        automated_decision = None
        consumed = False
        report["status"] = "hold"
        report.pop("adopted_version_id", None)
        return finish("validation budget exhausted")
    except (
        ProviderError,
        RuntimeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        return finish(f"workflow held: {exc}")
