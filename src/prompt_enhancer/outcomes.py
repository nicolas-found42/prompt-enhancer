"""Evidence-backed semantic outcomes for optimizer and history payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .score_vector import SCORE_DIMENSIONS

OUTCOME_VALUES = frozenset(
    {
        "converged",
        "improved_tested",
        "improved_unverified",
        "impossible",
        "failed_operational",
    }
)
CONTROL_STATES = frozenset(
    {"awaiting_approval", "stopped", "cancelled", "deadline_reached"}
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _convergence_is_proven(report: Mapping[str, Any]) -> bool:
    evidence = _mapping(report.get("convergence"))
    if not evidence:
        for round_record in reversed(_rounds(report)):
            evidence = _mapping(round_record.get("convergence"))
            if evidence:
                break
    if evidence.get("status") != "converged" or evidence.get("passed") is not True:
        return False
    scores = _mapping(evidence.get("scores"))
    floors = _mapping(evidence.get("floors"))
    if not all(
        isinstance(scores.get(name), (int, float))
        and not isinstance(scores.get(name), bool)
        and isinstance(floors.get(name), (int, float))
        and not isinstance(floors.get(name), bool)
        and float(scores[name]) >= float(floors[name])
        for name in SCORE_DIMENSIONS
    ):
        return False
    gain = evidence.get("gain")
    epsilon = evidence.get("epsilon")
    return gain is None or (
        isinstance(gain, (int, float))
        and not isinstance(gain, bool)
        and isinstance(epsilon, (int, float))
        and not isinstance(epsilon, bool)
        and float(gain) <= float(epsilon) + 1e-12
    )


def _rounds(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    for item in report.get("history", ()):
        if isinstance(item, Mapping):
            found.append(item)
    found.append(report)
    return found


def _accepted_candidate(
    report: Mapping[str, Any], final_prompt: str, original_prompt: str
) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]] | None:
    if not final_prompt or final_prompt == original_prompt:
        return None
    for round_record in reversed(_rounds(report)):
        evidence = _mapping(round_record.get("evidence"))
        selection = _mapping(
            round_record.get("selection_evidence") or evidence.get("selection_evidence")
        )
        selected_id = round_record.get("selected_candidate_id") or selection.get(
            "selected_candidate_id"
        )
        if not isinstance(selected_id, str) or not selected_id:
            continue
        selected = selection.get("selected_candidate")
        if not isinstance(selected, Mapping):
            continue
        if str(selected.get("candidate_id") or "") != selected_id:
            continue
        text = selected.get("text") or selected.get("prompt")
        if not isinstance(text, str) or text != final_prompt:
            continue
        metadata = _mapping(selected.get("metadata"))
        fidelity = _mapping(metadata.get("fidelity"))
        vector = _mapping(metadata.get("score_vector"))
        evaluation = _mapping(metadata.get("evaluation"))
        accept = _mapping(evaluation.get("accept"))
        if (
            fidelity.get("passed") is not True
            or vector.get("passed") is not True
            or evaluation.get("eligible") is not True
            or accept.get("accepted") is not True
        ):
            continue
        status = str(round_record.get("status") or "")
        if status not in {
            "improved",
            "improved_unverified",
            "converged",
            "stopped",
            "cancelled",
        }:
            continue
        return status, selected_id, evidence or round_record, evaluation
    return None


def _tested(
    evidence: Mapping[str, Any], evaluation: Mapping[str, Any], candidate_id: str
) -> bool:
    tests = evaluation.get("success_tests")
    outputs = evaluation.get("success_test_outputs")
    grade = evaluation.get("success_test_grade")
    strong = evidence.get("strong_check")
    strong_candidates = _mapping(strong).get("candidates")
    strong_passed = isinstance(strong_candidates, list) and any(
        isinstance(item, Mapping)
        and item.get("candidate_id") == candidate_id
        and item.get("passed") is True
        for item in strong_candidates
    )
    threshold = grade.get("threshold") if isinstance(grade, Mapping) else None
    worst = grade.get("worst") if isinstance(grade, Mapping) else None
    return (
        evaluation.get("downstream_verification") == "verified"
        and isinstance(tests, list)
        and bool(tests)
        and isinstance(outputs, list)
        and bool(outputs)
        and isinstance(grade, Mapping)
        and isinstance(worst, (int, float))
        and not isinstance(worst, bool)
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and float(worst) >= float(threshold)
        and grade.get("ungradable_outputs", 0) == 0
        and grade.get("detected_outputs", 0) == 0
        and grade.get("unresolved_screen_outputs", 0) == 0
        and grade.get("unresolved_grade_outputs", 0) == 0
        and strong_passed
    )


def apply_outcome_fields(
    report: Mapping[str, Any],
    *,
    original_prompt: str,
    final_prompt: str,
    control_state: str | None = None,
) -> dict[str, Any]:
    """Add the semantic outcome only when persisted evidence supports one."""
    result = dict(report)
    status = str(result.get("status") or "")
    failure = _mapping(result.get("failure"))
    if control_state is None and status in CONTROL_STATES:
        control_state = status
    if control_state in CONTROL_STATES:
        result["control_state"] = control_state
    else:
        result.pop("control_state", None)

    style = result.get("applied_style")
    result["applied_style"] = style if isinstance(style, str) and style else None

    outcome: str | None = None
    reason: str | None = None
    if status == "impossible" and failure.get("kind") == "impossible":
        outcome = "impossible"
        reason = str(
            failure.get("message")
            or failure.get("hint")
            or "Route proved this style incompatible with a hard requirement."
        )
    elif (
        status == "converged"
        and final_prompt != original_prompt
        and _accepted_candidate(result, final_prompt, original_prompt) is not None
        and _convergence_is_proven(result)
    ):
        evidence = _mapping(result.get("convergence"))
        if not evidence:
            for round_record in reversed(_rounds(result)):
                evidence = _mapping(round_record.get("convergence"))
                if evidence:
                    break
        style_name = result.get("applied_style") or "requested"
        if evidence.get("gain") is None:
            reason = f"The {style_name} prompt met every quality floor on its first round; no earlier round existed for a gain comparison."
        else:
            reason = str(
                result.get("summary")
                or "The selected prompt met every quality floor and its measured gain was within the configured epsilon."
            )
        outcome = "converged"
    elif (
        status == "failed"
        and failure
        and failure.get("kind") not in {"cancelled", "stopped", "impossible"}
    ):
        outcome = "failed_operational"
        reason = str(
            failure.get("hint")
            or failure.get("message")
            or "The run stopped before an accepted outcome was established."
        )
    else:
        accepted = _accepted_candidate(result, final_prompt, original_prompt)
        if accepted is not None:
            _round_status, candidate_id, evidence, evaluation = accepted
            style_name = result.get("applied_style") or "requested"
            if _tested(evidence, evaluation, candidate_id):
                outcome = "improved_tested"
                reason = f"The {style_name} rewrite passed the run's answer tests and candidate gates."
            elif evaluation.get("downstream_verification") == "unverified":
                outcome = "improved_unverified"
                reason = f"The {style_name} rewrite passed meaning and safety checks, but answer quality was not tested."

    result["outcome"] = outcome
    result["outcome_reason"] = reason
    return result


__all__ = ["CONTROL_STATES", "OUTCOME_VALUES", "apply_outcome_fields"]
