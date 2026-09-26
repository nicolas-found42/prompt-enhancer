"""Paired grading protocol comparison over saved, matched optimize results.

The comparison never treats token estimates as billed cost. Missing historical
observations and independent safety labels remain unavailable in the report.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _cases(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("cases"), list):
        items = value["cases"]
    elif isinstance(value, list):
        items = value
    elif isinstance(value, Mapping):
        items = [{"case_id": "single", "result": value}]
    else:
        raise ValueError("result input must be an optimize result or a cases array")
    cases: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("each comparison case must be an object")
        case_id = item.get("case_id")
        result = item.get("result")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(result, Mapping)
        ):
            raise ValueError("each case requires case_id and result")
        if case_id in cases:
            raise ValueError(f"duplicate comparison case: {case_id}")
        cases[case_id] = result
    return cases


def _grading(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(result.get("report")).get("grading_observation"))


def _screening(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(result.get("report")).get("test_screening"))


def _total_measured_judge_cost(result: Mapping[str, Any]) -> float | None:
    grading = _grading(result)
    screening = _mapping(_screening(result).get("screening_observation"))
    grade_cost = _number(grading.get("judge_cost_usd_measured"))
    screen_cost = _number(screening.get("judge_cost_usd_measured"))
    if grade_cost is None:
        return None
    return grade_cost + (screen_cost or 0.0)


def _role_costs(result: Mapping[str, Any]) -> Mapping[str, float]:
    cost = _mapping(result.get("cost"))
    values = _mapping(cost.get("cost_by_role"))
    return {
        str(role): amount
        for role, value in values.items()
        if (amount := _number(value)) is not None
    }


def _outputs_match(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    left_report = _mapping(before.get("report"))
    right_report = _mapping(after.get("report"))
    left_outputs = _mapping(_mapping(left_report.get("per_model")).get("panel")).get(
        "outputs"
    )
    right_outputs = _mapping(_mapping(right_report.get("per_model")).get("panel")).get(
        "outputs"
    )
    left_tests = left_report.get("tests")
    right_tests = right_report.get("tests")
    if not isinstance(left_outputs, list) or not isinstance(right_outputs, list):
        return False
    if not isinstance(left_tests, list) or not isinstance(right_tests, list):
        return False

    def content(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            item.get("prompt"),
            item.get("output"),
            item.get("model"),
            item.get("sample"),
        )

    def criterion(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            item.get("question"),
            item.get("kind"),
            item.get("expected"),
            item.get("options"),
            item.get("levels"),
        )

    return (
        len(left_outputs) == len(right_outputs)
        and all(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and content(left) == content(right)
            for left, right in zip(left_outputs, right_outputs, strict=True)
        )
        and len(left_tests) == len(right_tests)
        and all(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and criterion(left) == criterion(right)
            for left, right in zip(left_tests, right_tests, strict=True)
        )
    )


def _grade_scores(result: Mapping[str, Any]) -> dict[str, float]:
    selection = _mapping(_mapping(result.get("report")).get("selection_evidence"))
    scores: dict[str, float] = {}
    original = _number(_mapping(selection.get("original_score")).get("worst"))
    if original is not None:
        scores["original"] = original
    ranking = selection.get("ranking")
    if isinstance(ranking, list):
        for item in ranking:
            if not isinstance(item, Mapping):
                continue
            name = item.get("candidate_id")
            worst = _number(_mapping(item.get("grade")).get("worst"))
            if isinstance(name, str) and worst is not None:
                scores[name] = worst
    return scores


def compare_grading_results(
    before_input: Any,
    after_input: Any,
    *,
    unsafe_labels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compare matched cases without inferring accuracy from synthetic outputs."""
    before = _cases(before_input)
    after = _cases(after_input)
    if set(before) != set(after):
        raise ValueError("before and after must have identical case IDs")
    cases: list[dict[str, Any]] = []
    matched_scores = 0
    agreed_scores = 0
    labeled_unsafe = 0
    false_positives = 0
    for case_id in sorted(before):
        old, new = before[case_id], after[case_id]
        old_grade, new_grade = _grading(old), _grading(new)
        old_requests = _number(old_grade.get("gateway_batch_calls"))
        new_requests = _number(new_grade.get("gateway_batch_calls"))
        old_bytes = _number(old_grade.get("serialized_input_bytes_estimate"))
        new_bytes = _number(new_grade.get("serialized_input_bytes_estimate"))
        score_pairs = 0
        score_agreements = 0
        if _outputs_match(old, new):
            old_scores, new_scores = _grade_scores(old), _grade_scores(new)
            for key in old_scores.keys() & new_scores.keys():
                score_pairs += 1
                score_agreements += abs(old_scores[key] - new_scores[key]) <= 1e-9
        matched_scores += score_pairs
        agreed_scores += score_agreements
        labels = (
            set(unsafe_labels.get(case_id, ())) if unsafe_labels is not None else None
        )
        case_false_positives = 0
        if labels is not None:
            checks = _screening(new).get("screening_checks")
            if isinstance(checks, list):
                for check in checks:
                    if isinstance(check, Mapping) and check.get("test_id") in labels:
                        labeled_unsafe += 1
                        case_false_positives += check.get("accepted") is True
        false_positives += case_false_positives
        old_cost = _total_measured_judge_cost(old)
        new_cost = _total_measured_judge_cost(new)
        cost_increase = (
            new_cost / old_cost - 1.0
            if old_cost is not None and old_cost > 0 and new_cost is not None
            else None
        )
        cases.append(
            {
                "case_id": case_id,
                "grading_requests": {
                    "before": old_requests,
                    "after": new_requests,
                    "delta": new_requests - old_requests
                    if old_requests is not None and new_requests is not None
                    else None,
                },
                "serialized_input_bytes_estimate": {
                    "before": old_bytes,
                    "after": new_bytes,
                    "delta": new_bytes - old_bytes
                    if old_bytes is not None and new_bytes is not None
                    else None,
                },
                "grading_agreement": {
                    "matched_scores": score_pairs,
                    "agreed_scores": score_agreements,
                    "rate": score_agreements / score_pairs if score_pairs else None,
                },
                "screen_false_positives": case_false_positives
                if labels is not None
                else None,
                "per_role_cost_usd_measured": {
                    "before": dict(_role_costs(old)),
                    "after": dict(_role_costs(new)),
                },
                "phase_judge_cost_usd_measured": {
                    "before": old_cost,
                    "after": new_cost,
                    "screening_after": _number(
                        _mapping(_screening(new).get("screening_observation")).get(
                            "judge_cost_usd_measured"
                        )
                    ),
                    "grading_after": _number(new_grade.get("judge_cost_usd_measured")),
                },
                "judge_cost_increase_fraction": cost_increase,
                "above_ten_percent_increase": cost_increase is not None
                and cost_increase > 0.10,
                "remedy": (
                    "Inspect screening cache hits and per-output batch size; compare the measured screening and grading charges above."
                    if cost_increase is not None and cost_increase > 0.10
                    else None
                ),
            }
        )
    return {
        "status": "available" if cases else "unavailable",
        "matched_cases": len(cases),
        "grading_agreement": {
            "matched_scores": matched_scores,
            "agreed_scores": agreed_scores,
            "rate": agreed_scores / matched_scores if matched_scores else None,
        },
        "screen_false_positive_rate": {
            "labeled_unsafe_tests": labeled_unsafe,
            "false_positives": false_positives,
            "rate": false_positives / labeled_unsafe if labeled_unsafe else None,
        },
        "cases": cases,
        "evidence_note": "Request sizes and token counts are estimates; cost fields require provider usage. Grade agreement requires identical outputs and criteria, and screen false positives require independent labels.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--unsafe-labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    labels = (
        json.loads(args.unsafe_labels.read_text(encoding="utf-8"))
        if args.unsafe_labels is not None
        else None
    )
    report = compare_grading_results(
        json.loads(args.before.read_text(encoding="utf-8")),
        json.loads(args.after.read_text(encoding="utf-8")),
        unsafe_labels=labels,
    )
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
