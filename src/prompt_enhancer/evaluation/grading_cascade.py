"""Compare direct, confirmation, and full grading on independently labeled cases.

This report consumes a compact manifest of saved pair evidence and labels.
Known-answer fixtures verify the arithmetic; they do not measure live accuracy.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _rate(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _status(value: object) -> bool | None:
    if value == "confirmed_pass":
        return True
    if value == "confirmed_fail":
        return False
    return None


def evaluate_cascade_cases(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("grading cascade manifest requires a cases array")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise ValueError("each cascade case must be an object")
        case_id = raw.get("id")
        label = raw.get("label_pass")
        probability = raw.get("initial_pass_probability")
        cost = raw.get("measured_cost_usd")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("each cascade case needs a unique id")
        if not isinstance(label, bool):
            raise ValueError("label_pass must be boolean")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            raise ValueError("initial_pass_probability must be between zero and one")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            raise ValueError("measured_cost_usd must be non-negative")
        seen.add(case_id)
        direct = (
            bool(probability >= 0.5) if probability < 0.3 or probability > 0.7 else None
        )
        confirmation = _status(raw.get("confirmation_status"))
        full = _status(raw.get("full_status"))
        cases.append(
            {
                "id": case_id,
                "label_pass": label,
                "direct": direct,
                "confirmation": direct if direct is not None else confirmation,
                "full_cascade": direct if direct is not None else full,
                "escalated": raw.get("escalated") is True,
                "measured_cost_usd": float(cost),
            }
        )
    negatives = sum(case["label_pass"] is False for case in cases)

    def stage(name: str) -> dict[str, Any]:
        covered = [case for case in cases if case[name] is not None]
        agreement_pairs = [
            case
            for case in cases
            if case[name] is not None and case["direct"] is not None
        ]
        return {
            "coverage": _rate(len(covered), len(cases)),
            "abstention": _rate(len(cases) - len(covered), len(cases)),
            "correct_on_covered": _rate(
                sum(case[name] is case["label_pass"] for case in covered), len(covered)
            ),
            "false_pass_rate": _rate(
                sum(
                    case[name] is True and case["label_pass"] is False for case in cases
                ),
                negatives,
            ),
            "grade_agreement_with_direct": _rate(
                sum(case[name] is case["direct"] for case in agreement_pairs),
                len(agreement_pairs),
            ),
        }

    return {
        "fixture": str(manifest.get("name", "unnamed")),
        "provenance": str(manifest.get("provenance", "unspecified")),
        "case_count": len(cases),
        "direct": stage("direct"),
        "confirmation": stage("confirmation"),
        "full_cascade": stage("full_cascade"),
        "escalation_rate": _rate(sum(case["escalated"] for case in cases), len(cases)),
        "measured_cost_usd": sum(case["measured_cost_usd"] for case in cases),
        "production_accuracy": "unavailable"
        if manifest.get("provenance") == "synthetic_known_answer"
        else "requires_independent_labels",
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare direct and cascade grade evidence"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_cascade_cases(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
