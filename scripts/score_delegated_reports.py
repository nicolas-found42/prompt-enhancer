"""Score completed product reports against delegated real-prompt gap labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_review_common import save_json
from human_gap_review import GAPS


def score(dataset: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    if dataset.get("metadata", {}).get("reviewer_kind") != "user_delegated_model":
        raise ValueError("dataset does not record delegated-model provenance")
    labels = {case["id"]: case for case in dataset["cases"]}
    cases = report["cases"]
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate report case ID")
    if any(case["case_id"] not in labels for case in cases):
        raise ValueError("report contains a case outside the reviewed cohort")
    completed = [case for case in cases if case["status"] not in {"failed", "error"}]
    per_gap = {gap: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for gap in GAPS}
    for case in completed:
        expected = set(labels[case["case_id"]]["expected_gaps"])
        predicted = set(case["predicted_gaps"])
        if any(gap not in GAPS for gap in expected | predicted):
            raise ValueError(f"unknown gap key for {case['case_id']}")
        for gap in GAPS:
            key = (
                "tp"
                if gap in expected and gap in predicted
                else "fp"
                if gap in predicted
                else "fn"
                if gap in expected
                else "tn"
            )
            per_gap[gap][key] += 1
    micro = {
        key: sum(row[key] for row in per_gap.values()) for key in ("tp", "fp", "fn")
    }
    precision = (
        micro["tp"] / (micro["tp"] + micro["fp"]) if micro["tp"] + micro["fp"] else 0.0
    )
    recall = (
        micro["tp"] / (micro["tp"] + micro["fn"]) if micro["tp"] + micro["fn"] else 0.0
    )
    return {
        "label_provenance": "user_delegated_model",
        "report_dataset": report["dataset"],
        "cases": len(cases),
        "completed": len(completed),
        "source_counts": dict(
            Counter(labels[case["case_id"]]["source_dataset"] for case in completed)
        ),
        "task_counts": dict(
            Counter(labels[case["case_id"]]["task_stratum"] for case in completed)
        ),
        "micro": {
            **micro,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        },
        "per_gap": per_gap,
        "improvement": report["improvement"],
        "metered_openrouter_usd": report["cost"]["total"],
        "latency_ms": report["latency_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for specification in args.report:
        name, separator, filename = specification.partition("=")
        if not separator or not name or name in results:
            raise ValueError("report must be a unique NAME=PATH")
        results[name] = score(
            dataset, json.loads(Path(filename).read_text(encoding="utf-8"))
        )
    save_json(args.output, {"schema_version": 1, "reviews": results})


if __name__ == "__main__":
    main()
