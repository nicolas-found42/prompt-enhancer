"""Measure false flags of the exact outside_reference gap question.

The delegated review has no outside_reference labels, so recall cannot be
calibrated. Prompts judged to have no checklist gap at all still bound how
often the question flags a prompt that needs nothing from the user.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation_review_common import save_json

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.diagnosis import DEFAULT_RUBRIC, default_gap_question
from prompt_enhancer.gateway import GatewayConfig, HttpGateway

CUTOFFS = (0.9, 0.85, 0.8, 0.75)


def measure(dataset: dict[str, Any], decisions_path: Path) -> None:
    question = default_gap_question("outside_reference")
    saved: dict[str, Any] = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else {
        "schema_version": 1, "model": JEV_MODEL, "dataset_name": dataset["name"], "question": question, "rows": []}
    if saved["dataset_name"] != dataset["name"] or saved["question"] != question:
        raise ValueError("decisions belong to a different dataset or question wording")
    done = {row["case_id"] for row in saved["rows"]}
    todo = [case for case in dataset["cases"] if case["id"] not in done]
    gateway = HttpGateway(config=GatewayConfig.from_env())
    for start in range(0, len(todo), 20):
        chunk = todo[start:start + 20]
        answers = gateway.decide_batch([
            {"model": JEV_MODEL, "query": question, "state": {"prompt": case["prompt"]},
             "type": "noul", "key": f"gap:outside_reference:{case['id']}"}
            for case in chunk
        ])
        for case, answer in zip(chunk, answers, strict=True):
            saved["rows"].append({"case_id": case["id"], "source_group": case["source_group"],
                                  "probability": float(answer.get("noul", answer.get("probability_true")))})
        save_json(decisions_path, saved)
        print(f"outside_reference decisions {len(saved['rows'])}/{len(dataset['cases'])}", flush=True)


def summarize(dataset: dict[str, Any], decisions_path: Path) -> dict[str, Any]:
    cases = {case["id"]: case for case in dataset["cases"]}
    rows = json.loads(decisions_path.read_text(encoding="utf-8"))["rows"]
    if {row["case_id"] for row in rows} != set(cases):
        raise ValueError("incomplete outside_reference decisions")
    no_gap = [row["probability"] for row in rows if not cases[row["case_id"]]["expected_gaps"]]
    context = [row["probability"] for row in rows if "context" in cases[row["case_id"]]["expected_gaps"]]
    return {
        "dataset_name": dataset["name"], "question": default_gap_question("outside_reference"),
        "current_threshold": DEFAULT_RUBRIC.gap_threshold_for("outside_reference"),
        "no_gap_count": len(no_gap), "context_gap_count": len(context),
        "cutoffs": {
            str(cutoff): {"no_gap_flags": sum(value >= cutoff for value in no_gap),
                          "context_gap_flags": sum(value >= cutoff for value in context)}
            for cutoff in CUTOFFS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.measure:
        measure(dataset, args.decisions)
    save_json(args.output, summarize(dataset, args.decisions))


if __name__ == "__main__":
    main()
