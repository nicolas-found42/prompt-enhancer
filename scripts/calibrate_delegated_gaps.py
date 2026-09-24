"""Measure exact Jev gap questions against user-delegated model labels.

This calibration does not turn model judgments into human annotations. Source
groups stay together in the deterministic training/holdout split.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from delegated_review import SECRET
from evaluation_review_common import binary_metrics, in_holdout_group, save_json
from human_gap_review import GAPS, TASK_GAPS

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.diagnosis import DEFAULT_RUBRIC, default_gap_question
from prompt_enhancer.gateway import GatewayConfig, ModelGateway


def _probability(answer: Any) -> float:
    if not isinstance(answer, dict):
        raise TypeError("Jev answer is not an object")
    value = answer.get("noul", answer.get("probability_true"))
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
        raise ValueError("Jev answer lacks a valid yes probability")
    return float(value)


def measure(dataset_path: Path, decisions_path: Path) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if dataset.get("metadata", {}).get("reviewer_kind") != "user_delegated_model" or any(
        case.get("label_provenance") != "user_delegated_model" for case in cases
    ):
        raise ValueError("calibration requires a homogeneous delegated-model dataset")
    if any(SECRET.search(case["prompt"]) for case in cases):
        raise ValueError("secret-like prompt in calibration cohort")
    saved: dict[str, Any] = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else {
        "schema_version": 1, "model": JEV_MODEL, "dataset_name": dataset["name"], "rows": []}
    if saved["dataset_name"] != dataset["name"]:
        raise ValueError("decisions belong to a different dataset")
    done = {row["case_id"] for row in saved["rows"]}
    gateway = ModelGateway(config=GatewayConfig.from_env())
    for case in cases:
        if case["id"] in done:
            continue
        questions = [
            {"model": JEV_MODEL, "query": default_gap_question(gap),
             "state": {"prompt": case["prompt"]}, "type": "noul", "key": f"gap:{gap}"}
            for gap in TASK_GAPS[case["task_stratum"]]
        ]
        answers = gateway.jev_batch(questions)
        if len(answers) != len(questions):
            raise ValueError(f"wrong Jev answer count for {case['id']}")
        saved["rows"].append({"case_id": case["id"], "source_group": case["source_group"],
                              "probabilities": {gap: _probability(answer)
                                                for gap, answer in zip(TASK_GAPS[case["task_stratum"]], answers, strict=True)}})
        done.add(case["id"])
        save_json(decisions_path, saved)
        print(f"Jev gap decisions {len(done)}/{len(cases)}", flush=True)


SELECTIONS = {
    "f0_5": "Maximum training F0.5 on 0.80–0.95 grid; higher cutoff breaks positive-score ties; no selected cutoff when all scores are zero",
    "precision_floor": "Lowest cutoff on 0.60–0.95 grid whose training precision is at least {min_precision} with one or more true positives; none otherwise",
}


def _select(train: list[tuple[float, bool]], selection: str, min_precision: float) -> float | None:
    if selection == "precision_floor":
        grid = [value / 100 for value in range(60, 96)]
        eligible = [
            cutoff for cutoff in grid
            if (metrics := binary_metrics(train, cutoff))["tp"] >= 1 and metrics["precision"] >= min_precision
        ]
        return min(eligible) if eligible else None
    grid = [value / 100 for value in range(80, 96)]
    candidates = [(cutoff, binary_metrics(train, cutoff)) for cutoff in grid]
    best_f = max(metrics["f0_5"] for _, metrics in candidates)
    return max(cutoff for cutoff, metrics in candidates if metrics["f0_5"] == best_f) if best_f > 0 else None


def calibrate(
    dataset_path: Path, decisions_path: Path, *, selection: str = "f0_5", min_precision: float = 0.5
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("metadata", {}).get("reviewer_kind") != "user_delegated_model" or any(
        case.get("label_provenance") != "user_delegated_model" for case in dataset["cases"]
    ):
        raise ValueError("calibration requires a homogeneous delegated-model dataset")
    recorded = json.loads(decisions_path.read_text(encoding="utf-8"))
    if recorded["dataset_name"] != dataset["name"]:
        raise ValueError("dataset mismatch")
    cases = {case["id"]: case for case in dataset["cases"]}
    rows = {row["case_id"]: row for row in recorded["rows"]}
    if set(cases) != set(rows):
        raise ValueError("incomplete Jev decisions")
    by_gap: dict[str, dict[str, list[tuple[float, bool]]]] = defaultdict(lambda: {"train": [], "holdout": []})
    for case_id, case in cases.items():
        row = rows[case_id]
        if row["source_group"] != case["source_group"]:
            raise ValueError(f"source group mismatch for {case_id}")
        split = "holdout" if in_holdout_group(case["source_group"]) else "train"
        for gap in TASK_GAPS[case["task_stratum"]]:
            by_gap[gap][split].append((row["probabilities"][gap], gap in case["expected_gaps"]))
    result = {"dataset_name": dataset["name"], "label_provenance": "user_delegated_model",
              "decision_model": JEV_MODEL, "split": "SHA-256 source_group modulo 5; related participant prompts stay together",
              "selection": SELECTIONS[selection].format(min_precision=min_precision),
              "questions": {}}
    for gap in GAPS:
        train, holdout = by_gap[gap]["train"], by_gap[gap]["holdout"]
        if not train or not holdout:
            continue
        baseline = DEFAULT_RUBRIC.gap_threshold_for(gap)
        selected = _select(train, selection, min_precision)
        result["questions"][gap] = {
            "question": default_gap_question(gap), "train_count": len(train), "holdout_count": len(holdout),
            "train_positive": sum(label for _, label in train), "holdout_positive": sum(label for _, label in holdout),
            "baseline_threshold": baseline, "selected_threshold": selected,
            "baseline_train": binary_metrics(train, baseline),
            "selected_train": binary_metrics(train, selected) if selected is not None else None,
            "baseline_holdout": binary_metrics(holdout, baseline),
            "selected_holdout": binary_metrics(holdout, selected) if selected is not None else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--selection", choices=sorted(SELECTIONS), default="f0_5")
    parser.add_argument("--min-precision", type=float, default=0.5)
    args = parser.parse_args()
    if args.measure:
        measure(args.dataset, args.decisions)
    save_json(args.output, calibrate(args.dataset, args.decisions, selection=args.selection, min_precision=args.min_precision))


if __name__ == "__main__":
    main()
