"""Calibrate the exact default context question from a SOAR strict replay.

This is intentionally narrow: SOAR's human `missing context` label does not
provide ground truth for the other checklist questions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation_review_common import binary_metrics, in_holdout_group

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.diagnosis import default_gap_question
from prompt_enhancer.evaluation.datasets import load_dataset, replay_digest
from prompt_enhancer.gateway import ReplayGateway

CONTEXT_QUESTION = default_gap_question("context")
THRESHOLDS = tuple(round(0.8 + index / 100, 2) for index in range(16))
Row = tuple[str, float, bool, bool]


def _metrics(rows: list[Row], threshold: float) -> dict[str, float | int]:
    return binary_metrics(
        ((probability, expected) for _, probability, expected, _ in rows), threshold
    )


def calibrate(
    dataset_path: Path,
    replay_path: Path,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    question_text: str = CONTEXT_QUESTION,
) -> dict[str, Any]:
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("threshold grid must contain probabilities")
    dataset = load_dataset(dataset_path)
    recording = json.loads(replay_path.read_text(encoding="utf-8"))
    responses = recording.get("responses")
    if not isinstance(responses, dict):
        raise TypeError("strict replay requires an object of responses")
    rows: list[Row] = []
    for case in dataset.cases:
        if not case.labels_present or "context" not in case.metadata.get(
            "evaluation_gaps", []
        ):
            raise ValueError(f"case {case.id} lacks a reviewable context label")
        question = {
            "model": JEV_MODEL,
            "query": question_text,
            "state": {"prompt": case.prompt},
            "type": "noul",
            "key": "gap:context",
        }
        key = ReplayGateway.request_key("decide", JEV_MODEL, question, "judge")
        answer = responses.get(key)
        if answer is None:
            raise ValueError(f"missing context decision for case {case.id}")
        if not isinstance(answer, dict):
            raise TypeError(f"context decision for case {case.id} must be an object")
        probability = answer.get("noul", answer.get("probability_true"))
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (float, int))
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"invalid context probability for case {case.id}")
        holdout = in_holdout_group(case.id)
        rows.append(
            (case.id, float(probability), "context" in case.expected_gaps, holdout)
        )
    train = [row for row in rows if not row[3]]
    holdout = [row for row in rows if row[3]]
    selected = max(
        thresholds,
        key=lambda threshold: (_metrics(train, threshold)["f0_5"], threshold),
    )
    return {
        "question_id": "gap:context",
        "question": question_text,
        "dataset_digest": dataset.digest,
        "replay_digest": replay_digest(replay_path),
        "rows": len(rows),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "selection": f"Maximum training F0.5 over {min(thresholds):.2f} to {max(thresholds):.2f} in 0.01 steps; higher threshold breaks ties",
        "baseline_threshold": 0.9,
        "selected_threshold": selected,
        "baseline_train": _metrics(train, 0.9),
        "selected_train": _metrics(train, selected),
        "baseline_holdout": _metrics(holdout, 0.9),
        "selected_holdout": _metrics(holdout, selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-threshold", type=float, default=0.8)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--question", default=CONTEXT_QUESTION)
    args = parser.parse_args()
    low = round(args.min_threshold * 100)
    high = round(args.max_threshold * 100)
    if low < 0 or high > 100 or low > high:
        parser.error("threshold range must be within 0 to 1 and nonempty")
    grid = tuple(value / 100 for value in range(low, high + 1))
    rendered = (
        json.dumps(
            calibrate(
                args.dataset, args.replay, thresholds=grid, question_text=args.question
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
