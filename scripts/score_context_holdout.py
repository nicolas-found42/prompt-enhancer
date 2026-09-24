"""Score a frozen context cutoff on a separate, untouched SOAR holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_soar_context import _metrics

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.evaluation.datasets import load_dataset, replay_digest
from prompt_enhancer.gateway import ReplayGateway


def score(
    dataset_path: Path, replay_path: Path, calibration_path: Path
) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    selected = calibration["selected_threshold"]
    question_text = calibration.get("question")
    if (
        not isinstance(question_text, str)
        or not question_text
        or not isinstance(selected, (float, int))
    ):
        raise ValueError("calibration must name a context question and threshold")
    responses = json.loads(replay_path.read_text(encoding="utf-8"))["responses"]
    rows = []
    for case in dataset.cases:
        if not case.labels_present or "context" not in case.metadata.get(
            "evaluation_gaps", []
        ):
            raise ValueError(f"missing human context label for {case.id}")
        question = {
            "model": JEV_MODEL,
            "query": question_text,
            "state": {"prompt": case.prompt},
            "type": "noul",
            "key": "gap:context",
        }
        key = ReplayGateway.request_key("decide", JEV_MODEL, question, "judge")
        answer = responses.get(key)
        if not isinstance(answer, dict) or not isinstance(
            answer.get("noul"), (float, int)
        ):
            raise TypeError(f"missing or invalid context response for {case.id}")
        rows.append(
            (case.id, float(answer["noul"]), "context" in case.expected_gaps, False)
        )
    return {
        "question": question_text,
        "dataset_digest": dataset.digest,
        "replay_digest": replay_digest(replay_path),
        "calibration_dataset_digest": calibration["dataset_digest"],
        "selected_threshold": selected,
        "cases": len(rows),
        "selected": _metrics(rows, selected),
        "current_default_0_87": _metrics(rows, 0.87),
        "old_baseline_0_90": _metrics(rows, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = (
        json.dumps(
            score(args.dataset, args.replay, args.calibration), indent=2, sort_keys=True
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
