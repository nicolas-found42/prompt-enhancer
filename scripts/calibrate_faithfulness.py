"""Validate human test-faithfulness judgments and tune on conversation splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from evaluation_review_common import binary_metrics, in_holdout_group

from prompt_enhancer.success_tests import DEFAULT_FAITHFULNESS_THRESHOLD

# The product gate before the delegated faithfulness review.
PREVIOUS_THRESHOLD = 0.9


def calibrate(review_path: Path, evidence_path: Path, *, minimum: int = 80) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    source = evidence["rows"]
    with review_path.open(encoding="utf-8", newline="") as file:
        review = list(csv.DictReader(file))
    if len(review) != len(source) or {row["row_id"] for row in review} != set(source):
        raise ValueError("review rows differ from the blinded evidence queue")
    train: list[tuple[float, bool]] = []
    holdout: list[tuple[float, bool]] = []
    uncertain = 0
    provenances: set[str] = set()
    for row in review:
        row_id = row["row_id"]
        item = source[row_id]
        content = (row["prompt"] + "\n" + row["proposed_test"]).encode("utf-8")
        if hashlib.sha256(content).hexdigest() != item["content_digest"] or row["source_case_id"] != item["source_case_id"] or row["writer"] != item["writer"]:
            raise ValueError(f"source text or provenance changed for {row_id}")
        reviewer_kind = (row.get("reviewer_kind") or "human").strip()
        if reviewer_kind not in {"human", "user_delegated_model"}:
            raise ValueError(f"invalid reviewer_kind for {row_id}")
        label = (row.get("review_label") or row["human_faithful"]).strip().lower()
        if label == "uncertain":
            uncertain += 1
            continue
        if label not in {"yes", "no"}:
            raise ValueError(f"review label must be yes, no, or uncertain for {row_id}")
        reviewer = row.get("reviewer") or row["human_reviewer"]
        if not reviewer.strip() or not row["reviewed_at"].strip():
            raise ValueError(f"reviewer and date required for {row_id}")
        try:
            date.fromisoformat(row["reviewed_at"].strip())
        except ValueError as exc:
            raise ValueError(f"reviewed_at must be an ISO date for {row_id}") from exc
        grouped_holdout = in_holdout_group(row["source_case_id"])
        (holdout if grouped_holdout else train).append((float(item["probability"]), label == "yes"))
        provenances.add(reviewer_kind)
    if len(train) + len(holdout) < minimum:
        raise ValueError(f"only {len(train) + len(holdout)} yes/no judgments; require {minimum}")
    if not train or not holdout:
        raise ValueError("training and held-out groups must both contain judgments")
    if len(provenances) != 1:
        raise ValueError("calibration cannot mix reviewer kinds")
    thresholds = tuple(value / 100 for value in range(50, 96))
    selected = max(thresholds, key=lambda value: (binary_metrics(train, value)["f0_5"], value))
    return {
        "question": evidence["question"], "label_provenance": next(iter(provenances)), "reviewed": len(train) + len(holdout), "uncertain": uncertain,
        "train_count": len(train), "holdout_count": len(holdout), "selection": "Maximum training F0.5 on 0.50–0.95 grid; source conversations grouped; higher cutoff breaks ties",
        "previous_threshold": PREVIOUS_THRESHOLD, "current_threshold": DEFAULT_FAITHFULNESS_THRESHOLD, "selected_threshold": selected,
        "previous_train": binary_metrics(train, PREVIOUS_THRESHOLD), "previous_holdout": binary_metrics(holdout, PREVIOUS_THRESHOLD),
        "current_train": binary_metrics(train, DEFAULT_FAITHFULNESS_THRESHOLD), "selected_train": binary_metrics(train, selected),
        "current_holdout": binary_metrics(holdout, DEFAULT_FAITHFULNESS_THRESHOLD), "selected_holdout": binary_metrics(holdout, selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(calibrate(args.review, args.evidence), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
