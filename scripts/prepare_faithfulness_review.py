"""Make a blinded human review sheet for Jev success-test faithfulness.

Only first-turn SOAR prompts without obvious placeholders are sampled. The
reviewer CSV omits Jev probabilities; a private evidence JSON retains them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from prompt_enhancer.evaluation.datasets import load_dataset, replay_digest
from prompt_enhancer.evaluation.harness import HarnessOptions, default_engine_factory
from prompt_enhancer.jev import NoulDecision, parse_decision
from prompt_enhancer.optimizer import PromptOptimizer

_PLACEHOLDER = re.compile(r"\[[A-Z _]{3,}\]|<[^>]{3,}>|\b(?:placeholder|omitted|removed|redacted|code snippet)\b", re.IGNORECASE)
FIELDS = ("row_id", "source_case_id", "writer", "prompt", "proposed_test", "human_faithful", "human_notes", "human_reviewer", "reviewed_at")


def collect(dataset_path: Path, replays: dict[str, Path], *, per_writer: int = 50) -> tuple[list[dict[str, str]], dict[str, Any]]:
    dataset = load_dataset(dataset_path)
    review_rows: list[dict[str, str]] = []
    evidence: dict[str, Any] = {"dataset_digest": dataset.digest, "replay_digests": {}, "question": "Is this proposed success test faithful to the user's request, and does it test success rather than an invented requirement?", "rows": {}}
    for writer, replay_path in replays.items():
        engine = default_engine_factory(replay_path)
        if not isinstance(engine, PromptOptimizer):
            raise TypeError("expected product optimizer")
        evidence["replay_digests"][writer] = replay_digest(replay_path)
        options = HarnessOptions(tier="fast", model_overrides={"writer": writer} if writer != "space-bunny-free" else {})
        candidates = []
        for case in dataset.cases:
            if case.metadata.get("turn") != 1 or _PLACEHOLDER.search(case.prompt):
                continue
            engine.gateway.decision_log.clear()
            result = engine.optimize(case.prompt, options.optimize_options())
            if result["status"] == "failed":
                raise ValueError(f"strict replay failed for {case.id}/{writer}")
            for item in engine.gateway.decision_log:
                question = item["question"]
                if not str(question.get("key", "")).startswith("faithful:"):
                    continue
                decision = parse_decision(item["answer"])
                if not isinstance(decision, NoulDecision):
                    raise TypeError("faithfulness answer must be noul")
                proposed = question["state"]["proposed_test"]
                row_id = f"{case.id}/{writer}/{proposed['id']}"
                candidates.append((row_id, {
                    "row_id": row_id, "source_case_id": case.id, "writer": writer,
                    "prompt": case.prompt, "proposed_test": json.dumps(proposed, ensure_ascii=False),
                    "human_faithful": "", "human_notes": "", "human_reviewer": "", "reviewed_at": "",
                }, decision.probability))
        selected = sorted(candidates, key=lambda item: hashlib.sha256(item[0].encode()).hexdigest())[:per_writer]
        if len(selected) < per_writer:
            raise ValueError(f"only {len(selected)} faithfulness proposals for {writer}")
        for row_id, row, probability in selected:
            review_rows.append(row)
            content = (row["prompt"] + "\n" + row["proposed_test"]).encode("utf-8")
            evidence["rows"][row_id] = {
                "probability": probability, "source_case_id": row["source_case_id"], "writer": writer,
                "content_digest": hashlib.sha256(content).hexdigest(),
            }
    review_rows.sort(key=lambda row: row["row_id"])
    return review_rows, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bunny-replay", type=Path, required=True)
    parser.add_argument("--deepseek-replay", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    rows, evidence = collect(args.dataset, {"space-bunny-free": args.bunny_replay, "deepseek-v4.1-flash": args.deepseek_replay})
    args.review.parent.mkdir(parents=True, exist_ok=True)
    with args.review.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} blinded test judgments to {args.review}")


if __name__ == "__main__":
    main()
