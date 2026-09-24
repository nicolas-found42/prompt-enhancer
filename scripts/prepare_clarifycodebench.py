"""Build a local coding-gap evaluation set from ClarifyCodeBench.

These are human-annotated, deletion-edited LiveCodeBench tasks, so the harness
marks them synthetic rather than claiming they are spontaneous user prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "5e2d5b5ce6259daa034cebb69f65e5e4c6dec3e9"
SOURCE = (
    "https://raw.githubusercontent.com/fangz-cs/ClarifyCodeBench/"
    f"{COMMIT}/data/ClarifyCodeBench.jsonl"
)
CASE_COUNT = 150


def prepare(source: str) -> dict[str, object]:
    with urlopen(source, timeout=30) as response:
        rows = [json.loads(line) for line in response if line.strip()]
    if len(rows) < CASE_COUNT:
        raise ValueError(f"need at least {CASE_COUNT} ClarifyCodeBench tasks")
    rows.sort(key=lambda row: hashlib.sha256(row["task_id"].encode()).hexdigest())
    selected = rows[:CASE_COUNT]
    cases = [
        {
            "id": f"clarifycodebench-{row['task_id']}",
            "source": "synthetic",
            "prompt": row["modified_content"],
            "expected_gaps": row["keywords"],
            "evaluation_gaps": ["output_format"],
            "source_question_id": row["question_id"],
            "clarification_questions": row["qa_pairs"],
            "evaluation_notes": "Human deletion-edited coding task with annotated underspecification types; not a spontaneous user prompt.",
        }
        for row in sorted(selected, key=lambda row: row["task_id"])
    ]
    return {
        "schema_version": 1,
        "name": "clarifycodebench-human-edited-150",
        "metadata": {
            "source_url": SOURCE,
            "source_commit": COMMIT,
            "selection": "150 of 419 tasks selected by stable hash of task_id",
            "source_scope": "human deletion-edited LiveCodeBench coding tasks with fine-grained underspecification labels and key clarification answers",
            "license": "ClarifyCodeBench annotation repository MIT; underlying LiveCodeBench tasks have separate terms",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", default=SOURCE, help="pinned source JSONL URL or local file URL"
    )
    args = parser.parse_args()
    dataset = prepare(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {CASE_COUNT} human-annotated coding tasks to {args.output}")


if __name__ == "__main__":
    main()
