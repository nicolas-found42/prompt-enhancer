"""Build a local, stratified evaluation set from ClariQ's human labels.

The source data is fetched from a pinned commit and is not redistributed in
this repository. ``clarification_need`` is the only gold gap label supplied by
ClariQ; this script does not infer more specific rubric gaps.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "46885a544581a0af8aff0681d29e4971807e2912"
SOURCE = f"https://raw.githubusercontent.com/aliannejadi/ClariQ/{COMMIT}/data/train.tsv"
QUOTAS = {"1": 25, "2": 50, "3": 50, "4": 25}


def prepare(source: str, *, per_level: int | None = None) -> dict[str, object]:
    if per_level is not None and per_level < 1:
        raise ValueError("per_level must be positive")
    quotas = {level: per_level for level in QUOTAS} if per_level is not None else QUOTAS
    with urlopen(source, timeout=30) as response:
        text = response.read().decode("utf-8")
    topics: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        topic_id = row.get("topic_id", "")
        if topic_id and topic_id not in topics:
            topics[topic_id] = row
    by_level: dict[str, list[dict[str, str]]] = {level: [] for level in QUOTAS}
    for _topic_id, row in sorted(topics.items(), key=lambda item: int(item[0])):
        level = row.get("clarification_need", "")
        if level in by_level and row.get("initial_request", "").strip():
            by_level[level].append(row)
    for level, quota in quotas.items():
        if len(by_level[level]) < quota:
            raise ValueError(
                f"ClariQ level {level} has fewer than {quota} unique prompts"
            )
    cases = [
        {
            "id": f"clariq-{row['topic_id']}",
            "source": "real",
            "prompt": row["initial_request"].strip(),
            "expected_gaps": [] if level == "1" else ["clarification_need"],
            "evaluation_gaps": ["clarification_need"],
            "evaluation_notes": f"ClariQ human clarification_need={level}; no specific rubric gap labels are implied.",
        }
        for level, quota in quotas.items()
        for row in by_level[level][:quota]
    ]
    cases.sort(key=lambda case: case["id"])
    return {
        "schema_version": 1,
        "name": f"clariq-human-clarification-{sum(quotas.values())}",
        "metadata": {
            "source_url": SOURCE,
            "source_commit": COMMIT,
            "gold_label": "human-rated clarification_need (1=no clarification; 2-4=clarification needed)",
            "scope": "real search requests; binary clarification need, not detailed gap types",
            "selection": f"{quotas} unique topics by clarification-need rating",
            "redistribution": "source repository does not declare a license; generated file is local only",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", default=SOURCE, help="pinned ClariQ train.tsv URL or local file URL"
    )
    parser.add_argument(
        "--per-level", type=int, help="select this many topics from each of ratings 1–4"
    )
    args = parser.parse_args()
    dataset = prepare(args.source, per_level=args.per_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    count = (
        args.per_level * len(QUOTAS)
        if args.per_level is not None
        else sum(QUOTAS.values())
    )
    print(f"Wrote {count} real prompts to {args.output}")


if __name__ == "__main__":
    main()
