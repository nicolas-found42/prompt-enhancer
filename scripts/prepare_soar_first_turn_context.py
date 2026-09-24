"""Select 150 first-turn SOAR prompts for the exact context-gap question.

The source's processed prompt representation can hide code and error text.
Exclude obvious placeholders and retain source labels separately. This set is
only for context calibration, not for judging other checklist questions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from urllib.request import urlopen

from evaluation_review_common import has_soar_placeholder
from prepare_soar_prompt_gaps import COMMIT, SOURCE

POSITIVE_COUNT = 80
NEGATIVE_COUNT = 70


def prepare(text: str, *, split: str = "development") -> dict[str, object]:
    if split not in {"development", "holdout"}:
        raise ValueError("split must be development or holdout")
    csv.field_size_limit(sys.maxsize)
    positive: list[dict[str, object]] = []
    negative: list[dict[str, object]] = []
    for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        prompts = ast.literal_eval(row["prompts"])
        annotations = ast.literal_eval(row["annotated_gaps"])
        if not prompts or not annotations:
            continue
        prompt = prompts[0]
        labels = {str(label).lower() for label in annotations[0]}
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > 8000
            or has_soar_placeholder(prompt)
        ):
            continue
        case = {
            "id": f"soar-first-{row['conversation_id']}",
            "source": "real",
            "prompt": prompt,
            "expected_gaps": ["context"] if "missing context" in labels else [],
            "source_labels": sorted(labels),
            "evaluation_gaps": ["context"],
            "conversation_id": row["conversation_id"],
            "turn": 1,
        }
        if "missing context" in labels:
            positive.append(case)
        elif labels == {"no gap"}:
            negative.append(case)

    def unique_sorted(cases: list[dict[str, object]]) -> list[dict[str, object]]:
        seen_prompts: set[str] = set()
        selected: list[dict[str, object]] = []
        for case in sorted(
            cases, key=lambda item: hashlib.sha256(str(item["id"]).encode()).hexdigest()
        ):
            prompt = str(case["prompt"]).strip().casefold()
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            selected.append(case)
        return selected

    positive = unique_sorted(positive)
    negative = unique_sorted(negative)
    if len(positive) < POSITIVE_COUNT + 20 or len(negative) < NEGATIVE_COUNT + 40:
        raise ValueError(
            "source lacks enough first-turn context labels after screening"
        )
    if split == "development":
        chosen_positive, chosen_negative = (
            positive[:POSITIVE_COUNT],
            negative[:NEGATIVE_COUNT],
        )
    else:
        chosen_positive = positive[POSITIVE_COUNT : POSITIVE_COUNT + 20]
        chosen_negative = negative[NEGATIVE_COUNT : NEGATIVE_COUNT + 40]
    selected = sorted(
        chosen_positive + chosen_negative, key=lambda case: str(case["id"])
    )
    return {
        "schema_version": 1,
        "name": f"soar-first-turn-context-{split}-{len(selected)}",
        "metadata": {
            "source_url": SOURCE,
            "source_commit": COMMIT,
            "selection": f"{len(chosen_positive)} missing-context and {len(chosen_negative)} no-gap first turns in the {split} slice of stable SHA-256 ID ordering; obvious processed-text placeholders excluded",
            "source_scope": "single first turns from distinct developer conversations; exact source context label only",
            "redistribution": "source repository declares no license; generated file remains local",
        },
        "cases": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "holdout"), default="development"
    )
    args = parser.parse_args()
    with urlopen(SOURCE, timeout=30) as response:
        dataset = prepare(response.read().decode("utf-8-sig"), split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    count = POSITIVE_COUNT + NEGATIVE_COUNT if args.split == "development" else 60
    print(f"Wrote {count} first-turn context cases to {args.output}")


if __name__ == "__main__":
    main()
