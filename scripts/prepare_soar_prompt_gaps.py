"""Build a local evaluation set from human-annotated real developer prompts.

The MSR 2025 replication package contains public developer–ChatGPT conversations
shared in GitHub issues. Its labels apply to individual turns. Source prompts
are fetched from a pinned revision and are not redistributed in this repository.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

COMMIT = "39407a7a9ad83488f0ff593e9212149b72b6532d"
SOURCE = (
    "https://raw.githubusercontent.com/SOAR-Lab/prompt-knowledge-gap/"
    f"{COMMIT}/data/conversations.csv"
)
QUOTAS = {
    "multiple context": 8,
    "unclear instruction": 18,
    "missing specification": 20,
    "missing context": 52,
    "no gap": 52,
}
MAX_PROMPT_LENGTH = 8000


@dataclass(frozen=True)
class Candidate:
    id: str
    conversation_id: str
    turn: int
    prompt: str
    labels: frozenset[str]


def _labels(raw: object) -> frozenset[str]:
    if not isinstance(raw, set):
        raise TypeError("expected a set of labels per prompt turn")
    labels = {
        str(label).strip().lower().replace("missing specfications", "missing specification")
        for label in raw
    }
    labels.discard("")
    if not labels or not labels <= QUOTAS.keys():
        raise ValueError(f"unexpected gap labels: {labels}")
    if "no gap" in labels and len(labels) != 1:
        raise ValueError("no gap cannot occur with another label")
    return frozenset(labels)


def prepare(source: str) -> dict[str, object]:
    with urlopen(source, timeout=30) as response:
        raw_csv = response.read().decode("utf-8-sig")
    csv.field_size_limit(sys.maxsize)
    available: list[Candidate] = []
    for row in csv.DictReader(io.StringIO(raw_csv)):
        prompts = ast.literal_eval(row["prompts"])
        annotations = ast.literal_eval(row["annotated_gaps"])
        if not isinstance(prompts, list) or len(prompts) != len(annotations):
            raise ValueError(f"prompt/annotation mismatch in conversation {row['conversation_id']}")
        for index, (prompt, raw_labels) in enumerate(zip(prompts, annotations, strict=True), start=1):
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_LENGTH:
                continue
            available.append(Candidate(
                id=f"soar-{row['conversation_id']}-turn-{index}",
                conversation_id=row["conversation_id"],
                turn=index,
                prompt=prompt,
                labels=_labels(raw_labels),
            ))

    selected: list[Candidate] = []
    used_conversations: set[str] = set()
    for label, quota in QUOTAS.items():
        candidates = sorted(
            (case for case in available if label in case.labels),
            key=lambda case: hashlib.sha256(case.id.encode()).hexdigest(),
        )
        added = 0
        for case in candidates:
            conversation_id = case.conversation_id
            if conversation_id in used_conversations:
                continue
            selected.append(case)
            used_conversations.add(conversation_id)
            added += 1
            if added == quota:
                break
        if added != quota:
            raise ValueError(f"only {added} distinct conversations available for {label}; need {quota}")

    cases = [
        {
            "id": case.id,
            "source": "real",
            "prompt": case.prompt,
            "expected_gaps": sorted(case.labels - {"no gap"}),
            "source_labels": sorted(case.labels),
            "evaluation_gaps": ["context"],
            "conversation_id": case.conversation_id,
            "turn": case.turn,
        }
        for case in sorted(selected, key=lambda case: case.id)
    ]
    return {
        "schema_version": 1,
        "name": "soar-human-prompt-gaps-150",
        "metadata": {
            "source_url": SOURCE,
            "source_commit": COMMIT,
            "source_paper": "Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution (MSR 2025)",
            "selection": "150 distinct conversations, stratified by annotated gap type; prompts over 8000 characters excluded",
            "source_scope": "real developer prompts shared in GitHub issues; code and error text may be replaced with placeholders in the source",
            "redistribution": "source repository has no declared license; generated file is local only",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default=SOURCE, help="pinned source CSV URL or local file URL")
    args = parser.parse_args()
    dataset = prepare(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sum(QUOTAS.values())} real, human-labeled prompts to {args.output}")


if __name__ == "__main__":
    main()
