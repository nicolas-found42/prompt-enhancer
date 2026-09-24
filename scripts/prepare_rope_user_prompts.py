"""Select participant-written ROPE study prompts for a real, cross-task probe.

These are not gap-labeled. Keep each participant's four original prompts in
one split and exclude system-optimized and participant-iterated versions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "1ada01830031e5882f2585577720b182deac6246"
SOURCE = f"https://raw.githubusercontent.com/mqo00/rope/{COMMIT}/study_material/user_study_prompts.csv"
TASKS = ("Connect4", "TicTacToe", "OutlineAssistant", "TripAdvisor")


def prepare(text: str, *, subjects: int = 12) -> dict[str, object]:
    if not 1 <= subjects <= 30:
        raise ValueError("subjects must be between 1 and 30")
    originals = [row for row in csv.DictReader(io.StringIO(text)) if row["prompt_version"] == "original"]
    by_subject: dict[str, dict[str, dict[str, str]]] = {}
    for row in originals:
        subject = row["subject"]
        task = row["task"]
        if task not in TASKS or not row["prompt"].strip():
            continue
        tasks = by_subject.setdefault(subject, {})
        if task in tasks:
            raise ValueError(f"duplicate original for {subject}/{task}")
        tasks[task] = row
    if len(by_subject) != 30 or any(set(rows) != set(TASKS) for rows in by_subject.values()):
        raise ValueError("expected 30 participants with all four original tasks")
    selected = sorted(by_subject, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:subjects]
    cases = [
        {
            "id": f"rope-{subject}-{task.lower()}",
            "source": "real",
            "prompt": by_subject[subject][task]["prompt"].strip(),
            "task_stratum": task,
            "participant": subject,
            "labels_present": False,
        }
        for subject in sorted(selected)
        for task in TASKS
    ]
    if len({str(case["prompt"]).casefold() for case in cases}) != len(cases):
        raise ValueError("duplicate participant prompts in selected cohort")
    return {
        "schema_version": 1,
        "name": f"rope-original-{subjects}-participants",
        "metadata": {
            "source_url": SOURCE,
            "source_commit": COMMIT,
            "license": "Apache-2.0",
            "selection": f"{subjects} participants selected by SHA-256 subject hash, four original prompts per participant",
            "scope": "participant-authored study prompts across two game, outline-assistant, and travel-assistant tasks; no per-prompt gap gold",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=12)
    args = parser.parse_args()
    with urlopen(SOURCE, timeout=30) as response:
        dataset = prepare(response.read().decode("utf-8"), subjects=args.subjects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.subjects * len(TASKS)} participant prompts to {args.output}")


if __name__ == "__main__":
    main()
