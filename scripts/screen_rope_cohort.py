"""Exclude unresolved template placeholders from a recorded ROPE cohort."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_PLACEHOLDER = re.compile(
    r"\b(?:XXX|TBD|placeholder|fill in)\b|\[[^\]]+\]|<[^>]+>|\{[^{}]+\}", re.IGNORECASE
)


def screen(dataset: dict[str, Any]) -> dict[str, Any]:
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise TypeError("ROPE dataset requires cases")
    retained = [
        case
        for case in cases
        if isinstance(case, dict)
        and not _PLACEHOLDER.search(str(case.get("prompt", "")))
    ]
    retained_ids = {case["id"] for case in retained}
    excluded = [str(case["id"]) for case in cases if case["id"] not in retained_ids]
    metadata = dict(dataset.get("metadata", {}))
    metadata["screen"] = "excluded unresolved template placeholders by fixed regex"
    metadata["excluded_case_ids"] = excluded
    metadata["task_counts"] = dict(
        Counter(str(case["task_stratum"]) for case in retained)
    )
    return {
        **dataset,
        "name": str(dataset["name"]) + "-screened",
        "metadata": metadata,
        "cases": retained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = screen(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Retained {len(result['cases'])} cases after placeholder screening")


if __name__ == "__main__":
    main()
