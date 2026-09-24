"""Compare two paired evaluation reports without treating unavailable scores as ties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _comparable_options(options: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(options))
    copied.get("model_overrides", {}).pop("writer", None)
    return copied


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if (
        left["run_identity"]["dataset_digest"]
        != right["run_identity"]["dataset_digest"]
    ):
        raise ValueError("writer reports must use the same dataset")
    if _comparable_options(left["options"]) != _comparable_options(right["options"]):
        raise ValueError("writer reports differ in an option besides writer")
    left_cases = {case["case_id"]: case for case in left["cases"]}
    right_cases = {case["case_id"]: case for case in right["cases"]}
    if set(left_cases) != set(right_cases):
        raise ValueError("writer reports require the same case IDs")
    paired = left_wins = right_wins = ties = 0
    for case_id, a in left_cases.items():
        b = right_cases[case_id]
        if a["status"] in {"failed", "error"} or b["status"] in {"failed", "error"}:
            continue
        if a["score_delta"] is None or b["score_delta"] is None:
            continue
        paired += 1
        if a["score_delta"] > b["score_delta"] + 1e-12:
            left_wins += 1
        elif b["score_delta"] > a["score_delta"] + 1e-12:
            right_wins += 1
        else:
            ties += 1
    return {
        "dataset_digest": left["run_identity"]["dataset_digest"],
        "cases": len(left_cases),
        "both_scored": paired,
        "unavailable_pairwise": len(left_cases) - paired,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "left": {
            "writer": left["options"]["model_overrides"].get("writer", "default"),
            "completed": sum(
                case["status"] not in {"failed", "error"}
                for case in left_cases.values()
            ),
            "scored": left["improvement"]["comparable_cases"],
            "improvement": left["improvement"],
            "metered_openrouter_usd": left["cost"]["total"],
            "median_case_latency_ms": left["latency_ms"]["p50"],
        },
        "right": {
            "writer": right["options"]["model_overrides"].get("writer", "default"),
            "completed": sum(
                case["status"] not in {"failed", "error"}
                for case in right_cases.values()
            ),
            "scored": right["improvement"]["comparable_cases"],
            "improvement": right["improvement"],
            "metered_openrouter_usd": right["cost"]["total"],
            "median_case_latency_ms": right["latency_ms"]["p50"],
        },
        "cost_note": "Metered OpenRouter dollars exclude OpenCode Go subscription consumption; no per-call Go price is inferred.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    rendered = json.dumps(compare(left, right), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
