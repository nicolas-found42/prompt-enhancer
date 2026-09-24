"""Summarize stage attrition from strict product-engine evaluation replay.

The output contains counts and case IDs only, never prompt or model text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from prompt_enhancer.evaluation.datasets import load_dataset, replay_digest
from prompt_enhancer.evaluation.harness import HarnessOptions, default_engine_factory
from prompt_enhancer.optimizer import PromptOptimizer


def analyze(
    dataset_path: Path, replay_path: Path, options: HarnessOptions
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    engine = cast(PromptOptimizer, default_engine_factory(replay_path))
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    paths: dict[str, list[str]] = {}
    for case in dataset.cases:
        engine.gateway.decision_log.clear()
        result = engine.optimize(case.prompt, options.optimize_options())
        report = result.get("report", {})
        status = result.get("status")
        counts["cases"] += 1
        counts[f"status:{status}"] += 1
        if status == "failed":
            paths.setdefault("failed", []).append(case.id)
            continue
        diagnosis = report.get("diagnosis", {})
        gaps = diagnosis.get("confirmed_gaps", [])
        accepted = report.get("tests", [])
        proposed = sum(
            str(item["question"].get("key", "")).startswith("faithful:")
            for item in engine.gateway.decision_log
        )
        counts["cases_with_confirmed_gap"] += bool(gaps)
        counts["cases_with_test_proposal"] += proposed > 0
        counts["proposed_tests"] += proposed
        counts["accepted_tests"] += len(accepted)
        counts["cases_with_accepted_test"] += bool(accepted)
        counts["cases_with_both"] += bool(gaps and accepted)
        candidates = report.get("candidates", [])
        counts["cases_with_candidate"] += bool(candidates)
        counts["candidates"] += len(candidates)
        counts["fidelity_passed"] += sum(
            candidate.get("metadata", {}).get("fidelity", {}).get("passed", False)
            for candidate in candidates
        )
        strong = report.get("strong_check", {})
        counts["strong_attempted"] += len(strong.get("candidates", []))
        counts["strong_passed"] += sum(
            item.get("passed", False) for item in strong.get("candidates", [])
        )
        counts["cases_with_comparable_score"] += bool(report.get("selection_evidence"))
        for candidate in candidates:
            reasons.update(candidate.get("rejection_reasons", []))
        if not gaps:
            path = "no_confirmed_gap"
        elif not accepted:
            path = "gap_without_accepted_test"
        elif not candidates:
            path = "no_candidate"
        else:
            path = "scored_candidate"
        paths.setdefault(path, []).append(case.id)
    return {
        "dataset_digest": dataset.digest,
        "replay_digest": replay_digest(replay_path),
        "options": options.to_dict(),
        "counts": dict(counts),
        "candidate_rejection_reasons": dict(reasons),
        "case_ids_by_path": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--tier", choices=("fast", "standard", "deep"), default="fast")
    parser.add_argument("--writer-model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    overrides = {"writer": args.writer_model} if args.writer_model else {}
    report = analyze(
        args.dataset,
        args.replay,
        HarnessOptions(tier=args.tier, model_overrides=overrides),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
