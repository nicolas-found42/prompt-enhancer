"""Maintainer CLI for offline CatBoost prompt-quality training.

Run with::

    python -m prompt_enhancer.training --database runs.sqlite3 \
        --artifact prompt-quality.cbm --report training-report.json

The command only reads a local JSON export and writes local artifacts.  It has
no provider credentials, gateway, or optimizer options.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .failure_prediction import (
    TrainingConfig,
    TrainingDataError,
    load_logged_runs,
    train_and_save,
)
from .quality_model import train_quality_model
from .store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the offline weak-panel failure predictor from logged runs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Local JSON export containing a list of logged runs.",
    )
    source.add_argument("--database", type=Path, help="Local RunStore SQLite database.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("prompt-quality.cbm"),
        help="Output path for the CatBoost model artifact.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("failure-predictor-report.json"),
        help="Output path for metrics, manifest, and priority queue.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument("--max-stumps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-split-gain", type=float, default=1e-4)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--legacy-stumps",
        action="store_true",
        help="Run the older binary stump trainer for compatibility.",
    )
    parser.add_argument(
        "--code-version",
        help="Release, commit, or other caller-supplied immutable code version.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.database is not None:
            store = RunStore(args.database)
            try:
                runs = store.all_runs()
            finally:
                store.close()
        else:
            runs = load_logged_runs(args.input)
        if not args.legacy_stumps:
            report = train_quality_model(
                runs,
                artifact_path=args.artifact,
                report_path=args.report,
                seed=args.seed,
                holdout_fraction=args.holdout_fraction,
                iterations=args.iterations,
                code_version=args.code_version,
            )
            print(
                json.dumps(
                    {
                        "artifact": report["manifest"]["artifact"],
                        "report": str(args.report),
                        "held_out_runs": len(report["evaluation"]["held_out_runs"]),
                        "model_mae": report["evaluation"]["model"]["mae"],
                        "baseline_mae": report["evaluation"]["baseline"]["mae"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        config = TrainingConfig(
            seed=args.seed,
            holdout_fraction=args.holdout_fraction,
            failure_threshold=args.failure_threshold,
            max_stumps=args.max_stumps,
            learning_rate=args.learning_rate,
            min_split_gain=args.min_split_gain,
            calibration_bins=args.calibration_bins,
            code_version=args.code_version,
        )
        report = train_and_save(
            runs,
            artifact_path=args.artifact,
            report_path=args.report,
            config=config,
        )
    except (TrainingDataError, OSError, ValueError) as exc:
        print(f"failure predictor training failed: {exc}", file=sys.stderr)
        return 2

    model_metrics = report["evaluation"]["model"]
    baseline_metrics = report["evaluation"]["baseline"]
    print(
        json.dumps(
            {
                "artifact": report["manifest"]["artifact"],
                "report": str(args.report),
                "held_out_runs": len(report["evaluation"]["held_out_runs"]),
                "model_roc_auc": model_metrics["discrimination"]["roc_auc"],
                "baseline_roc_auc": baseline_metrics["discrimination"]["roc_auc"],
                "model_brier_score": model_metrics["calibration"]["brier_score"],
                "model_feature_coverage": model_metrics["coverage"][
                    "observed_feature_fraction"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
