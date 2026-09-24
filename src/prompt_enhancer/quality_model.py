"""Offline CatBoost regression for original weak-panel pass rate.

This job reads local run evidence only. It does not initialize a model gateway or
make provider requests. CatBoost is an optional training dependency.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from typing import Any

from .failure_prediction import (
    TrainingConfig,
    TrainingDataError,
    _normalize_examples,
    _prediction_metrics,
    _split_examples,
)


def _source_digest() -> str:
    source_root = Path(__file__).parent
    digest = sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def train_quality_model(
    runs: Iterable[Mapping[str, Any]],
    *,
    artifact_path: str | Path,
    report_path: str | Path,
    seed: int = 1729,
    holdout_fraction: float = 0.25,
    iterations: int = 100,
    code_version: str | None = None,
) -> dict[str, Any]:
    """Fit a continuous pass-rate model and write a held-out audit report."""
    try:
        from catboost import CatBoostRegressor, Pool  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise TrainingDataError(
            "CatBoost is required; install prompt-enhancer[training]"
        ) from exc
    if iterations < 1:
        raise ValueError("iterations must be positive")
    config = TrainingConfig(seed=seed, holdout_fraction=holdout_fraction)
    examples = _normalize_examples(runs, config.failure_threshold)
    run_set_digest = sha256(
        json.dumps(
            [
                {
                    "run_id": row.run_id,
                    "features": row.features,
                    "pass_rate": row.weak_panel_pass_rate,
                }
                for row in sorted(examples, key=lambda example: example.run_id)
            ],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if len(examples) < config.minimum_training_runs + 1:
        raise TrainingDataError(
            "at least three runs with original weak-panel scores are required"
        )
    training, held_out = _split_examples(examples, config)
    names = tuple(sorted({key for example in training for key in example.features}))
    if not names:
        raise TrainingDataError("training runs contain no Jev or score features")

    def matrix(rows: list[Any]) -> list[list[float]]:
        return [
            [row.features.get(name, float("nan")) for name in names] for row in rows
        ]

    model = CatBoostRegressor(
        iterations=iterations,
        depth=4,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        Pool(
            matrix(training),
            [row.weak_panel_pass_rate for row in training],
            feature_names=list(names),
        )
    )
    predictions = [
        max(0.0, min(1.0, float(value)))
        for value in model.predict(Pool(matrix(held_out), feature_names=list(names)))
    ]
    labels = [row.weak_panel_pass_rate for row in held_out]
    baseline = fmean(row.weak_panel_pass_rate for row in training)
    failure_labels = [row.failure_label for row in held_out]

    artifact = Path(artifact_path)
    report = Path(report_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=artifact.parent, suffix=".cbm", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        model.save_model(str(temporary_path), format="cbm")
        os.replace(temporary_path, artifact)
    finally:
        temporary_path.unlink(missing_ok=True)
    digest = sha256(artifact.read_bytes()).hexdigest()
    priority_queue = [
        {
            "run_id": row.run_id,
            "observed_pass_rate": row.weak_panel_pass_rate,
            "predicted_pass_rate": prediction,
            "absolute_error": abs(row.weak_panel_pass_rate - prediction),
            "observation": "Model overestimated prompt quality"
            if prediction > row.weak_panel_pass_rate
            else "Model underestimated prompt quality",
        }
        for row, prediction in zip(held_out, predictions, strict=True)
    ]
    priority_queue.sort(
        key=lambda item: (-float(item["absolute_error"]), str(item["run_id"]))
    )
    payload = {
        "schema_version": 1,
        "offline": True,
        "network_calls": 0,
        "manifest": {
            "algorithm": "CatBoostRegressor",
            "label": "original_weak_panel.mean_pass_rate",
            "code_version": code_version or _source_digest(),
            "run_set": {"count": len(examples), "sha256": f"sha256:{run_set_digest}"},
            "config": {
                "seed": seed,
                "holdout_fraction": holdout_fraction,
                "failure_threshold": config.failure_threshold,
                "calibration_bins": config.calibration_bins,
                "iterations": iterations,
                "depth": 4,
                "learning_rate": 0.05,
                "loss_function": "RMSE",
                "thread_count": 1,
            },
            "feature_names": list(names),
            "training_run_ids": [row.run_id for row in training],
            "held_out_run_ids": [row.run_id for row in held_out],
            "artifact": {"path": str(artifact), "sha256": f"sha256:{digest}"},
        },
        "evaluation": {
            "held_out_runs": [row.run_id for row in held_out],
            "model": {
                **_regression_metrics(labels, predictions),
                **_prediction_metrics(
                    failure_labels,
                    [1.0 - value for value in predictions],
                    names,
                    held_out,
                    config.calibration_bins,
                ),
            },
            "baseline": {
                **_regression_metrics(labels, [baseline] * len(labels)),
                **_prediction_metrics(
                    failure_labels,
                    [1.0 - baseline] * len(labels),
                    names,
                    held_out,
                    config.calibration_bins,
                ),
            },
        },
        "priority_queue": priority_queue,
    }
    with NamedTemporaryFile(
        dir=report.parent, mode="w", encoding="utf-8", delete=False
    ) as temp:
        temporary_report = Path(temp.name)
        json.dump(payload, temp, ensure_ascii=False, sort_keys=True, indent=2)
        temp.write("\n")
    try:
        os.replace(temporary_report, report)
    finally:
        temporary_report.unlink(missing_ok=True)
    return payload


def _regression_metrics(
    labels: list[float], predictions: list[float]
) -> dict[str, float]:
    errors = [
        actual - prediction
        for actual, prediction in zip(labels, predictions, strict=True)
    ]
    return {
        "mae": fmean(abs(error) for error in errors),
        "rmse": math.sqrt(fmean(error * error for error in errors)),
    }
