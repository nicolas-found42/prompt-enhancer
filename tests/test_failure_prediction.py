from __future__ import annotations

import json
from pathlib import Path

from prompt_enhancer.failure_prediction import (
    FailurePredictionModel,
    TrainingConfig,
    load_logged_runs,
    train_and_save,
    train_failure_predictor,
)
from prompt_enhancer.training import main

FIXTURE = Path(__file__).parent / "fixtures" / "failure_prediction_runs.json"


def test_trains_from_jev_scores_and_weak_outcomes_with_held_out_report() -> None:
    runs = load_logged_runs(FIXTURE)
    result = train_failure_predictor(
        runs,
        TrainingConfig(seed=7, holdout_fraction=0.25, code_version="test-revision"),
    )

    training_ids = set(result.evaluation["training_runs"])
    held_out_ids = set(result.evaluation["held_out_runs"])
    assert training_ids.isdisjoint(held_out_ids)
    assert training_ids | held_out_ids == {run["run_id"] for run in runs}
    assert "jev/noul/goal.missing/probability" in result.model.feature_names
    assert "score/fidelity/mean" in result.model.feature_names
    assert "score/fidelity/spread" in result.model.feature_names
    assert result.model.training_config["label_source"] == (
        "original_weak_panel_pass_rate"
    )

    model_metrics = result.evaluation["model"]
    assert {"discrimination", "calibration", "coverage"} <= model_metrics.keys()
    assert model_metrics["discrimination"]["roc_auc"] is not None
    assert 0 <= model_metrics["calibration"]["brier_score"] <= 1
    assert model_metrics["coverage"]["sample_fraction"] == 1
    assert model_metrics["coverage"]["observed_feature_fraction"] == 1
    assert result.evaluation["baseline"]["discrimination"]["roc_auc"] is None
    assert len(result.priority_queue) == len(held_out_ids)
    assert {"run_id", "reason", "top_contributions"} <= result.priority_queue[0].keys()


def test_repeated_training_writes_reproducible_artifact_and_manifest(
    tmp_path: Path,
) -> None:
    runs = load_logged_runs(FIXTURE)
    config = TrainingConfig(
        seed=19,
        holdout_fraction=0.25,
        max_stumps=8,
        code_version="fixture-revision",
    )
    first = train_and_save(
        runs, tmp_path / "first.json", tmp_path / "first-report.json", config
    )
    second = train_and_save(
        runs, tmp_path / "second.json", tmp_path / "second-report.json", config
    )

    first_artifact = json.loads((tmp_path / "first.json").read_text())
    second_artifact = json.loads((tmp_path / "second.json").read_text())
    assert (
        first["manifest"]["artifact"]["sha256"]
        == second["manifest"]["artifact"]["sha256"]
    )
    assert first["manifest"]["run_set"] == second["manifest"]["run_set"]
    assert first["manifest"]["config"] == second["manifest"]["config"]
    assert first["manifest"]["code_version"] == "fixture-revision"
    assert first["evaluation"] == second["evaluation"]
    assert FailurePredictionModel.from_dict(first_artifact).predict(
        {"jev/noul/goal.missing/probability": 0.9}
    ) == FailurePredictionModel.from_dict(second_artifact).predict(
        {"jev/noul/goal.missing/probability": 0.9}
    )


def test_cli_trains_from_local_json_without_credentials(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_KEY", raising=False)
    artifact = tmp_path / "predictor.json"
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--input",
            str(FIXTURE),
            "--legacy-stumps",
            "--artifact",
            str(artifact),
            "--report",
            str(report_path),
            "--seed",
            "23",
            "--code-version",
            "cli-revision",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text())
    assert output["held_out_runs"] == len(report["evaluation"]["held_out_runs"])
    assert report["offline"] is True
    assert report["network_calls"] == 0
    assert report["manifest"]["code_version"] == "cli-revision"
    assert report["manifest"]["artifact"]["path"] == str(artifact)
    assert artifact.exists()


def test_catboost_cli_trains_continuous_pass_rate_from_local_run_store(
    tmp_path: Path, capsys
) -> None:
    import pytest

    pytest.importorskip("catboost")
    from catboost import CatBoostRegressor

    from prompt_enhancer.store import RunStore

    database = tmp_path / "runs.sqlite3"
    store = RunStore(database)
    for run in load_logged_runs(FIXTURE):
        store.save_run({**run, "prompt": f"Original prompt for {run['run_id']}"})
    store.close()
    artifact = tmp_path / "quality.cbm"
    report_path = tmp_path / "quality-report.json"

    exit_code = main(
        [
            "--database",
            str(database),
            "--artifact",
            str(artifact),
            "--report",
            str(report_path),
            "--iterations",
            "20",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text())
    assert output["held_out_runs"] == 2
    assert report["manifest"]["algorithm"] == "CatBoostRegressor"
    assert report["manifest"]["label"] == "original_weak_panel.mean_pass_rate"
    assert report["manifest"]["code_version"].startswith("sha256:")
    assert report["manifest"]["run_set"]["count"] == 8
    assert report["manifest"]["run_set"]["sha256"].startswith("sha256:")
    assert report["manifest"]["config"]["depth"] == 4
    assert {"discrimination", "calibration", "coverage"} <= report["evaluation"][
        "model"
    ].keys()
    assert report["evaluation"]["model"]["mae"] >= 0
    assert artifact.exists()
    model = CatBoostRegressor()
    model.load_model(str(artifact))
    assert model.tree_count_ > 0
