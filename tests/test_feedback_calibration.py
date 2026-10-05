"""Synthetic keep/reject feedback tests for explicit floor calibration."""

import pytest
from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.config import Settings
from prompt_enhancer.feedback_labels import (
    MINIMUM_LABELED_SAMPLE,
    calibrate_score_floors,
)
from prompt_enhancer.history import RunHistory
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.score_vector import SCORE_DIMENSIONS
from prompt_enhancer.store import RunStore


def _feedback(count: int, *, keep: float, reject: float) -> list[dict]:
    records = []
    for index in range(count):
        decision = "accept" if index % 2 == 0 else "reject"
        score = keep if decision == "accept" else reject
        records.append(
            {
                "run_id": f"run-{index}",
                "feedback_labels": {
                    "decision": decision,
                    "status": "linked",
                    "candidate_id": f"candidate-{index}",
                    "score_vector": {
                        dimension: score for dimension in SCORE_DIMENSIONS
                    },
                    "weak_dimensions": [],
                },
            }
        )
    return records


def test_insufficient_labeled_sample_leaves_floors_unchanged() -> None:
    shipped = {dimension: 0.6 for dimension in SCORE_DIMENSIONS}
    current = {dimension: 0.7 for dimension in SCORE_DIMENSIONS}

    result = calibrate_score_floors(
        _feedback(MINIMUM_LABELED_SAMPLE - 1, keep=0.9, reject=0.3),
        current,
        conservative_floors=shipped,
    )

    assert result.adjusted_floors == current
    assert result.sample_size == MINIMUM_LABELED_SAMPLE - 1
    assert result.changed is False
    assert all(value == 0.0 for value in result.movement.values())


def test_unlinked_feedback_vectors_do_not_count_toward_sample() -> None:
    shipped = {dimension: 0.6 for dimension in SCORE_DIMENSIONS}
    records = _feedback(MINIMUM_LABELED_SAMPLE, keep=0.9, reject=0.3)
    records[0]["feedback_labels"]["status"] = "unavailable"
    records[0]["feedback_labels"]["candidate_id"] = None

    result = calibrate_score_floors(records, shipped)

    assert result.sample_size == MINIMUM_LABELED_SAMPLE - 1
    assert result.applied is False


def test_valid_labeled_feedback_moves_floor_and_logs_sample_and_movement() -> None:
    shipped = {dimension: 0.6 for dimension in SCORE_DIMENSIONS}

    result = calibrate_score_floors(
        _feedback(MINIMUM_LABELED_SAMPLE, keep=0.95, reject=0.45), shipped
    )

    assert result.adjusted_floors["clarity"] == 0.7
    assert result.sample_size == MINIMUM_LABELED_SAMPLE
    assert result.changed is True
    assert result.movement["clarity"] == 0.1


def test_calibrated_floor_respects_conservative_lower_bound() -> None:
    shipped = {dimension: 0.6 for dimension in SCORE_DIMENSIONS}

    result = calibrate_score_floors(
        _feedback(MINIMUM_LABELED_SAMPLE, keep=0.5, reject=0.2), shipped
    )

    assert result.adjusted_floors["clarity"] == pytest.approx(0.45)


def test_explicit_api_recalibration_persists_for_next_optimizer(tmp_path) -> None:
    database = tmp_path / "feedback.sqlite3"
    config = Settings(database_path=str(database))
    store = RunStore(database)
    history = RunHistory(store)
    for record in _feedback(MINIMUM_LABELED_SAMPLE, keep=0.95, reject=0.45):
        history.save_run(
            {
                "run_id": record["run_id"],
                "prompt": "Test floor recalibration",
                "result": {
                    "status": "completed",
                    "final_prompt": "A calibrated prompt",
                },
                "feedback": record["feedback_labels"]["decision"],
                "feedback_labels": record["feedback_labels"],
            }
        )
    client = TestClient(create_app(store=store, settings=config))

    response = client.post("/api/quality/floors/recalibrate")

    assert response.status_code == 200
    assert response.json()["sample_size"] == MINIMUM_LABELED_SAMPLE
    assert response.json()["adjusted_floors"]["clarity"] == pytest.approx(0.7)
    optimizer = PromptOptimizer(
        store=RunStore(database), config=Settings(database_path=str(database))
    )
    assert optimizer.config.score_floors["clarity"] == pytest.approx(0.7)
    assert optimizer.get_model_settings()["score_floors"]["clarity"] == pytest.approx(
        0.7
    )


def test_insufficient_api_recalibration_preserves_persisted_active_floors(
    tmp_path,
) -> None:
    database = tmp_path / "existing-calibration.sqlite3"
    config = Settings(database_path=str(database), score_floor_clarity=0.7)
    store = RunStore(database)
    history = RunHistory(store)
    for record in _feedback(MINIMUM_LABELED_SAMPLE - 1, keep=0.9, reject=0.3):
        history.save_run(
            {
                "run_id": record["run_id"],
                "prompt": "Keep the existing calibrated floor",
                "result": {
                    "status": "completed",
                    "final_prompt": "Saved result",
                },
                "feedback": record["feedback_labels"]["decision"],
                "feedback_labels": record["feedback_labels"],
            }
        )
    optimizer = PromptOptimizer(store=store, config=config)

    result = optimizer.recalibrate_score_floors()

    assert result["applied"] is False
    assert result["sample_size"] == MINIMUM_LABELED_SAMPLE - 1
    assert result["floors_before"]["clarity"] == 0.7
    assert result["adjusted_floors"]["clarity"] == 0.7
    restarted = PromptOptimizer(
        store=RunStore(database), config=Settings(database_path=str(database))
    )
    assert restarted.config.score_floors["clarity"] == pytest.approx(0.7)
