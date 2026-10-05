"""Store migrations keep pre-consolidation rows explicitly historical."""

from __future__ import annotations

import json
import sqlite3

from prompt_enhancer.history import RunHistory
from prompt_enhancer.store import RunStore


def test_legacy_tier_row_is_copied_losslessly_into_legacy_metadata(tmp_path) -> None:
    database = tmp_path / "pre-consolidation.sqlite3"
    created_at = "2026-10-01T10:11:12+00:00"
    options = {"tier": "standard", "seed": 9, "model_overrides": {"weak": ["a"]}}
    result = {
        "status": "completed",
        "final_prompt": "Old final prompt",
        "original_kept": False,
        "report": {
            "status": "unverified",
            "outcome": "improved_unverified",
            "applied_style": "clearer",
            "summary": "We couldn't test this prompt",
            "evaluation_evidence": {"opaque": [1, "raw"]},
            "tier": "standard",
        },
    }
    feedback = {"decision": "reject", "source": "user"}
    feedback_at = "2026-10-02T11:12:13+00:00"
    cost = {"total": 0.25}
    timing = {"total_ms": 321}
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            prompt TEXT NOT NULL,
            tier TEXT NOT NULL,
            options_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            cost_json TEXT NOT NULL,
            timing_json TEXT NOT NULL,
            feedback_json TEXT,
            feedback_at TEXT,
            evidence_json TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-1",
            created_at,
            "Old original prompt",
            "Standard",
            json.dumps(options),
            json.dumps(result),
            json.dumps(cost),
            json.dumps(timing),
            json.dumps(feedback),
            feedback_at,
            json.dumps({"extra": {"keep": True}}),
        ),
    )
    connection.commit()
    connection.close()

    store = RunStore(database)
    record = store.get_run("legacy-1")
    assert record is not None
    assert record["run_id"] == "legacy-1"
    assert record["created_at"] == created_at
    assert record["prompt"] == "Old original prompt"
    assert record["cost"] == cost
    assert record["timing"] == timing
    assert record["feedback"] == feedback
    assert record["feedback_at"] == feedback_at
    assert "extra" not in record
    assert "tier" not in record
    assert record["options"] == {"seed": 9, "model_overrides": {"weak": ["a"]}}
    assert record["result"] == {
        "status": "legacy",
        "legacy": True,
        "final_prompt": "Old final prompt",
        "original_kept": False,
    }
    assert record["legacy_metadata"] == {
        "tier": "Standard",
        "options": options,
        "result": result,
        "cost": cost,
        "timing": timing,
        "feedback": feedback,
        "feedback_at": feedback_at,
        "evidence": {"extra": {"keep": True}},
        "recorded_outcome": "improved_unverified",
        "recorded_applied_style": "clearer",
    }
    columns = {row[1] for row in store._connection.execute("PRAGMA table_info(runs)")}
    assert "tier" not in columns

    reopened = RunStore(database)
    assert reopened.get_run("legacy-1") == record
    historical = RunHistory(reopened).get_run("legacy-1")
    assert historical is not None
    assert historical["outcome"] is None
    assert historical["outcome_reason"] is None
    assert historical["applied_style"] == "clearer"
    assert historical["report"] == {}
    assert historical["legacy_metadata"]["result"] == result
    assert "couldn't test" not in str(historical["report"]).casefold()

    new_id = store.save_run(
        {
            "run_id": "new-1",
            "created_at": created_at,
            "prompt": "Current prompt",
            "options": {"seed": 4},
            "result": {"status": "completed"},
        }
    )
    assert store.get_run(new_id) == {
        "run_id": "new-1",
        "created_at": created_at,
        "prompt": "Current prompt",
        "options": {"seed": 4},
        "result": {"status": "completed"},
        "cost": {},
        "timing": {},
    }


def test_earliest_tier_schema_migrates_without_optional_columns(tmp_path) -> None:
    database = tmp_path / "earliest.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            prompt TEXT NOT NULL,
            tier TEXT NOT NULL,
            options_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            cost_json TEXT NOT NULL,
            timing_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "earliest-1",
            "2026-09-01T00:00:00+00:00",
            "Prompt",
            "Fast",
            '{"tier":"fast","seed":1}',
            '{"status":"completed","report":{"status":"clarified"}}',
            '{"total":0.1}',
            '{"total_ms":15}',
        ),
    )
    connection.commit()
    connection.close()

    store = RunStore(database)
    record = store.get_run("earliest-1")
    assert record is not None
    assert record["legacy_metadata"]["tier"] == "Fast"
    assert record["legacy_metadata"]["options"] == {"tier": "fast", "seed": 1}
    columns = {row[1] for row in store._connection.execute("PRAGMA table_info(runs)")}
    assert {
        "feedback_json",
        "feedback_at",
        "evidence_json",
        "legacy_metadata_json",
    } <= columns
    assert "tier" not in columns


def test_invalid_legacy_json_rolls_back_schema_copy(tmp_path) -> None:
    database = tmp_path / "invalid-legacy-json.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            prompt TEXT NOT NULL,
            tier TEXT NOT NULL,
            options_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            cost_json TEXT NOT NULL,
            timing_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO runs VALUES ('bad-1', 'time', 'prompt', 'Deep', '{bad', '{}', '{}', '{}')"
    )
    connection.commit()
    connection.close()

    try:
        RunStore(database)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("invalid legacy JSON should stop the migration")

    connection = sqlite3.connect(database)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    assert "tier" in columns
    assert connection.execute("SELECT run_id FROM runs").fetchone() == ("bad-1",)
    connection.close()
