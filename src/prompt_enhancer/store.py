"""SQLite persistence for completed or paused optimization runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunStore:
    """A local repository for run inputs, options, results, and feedback.

    Reports remain JSON so the schema can evolve while the frequently searched
    scalar fields stay indexed. Feedback is stored explicitly so a restart does
    not lose a user's accept/reject decision.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                prompt TEXT NOT NULL,
                options_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                cost_json TEXT NOT NULL,
                timing_json TEXT NOT NULL,
                feedback_json TEXT,
                feedback_at TEXT,
                evidence_json TEXT,
                legacy_metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "tier" in columns:
            self._migrate_tier_rows()
            columns = {
                row[1]
                for row in self._connection.execute(
                    "PRAGMA table_info(runs)"
                ).fetchall()
            }
        if "feedback_json" not in columns:
            self._connection.execute("ALTER TABLE runs ADD COLUMN feedback_json TEXT")
        if "feedback_at" not in columns:
            self._connection.execute("ALTER TABLE runs ADD COLUMN feedback_at TEXT")
        if "evidence_json" not in columns:
            self._connection.execute("ALTER TABLE runs ADD COLUMN evidence_json TEXT")
        if "legacy_metadata_json" not in columns:
            self._connection.execute(
                "ALTER TABLE runs ADD COLUMN legacy_metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS runs_created_at_idx ON runs(created_at DESC)"
        )
        self._connection.commit()

    def _migrate_tier_rows(self) -> None:
        """Copy the pre-consolidation schema without rewriting historical JSON."""
        old_rows = self._connection.execute("SELECT * FROM runs").fetchall()
        self._connection.execute("BEGIN")
        self._connection.execute("ALTER TABLE runs RENAME TO runs_legacy")
        self._connection.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                prompt TEXT NOT NULL,
                options_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                cost_json TEXT NOT NULL,
                timing_json TEXT NOT NULL,
                feedback_json TEXT,
                feedback_at TEXT,
                evidence_json TEXT,
                legacy_metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        try:
            for row in old_rows:
                values = dict(row)
                raw_options = json.loads(values.get("options_json") or "{}")
                raw_result = json.loads(values.get("result_json") or "{}")
                raw_evidence = (
                    json.loads(values["evidence_json"])
                    if values.get("evidence_json") is not None
                    else {}
                )
                report = (
                    raw_result.get("report") if isinstance(raw_result, dict) else None
                )
                legacy = {
                    "tier": values.get("tier"),
                    "options": raw_options,
                    "result": raw_result,
                    "cost": json.loads(values.get("cost_json") or "{}"),
                    "timing": json.loads(values.get("timing_json") or "{}"),
                    "feedback": (
                        json.loads(values["feedback_json"])
                        if values.get("feedback_json") is not None
                        else None
                    ),
                    "feedback_at": values.get("feedback_at"),
                    "evidence": raw_evidence,
                }
                if isinstance(report, dict):
                    if isinstance(report.get("outcome"), str):
                        legacy["recorded_outcome"] = report["outcome"]
                    if isinstance(report.get("applied_style"), str):
                        legacy["recorded_applied_style"] = report["applied_style"]
                current_options = dict(raw_options)
                current_options.pop("tier", None)
                historical_result: dict[str, Any] = {
                    "status": "legacy",
                    "legacy": True,
                }
                if isinstance(raw_result, dict):
                    final_prompt = raw_result.get("final_prompt")
                    if isinstance(final_prompt, str):
                        historical_result["final_prompt"] = final_prompt
                    if isinstance(raw_result.get("original_kept"), bool):
                        historical_result["original_kept"] = raw_result["original_kept"]
                self._connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, created_at, prompt, options_json, result_json, cost_json,
                        timing_json, feedback_json, feedback_at, evidence_json,
                        legacy_metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        values["run_id"],
                        values["created_at"],
                        values["prompt"],
                        json.dumps(current_options, ensure_ascii=False),
                        json.dumps(historical_result, ensure_ascii=False),
                        values.get("cost_json") or "{}",
                        values.get("timing_json") or "{}",
                        values.get("feedback_json"),
                        values.get("feedback_at"),
                        "{}",
                        json.dumps(legacy, ensure_ascii=False),
                    ),
                )
            self._connection.execute("DROP TABLE runs_legacy")
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def save_run(self, record: dict[str, Any]) -> str:
        run_id = str(record["run_id"])
        created_at = (
            str(record["created_at"])
            if "created_at" in record
            else datetime.now(UTC).isoformat()
        )
        result = dict(record.get("result") or {})
        feedback = record.get("feedback", result.get("feedback"))
        feedback_at = record.get("feedback_at", result.get("feedback_at"))
        known = {
            "run_id",
            "created_at",
            "prompt",
            "tier",
            "options",
            "result",
            "cost",
            "timing",
            "feedback",
            "feedback_at",
            "legacy_metadata",
        }
        evidence = {key: value for key, value in record.items() if key not in known}
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, created_at, prompt, options_json, result_json, cost_json,
                     timing_json, feedback_json, feedback_at, evidence_json, legacy_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    str(record["prompt"]),
                    json.dumps(record.get("options") or {}, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    json.dumps(
                        record.get("cost") or result.get("cost") or {},
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        record.get("timing") or result.get("timing") or {},
                        ensure_ascii=False,
                    ),
                    json.dumps(feedback, ensure_ascii=False)
                    if feedback is not None
                    else None,
                    str(feedback_at) if feedback_at is not None else None,
                    json.dumps(evidence, ensure_ascii=False),
                    json.dumps(record.get("legacy_metadata") or {}, ensure_ascii=False),
                ),
            )
            self._connection.commit()
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 200))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (bounded,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def all_runs(self) -> list[dict[str, Any]]:
        """Read every locally persisted run for offline training."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM runs ORDER BY created_at, run_id"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
        keys = set(row.keys())
        record: dict[str, Any] = {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "prompt": row["prompt"],
            "options": json.loads(row["options_json"]),
            "result": json.loads(row["result_json"]),
            "cost": json.loads(row["cost_json"]),
            "timing": json.loads(row["timing_json"]),
        }
        if "feedback_json" in keys and row["feedback_json"] is not None:
            record["feedback"] = json.loads(row["feedback_json"])
        if "feedback_at" in keys and row["feedback_at"] is not None:
            record["feedback_at"] = row["feedback_at"]
        legacy: dict[str, Any] = {}
        if "legacy_metadata_json" in keys and row["legacy_metadata_json"]:
            parsed_legacy = json.loads(row["legacy_metadata_json"])
            if isinstance(parsed_legacy, dict):
                legacy = parsed_legacy
        if "evidence_json" in keys and row["evidence_json"] is not None and not legacy:
            record.update(json.loads(row["evidence_json"]))
        if legacy:
            record["legacy_metadata"] = legacy
        return record


RunRepository = RunStore
