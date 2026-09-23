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
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "feedback_json" not in columns:
            self._connection.execute("ALTER TABLE runs ADD COLUMN feedback_json TEXT")
        if "feedback_at" not in columns:
            self._connection.execute("ALTER TABLE runs ADD COLUMN feedback_at TEXT")
        if "evidence_json" not in columns:
            self._connection.execute("ALTER TABLE runs ADD COLUMN evidence_json TEXT")
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS runs_created_at_idx ON runs(created_at DESC)"
        )
        self._connection.commit()

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
        known = {"run_id", "created_at", "prompt", "tier", "options", "result", "cost", "timing", "feedback", "feedback_at"}
        evidence = {key: value for key, value in record.items() if key not in known}
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, created_at, prompt, tier, options_json,
                     result_json, cost_json, timing_json, feedback_json, feedback_at, evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    str(record["prompt"]),
                    str(record.get("tier", "standard")),
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
            "tier": row["tier"],
            "options": json.loads(row["options_json"]),
            "result": json.loads(row["result_json"]),
            "cost": json.loads(row["cost_json"]),
            "timing": json.loads(row["timing_json"]),
        }
        if "feedback_json" in keys and row["feedback_json"] is not None:
            record["feedback"] = json.loads(row["feedback_json"])
        if "feedback_at" in keys and row["feedback_at"] is not None:
            record["feedback_at"] = row["feedback_at"]
        if "evidence_json" in keys and row["evidence_json"] is not None:
            record.update(json.loads(row["evidence_json"]))
        return record


RunRepository = RunStore
