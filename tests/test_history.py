from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from prompt_enhancer.history import RunHistory, register_history_routes
from prompt_enhancer.store import RunStore


def _run(run_id: str = "run-1") -> dict:
    return {
        "run_id": run_id,
        "prompt": "Make a launch checklist for the API team",
        "tier": "Standard",
        "options": {"models": {"writer": "fake-writer", "judge": "typesafe/jev-1.13"}},
        "result": {
            "status": "completed",
            "final_prompt": "Create a concise API launch checklist with owners and dates.",
            "original_kept": False,
            "report": {
                "diagnosis": {"task": "planning", "gaps": ["owners"]},
                "tests": ["Every item has an owner"],
                "candidates": [{"prompt": "Create a concise API launch checklist."}],
                "per_model": {"fake-weak": {"pass_rate": 1.0}},
                "grades": [{"candidate": 0, "score": 1.0}],
            },
        },
        "cost": {"total": 0.006, "by_role": {"judge": 0.0001}},
        "timing": {"elapsed_ms": 42},
    }


def test_history_search_detail_feedback_and_restart(tmp_path: Path):
    database = tmp_path / "history.sqlite3"
    store = RunStore(database)
    history = RunHistory(store)
    history.save_run(_run())

    assert [item["run_id"] for item in history.list_runs("API team")] == ["run-1"]
    detail = history.get_run("run-1")
    assert detail is not None
    assert detail["original_prompt"].startswith("Make a launch")
    assert detail["final_prompt"].startswith("Create a concise")
    assert detail["tier"] == "Standard"
    assert detail["models"]["writer"] == "fake-writer"
    assert detail["diagnosis"]["task"] == "planning"
    assert detail["tests"] == ["Every item has an owner"]
    assert detail["candidates"]
    assert detail["outputs"] == {"fake-weak": {"pass_rate": 1.0}}
    assert detail["grades"]
    assert detail["cost"]["total"] == 0.006
    assert detail["timings"]["elapsed_ms"] == 42

    accepted = history.record_feedback("run-1", "accepted")
    assert accepted["feedback"] == "accept"
    assert accepted["feedback_at"]

    reopened = RunHistory(RunStore(database))
    assert reopened.get_run("run-1")["feedback"] == "accept"
    assert reopened.search_runs("planning")[0]["feedback"] == "accept"


def test_history_http_contract(tmp_path: Path):
    store = RunStore(tmp_path / "history-http.sqlite3")
    history = RunHistory(store)
    history.save_run(_run("run-http"))

    app = FastAPI()
    register_history_routes(app, store)
    client = TestClient(app)

    listing = client.get("/api/runs", params={"q": "checklist"})
    assert listing.status_code == 200
    assert listing.json()["runs"][0]["run_id"] == "run-http"

    detail = client.get("/api/runs/run-http")
    assert detail.status_code == 200
    assert detail.json()["final_prompt"].startswith("Create a concise")

    feedback = client.post("/api/runs/run-http/feedback", json={"decision": "reject"})
    assert feedback.status_code == 200
    assert feedback.json()["feedback"] == "reject"
    assert client.get("/api/runs/run-http").json()["feedback"] == "reject"


def test_history_summary_exposes_outcome_without_rewriting_legacy_status(
    tmp_path: Path,
):
    history = RunHistory(RunStore(tmp_path / "history-outcomes.sqlite3"))
    history.save_run(
        {
            "run_id": "run-cancelled",
            "prompt": "Keep the original prompt.",
            "result": {
                "status": "failed",
                "final_prompt": "Keep the original prompt.",
                "report": {
                    "status": "cancelled",
                    "failure": {"kind": "cancelled"},
                },
            },
        }
    )
    history.save_run(
        {
            "run_id": "run-failed",
            "prompt": "Write a useful reply.",
            "result": {
                "status": "failed",
                "report": {
                    "status": "failed",
                    "failure": {"kind": "provider_error"},
                },
            },
        }
    )

    summaries = {run["run_id"]: run for run in history.list_runs()}

    assert summaries["run-cancelled"]["status"] == "failed"
    assert summaries["run-cancelled"]["outcome"] == "cancelled"
    assert summaries["run-failed"]["status"] == "failed"
    assert summaries["run-failed"]["outcome"] == "failed"


def test_history_rejects_feedback_for_incomplete_run(tmp_path: Path):
    history = RunHistory(RunStore(tmp_path / "history-incomplete.sqlite3"))
    history.save_run(
        {
            "run_id": "run-paused",
            "prompt": "A prompt",
            "result": {"status": "needs_input"},
        }
    )

    try:
        history.record_feedback("run-paused", "accept")
    except ValueError as error:
        assert "completed" in str(error)
    else:
        raise AssertionError("paused runs must not accept feedback")
