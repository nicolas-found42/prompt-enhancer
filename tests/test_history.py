from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prompt_enhancer.history import RunHistory, register_history_routes
from prompt_enhancer.score_vector import SCORE_DIMENSIONS
from prompt_enhancer.store import RunStore


def _run(run_id: str = "run-1") -> dict:
    return {
        "run_id": run_id,
        "prompt": "Make a launch checklist for the API team",
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
    assert detail["outcome"] is None
    assert detail["applied_style"] is None
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


def test_feedback_links_rejection_to_selected_candidate_vector(tmp_path: Path):
    history = RunHistory(RunStore(tmp_path / "linked-feedback.sqlite3"))
    run = _run("run-linked")
    run["result"]["selected_candidate_id"] = "candidate-winner"
    run["result"]["report"]["candidates"] = [
        {
            "candidate_id": "candidate-winner",
            "text": run["result"]["final_prompt"],
            "selected": True,
            "metadata": {
                "score_vector": {
                    "scores": {
                        dimension: score
                        for dimension, score in zip(
                            SCORE_DIMENSIONS,
                            (0.9, 0.8, 0.4, 0.7, 0.6, 0.9),
                            strict=True,
                        )
                    },
                    "floors": {dimension: 0.5 for dimension in SCORE_DIMENSIONS},
                }
            },
        }
    ]
    history.save_run(run)

    rejected = history.record_feedback("run-linked", "reject")

    assert rejected["feedback_labels"]["status"] == "linked"
    assert rejected["feedback_labels"]["candidate_id"] == "candidate-winner"
    assert rejected["feedback_labels"]["score_vector"]["clarity"] == 0.4
    assert rejected["feedback_labels"]["weak_dimensions"] == [
        "clarity",
        "coherence",
        "specificity",
    ]


@pytest.mark.parametrize("selected_id", ["original", "candidate-best-round-1"])
def test_feedback_links_selected_vector_when_history_omits_it_from_candidates(
    tmp_path: Path, selected_id: str
):
    history = RunHistory(RunStore(tmp_path / f"linked-{selected_id}.sqlite3"))
    run = _run(f"run-{selected_id}")
    if selected_id == "original":
        run["result"]["final_prompt"] = run["prompt"]
    vector = {
        "scores": {
            dimension: score
            for dimension, score in zip(
                SCORE_DIMENSIONS, (0.8, 0.7, 0.3, 0.6, 0.5, 0.9), strict=True
            )
        }
    }
    run["result"]["selected_candidate_id"] = selected_id
    run["result"]["report"]["selection_evidence"] = {
        "selected_candidate_id": selected_id,
        "selected_candidate": {
            "candidate_id": selected_id,
            "text": run["result"]["final_prompt"],
            "metadata": {"score_vector": vector},
        },
    }
    # The winning baseline or historical candidate is represented directly
    # in selection evidence; terminal-round candidates can be a different set.
    run["result"]["report"]["candidates"] = [
        {"candidate_id": "terminal-round-loser", "metadata": {}}
    ]
    history.save_run(run)

    rejected = history.record_feedback(run["run_id"], "reject")

    assert rejected["feedback_labels"]["status"] == "linked"
    assert rejected["feedback_labels"]["candidate_id"] == selected_id
    assert rejected["feedback_labels"]["score_vector"]["clarity"] == 0.3
    assert rejected["feedback_labels"]["weak_dimensions"] == [
        "clarity",
        "coherence",
        "specificity",
    ]


def test_feedback_without_selected_vector_remains_visible_but_unlabeled(
    tmp_path: Path,
):
    history = RunHistory(RunStore(tmp_path / "unlinked-feedback.sqlite3"))
    history.save_run(_run("run-unlinked"))

    accepted = history.record_feedback("run-unlinked", "accept")

    assert accepted["feedback"] == "accept"
    assert accepted["feedback_labels"]["status"] == "unavailable"
    assert accepted["feedback_labels"]["score_vector"] is None


def test_feedback_for_manually_edited_final_prompt_has_no_stale_vector_label(
    tmp_path: Path,
):
    history = RunHistory(RunStore(tmp_path / "edited-feedback.sqlite3"))
    run = _run("run-edited-feedback")
    run["result"]["report"]["status"] = "edited"
    run["result"]["report"]["selection_evidence"] = {
        "selected_candidate_id": "candidate-winner",
        "selected_candidate": {
            "candidate_id": "candidate-winner",
            "text": run["result"]["final_prompt"],
            "metadata": {
                "score_vector": {
                    "scores": {dimension: 0.9 for dimension in SCORE_DIMENSIONS}
                }
            },
        },
    }
    run["result"]["selected_candidate_id"] = "candidate-winner"
    run["result"]["final_prompt"] = "Manually edited prompt text."
    history.save_run(run)

    rejected = history.record_feedback("run-edited-feedback", "reject")

    assert rejected["feedback"] == "reject"
    assert rejected["feedback_labels"]["status"] == "unavailable"
    assert rejected["feedback_labels"]["score_vector"] is None


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
    assert summaries["run-cancelled"]["outcome"] is None
    assert summaries["run-cancelled"]["control_state"] == "cancelled"
    assert summaries["run-failed"]["status"] == "failed"
    assert summaries["run-failed"]["outcome"] == "failed_operational"
    assert summaries["run-failed"]["outcome_reason"]


def test_history_summary_marks_unverified_completed_runs(tmp_path: Path):
    history = RunHistory(RunStore(tmp_path / "history-unverified.sqlite3"))
    history.save_run(
        {
            "run_id": "run-unverified",
            "prompt": "Help with my homework.",
            "result": {
                "status": "completed",
                "original_kept": False,
                "final_prompt": "Help with my homework.\n\nClarifications:\nContext: notes",
                "report": {"status": "unverified"},
            },
        }
    )
    history.save_run(
        {
            "run_id": "run-improved-unverified",
            "prompt": "Help with my homework.",
            "result": {
                "status": "completed",
                "original_kept": False,
                "final_prompt": "Help with my homework, please.",
                "report": {"status": "improved_unverified"},
            },
        }
    )
    history.save_run(
        {
            "run_id": "run-clarified",
            "prompt": "Help with my homework.",
            "result": {
                "status": "completed",
                "original_kept": False,
                "final_prompt": "Help with my homework.\n\nClarifications:\nContext: notes",
                "report": {"status": "clarified"},
            },
        }
    )
    history.save_run(
        {
            "run_id": "run-edited",
            "prompt": "Help with my homework.",
            "result": {
                "status": "completed",
                "original_kept": False,
                "final_prompt": "Write a homework plan.",
                "report": {"status": "edited"},
            },
        }
    )

    summaries = {run["run_id"]: run for run in history.list_runs()}

    assert summaries["run-unverified"]["status"] == "completed"
    assert summaries["run-unverified"]["outcome"] is None
    assert summaries["run-improved-unverified"]["outcome"] is None
    assert summaries["run-clarified"]["outcome"] is None
    assert summaries["run-edited"]["outcome"] is None


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
