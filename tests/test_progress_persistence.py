"""Progress writes must preserve large evidence without decoding it per event."""

import json
import threading

from prompt_enhancer.jobs import RunJobs
from prompt_enhancer.publication import publication_allowed
from prompt_enhancer.store import RunStore


def test_progress_preserves_evidence_and_restarts_without_full_record_reads(
    tmp_path, monkeypatch
):
    path = tmp_path / "runs.sqlite"
    store = RunStore(path)
    evidence = {"question": {"state": "x" * 1_000_000}, "answer": {"noul": 0.2}}
    store.save_run(
        {
            "run_id": "large",
            "prompt": "Preserve this evidence.",
            "result": {},
            "jev_answers": [evidence],
            "configuration_history": [{"writer": "original"}],
        }
    )
    jobs = RunJobs(store=store)
    entered = threading.Event()
    release = threading.Event()
    reads = []
    decode = store._row_to_record

    def counted(row):
        reads.append(row["run_id"])
        return decode(row)

    monkeypatch.setattr(store, "_row_to_record", counted)

    def work(progress, cancel, observe):
        for i in range(20):
            progress("activity", {"kind": "waiting", "summary": f"Event {i}"})
        entered.set()
        assert release.wait(5)
        return {"status": "completed", "final_prompt": "Preserve this evidence."}

    try:
        jobs.submit("large", "optimize", work, lambda e: {"status": "failed"})
        assert entered.wait(5)
        # Submission reads and initializes the row; progress never reads its payload.
        assert len(reads) <= 2
        saved = store.get_run("large")
        assert saved["jev_answers"] == [evidence]
        assert saved["configuration_history"] == [{"writer": "original"}]
        assert [e["summary"] for e in saved["job"]["events"]][1:] == [
            f"Event {i}" for i in range(20)
        ]
    finally:
        release.set()
        jobs._executor.submit(lambda: None).result(5)
        jobs.close()
    store.close()
    reopened = RunStore(path)
    recovered = reopened.get_run("large")
    assert recovered["jev_answers"] == [evidence]
    assert recovered["job"]["state"] == "done"
    assert recovered["result"]["final_prompt"] == "Preserve this evidence."
    reopened.close()


def test_progress_update_honors_publication_owner_and_does_not_create_runs(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    assert store.update_job("missing", {"state": "running"}) is False
    store.save_run({"run_id": "r", "prompt": "p", "job": {"state": "queued"}})
    token = publication_allowed.set(lambda: False)
    try:
        assert store.update_job("r", {"state": "running"}) is False
    finally:
        publication_allowed.reset(token)
    assert store.get_run("r")["job"] == {"state": "queued"}
    assert store.update_job("r", {"state": "running", "events": []}) is True
    assert store.get_run("r")["job"] == {"state": "running", "events": []}
    # The evidence column never receives a second copy of the progress log.
    raw = store._connection.execute("SELECT evidence_json FROM runs").fetchone()[0]
    assert "job" not in json.loads(raw)
    store.close()


def test_legacy_progress_in_evidence_column_survives_migration(tmp_path):
    path = tmp_path / "legacy.sqlite"
    store = RunStore(path)
    store.save_run({"run_id": "r", "prompt": "p", "marker": ["raw evidence"]})
    store._connection.execute(
        "UPDATE runs SET evidence_json = ?, job_json = NULL",
        (json.dumps({"marker": ["raw evidence"], "job": {"state": "running"}}),),
    )
    store._connection.commit()
    store._connection.execute("ALTER TABLE runs DROP COLUMN job_json")
    store.close()
    reopened = RunStore(path)
    assert reopened.get_run("r")["job"] == {"state": "running"}
    assert reopened.update_job("r", {"state": "done"}) is True
    assert reopened.get_run("r")["job"] == {"state": "done"}
    assert reopened.get_run("r")["marker"] == ["raw evidence"]
    reopened.save_run(reopened.get_run("r"))
    raw = reopened._connection.execute("SELECT evidence_json FROM runs").fetchone()[0]
    assert "job" not in json.loads(raw)
    reopened.close()


def test_stale_watchdog_and_late_checkpoint_cannot_publish(tmp_path):
    import pytest

    from prompt_enhancer.failures import RunCancelled
    from prompt_enhancer.publication import checkpoint_transaction

    store = RunStore(tmp_path / "late-checkpoint.sqlite")
    jobs = RunJobs(store=store)
    callbacks = []

    def work(progress, cancel, observe):
        callbacks.append(checkpoint_transaction.get())
        return {"status": "completed", "final_prompt": "First result."}

    try:
        jobs.submit("r", "optimize", work, lambda exc: {"status": "failed"})
        jobs._executor.submit(lambda: None).result(5)
        first = jobs._jobs["r"]
        original = store.get_run("r")
        transitions = []
        with pytest.raises(RunCancelled):
            callbacks[0](lambda: transitions.append("invalid publication"))
        assert not transitions
        jobs._expire(first)
        assert store.get_run("r") == original
    finally:
        jobs.close()
        store.close()


def test_feedback_preserves_progress_advanced_after_the_edit_read(monkeypatch):
    from prompt_enhancer.history import RunHistory

    store = RunStore(":memory:")
    store.save_run(
        {
            "run_id": "feedback-race",
            "prompt": "Original",
            "result": {"status": "completed", "final_prompt": "Original"},
            "job": {"events": ["old"]},
        }
    )
    read = store.get_run
    reads = 0

    def interleaved(run_id):
        nonlocal reads
        snapshot = read(run_id)
        reads += 1
        if reads == 2:
            store.update_job(run_id, {"events": ["old", "new"]})
        return snapshot

    monkeypatch.setattr(store, "get_run", interleaved)
    RunHistory(store).record_feedback("feedback-race", "accept")
    assert store.get_run("feedback-race")["job"]["events"] == ["old", "new"]
    store.close()


def test_assumption_edit_preserves_progress_advanced_during_meaning_check(monkeypatch):
    from prompt_enhancer.gateway import ScriptedGateway
    from prompt_enhancer.optimizer import PromptOptimizer

    store = RunStore(":memory:")
    store.save_run(
        {
            "run_id": "assumption-race",
            "prompt": "Help me.",
            "result": {
                "status": "completed",
                "final_prompt": "Help me. Clarifications: goal: Summarize",
                "report": {
                    "assumptions": [
                        {
                            "key": "goal",
                            "value": "Summarize",
                            "source": "user_confirmed",
                        }
                    ]
                },
            },
            "job": {"events": ["old"]},
        }
    )
    engine = PromptOptimizer(store=store, gateway=ScriptedGateway())

    def meaning_check(*args):
        store.update_job("assumption-race", {"events": ["old", "new"]})
        return {"passed": True}

    monkeypatch.setattr(engine, "_assumption_meaning_check", meaning_check)
    result = engine.update_assumption(
        "assumption-race", {"key": "goal", "value": "Analyze"}
    )
    assert "goal: Analyze" in result["final_prompt"]
    assert store.get_run("assumption-race")["job"]["events"] == ["old", "new"]
    store.close()
