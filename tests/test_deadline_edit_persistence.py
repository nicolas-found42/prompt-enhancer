"""Late deadline workers must not overwrite a subsequent durable edit."""

import threading

import pytest
from active_clock import TickingClock

from prompt_enhancer.jobs import RunJobs
from prompt_enhancer.store import RunStore


def test_late_worker_completion_preserves_edit_after_deadline_publication():
    store = RunStore(":memory:")
    clock = TickingClock()
    jobs = RunJobs(store=store, monotonic=clock)
    entered = threading.Event()
    release = threading.Event()

    def work(*args):
        entered.set()
        assert release.wait(5)
        return {"status": "completed", "final_prompt": "Stale worker prompt"}

    try:
        jobs.submit("run", "optimize", work, lambda exc: {}, prompt="Original prompt")
        assert entered.wait(3)
        clock.now = 150
        deadline = jobs.get("run")["result"]
        assert deadline["report"]["control_state"] == "deadline_reached"
        assert store.get_run("run")["result"] == deadline
        edited = {
            **deadline,
            "final_prompt": "A later user correction",
            "report": {**deadline["report"], "status": "edited"},
        }
        store.update_run("run", lambda record: {**record, "result": edited})
        release.set()
        # The single-worker queue makes this a witness that _run's finally
        # block has finished, rather than only observing its terminal status.
        jobs._executor.submit(lambda: None).result(3)
        jobs.close()
        assert store.get_run("run")["result"] == edited
    finally:
        release.set()
        jobs.close()


@pytest.mark.parametrize("saved_report", [None, "legacy text", 7, ["partial"]])
def test_deadline_replaces_malformed_prior_report_without_losing_its_evidence(
    saved_report,
):
    store = RunStore(":memory:")
    clock = TickingClock()
    jobs = RunJobs(store=store, monotonic=clock)
    entered = threading.Event()
    release = threading.Event()

    def work(*args):
        entered.set()
        assert release.wait(5)
        return {"status": "completed"}

    try:
        jobs.submit("run", "optimize", work, lambda exc: {}, prompt="Original prompt")
        assert entered.wait(3)
        store.update_run(
            "run",
            lambda record: {
                **record,
                "result": {"status": "completed", "report": saved_report},
            },
        )
        clock.now = 150
        deadline = jobs.get("run")["result"]
        assert deadline["report"]["control_state"] == "deadline_reached"
        assert store.get_run("run")["result"] == deadline
        release.set()
        jobs._executor.submit(lambda: None).result(3)
        assert store.get_run("run")["result"] == deadline
    finally:
        release.set()
        jobs.close()
