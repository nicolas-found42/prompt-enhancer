"""Deadline and event continuity at the public background-job seam."""

import threading

import pytest
from active_clock import TickingClock

from prompt_enhancer.jobs import RunJobs
from prompt_enhancer.store import RunStore


def test_matched_service_fallback_events_and_samples_survive_reload(tmp_path):
    from active_clock import advancing_chat
    from test_always_attempt import _gateway

    from prompt_enhancer.gateway import ProviderError
    from prompt_enhancer.optimizer import PromptOptimizer

    clock = TickingClock()
    store = RunStore(tmp_path / "fallback.sqlite")
    jobs = RunJobs(store=store, monotonic=clock)
    gateway = _gateway(
        candidate_text="Respond with exactly PING and nothing else.", weak_output="PING"
    )
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] == "weak" and params["provider"]["only"] == ["novita"]:
            raise ProviderError("openrouter", model, 503, role="weak")
        return handler(model, messages, **params)

    gateway.chat_handler = advancing_chat(chat, clock, 20)
    optimizer = PromptOptimizer(store=store, clock=clock, gateway=gateway)
    prompt = "Reply with exactly PING and nothing else."
    jobs.submit(
        "fallback",
        "optimize",
        lambda progress, _cancel, _observe: optimizer.optimize(
            prompt,
            {"evaluation_profile": "llama-tuning-v1"},
            run_id="fallback",
            progress=progress,
        ),
        lambda _exc: {},
        prompt=prompt,
    )
    done = jobs.wait("fallback")
    fallback = next(item for item in done["events"] if item["kind"] == "fallback")
    assert "baseline and drafts on Groq" in fallback["summary"]
    assert fallback["comparison"]["attempts"][0]["status"] == "incomplete"
    assert len(fallback["comparison"]["attempts"][0]["requests"]) == 3
    comparison = done["result"]["report"]["comparisons"][0]
    assert comparison["attempts"][-1]["status"] == "completed"
    assert len(comparison["attempts"][-1]["outputs"]) >= 6
    restored = RunJobs(
        store=RunStore(tmp_path / "fallback.sqlite"), monotonic=clock
    ).get("fallback")
    assert restored["events"] == done["events"]
    assert (
        restored["result"]["report"]["comparisons"]
        == done["result"]["report"]["comparisons"]
    )


def test_requirement_check_events_keep_candidate_evidence_across_reload(tmp_path):
    from test_always_attempt import _gateway

    from prompt_enhancer.optimizer import PromptOptimizer

    clock = TickingClock()
    store = RunStore(tmp_path / "checks.sqlite")
    jobs = RunJobs(store=store, monotonic=clock)
    optimizer = PromptOptimizer(
        store=store,
        clock=clock,
        gateway=_gateway(
            candidate_text="Respond with exactly PING and nothing else.",
            weak_output="PING",
        ),
    )
    prompt = "Reply with exactly PING and nothing else."
    jobs.submit(
        "checks",
        "optimize",
        lambda progress, _cancel, _observe: optimizer.optimize(
            prompt,
            run_id="checks",
            progress=progress,
        ),
        lambda _exc: {},
        prompt=prompt,
    )
    done = jobs.wait("checks")
    selected = done["result"]["report"]["selection_evidence"]["selected_candidate"]
    events = [item for item in done["events"] if item["kind"] == "checks"]
    own = next(
        item for item in events if item["candidate_id"] == selected["candidate_id"]
    )
    assert own["checks"][0]["source"] == prompt
    assert own["checks"][0]["tested"] == 16
    assert own["checks"][0]["failed"] == own["checks"][0]["untestable"] == 0
    restored = RunJobs(store=RunStore(tmp_path / "checks.sqlite"), monotonic=clock).get(
        "checks"
    )
    assert restored["events"] == done["events"]


def test_queue_time_counts_and_late_work_cannot_replace_terminal_state():
    clock = TickingClock()
    store = RunStore(":memory:")
    jobs = RunJobs(store=store, monotonic=clock)
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    dispatched = []

    def first(progress, _cancel, _observe):
        progress("writing_candidates", {"round": 1})
        progress("writing_candidates", {"round": 2})
        entered.set()
        release.wait(5)
        record = store.get_run("first")
        record["result"] = {"status": "completed", "final_prompt": "LATE"}
        store.save_run(record)
        exited.set()
        return record["result"]

    def second(*_args):
        dispatched.append("second")
        return {"status": "completed"}

    try:
        jobs.submit("first", "optimize", first, lambda _exc: {}, prompt="First prompt")
        assert entered.wait(2)
        jobs.submit(
            "second", "optimize", second, lambda _exc: {}, prompt="Second prompt"
        )
        clock.now = 150
        queued = jobs.get("second")
        running = jobs.get("first")
        assert queued["state"] == running["state"] == "done"
        assert queued["elapsed_ms"] == 150_000
        assert queued["remaining_active_ms"] == 0
        assert queued["result"]["report"]["control_state"] == "deadline_reached"
        assert [
            event["round"] for event in running["events"] if event["kind"] == "started"
        ] == [1, 2]
        release.set()
        assert exited.wait(2)
        assert store.get_run("first")["result"] == running["result"]
        assert jobs.get("first")["result"] == running["result"]
        assert not dispatched
    finally:
        release.set()


def test_resume_preserves_budget_and_event_cursors_after_human_wait():
    clock = TickingClock()
    store = RunStore(":memory:")
    jobs = RunJobs(store=store, monotonic=clock)

    def clarify(progress, _cancel, _observe):
        clock.now = 45
        progress("clarifying", {})
        return {"status": "needs_input", "timing": {"total_ms": 45_000}}

    jobs.submit("run", "optimize", clarify, lambda _exc: {}, prompt="Prompt")
    initial = jobs.wait("run")
    clock.now += 300
    resumed_seen = []

    def resume(progress, _cancel, _observe):
        resumed_seen.append(jobs.get("run")["remaining_active_ms"])
        progress("writing_candidates", {"round": 1})
        return {"status": "completed"}

    jobs.submit("run", "resume", resume, lambda _exc: {}, prompt="Prompt")
    resumed = jobs.wait("run")
    assert resumed_seen == [105_000]
    assert resumed["events"][: len(initial["events"])] == initial["events"]
    assert [event["cursor"] for event in resumed["events"]] == list(
        range(1, resumed["event_cursor"] + 1)
    )


def test_expiry_waits_for_atomic_qualified_round_checkpoint():
    from concurrent.futures import ThreadPoolExecutor

    from test_always_attempt import _gateway

    from prompt_enhancer.optimizer import PromptOptimizer

    entered = threading.Event()
    release = threading.Event()

    class CheckpointStore(RunStore):
        def update_run(self, run_id, update):
            def delayed(record):
                updated = update(record)
                if (updated.get("checkpoint") or {}).get(
                    "history"
                ) and not entered.is_set():
                    entered.set()
                    assert release.wait(5)
                return updated

            return super().update_run(run_id, delayed)

    clock = TickingClock()
    store = CheckpointStore(":memory:")
    jobs = RunJobs(store=store, monotonic=clock)
    optimizer = PromptOptimizer(store=store, gateway=_gateway(), clock=clock)
    prompt = "whats 2 plus 2"

    def work(progress, _cancel, _observe):
        return optimizer.optimize(prompt, run_id="run", progress=progress)

    try:
        jobs.submit("run", "optimize", work, lambda _exc: {}, prompt=prompt)
        assert entered.wait(3)
        clock.now = 150
        with ThreadPoolExecutor(max_workers=1) as pool:
            expired = pool.submit(jobs.get, "run")
            assert not expired.done()
            release.set()
            result = expired.result(3)["result"]
        assert result["final_prompt"] != prompt
        assert result["report"]["control_state"] == "deadline_reached"
        assert result["report"]["outcome"] != "converged"
        selected = result["report"]["selection_evidence"]["selected_candidate"]
        assert selected["text"] == result["final_prompt"]
        assert selected["metadata"]["score_vector"]["passed"] is True
        assert store.get_run("run")["result"] == result
    finally:
        release.set()


def test_event_cursor_polling_does_not_replay_delivered_events():
    jobs = RunJobs(store=RunStore(":memory:"))

    def work(progress, _cancel, _observe):
        progress("writing_candidates", {"round": 1})
        return {"status": "completed"}

    jobs.submit("run", "optimize", work, lambda _exc: {}, prompt="Prompt")
    snapshot = jobs.wait("run")
    cursor = snapshot["events"][0]["cursor"]
    incremental = jobs.get("run", after_cursor=cursor)
    assert incremental["events"] == snapshot["events"][1:]
    assert incremental["event_cursor"] == snapshot["event_cursor"]
    assert jobs.get("run", after_cursor=snapshot["event_cursor"])["events"] == []


def test_terminal_poll_waits_for_durable_publication():
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()

    class PublicationStore(RunStore):
        def update_run(self, run_id, update):
            def delayed(record):
                updated = update(record)
                if (updated.get("job") or {}).get("state") == "done":
                    entered.set()
                    assert release.wait(5)
                return updated

            return super().update_run(run_id, delayed)

    store = PublicationStore(":memory:")
    jobs = RunJobs(store=store)
    try:
        jobs.submit(
            "run",
            "optimize",
            lambda *_args: {"status": "completed", "final_prompt": "Changed"},
            lambda _exc: {},
            prompt="Prompt",
        )
        assert entered.wait(2)
        with ThreadPoolExecutor(max_workers=1) as pool:
            poll = pool.submit(jobs.get, "run")
            assert not poll.done()
            release.set()
            result = poll.result(2)["result"]
        assert store.get_run("run")["result"] == result
    finally:
        release.set()


def test_approval_wait_uses_budget_and_expires_without_further_calls():
    clock = TickingClock()
    jobs = RunJobs(store=RunStore(":memory:"), monotonic=clock)

    def work(progress, _cancel, _observe):
        clock.now = 45
        progress("writing_candidates", {"round": 1})
        return {
            "status": "needs_input",
            "report": {"status": "awaiting_approval"},
            "timing": {"total_ms": 45000},
        }

    jobs.submit("run", "optimize", work, lambda _exc: {}, prompt="Prompt")
    assert jobs.wait("run")["result"]["status"] == "needs_input"
    clock.now = 100
    assert jobs.get("run")["remaining_active_ms"] == 50000
    clock.now = 150
    expired = jobs.get("run")
    assert expired["remaining_active_ms"] == 0
    assert expired["result"]["report"]["control_state"] == "deadline_reached"


def test_restart_counts_approval_wait_and_retains_event_prefix(tmp_path):
    clock = TickingClock()
    wall = [1000.0]
    store = RunStore(tmp_path / "runs.sqlite3")
    jobs = RunJobs(store=store, monotonic=clock, wall_time=lambda: wall[0])

    def work(progress, _cancel, _observe):
        clock.now = 45
        progress("writing_candidates", {"round": 1})
        return {
            "status": "needs_input",
            "report": {"status": "awaiting_approval"},
            "timing": {"total_ms": 45000},
        }

    jobs.submit("run", "optimize", work, lambda _exc: {}, prompt="Prompt")
    initial = jobs.wait("run")
    wall[0] += 100
    restarted = RunJobs(
        store=RunStore(store.path), monotonic=TickingClock(), wall_time=lambda: wall[0]
    )
    recovered = restarted.get("run")
    assert recovered["remaining_active_ms"] == 5000
    assert recovered["events"] == initial["events"]


def test_replacing_store_during_approval_cannot_cancel_its_deadline():
    from test_run_control import _rejected_gateway

    from prompt_enhancer.optimizer import PromptOptimizer

    clock = TickingClock()
    gateway = _rejected_gateway(cost_per_call=1.0)
    chat = gateway.chat_handler

    def delayed_round(*args, **kwargs):
        if kwargs.get("role") == "writer":
            clock.now += 20
        return chat(*args, **kwargs)

    gateway.chat_handler = delayed_round
    original_store = RunStore(":memory:")
    optimizer = PromptOptimizer(
        store=original_store,
        gateway=gateway,
        clock=clock,
        writer_instruction_version=4,
    )
    pending = optimizer.optimize("whats 2 plus 2", {"spend_limit_usd": 0})
    assert pending["report"]["status"] == "awaiting_approval"
    jobs = optimizer.jobs
    optimizer.store = RunStore(":memory:")
    with pytest.raises(RuntimeError, match="while a run is active"):
        _ = optimizer.jobs
    clock.now = 150
    expired = jobs.get(pending["run_id"])
    assert expired["result"]["report"]["control_state"] == "deadline_reached"
    assert original_store.get_run(pending["run_id"])["result"] == expired["result"]


def test_explicit_stop_cannot_be_overwritten_by_the_approval_watchdog():
    from test_run_control import _rejected_gateway

    from prompt_enhancer.optimizer import PromptOptimizer

    clock = TickingClock()
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(
        store=store,
        gateway=_rejected_gateway(),
        clock=clock,
        writer_instruction_version=4,
    )
    pending = optimizer.optimize("whats 2 plus 2", {"time_limit_s": 0})
    assert pending["report"]["status"] == "awaiting_approval"
    stopped = optimizer.stop_run(pending["run_id"])
    clock.now = 150
    job = optimizer.jobs.get(pending["run_id"])
    assert job["result"] == stopped
    assert stopped["report"]["control_state"] == "stopped"
    assert store.get_run(pending["run_id"])["result"] == stopped
    assert optimizer.jobs.has_active_budget() is False


def test_expiry_during_clarification_planning_retains_completed_diagnosis():
    from test_engine import _clarification_gateway

    from prompt_enhancer.optimizer import PromptOptimizer

    clock = TickingClock()
    entered = threading.Event()
    release = threading.Event()
    gateway = _clarification_gateway(0.95)
    chat = gateway.chat_handler

    def blocked_planning(*args, **kwargs):
        # Extraction now precedes diagnosis. Block the actual clarification
        # proposal, so this control still exercises completed diagnosis evidence.
        if (
            kwargs.get("role") == "writer"
            and "Suggest two or three plausible values" in args[1][0]["content"]
        ):
            entered.set()
            assert release.wait(5)
        return chat(*args, **kwargs)

    gateway.chat_handler = blocked_planning
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(store=store, gateway=gateway, clock=clock)
    try:
        optimizer.jobs.submit(
            "run",
            "optimize",
            lambda progress, _cancel, _observe: optimizer.optimize(
                "Write a concise report.", run_id="run", progress=progress
            ),
            lambda _exc: {},
            prompt="Write a concise report.",
            evidence=optimizer.deadline_evidence_for("run", "optimize"),
        )
        assert entered.wait(3)
        clock.now = 150
        expired = optimizer.jobs.get("run")["result"]
        assert expired["report"]["control_state"] == "deadline_reached"
        assert expired["report"]["diagnosis"]["confirmed_gaps"]
        assert expired["report"]["jev_answers"]
        assert expired["report"]["models"]
        assert expired["report"]["history"] == []
        assert store.get_run("run")["result"] == expired
    finally:
        release.set()
