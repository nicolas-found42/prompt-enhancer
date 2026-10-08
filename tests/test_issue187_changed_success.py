"""Changed-only success and the 150-second active budget (issue #187).

The original prompt is reference evidence, never a successful result. A run
that cannot qualify a changed prompt keeps searching until its active budget
ends; the deadline is a terminal stop, not a pause for approval.
"""

import threading
import time
from functools import partial

import pytest
from active_clock import TickingClock, advancing_chat
from test_always_attempt import _gateway
from test_engine import _clarification_gateway

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.catalog import StaticModelCatalog
from prompt_enhancer.gateway import GatewayConfig, HttpGateway, ScriptedGateway

PromptOptimizer = partial(PromptOptimizer, writer_instruction_version=4)

PROMPT = "whats 2 plus 2"


def _ticking(gateway: ScriptedGateway, clock: TickingClock, step_s: float):
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, step_s)
    return gateway


def test_expiry_during_diagnosis_prevents_another_model_call() -> None:
    clock = TickingClock()
    gateway = _gateway()
    decide = gateway.decision_handler
    dispatched = []

    def slow_decision(request, **kwargs):
        dispatched.append(request["key"])
        clock.now += 150
        return decide(request, **kwargs)

    gateway.decision_handler = slow_decision
    store = RunStore(":memory:")
    result = PromptOptimizer(store=store, gateway=gateway, clock=clock).optimize(PROMPT)

    assert len(dispatched) == 1
    assert result["report"]["control_state"] == "deadline_reached"
    assert result["report"]["outcome"] is None
    assert store.get_run(result["run_id"])["result"] == result


def test_rejected_strategy_bundle_still_attempts_faithful_presentation() -> None:
    gateway = _gateway(recheck_probability=0.01)
    store = RunStore(":memory:")
    result = PromptOptimizer(
        store=store, gateway=gateway, writer_instruction_version=15
    ).optimize(PROMPT)

    assert result["final_prompt"] != PROMPT
    assert result["report"]["outcome"] == "converged"
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    assert selected["strategy"] == "faithful_presentation"
    assert (
        selected["metadata"]["evaluation"]["comparison"]["useful_change"]["probability"]
        >= 0.8
    )


def test_uncooperative_transport_returns_by_the_remaining_run_deadline() -> None:
    release = threading.Event()
    requests = []
    origin = time.monotonic()
    reads = 0

    def clock():
        nonlocal reads
        reads += 1
        return time.monotonic() - origin + (0 if reads == 1 else 149.9)

    class Transport:
        def request(self, url, **kwargs):
            requests.append(kwargs)
            release.wait(5)
            return {"status_code": 200, "json": {}}

    gateway = HttpGateway(
        Transport(),
        config=GatewayConfig(openrouter_api_key="test-key"),
        catalog=StaticModelCatalog((), ()),
    )
    store = RunStore(":memory:")
    try:
        result = PromptOptimizer(store=store, gateway=gateway, clock=clock).optimize(
            PROMPT
        )
        assert time.monotonic() - origin < 1
        assert len(requests) == 1
        assert requests[0]["timeout"] <= 0.1
        assert result["report"]["control_state"] == "deadline_reached"
        saved = store.get_run(result["run_id"])["result"]
        release.set()
        assert store.get_run(result["run_id"])["result"] == saved
    finally:
        release.set()


def test_floor_passing_original_is_not_a_successful_result() -> None:
    clock = TickingClock()
    gateway = _ticking(
        _gateway(
            support="new_requirement",
            meaning_probability=0.01,
            baseline_score_probability=0.99,
        ),
        clock,
        step_s=80.0,
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    )

    result = optimizer.optimize(PROMPT)

    report = result["report"]
    assert report["status"] != "converged"
    assert report["outcome"] is None
    assert result["original_kept"] is True
    assert result["final_prompt"] == PROMPT
    assert all(entry["status"] != "converged" for entry in report["history"]), (
        "the original baseline must not converge on its own evidence"
    )


def test_deadline_is_a_terminal_stop_not_an_approval_pause() -> None:
    clock = TickingClock()
    gateway = _ticking(
        _gateway(
            support="new_requirement",
            meaning_probability=0.01,
            baseline_score_probability=0.99,
        ),
        clock,
        step_s=80.0,
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    )

    result = optimizer.optimize(PROMPT)

    assert result["status"] == "completed"
    report = result["report"]
    assert report["control_state"] == "deadline_reached"
    assert report["deadline"]["limit_s"] == 150
    assert report["deadline"]["elapsed_ms"] >= 150_000
    assert "pause" not in report
    assert result["status"] != "needs_input"


def test_clarification_resume_stops_at_the_active_deadline() -> None:
    clock = TickingClock()
    gateway = _clarification_gateway(0.95)
    chat = gateway.chat_handler
    writer_calls = 0

    def bounded_chat(model, messages, **kwargs):
        nonlocal writer_calls
        if kwargs.get("role") == "writer":
            writer_calls += 1
            # Bound the broken path too: an expired run must finish before
            # a runaway scripted loop reaches this transport failure.
            if writer_calls > 6:
                raise RuntimeError("resume continued after its active deadline")
            clock.now += 80.0
        return chat(model, messages, **kwargs)

    gateway.chat_handler = bounded_chat
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(store=store, gateway=gateway, clock=clock)
    initial = optimizer.optimize("Write a concise report.")
    assert initial["status"] == "needs_input"

    resumed = optimizer.resume(initial["run_id"], {"goal": "summarize"})

    assert resumed["run_id"] == initial["run_id"]
    assert resumed["report"]["control_state"] == "deadline_reached"
    assert resumed["report"]["deadline"]["elapsed_ms"] >= 150_000
    assert writer_calls <= 6
    assert store.get_run(initial["run_id"])["result"] == resumed


@pytest.mark.parametrize("skip", [False, True])
def test_clarification_restart_keeps_active_time_and_excludes_human_wait(
    tmp_path, skip
) -> None:
    clock = TickingClock()
    gateway = _clarification_gateway(0.95)
    chat = gateway.chat_handler

    def initial_chat(model, messages, **kwargs):
        if kwargs.get("role") == "writer":
            clock.now = 45.0
        return chat(model, messages, **kwargs)

    gateway.chat_handler = initial_chat
    database = str(tmp_path / "runs.sqlite3")
    initial = PromptOptimizer(
        store=RunStore(database), gateway=gateway, clock=clock
    ).optimize("Write a concise report.")
    assert initial["status"] == "needs_input"
    assert initial["timing"]["total_ms"] == 45_000
    clock.now += 300.0  # The human takes five minutes to reply.

    resumed_clock = TickingClock()
    resumed_clock.now = 1_000.0  # A restarted process has a new clock origin.
    resumed_gateway = _ticking(_clarification_gateway(0.95), resumed_clock, step_s=40.0)
    store = RunStore(database)
    optimizer = PromptOptimizer(
        store=store, gateway=resumed_gateway, clock=resumed_clock
    )
    resumed = (
        optimizer.skip_clarification(initial["run_id"])
        if skip
        else optimizer.resume(initial["run_id"], {"goal": "summarize"})
    )

    assert resumed["report"]["control_state"] == "deadline_reached"
    active_ms = 45_000 + round((resumed_clock.now - 1_000.0) * 1_000)
    assert resumed["timing"]["total_ms"] == active_ms
    assert resumed["report"]["deadline"]["elapsed_ms"] == active_ms
    assert 150_000 <= active_ms < 300_000
    assert store.get_run(initial["run_id"])["timing"]["total_ms"] == active_ms
