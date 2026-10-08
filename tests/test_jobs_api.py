import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.failures import RunCancelled
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    ProviderError,
    ScriptedGateway,
)
from prompt_enhancer.history import RunHistory
from prompt_enhancer.jobs import RunJobs
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def _decide(request, **_kwargs):
    key = str(request.get("key", ""))
    if request.get("type") == "choice":
        if key.endswith(":verbosity_direction"):
            choice = "same"
        elif key == "task_type":
            choice = "general"
        elif key == "strategy_choice":
            choice = "specify_output_format"
        elif key.startswith("fidelity:sentence:"):
            choice = "supported_by_original"
        else:
            choice = "none"
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: 1.0},
            "confidence": 1.0,
        }
    if key.startswith("score:"):
        probability = 0.99
    elif key.startswith(("fidelity:", "strategy_recheck:", "evaluate:")):
        probability = 0.99
    elif key.startswith("gap:"):
        probability = 0.01
    else:
        probability = 0.01
    return {"type": "noul", "probability_true": probability, "confidence": 1.0}


def _client(gateway) -> tuple[TestClient, object]:
    app = create_app(
        optimizer=PromptOptimizer(
            store=RunStore(":memory:"),
            gateway=gateway,
            writer_instruction_version=4,
        )
    )
    return TestClient(app), app.state.jobs


def _clarification_gateway() -> ScriptedGateway:
    def initial_chat(*_args, **_kwargs):
        return (
            '{"gaps":{"goal":{"question":"What should the assistant do?",'
            '"options":[{"value":"summarize","label":"Summarize"},'
            '{"value":"analyze","label":"Analyze"}]}},"tests":[]}'
        )

    def chat(model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                return json.dumps(
                    {
                        strategy["name"]: state["prompt"]
                        for strategy in state["strategies"]
                    }
                )
            return initial_chat(model, messages, role=role)
        return "pass"

    def decide(request, **_kwargs):
        key = request.get("key")
        if key == "task_type":
            choice = "general"
        elif key == "infer:goal":
            choice = "unknown"
        elif key == "strategy_choice":
            choice = "specify_output_format"
        elif key.endswith(":verbosity_direction"):
            choice = "same"
        elif key.startswith("fidelity:sentence:"):
            choice = "supported_by_original"
        elif request.get("type") == "choice":
            choice = "none"
        else:
            choice = None
        if choice is not None:
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        if key == "gap:goal":
            probability = 0.99
        elif key.startswith(("score:", "fidelity:", "strategy_recheck:", "evaluate:")):
            probability = 0.99
        else:
            probability = 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(chat=chat, decision=decide)


def test_optimize_job_returns_run_id_at_once_and_finishes_with_the_result() -> None:
    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                return json.dumps(
                    {
                        strategy["name"]: state["prompt"]
                        for strategy in state["strategies"]
                    }
                )
            return '{"tests":[]}'
        return "pass"

    client, jobs = _client(ScriptedGateway(chat=chat, decision=_decide))

    started = client.post("/api/jobs/optimize", json={"prompt": "Explain recursion."})

    assert started.status_code == 202
    run_id = started.json()["run_id"]
    jobs.wait(run_id)
    job = client.get(f"/api/jobs/{run_id}").json()
    assert job["state"] == "done"
    assert job["prompt"] == "Explain recursion."
    # The writer echoes the original and supplies no tests. The job finishes
    # at its deadline without presenting that baseline as an improvement.
    assert job["result"]["status"] == "completed"
    assert job["result"]["report"]["status"] == "deadline_reached"
    assert job["result"]["report"]["outcome"] is None
    assert job["result"]["final_prompt"] == "Explain recursion."
    assert "diagnosing" in job["stages_seen"]
    assert client.get(f"/api/runs/{run_id}").status_code == 200


def test_optimize_job_rejects_invalid_requests_before_starting() -> None:
    client, _jobs = _client(
        ScriptedGateway(chat=lambda *_a, **_k: "{}", decision=_decide)
    )

    assert client.post("/api/jobs/optimize", json={"prompt": "  "}).status_code == 422
    assert client.get("/api/jobs/unknown").status_code == 404


def test_invalid_resume_answer_is_rejected_before_a_job_and_can_be_corrected() -> None:
    client, jobs = _client(_clarification_gateway())
    pending = client.post(
        "/api/optimize", json={"prompt": "Write a concise report."}
    ).json()
    assert pending["status"] == "needs_input"
    run_id = pending["run_id"]

    rejected = client.post(
        f"/api/jobs/{run_id}/resume",
        json={"answers": {"goal": {"value": "other", "text": "  "}}},
    )

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == {
        "code": "invalid_answer",
        "question_id": "goal",
        "message": "Answer for 'goal' requires other text",
    }
    preserved = client.get(f"/api/jobs/{run_id}")
    assert preserved.status_code == 200
    assert preserved.json()["result"]["status"] == "needs_input"

    corrected = client.post(
        f"/api/jobs/{run_id}/resume",
        json={
            "answers": {
                question["id"]: {
                    "value": "other",
                    "text": "Summarize"
                    if question["id"] == "goal"
                    else "Use concise language",
                }
                for question in pending["questions"]
            }
        },
    )
    assert corrected.status_code == 202, corrected.text
    assert jobs.wait(run_id)["result"]["status"] == "completed"


def test_refused_provider_produces_a_failed_result_with_a_plain_hint() -> None:
    def chat(model, *_args, role, **_kwargs):
        raise ProviderError("go", model, 403, role=role)

    client, jobs = _client(ScriptedGateway(chat=chat, decision=_decide))
    run_id = client.post(
        "/api/jobs/optimize", json={"prompt": "Write a reply."}
    ).json()["run_id"]

    result = jobs.wait(run_id)["result"]

    assert result["status"] == "failed"
    failure = result["report"]["failure"]
    assert failure["headline"] == "OpenCode Go refused the request"
    assert "subscription" in failure["hint"]
    assert failure["http_status"] == 403
    assert client.get("/api/runs").json()[0]["status"] == "failed"


def test_cancel_stops_the_run_at_the_next_stage() -> None:
    entered = threading.Event()
    release = threading.Event()

    def decide(request, **kwargs):
        if request.get("key") == "task_type":
            entered.set()
            release.wait(5)
        return _decide(request, **kwargs)

    client, jobs = _client(
        ScriptedGateway(chat=lambda *_a, **_k: '{"tests":[]}', decision=decide)
    )
    run_id = client.post(
        "/api/jobs/optimize", json={"prompt": "Write a reply."}
    ).json()["run_id"]
    assert entered.wait(5)

    cancellation = client.post(f"/api/jobs/{run_id}/cancel").json()
    assert cancellation["cancel_requested"] is True
    assert cancellation["cancellation_pending"] is True
    assert cancellation["state"] == "running"
    release.set()
    terminal = jobs.wait(run_id)
    result = terminal["result"]
    assert terminal["cancellation_pending"] is False

    assert result["status"] == "failed"
    assert result["report"]["status"] == "cancelled"
    assert result["final_prompt"] == "Write a reply."
    summary = client.get("/api/runs").json()[0]
    assert summary["status"] == "failed"
    assert summary["control_state"] == "cancelled"
    assert summary["outcome"] is None


def test_running_job_elapsed_time_keeps_advancing_between_progress_events() -> None:
    now = [10.0]
    entered = threading.Event()
    release = threading.Event()
    jobs = RunJobs(monotonic=lambda: now[0])

    def work(progress, _cancel_check, _observe_operation):
        progress("writing_tests", {"round": 1, "elapsed_ms": 100})
        entered.set()
        release.wait(5)
        return {"status": "completed"}

    jobs.submit("elapsed-run", "optimize", work, lambda _exc: {})
    assert entered.wait(5)
    now[0] = 10.75

    snapshot = jobs.get("elapsed-run")
    release.set()
    jobs.wait("elapsed-run")

    assert snapshot["elapsed_ms"] == 750
    assert snapshot["last_progress_age_ms"] == 750


def test_started_job_is_saved_and_restart_recovers_it_as_interrupted(tmp_path) -> None:
    store_path = tmp_path / "runs.sqlite"
    store = RunStore(store_path)
    jobs = RunJobs(store=store)

    jobs.submit(
        "durable-run",
        "optimize",
        lambda _progress, _cancel_check, _observe_operation: {"status": "completed"},
        lambda _exc: {},
        prompt="Explain a seed.",
    )
    jobs.wait("durable-run")

    assert store.get_run("durable-run") is not None
    crashed = store.get_run("durable-run")
    crashed["job"]["state"] = "running"
    crashed["job"].update(
        {
            "stage": "writing_tests",
            "round": {"round": 1},
            "elapsed_ms": 400,
            "cost_total": 0.02,
        }
    )
    crashed["result"] = {"run_id": "durable-run", "status": "running"}
    crashed["cost"] = {"total": 0.02}
    store.save_run(crashed)
    store.save_run(
        {
            "run_id": "cancel-before-crash",
            "created_at": "2026-10-06T00:00:00+00:00",
            "prompt": "Explain a seed.",
            "options": {},
            "result": {"run_id": "cancel-before-crash", "status": "running"},
            "cost": {},
            "timing": {},
            "job": {
                "state": "running",
                "kind": "optimize",
                "started_at": 1.0,
                "elapsed_ms": 400,
                "cancel_requested": True,
            },
        }
    )
    store.close()

    recovered_store = RunStore(store_path)
    recovered_app = create_app(
        optimizer=PromptOptimizer(
            store=recovered_store,
            gateway=ScriptedGateway(chat=lambda *_a, **_k: "{}", decision=_decide),
        )
    )
    recovered_jobs = recovered_app.state.jobs
    recovered = recovered_jobs.get("durable-run")

    assert recovered["state"] == "interrupted"
    assert recovered["result"]["report"]["failure"]["kind"] == "interrupted"
    fixture_path = Path(__file__).parent / "fixtures/recovered-interrupted-job.json"
    expected_api_contract = json.loads(fixture_path.read_text())
    assert TestClient(recovered_app).get("/api/jobs/durable-run").json() == (
        expected_api_contract
    )
    record = recovered_store.get_run("durable-run")
    assert record["result"]["status"] == "failed"
    assert record["result"]["report"]["status"] == "failed"
    assert record["result"]["report"]["outcome"] == "failed_operational"
    assert record["result"]["cost"]["total"] == 0.02
    assert record["result"]["timing"]["total_ms"] == 400
    recovered_history = RunHistory(recovered_store).get_run("durable-run")
    assert recovered_history["status"] == "failed"
    assert recovered_history["outcome"] == "failed_operational"
    cancelled_before_crash = recovered_jobs.get("cancel-before-crash")
    assert cancelled_before_crash["state"] == "interrupted"
    assert cancelled_before_crash["cancel_requested"] is True
    assert cancelled_before_crash["cancellation_pending"] is False
    assert (
        cancelled_before_crash["result"]["report"]["failure"]["kind"] == "interrupted"
    )


def test_job_stage_cancel_and_terminal_transitions_are_durable(tmp_path) -> None:
    store = RunStore(tmp_path / "transitions.sqlite")
    jobs = RunJobs(store=store)
    entered = threading.Event()
    release = threading.Event()

    def work(progress, cancel_check, _observe_operation):
        progress("writing_tests", {"round": 1})
        entered.set()
        release.wait(5)
        if cancel_check():
            raise RuntimeError("cancelled by test worker")
        return {"status": "completed"}

    jobs.submit(
        "transition-run",
        "optimize",
        work,
        lambda _exc: {"status": "failed"},
        prompt="Explain a seed.",
    )
    assert entered.wait(5)
    running = store.get_run("transition-run")
    assert running["job"]["state"] == "running"
    assert running["job"]["stage"] == "writing_tests"
    assert running["result"] == {}
    active_history = RunHistory(store).get_run("transition-run")
    assert active_history["status"] == "running"
    assert active_history["outcome"] is None

    cancelled = jobs.cancel("transition-run")
    persisted_cancel = store.get_run("transition-run")
    assert cancelled["cancellation_pending"] is True
    assert persisted_cancel["job"]["cancel_requested"] is True

    release.set()
    jobs.wait("transition-run")
    terminal = store.get_run("transition-run")
    assert terminal["job"]["state"] == "done"
    assert terminal["result"]["status"] == "failed"


def test_concurrent_cancel_save_cannot_overwrite_a_terminal_job(tmp_path) -> None:
    store_path = tmp_path / "concurrent-transitions.sqlite"
    store = RunStore(store_path)
    jobs = RunJobs(store=store)
    entered = threading.Event()
    release_work = threading.Event()
    stale_ready = threading.Event()
    release_stale = threading.Event()
    terminal_saved = threading.Event()
    original_save = store.save_run

    def controlled_save(record):
        if threading.current_thread().name == "stale-cancel":
            stale_ready.set()
            assert release_stale.wait(5)
        result = original_save(record)
        if record.get("job", {}).get("state") == "done":
            terminal_saved.set()
        return result

    store.save_run = controlled_save

    def work(progress, cancel_check, _observe_operation):
        progress("writing_tests", {"round": 1})
        entered.set()
        assert release_work.wait(5)
        if cancel_check():
            raise RunCancelled("concurrent-run")
        return {"status": "completed"}

    jobs.submit(
        "concurrent-run",
        "optimize",
        work,
        lambda _exc: {"status": "cancelled"},
        prompt="Explain a seed.",
    )
    canceller = threading.Thread(
        target=lambda: jobs.cancel("concurrent-run"), name="stale-cancel"
    )
    try:
        assert entered.wait(5)
        canceller.start()
        assert stale_ready.wait(5)
        release_work.set()
        # Permit the terminal write to overtake the held stale write on the
        # broken implementation. With atomic updates it must wait instead.
        # This wait only bounds the probe; the assertion is durable state.
        terminal_saved.wait(1)
    finally:
        release_work.set()
        release_stale.set()
        if canceller.ident is not None:
            canceller.join(5)
        jobs._executor.shutdown(wait=True)
    assert not canceller.is_alive()
    terminal = store.get_run("concurrent-run")
    assert terminal["job"]["state"] == "done"
    assert terminal["job"]["cancel_requested"] is True
    assert terminal["result"]["status"] == "cancelled"
    store.close()
    recovered_store = RunStore(store_path)
    recovered_jobs = RunJobs(store=recovered_store)
    assert recovered_jobs.active() == []
    assert recovered_store.get_run("concurrent-run")["job"]["state"] == "done"
    assert recovered_store.get_run("concurrent-run")["result"]["status"] == "cancelled"
    recovered_jobs._executor.shutdown(wait=True)
    recovered_store.close()


def test_api_cancellation_reaches_the_active_gateway_operation() -> None:
    entered = threading.Event()
    release = threading.Event()

    class OperationAwareGateway(ScriptedGateway):
        def cancel_check(self):
            return False

        def observer(self, _event):
            return None

        @contextmanager
        def operation_context(self, *, cancel_check=None, observer=None, substage=None):
            self.cancel_check = cancel_check or self.cancel_check
            self.observer = observer or self.observer
            self.observer({"event": "start", "operation": "test.writer"})
            try:
                yield
            finally:
                self.cancel_check = type(self).cancel_check.__get__(self)

    gateway_ref = {}

    first_writer_call = True

    def chat(_model, messages, *, role, **_kwargs):
        nonlocal first_writer_call
        if role == "writer":
            if first_writer_call:
                first_writer_call = False
                entered.set()
                release.wait(5)
                gateway = gateway_ref["gateway"]
                if gateway.cancel_check():
                    gateway.observer(
                        {"event": "cancel_pending", "operation": "test.writer"}
                    )
                    raise RunCancelled()
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                return json.dumps(
                    {
                        strategy["name"]: state["prompt"]
                        for strategy in state["strategies"]
                    }
                )
            return '{"tests":[]}'
        return "pass"

    gateway = OperationAwareGateway(chat=chat, decision=_decide)
    gateway_ref["gateway"] = gateway
    app = create_app(
        optimizer=PromptOptimizer(store=RunStore(":memory:"), gateway=gateway)
    )
    client = TestClient(app)
    run_id = client.post(
        "/api/jobs/optimize", json={"prompt": "Write a reply."}
    ).json()["run_id"]
    assert entered.wait(5), app.state.jobs.get(run_id)["result"]["report"]["failure"]

    pending = client.post(f"/api/jobs/{run_id}/cancel").json()
    assert pending["cancellation_pending"] is True
    assert pending["operation"]["operation"] == "test.writer"
    release.set()

    terminal = app.state.jobs.wait(run_id)
    assert terminal["state"] == "done"
    assert terminal["cancellation_pending"] is False
    assert terminal["result"]["report"]["status"] == "cancelled"


def test_invalid_writer_reply_is_not_reported_as_a_network_error() -> None:
    client, jobs = _client(
        ScriptedGateway(chat=lambda *_a, **_k: "not json", decision=_decide)
    )
    run_id = client.post(
        "/api/jobs/optimize", json={"prompt": "Write a reply."}
    ).json()["run_id"]

    failure = jobs.wait(run_id)["result"]["report"]["failure"]

    assert failure["kind"] == "invalid_response"
    assert "network" not in failure["message"]
    assert failure["headline"] == "The writer model gave an unusable reply"


def test_retired_deep_job_endpoint_is_not_exposed() -> None:
    client, _jobs = _client(
        ScriptedGateway(chat=lambda *_a, **_k: "{}", decision=_decide)
    )

    assert client.post("/api/jobs/missing/deep").status_code == 404
    assert client.get("/api/estimates").status_code == 404


def test_provider_probe_marks_a_refused_provider_unavailable_without_recording_cost() -> (
    None
):
    class Transport:
        def request(self, url, **_kwargs):
            return {"status_code": 403, "json": {}}

    gateway = HttpGateway(Transport(), config=GatewayConfig(), go_models=["go-writer"])

    health = gateway.provider_health(probe_models=["go-writer"])

    assert health["go"]["status"] == "unavailable"
    assert health["go"]["http_status"] == 403
    assert gateway.usage_report()["total"] == 0.0


def test_providers_endpoint_offers_fallback_models() -> None:
    client, _jobs = _client(
        ScriptedGateway(chat=lambda *_a, **_k: "{}", decision=_decide)
    )

    body = client.get("/api/providers?probe=true").json()

    assert body["providers"] == {}
    assert body["fallback"]["writer"] and body["fallback"]["strong"]


def test_stale_cancel_cannot_persist_over_a_replacement_job(tmp_path):
    store = RunStore(tmp_path / "replacement.sqlite")
    jobs = RunJobs(store=store)
    first_entered = threading.Event()
    first_release = threading.Event()
    stale_entered = threading.Event()
    stale_release = threading.Event()
    replacement_entered = threading.Event()
    replacement_release = threading.Event()
    persist = jobs._persist

    def hold_old_cancel(job, **kwargs):
        if threading.current_thread().name == "held-cancel":
            stale_entered.set()
            assert stale_release.wait(5)
        return persist(job, **kwargs)

    jobs._persist = hold_old_cancel

    def first(_progress, _cancel, _observe):
        first_entered.set()
        assert first_release.wait(5)
        return {"status": "completed"}

    def replacement(_progress, _cancel, observe):
        observe({"event": "start", "operation": "replacement"})
        replacement_entered.set()
        assert replacement_release.wait(5)
        return {"status": "completed"}

    jobs.submit("same-run", "optimize", first, lambda exc: {})
    canceller = threading.Thread(
        target=lambda: jobs.cancel("same-run"), name="held-cancel"
    )
    try:
        assert first_entered.wait(5)
        canceller.start()
        assert stale_entered.wait(5)
        first_release.set()
        jobs.wait("same-run")
        jobs.submit("same-run", "continue", replacement, lambda exc: {})
        assert replacement_entered.wait(5)
        stale_release.set()
        canceller.join(5)
        assert not canceller.is_alive()
        saved = store.get_run("same-run")["job"]
        assert saved["kind"] == "continue"
        assert saved["state"] == "running"
        assert saved["cancel_requested"] is False
        assert saved["operation"]["operation"] == "replacement"
    finally:
        first_release.set()
        stale_release.set()
        replacement_release.set()
        canceller.join(5)
        jobs._executor.shutdown(wait=True)
        store.close()


@pytest.mark.parametrize("kind", ["resume", "skip", "continue", "optimize"])
@pytest.mark.parametrize("status", ["paused", "completed"])
def test_restart_preserves_saved_canonical_result(tmp_path, kind, status):
    from prompt_enhancer.optimizer import PAUSED_REPORT_STATUS, RESUME_CONTEXT_KEY

    store = RunStore(tmp_path / "canonical.sqlite")
    result = {
        "status": status,
        "final_prompt": "Accepted changed prompt.",
        "report": {
            "status": PAUSED_REPORT_STATUS if status == "paused" else "completed"
        },
    }
    context = {"round": 2}
    store.save_run(
        {
            "run_id": "canonical",
            "prompt": "Original.",
            "result": result,
            "timing": {"finished_at": "saved-time"},
            RESUME_CONTEXT_KEY: context,
            "job": {"state": "running", "kind": kind},
        }
    )
    jobs = RunJobs(store=store)
    try:
        assert jobs.get("canonical")["state"] == "interrupted"
        assert jobs.get("canonical")["result"] == result
        saved = store.get_run("canonical")
        assert saved["result"] == result
        assert saved["timing"]["finished_at"] == "saved-time"
        assert saved[RESUME_CONTEXT_KEY] == context
        if status == "paused":
            optimizer = PromptOptimizer(store=store, gateway=ScriptedGateway())
            assert optimizer._paused_record("canonical")[1:] == (result, context)
    finally:
        jobs._executor.shutdown(wait=True)
        store.close()


def test_cancelled_api_job_persists_completed_provider_spend(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class PaidTransport:
        def request(self, _url, **_kwargs):
            entered.set()
            assert release.wait(5)
            return {
                "status_code": 200,
                "json": {
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "cost": 0.03}
                },
            }

    gateway = HttpGateway(PaidTransport(), config=GatewayConfig(max_retries=0))
    store = RunStore(tmp_path / "paid.sqlite")
    app = create_app(optimizer=PromptOptimizer(store=store, gateway=gateway))
    client = TestClient(app)
    run_id = client.post(
        "/api/jobs/optimize", json={"prompt": "Explain a seed."}
    ).json()["run_id"]
    try:
        assert entered.wait(5)
        client.post(f"/api/jobs/{run_id}/cancel")
        release.set()
        result = app.state.jobs.wait(run_id)["result"]
        assert result["report"]["status"] == "cancelled"
        assert result["cost"]["total"] == 0.03
        assert store.get_run(run_id)["cost"]["total"] == 0.03
        assert RunHistory(store).get_run(run_id)["cost"]["total"] == 0.03
    finally:
        release.set()
        app.state.jobs._executor.shutdown(wait=True)
        store.close()


@pytest.mark.parametrize("operation", ["get", "active", "cancel"])
def test_job_status_and_cancel_signal_do_not_wait_for_sqlite_io(tmp_path, operation):
    from concurrent.futures import ThreadPoolExecutor

    store = RunStore(tmp_path / "slow-io.sqlite")
    jobs = RunJobs(store=store)
    writing = threading.Event()
    release_write = threading.Event()
    save = store.save_run

    def slow_operation_write(record):
        if (record.get("job", {}).get("operation") or {}).get(
            "operation"
        ) == "held-write" and not writing.is_set():
            writing.set()
            assert release_write.wait(5)
        return save(record)

    store.save_run = slow_operation_write

    def work(_progress, cancel_check, observe):
        observe({"event": "start", "operation": "held-write"})
        if cancel_check():
            raise RunCancelled("slow")
        return {"status": "completed"}

    jobs.submit("slow", "optimize", work, lambda exc: {"status": "cancelled"})
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        assert writing.wait(2)
        if operation == "cancel":
            future = pool.submit(jobs.cancel, "slow")
            # The cancellation response may await its durable write, but the
            # provider-facing signal must not wait for unrelated SQLite I/O.
            assert jobs._jobs["slow"].cancel.wait(0.5)
        elif operation == "get":
            future = pool.submit(jobs.get, "slow")
            assert future.result(timeout=0.5)["state"] == "running"
        else:
            future = pool.submit(jobs.active)
            assert future.result(timeout=0.5)[0]["run_id"] == "slow"
    finally:
        release_write.set()
        pool.shutdown(wait=True)
        jobs._executor.shutdown(wait=True)
        store.close()
