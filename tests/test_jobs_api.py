import threading

from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app, run_estimates
from prompt_enhancer.gateway import (
    GatewayConfig,
    ModelGateway,
    ProviderError,
    ScriptedGateway,
)
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def _decide(request, **_kwargs):
    if request.get("type") == "choice":
        choice = "general" if request.get("key") == "task_type" else "none"
        return {"type": "choice", "choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0}
    return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}


def _client(gateway) -> tuple[TestClient, object]:
    app = create_app(optimizer=PromptOptimizer(store=RunStore(":memory:"), gateway=gateway))
    return TestClient(app), app.state.jobs


def test_optimize_job_returns_run_id_at_once_and_finishes_with_the_result() -> None:
    client, jobs = _client(ScriptedGateway(chat=lambda *_a, **_k: '{"tests":[]}', decision=_decide))

    started = client.post("/api/jobs/optimize", json={"prompt": "Explain recursion.", "tier": "fast"})

    assert started.status_code == 202
    run_id = started.json()["run_id"]
    jobs.wait(run_id)
    job = client.get(f"/api/jobs/{run_id}").json()
    assert job["state"] == "done"
    assert job["result"]["status"] == "completed"
    assert "diagnosing" in job["stages_seen"]
    assert client.get(f"/api/runs/{run_id}").status_code == 200


def test_optimize_job_rejects_invalid_requests_before_starting() -> None:
    client, _jobs = _client(ScriptedGateway(chat=lambda *_a, **_k: "{}", decision=_decide))

    assert client.post("/api/jobs/optimize", json={"prompt": "  "}).status_code == 422
    assert client.get("/api/jobs/unknown").status_code == 404


def test_refused_provider_produces_a_failed_result_with_a_plain_hint() -> None:
    def chat(model, *_args, role, **_kwargs):
        raise ProviderError("go", model, 403, role=role)

    client, jobs = _client(ScriptedGateway(chat=chat, decision=_decide))
    run_id = client.post("/api/jobs/optimize", json={"prompt": "Write a reply.", "tier": "fast"}).json()["run_id"]

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

    client, jobs = _client(ScriptedGateway(chat=lambda *_a, **_k: '{"tests":[]}', decision=decide))
    run_id = client.post("/api/jobs/optimize", json={"prompt": "Write a reply.", "tier": "fast"}).json()["run_id"]
    assert entered.wait(5)

    assert client.post(f"/api/jobs/{run_id}/cancel").json()["cancel_requested"] is True
    release.set()
    result = jobs.wait(run_id)["result"]

    assert result["status"] == "failed"
    assert result["report"]["status"] == "cancelled"
    assert result["final_prompt"] == "Write a reply."


def test_invalid_writer_reply_is_not_reported_as_a_network_error() -> None:
    client, jobs = _client(ScriptedGateway(chat=lambda *_a, **_k: "not json", decision=_decide))
    run_id = client.post("/api/jobs/optimize", json={"prompt": "Write a reply.", "tier": "fast"}).json()["run_id"]

    failure = jobs.wait(run_id)["result"]["report"]["failure"]

    assert failure["kind"] == "invalid_response"
    assert "network" not in failure["message"]
    assert failure["headline"] == "The writer model gave an unusable reply"


def test_estimates_use_completed_runs_per_tier() -> None:
    runs = [
        {"status": "completed", "tier": "standard", "timings": {"total_ms": 60000 * minutes}, "cost": {"total": minutes / 100}}
        for minutes in (2, 4, 6, 8, 10)
    ] + [{"status": "failed", "tier": "standard", "timings": {"total_ms": 1000}, "cost": {"total": 0.0}}]

    estimate = run_estimates(runs)["standard"]

    assert estimate["runs"] == 5
    assert estimate["minutes"][0] == 6
    assert 8 < estimate["minutes"][1] <= 10


def test_provider_probe_marks_a_refused_provider_unavailable_without_recording_cost() -> None:
    class Transport:
        def request(self, url, **_kwargs):
            return {"status_code": 403, "json": {}}

    gateway = ModelGateway(Transport(), config=GatewayConfig(), go_models=["go-writer"])

    health = gateway.provider_health(probe_models=["go-writer"])

    assert health["go"]["status"] == "unavailable"
    assert health["go"]["http_status"] == 403
    assert gateway.usage_report()["total"] == 0.0


def test_providers_endpoint_offers_fallback_models() -> None:
    client, _jobs = _client(ScriptedGateway(chat=lambda *_a, **_k: "{}", decision=_decide))

    body = client.get("/api/providers?probe=true").json()

    assert body["providers"] == {}
    assert body["fallback"]["writer"] and body["fallback"]["strong"]
