import json
import threading

from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    ProviderError,
    ScriptedGateway,
)
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
    # The stub keeps the original prompt and supplies no tests, so the run
    # converges on its measured baseline with an unverified result.
    assert job["result"]["status"] == "completed"
    assert job["result"]["report"]["status"] == "converged"
    assert job["result"]["report"]["convergence"]["verification"] == "unverified"
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
    assert client.get(f"/api/jobs/{run_id}").status_code == 404

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

    assert client.post(f"/api/jobs/{run_id}/cancel").json()["cancel_requested"] is True
    release.set()
    result = jobs.wait(run_id)["result"]

    assert result["status"] == "failed"
    assert result["report"]["status"] == "cancelled"
    assert result["final_prompt"] == "Write a reply."
    summary = client.get("/api/runs").json()[0]
    assert summary["status"] == "failed"
    assert summary["control_state"] == "cancelled"
    assert summary["outcome"] is None


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
