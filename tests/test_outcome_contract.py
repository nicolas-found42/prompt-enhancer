"""Public result, history, control and feedback use the same outcome evidence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_always_attempt import _gateway as improvement_gateway
from test_jobs_api import _clarification_gateway
from test_understand_route import _gateway as route_gateway

from prompt_enhancer.api import create_app
from prompt_enhancer.gateway import ProviderError, ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore

ORIGINAL = "whats 2 plus 2"
FINAL = "What is 2 + 2?"


def _client(gateway):
    optimizer = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), writer_instruction_version=4
    )
    return TestClient(create_app(optimizer=optimizer))


def _assert_history_matches(client, result, expected, style):
    detail = client.get(f"/api/runs/{result['run_id']}").json()
    report = result["report"]
    for field in ("outcome", "outcome_reason", "applied_style", "control_state"):
        assert detail.get(field) == report.get(field), field
    assert report["outcome"] == expected
    assert report["outcome_reason"]
    assert report["applied_style"] == style
    return detail


@pytest.mark.parametrize("kind", ["converged", "impossible", "failed_operational"])
def test_public_result_and_history_agree_on_terminal_outcome(kind):
    prompt, style = ORIGINAL, "clearer"
    if kind == "impossible":
        gateway = route_gateway()
        prompt, style = 'Reply with exactly: "OK"', "creative"
    else:
        gateway = improvement_gateway()
        if kind == "failed_operational":
            original_decide = gateway.decision_handler

            def fail_at_score(request, **kwargs):
                if str(request.get("key", "")).startswith("score:"):
                    raise ProviderError(
                        "scripted", "judge", 503, "Provider unavailable"
                    )
                return original_decide(request, **kwargs)

            gateway.decision_handler = fail_at_score
    client = _client(gateway)
    response = client.post(
        "/api/optimize",
        json={
            "prompt": prompt,
            "improvement_style": style,
            "clarification_allowed": False,
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    _assert_history_matches(client, result, kind, style)
    if kind == "converged":
        assert result["final_prompt"] == FINAL
    else:
        assert result["final_prompt"] == prompt


def _accepted_then_paused_gateway(*, tested):
    base = improvement_gateway(with_test=tested, baseline_score_probability=0.05)
    batches = 0
    gateway = None

    def chat(model, messages, *, role, **kwargs):
        nonlocal batches
        if role == "writer":
            if "state.strategies" in messages[0]["content"]:
                batches += 1
                if batches == 2:
                    gateway.usage.record(
                        role="writer", provider="scripted", model=model, cost=2.0
                    )
            return base.chat_handler(model, messages, role=role, **kwargs)
        return "fail" if messages[0]["content"] == ORIGINAL else "pass"

    def decide(request, **kwargs):
        key = str(request.get("key", ""))
        if key.startswith("score:"):
            state = request["state"]
            probability = (
                0.05
                if state["candidate_prompt"] == state["original_prompt"]
                else 0.1
                if batches == 1
                else 0.99
            )
            return {"type": "noul", "probability_true": probability, "confidence": 1.0}
        return base.decision_handler(request, **kwargs)

    gateway = ScriptedGateway(chat=chat, decision=decide)
    return gateway


@pytest.mark.parametrize("tested", [False, True])
def test_accepted_stop_keeps_outcome_and_feedback_but_pause_requires_approval(tested):
    client = _client(_accepted_then_paused_gateway(tested=tested))
    response = client.post(
        "/api/optimize",
        json={
            "prompt": ORIGINAL,
            "improvement_style": "clearer",
            "clarification_allowed": False,
            "spend_limit_usd": 1,
        },
    )
    assert response.status_code == 200, response.text
    paused = response.json()
    assert paused["status"] == "needs_input", paused["report"]
    assert paused["report"]["control_state"] == "awaiting_approval"
    assert paused["final_prompt"] == FINAL
    run_id = paused["run_id"]
    assert (
        client.post(
            f"/api/runs/{run_id}/feedback", json={"decision": "accept"}
        ).status_code
        == 409
    )
    stopped_response = client.post(f"/api/runs/{run_id}/stop")
    assert stopped_response.status_code == 200, stopped_response.text
    stopped = stopped_response.json()
    expected = "improved_tested" if tested else "improved_unverified"
    _assert_history_matches(client, stopped, expected, "clearer")
    assert stopped["report"]["control_state"] == "stopped"
    assert stopped["final_prompt"] == FINAL
    feedback = client.post(f"/api/runs/{run_id}/feedback", json={"decision": "accept"})
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["feedback"] == "accept"
    assert feedback.json()["feedback_labels"]["status"] == "linked"


def test_assumption_edit_invalidates_active_outcome_and_feedback_vector():
    gateway = _clarification_gateway()
    original_decide = gateway.decision_handler

    def confirm_supported_user_edit(request, **kwargs):
        if request.get("key") == "assumption_meaning":
            assert "goal: Analyze" in request["state"]["updated_prompt"]
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        return original_decide(request, **kwargs)

    gateway.decision_handler = confirm_supported_user_edit
    client = _client(gateway)
    initial = client.post("/api/optimize", json={"prompt": "Help me with this."}).json()
    answers = {
        question["id"]: question["options"][0]["value"]
        for question in initial["questions"]
    }
    resumed_response = client.post(
        f"/api/runs/{initial['run_id']}/resume", json={"answers": answers}
    )
    assert resumed_response.status_code == 200, resumed_response.text
    resumed = resumed_response.json()
    _assert_history_matches(client, resumed, "converged", "clearer")
    edited_response = client.post(
        f"/api/runs/{initial['run_id']}/assumption",
        json={"assumption": {"key": "goal", "value": "Analyze"}},
    )
    assert edited_response.status_code == 200, edited_response.text
    edited = edited_response.json()
    detail = client.get(f"/api/runs/{initial['run_id']}").json()
    assert edited["final_prompt"] != resumed["final_prompt"]
    assert edited["report"]["outcome"] is detail["outcome"] is None
    assert "original optimization" in edited["report"]["summary"]
    feedback = client.post(
        f"/api/runs/{initial['run_id']}/feedback", json={"decision": "accept"}
    )
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["feedback_labels"]["status"] == "unavailable"
