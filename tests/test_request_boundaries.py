"""Malformed public requests fail before jobs, persistence or Gateway calls."""

import pytest
from fastapi.testclient import TestClient

from prompt_enhancer.api import ContinueRequest, create_app
from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


@pytest.mark.parametrize("endpoint", ["/api/optimize", "/api/jobs/optimize"])
@pytest.mark.parametrize(
    "extra",
    [
        {"options": {"model_overrides": 5}},
        {"options": {"model_overrides": ["writer"]}},
        {"options": {"model_overrides": False}},
        {"options": {"model_overrides": None}},
        {"options": {"model_overrides": []}},
        {
            "options": {"model_overrides": 5},
            "model_overrides": {"writer": "test-model"},
        },
        {"time_limit_s": True},
        {"time_limit_s": "0"},
        {"spend_limit_usd": False},
        {"spend_limit_usd": "0"},
        {"clarification_allowed": "false"},
        {"options": {"clarification_allowed": "false"}},
        {"options": {"clarification_allowed": None}},
        {"options": {"model_overrides": {"writer": " "}}},
        {"options": {"model_overrides": {"strong": "\t"}}},
    ],
)
def test_invalid_request_has_no_side_effects(endpoint, extra):
    calls = []
    store = RunStore(":memory:")
    gateway = ScriptedGateway(
        chat=lambda *args, **kwargs: calls.append("chat") or "{}",
        decision=lambda *args, **kwargs: calls.append("decision") or {},
    )
    optimizer = PromptOptimizer(config=Settings(), store=store, gateway=gateway)
    client = TestClient(create_app(optimizer=optimizer), raise_server_exceptions=False)

    response = client.post(endpoint, json={"prompt": "Explain recursion.", **extra})
    if response.status_code == 202:
        optimizer.jobs.wait(response.json()["run_id"], timeout=5)

    assert response.status_code == 422, response.text
    assert response.json()["detail"]
    assert calls == []
    assert store.list_runs() == []
    assert optimizer.jobs.active() == []


@pytest.mark.parametrize("value", [False, "false", None, 0, []])
def test_library_rejects_non_mapping_model_overrides(value):
    optimizer = PromptOptimizer(config=Settings(), store=RunStore(":memory:"))
    with pytest.raises(ValueError, match="model_overrides must be a mapping"):
        optimizer.validate_request("Explain recursion.", {"model_overrides": value})


@pytest.mark.parametrize("value", [None, "false", 0, [], {}])
def test_library_rejects_non_boolean_clarification(value):
    optimizer = PromptOptimizer(config=Settings(), store=RunStore(":memory:"))
    with pytest.raises(ValueError, match="clarification_allowed must be a boolean"):
        optimizer.validate_request(
            "Explain recursion.", {"clarification_allowed": value}
        )


def test_valid_overrides_merge_with_top_level_precedence():
    from prompt_enhancer.api import OptimizeRequest, _optimize_options

    request = OptimizeRequest(
        prompt="Explain recursion.",
        options={"model_overrides": {"writer": "old", "strong": "strong-test"}},
        model_overrides={"writer": "new"},
        clarification_allowed=False,
        time_limit_s=0,
        spend_limit_usd=0.5,
    )
    options = _optimize_options(request)
    optimizer = PromptOptimizer(config=Settings(), store=RunStore(":memory:"))
    optimizer.validate_request(request.prompt, options)
    assert options["model_overrides"] == {"writer": "new", "strong": "strong-test"}
    assert options["clarification_allowed"] is False
    assert options["time_limit_s"] == 0
    assert options["spend_limit_usd"] == 0.5
    assert request.options["model_overrides"]["writer"] == "old"


@pytest.mark.parametrize("field", ["time_limit_s", "spend_limit_usd"])
@pytest.mark.parametrize("value", [True, False, "0"])
def test_continue_uses_the_same_strict_budget_types(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ContinueRequest.model_validate({field: value})


@pytest.mark.parametrize("endpoint", ["/api/optimize", "/api/jobs/optimize"])
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "-1"])
def test_nonfinite_and_negative_budget_json_has_no_side_effects(endpoint, literal):
    store = RunStore(":memory:")
    calls = []
    optimizer = PromptOptimizer(
        config=Settings(),
        store=store,
        gateway=ScriptedGateway(
            decision=lambda *args, **kwargs: calls.append("decision") or {}
        ),
    )
    client = TestClient(create_app(optimizer=optimizer), raise_server_exceptions=False)
    response = client.post(
        endpoint,
        content='{"prompt":"Explain recursion.","time_limit_s":' + literal + "}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    assert calls == []
    assert store.list_runs() == []
