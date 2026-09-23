from __future__ import annotations

import json

import pytest

from prompt_enhancer.catalog import JEV_MODEL, ModelInfo, StaticModelCatalog
from prompt_enhancer.gateway import (
    GatewayConfig,
    ModelGateway,
    ReplayGateway,
    ScriptedGateway,
)
from prompt_enhancer.usage import UsageLedger


class Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class QueueTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, url, **kwargs):
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_routes_go_and_openrouter_with_stable_session_and_fixed_jev():
    transport = QueueTransport(
        [
            Response(200, {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}),
            Response(200, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
            Response(200, {"ok": True}),
        ]
    )
    catalog = StaticModelCatalog(
        [
            ModelInfo(
                "go-writer", "go", input_cost_per_token=0.1, output_cost_per_token=0.2
            )
        ],
        [
            ModelInfo(
                "or-writer",
                "openrouter",
                input_cost_per_token=0.01,
                output_cost_per_token=0.02,
            )
        ],
    )
    gateway = ModelGateway(
        transport,
        config=GatewayConfig(
            go_api_key="go-secret",
            openrouter_api_key="or-secret",
            go_base_url="https://go.test/v1",
            openrouter_base_url="https://or.test/v1",
            max_retries=0,
        ),
        catalog=catalog,
    )
    gateway.new_run("run-1")
    gateway.chat("go-writer", "hello", role="writer")
    gateway.chat("or-writer", "hello", role="strong", run_id="run-1")
    gateway.decide(
        {"state": "the user prompt", "instructions": "classify"}, role="judge"
    )

    go_request, openrouter_request = transport.requests[:2]
    assert go_request["url"] == "https://go.test/v1/chat/completions"
    assert go_request["headers"]["x-opencode-session"] == "run-1"
    assert go_request["headers"]["User-Agent"]
    assert openrouter_request["url"] == "https://or.test/v1/chat/completions"
    assert "x-opencode-session" not in openrouter_request["headers"]
    assert transport.requests[2]["json"]["model"] == JEV_MODEL
    assert transport.requests[2]["json"]["state"] == "the user prompt"
    assert gateway.usage.role_cost("writer") == pytest.approx(0.8)
    assert gateway.usage.role_cost("strong") == pytest.approx(0.03)
    assert "go-secret" not in json.dumps(gateway.usage_report())


def test_retries_provider_error_without_returning_key():
    transport = QueueTransport(
        [
            Response(503, {"error": {"message": "try later"}}),
            Response(200, {"ok": True}),
        ]
    )
    gateway = ModelGateway(
        transport,
        config=GatewayConfig(openrouter_api_key="never-return-me", max_retries=1),
    )
    assert gateway.chat("model", "prompt", role="writer") == {"ok": True}
    assert len(transport.requests) == 2
    assert "never-return-me" not in json.dumps(gateway.calls)


def test_settings_overrides_are_per_run_and_persist(tmp_path):
    from prompt_enhancer.settings import SettingsStore

    store = SettingsStore(tmp_path / "settings.json")
    saved = store.update_defaults(
        writer="writer-v2", strong="strong-v2", weak=["weak-v2"]
    )
    assert saved.defaults.writer == "writer-v2"
    assert store.resolve({"writer": "one-run"}).defaults.writer == "one-run"
    assert store.load().defaults.writer == "writer-v2"


def test_scripted_and_replay_gateways_are_deterministic():
    scripted = ScriptedGateway([{"text": "one"}])
    assert scripted.complete("m", "hello") == {"text": "one"}
    replay = ReplayGateway({("chat", "m", "writer"): {"text": "replayed"}})
    assert replay.complete("m", "hello", role="writer") == {"text": "replayed"}
    assert replay.calls[0]["operation"] == "chat"


def test_usage_ledger_splits_roles():
    ledger = UsageLedger()
    ledger.record(
        role="writer",
        provider="go",
        model="m",
        response={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        input_cost_per_token=0.1,
        output_cost_per_token=0.2,
    )
    ledger.record(
        role="strong",
        provider="openrouter",
        model="s",
        response={"usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        input_cost_per_token=0.01,
        output_cost_per_token=0.02,
    )
    report = ledger.to_dict()
    assert report["cost_by_role"]["writer"] == pytest.approx(2.0)
    assert report["cost_by_role"]["strong"] == pytest.approx(0.04)
