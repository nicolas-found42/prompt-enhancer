from __future__ import annotations

import json

import pytest

from prompt_enhancer.catalog import (
    JEV_MODEL,
    LiveModelCatalog,
    ModelInfo,
    StaticModelCatalog,
)
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    ProviderError,
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


def test_default_go_route_uses_subscription_endpoint():
    gateway = HttpGateway(config=GatewayConfig(), go_models=["go-writer"])

    assert (
        gateway.route_model("go-writer").url
        == "https://opencode.ai/zen/go/v1/chat/completions"
    )
    assert (
        GatewayConfig.from_env({}).go_models_url
        == "https://opencode.ai/zen/go/v1/models"
    )


def test_gateway_reads_provider_credentials_from_environment():
    gateway = HttpGateway.from_env(
        {"OPENCODE_GO_KEY": "go-test-key", "OPENROUTER_API_KEY": "router-test-key"}
    )

    assert gateway.config.go_api_key == "go-test-key"
    assert gateway.config.openrouter_api_key == "router-test-key"


def test_known_go_defaults_do_not_fall_back_to_openrouter_without_catalog():
    gateway = HttpGateway(config=GatewayConfig())

    assert gateway.route_model("deepseek-v4.1-flash").provider == "go"
    assert gateway.route_model("space-bunny-free").provider == "go"
    assert gateway.route_model("glm-5.3-flash").provider == "go"
    assert gateway.route_model("qwen3.8-flash").url.endswith("/messages")
    assert gateway.route_model("muse-spark-1.3-contributor").url.endswith("/responses")


def test_go_catalog_sends_its_user_agent():
    transport = QueueTransport(
        [
            Response(200, {"data": [{"id": "go-writer"}]}),
            Response(200, {"data": [{"id": "or-writer"}]}),
        ]
    )
    catalog = LiveModelCatalog(
        transport,
        go_url="https://go.test/models",
        openrouter_url="https://or.test/models",
    )

    assert catalog.fetch().go[0].id == "go-writer"
    assert transport.requests[0]["headers"]["User-Agent"] == "prompt-enhancer/0.1"


def test_go_anthropic_model_uses_messages_endpoint_and_normalizes_output():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "content": [{"type": "text", "text": "Done"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            )
        ]
    )
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(go_api_key="go-test-key"),
        catalog=StaticModelCatalog(["qwen3.8-flash"]),
    )

    response = gateway.chat(
        "qwen3.8-flash",
        [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Say done"},
        ],
        seed=9,
    )

    request = transport.requests[0]
    assert request["url"] == "https://opencode.ai/zen/go/v1/messages"
    assert request["json"]["system"] == "Be concise"
    assert "seed" not in request["json"]
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert request["headers"]["x-api-key"] == "go-test-key"
    assert "Authorization" not in request["headers"]
    assert response["choices"][0]["message"]["content"] == "Done"
    assert gateway.usage.for_role("writer")[0].output_tokens == 1


def test_go_responses_model_uses_responses_endpoint_and_normalizes_output():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Done"}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            )
        ]
    )
    gateway = HttpGateway(transport, catalog=StaticModelCatalog(["grok-4.6"]))

    response = gateway.chat("grok-4.6", "Say done", max_tokens=32, seed=9)

    request = transport.requests[0]
    assert request["url"] == "https://opencode.ai/zen/go/v1/responses"
    assert request["json"]["max_output_tokens"] == 32
    assert "seed" not in request["json"]
    assert response["choices"][0]["message"]["content"] == "Done"


def test_muse_reserves_room_for_reasoning_before_visible_output():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "OK"}],
                        }
                    ],
                },
            )
        ]
    )
    gateway = HttpGateway(
        transport, catalog=StaticModelCatalog(["muse-spark-1.3-contributor"])
    )

    gateway.chat("muse-spark-1.3-contributor", "Return OK", role="weak")

    assert transport.requests[0]["json"]["max_output_tokens"] == 4096


def test_routes_go_and_openrouter_with_stable_session_and_fixed_jev():
    transport = QueueTransport(
        [
            Response(200, {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}),
            Response(200, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
            Response(
                200,
                {
                    "model": JEV_MODEL,
                    "answers": {"decision": {"type": "noul", "noul": 0.9}},
                },
            ),
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
    gateway = HttpGateway(
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
    gateway = HttpGateway(
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


def test_scripted_gateway_answers_in_order_then_reports_exhaustion():
    scripted = ScriptedGateway([{"text": "one"}])
    messages = [{"role": "user", "content": "hello"}]

    assert scripted.chat("m", messages, role="writer") == {"text": "one"}
    with pytest.raises(ProviderError, match="no scripted response remains"):
        scripted.chat("m", messages, role="writer")


def test_strict_replay_requires_the_recorded_request() -> None:
    payload = {
        "model": "writer",
        "messages": [{"role": "user", "content": "First prompt"}],
    }
    key = ReplayGateway.request_key("chat", "writer", payload, "writer")
    replay = ReplayGateway({key: {"text": "First result"}})

    assert replay.chat("writer", payload["messages"], role="writer") == {
        "text": "First result"
    }
    assert replay.calls[0]["operation"] == "chat"
    with pytest.raises(ProviderError, match="no recorded response"):
        replay.chat(
            "writer", [{"role": "user", "content": "Another prompt"}], role="writer"
        )

    single = ReplayGateway({"some-other-key": {"text": "generic"}})
    with pytest.raises(ProviderError, match="no recorded response"):
        single.chat("writer", payload["messages"], role="writer")


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


def test_jev_requests_use_decisions_api_and_return_typed_answer_payload():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "model": JEV_MODEL,
                    "answers": {"meaning": {"type": "noul", "noul": 0.95}},
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 1,
                        "cost": 0.0000042,
                    },
                },
            ),
        ]
    )
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(openrouter_api_key="or-secret", max_retries=0),
    )

    answer = gateway.decide(
        {
            "key": "meaning",
            "type": "noul",
            "query": "Does the rewrite preserve the request?",
            "state": {"original": "A", "rewrite": "B"},
        }
    )

    assert answer == {"type": "noul", "noul": 0.95}
    request = transport.requests[0]
    assert request["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert request["json"] == {
        "model": JEV_MODEL,
        "state": {"original": "A", "rewrite": "B"},
        "questions": {
            "meaning": {
                "type": "noul",
                "instructions": "Does the rewrite preserve the request?",
            }
        },
    }
    assert gateway.usage_report()["total"] == pytest.approx(0.0000042)
    assert gateway.decision_log[0]["answered_by"] == JEV_MODEL
    assert gateway.decision_log[0]["usage"]["cost"] == pytest.approx(0.0000042)


def test_decide_batch_sends_one_request_for_multiple_questions():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "model": JEV_MODEL,
                    "answers": {
                        "gap:goal": {"type": "noul", "noul": 0.91},
                        "gap:context": {"type": "noul", "noul": 0.12},
                    },
                },
            ),
        ]
    )
    gateway = HttpGateway(transport, config=GatewayConfig(max_retries=0))
    answers = gateway.decide_batch(
        [
            {
                "key": "gap:goal",
                "type": "noul",
                "query": "Is goal missing?",
                "state": {"prompt": "A"},
            },
            {
                "key": "gap:context",
                "type": "noul",
                "query": "Is context missing?",
                "state": {"prompt": "A"},
            },
        ]
    )
    assert [answer["noul"] for answer in answers] == [0.91, 0.12]
    assert len(transport.requests) == 1
    assert all(entry["answered_by"] == JEV_MODEL for entry in gateway.decision_log)


def test_jev_response_without_model_snapshot_is_rejected():
    gateway = HttpGateway(
        QueueTransport(
            [Response(200, {"answers": {"decision": {"type": "noul", "noul": 0.5}}})]
        ),
        config=GatewayConfig(max_retries=0),
    )
    with pytest.raises(ProviderError, match="missing model snapshot"):
        gateway.decide({"state": "prompt", "instructions": "judge"})


def test_529_retry_honors_retry_after():
    delays = []
    transport = QueueTransport(
        [
            {"status_code": 529, "headers": {"Retry-After": "2"}},
            Response(200, {"ok": True}),
        ]
    )
    gateway = HttpGateway(
        transport, config=GatewayConfig(max_retries=1), sleep=delays.append
    )
    assert gateway.chat("model", "prompt") == {"ok": True}
    assert delays == [2.0]
    assert len(transport.requests) == 2


def test_configured_jev_pin_is_sent_to_decisions_api():
    pin = "typesafe/jev-1.13-20261001"
    transport = QueueTransport(
        [Response(200, {"model": pin, "answers": {"decision": True}})]
    )
    gateway = HttpGateway(transport, config=GatewayConfig(jev_model=pin, max_retries=0))
    assert gateway.decide({"state": "prompt", "instructions": "judge"}) is True
    assert transport.requests[0]["json"]["model"] == pin
    assert gateway.decision_log[0]["answered_by"] == pin
