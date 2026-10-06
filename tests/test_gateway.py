from __future__ import annotations

import contextvars
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from prompt_enhancer.catalog import (
    JEV_MODEL,
    CatalogSnapshot,
    LiveModelCatalog,
    ModelInfo,
    StaticModelCatalog,
)
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    HttpTransport,
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
    assert GatewayConfig.from_env({}).operation_timeout_s == 180.0
    assert (
        GatewayConfig.from_env(
            {"PROMPT_ENHANCER_OPERATION_TIMEOUT": "45"}
        ).operation_timeout_s
        == 45.0
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


def test_jev_preserves_structured_instructions_and_choice_descriptions():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "model": JEV_MODEL,
                    "answers": {
                        "kind": {
                            "type": "choice",
                            "choice": "direct",
                            "probabilities": {"direct": 0.8, "other": 0.2},
                            "confidence": 0.6,
                        }
                    },
                },
            )
        ]
    )
    gateway = HttpGateway(transport, config=GatewayConfig(max_retries=0))
    instructions = {"question": "Which kind?", "focus": ["tone", "intent"]}
    criteria = {"direct": {"what": "A direct request"}, "other": None}

    gateway.decide(
        {
            "key": "kind",
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
            "state": {"prompt": "Please answer"},
        }
    )

    assert transport.requests[0]["json"] == {
        "model": JEV_MODEL,
        "state": {"prompt": "Please answer"},
        "questions": {
            "kind": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }


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
    assert [entry["question"]["key"] for entry in gateway.decision_log] == [
        "gap:goal",
        "gap:context",
    ]


def test_decide_batch_logs_only_received_answers_before_incomplete_response_error():
    received = {"type": "noul", "noul": 0.91, "confidence": 0.88}
    gateway = HttpGateway(
        QueueTransport(
            [
                Response(
                    200,
                    {
                        "model": JEV_MODEL,
                        "usage": {"total_tokens": 17},
                        "answers": {"answered": received},
                    },
                )
            ]
        ),
        config=GatewayConfig(max_retries=0),
    )
    requests = [
        {"key": "answered", "type": "noul", "query": "q1", "state": {"prompt": "A"}},
        {"key": "missing", "type": "noul", "query": "q2", "state": {"prompt": "B"}},
    ]

    with pytest.raises(ProviderError, match="decision answers are missing"):
        gateway.decide_batch(requests)

    assert [entry["question"]["key"] for entry in gateway.decision_log] == ["answered"]
    assert gateway.decision_log[0]["question"] == requests[0]
    assert gateway.decision_log[0]["answer"] == received
    assert gateway.decision_log[0]["answered_by"] == JEV_MODEL
    assert gateway.decision_log[0]["usage"] == {"total_tokens": 17}


def test_partial_decide_batch_without_model_logs_received_answer_as_unknown():
    received = {"type": "noul", "noul": 0.91, "confidence": 0.88}
    gateway = HttpGateway(
        QueueTransport(
            [
                Response(
                    200,
                    {"answers": {"answered": received}},
                )
            ]
        ),
        config=GatewayConfig(max_retries=0),
    )

    with pytest.raises(ProviderError, match="decision answers are missing"):
        gateway.decide_batch(
            [
                {"key": "answered", "type": "noul", "query": "q1", "state": {}},
                {"key": "missing", "type": "noul", "query": "q2", "state": {}},
            ]
        )

    assert len(gateway.decision_log) == 1
    assert gateway.decision_log[0]["answer"] == received
    assert gateway.decision_log[0]["answered_by"] is None
    assert gateway.decision_log[0]["usage"] == {}


def test_decide_batch_addresses_distinct_states_with_structured_instructions():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "model": JEV_MODEL,
                    "answers": {
                        "first": {
                            "type": "choice",
                            "choice": "direct",
                            "probabilities": {"direct": 0.9, "other": 0.1},
                            "confidence": 0.8,
                        },
                        "second": {"type": "noul", "noul": 0.1},
                    },
                },
            )
        ]
    )
    gateway = HttpGateway(transport, config=GatewayConfig(max_retries=0))
    object_instructions = {"question": "Is it clear?", "focus": ["intent"]}
    array_instructions = ["Check the intent", "Check the audience"]
    criteria = {"direct": {"what": "An explicit request"}, "other": None}

    gateway.decide_batch(
        [
            {
                "key": "first",
                "type": "choice",
                "query": object_instructions,
                "criteria": criteria,
                "state": {"prompt": "A"},
            },
            {
                "key": "second",
                "type": "noul",
                "query": array_instructions,
                "state": {"prompt": "B"},
            },
        ]
    )

    payload = transport.requests[0]["json"]
    assert payload["state"] == {
        "items": {"first": {"prompt": "A"}, "second": {"prompt": "B"}}
    }
    assert payload["questions"]["first"]["instructions"] == {
        **object_instructions,
        "item": "state.items['first']",
    }
    assert payload["questions"]["first"]["criteria"] == criteria
    assert payload["questions"]["second"]["instructions"] == {
        "item": "state.items['second']",
        "question": array_instructions,
    }
    assert object_instructions == {"question": "Is it clear?", "focus": ["intent"]}


def test_jev_rejects_empty_instructions_and_conflicting_batch_item():
    gateway = HttpGateway(QueueTransport([]), config=GatewayConfig(max_retries=0))
    for value in (None, " ", {}, []):
        with pytest.raises(ValueError, match="instructions are required"):
            gateway.decide({"query": value, "instructions": "fallback", "state": "A"})

    with pytest.raises(ValueError, match="item field conflicts with state"):
        gateway.decide_batch(
            [
                {"key": "first", "query": {"item": "other"}, "state": "A"},
                {"key": "second", "query": "Is it B?", "state": "B"},
            ]
        )


def test_decide_batch_accepts_matching_item_state_path():
    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "model": JEV_MODEL,
                    "answers": {
                        "first": {"type": "noul", "noul": 0.9},
                        "second": {"type": "noul", "noul": 0.1},
                    },
                },
            )
        ]
    )
    gateway = HttpGateway(transport, config=GatewayConfig(max_retries=0))
    instructions = {"item": "state.items['first']", "question": "Is it clear?"}

    gateway.decide_batch(
        [
            {"key": "first", "query": instructions, "state": "A"},
            {"key": "second", "query": "Is it B?", "state": "B"},
        ]
    )

    assert transport.requests[0]["json"]["questions"]["first"]["instructions"] == (
        instructions
    )


def test_jev_response_without_model_snapshot_is_rejected():
    answer = {"type": "noul", "noul": 0.5}
    gateway = HttpGateway(
        QueueTransport([Response(200, {"answers": {"decision": answer}})]),
        config=GatewayConfig(max_retries=0),
    )
    with pytest.raises(ProviderError, match="missing model snapshot"):
        gateway.decide({"state": "prompt", "instructions": "judge"})
    assert len(gateway.decision_log) == 1
    assert gateway.decision_log[0]["answer"] == answer
    assert gateway.decision_log[0]["answered_by"] is None


def test_complete_decide_batch_without_model_snapshot_is_still_rejected():
    first = {"type": "noul", "noul": 0.9}
    second = {"type": "noul", "noul": 0.1}
    gateway = HttpGateway(
        QueueTransport(
            [
                Response(
                    200,
                    {
                        "answers": {
                            "first": first,
                            "second": second,
                        }
                    },
                )
            ]
        ),
        config=GatewayConfig(max_retries=0),
    )

    with pytest.raises(ProviderError, match="missing model snapshot"):
        gateway.decide_batch(
            [
                {"key": "first", "type": "noul", "query": "q1", "state": {}},
                {"key": "second", "type": "noul", "query": "q2", "state": {}},
            ]
        )

    assert [entry["question"]["key"] for entry in gateway.decision_log] == [
        "first",
        "second",
    ]
    assert [entry["answer"] for entry in gateway.decision_log] == [first, second]
    assert all(entry["answered_by"] is None for entry in gateway.decision_log)


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


def test_gateway_operation_deadline_bounds_each_transport_attempt():
    now = [0.0]

    class SlowFailureTransport:
        requests = []

        def request(self, url, **kwargs):
            self.requests.append(kwargs)
            now[0] += 0.6
            raise TimeoutError("slow transport")

    transport = SlowFailureTransport()
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(timeout=10.0, operation_timeout_s=0.5, max_retries=3),
        monotonic=lambda: now[0],
    )

    with pytest.raises(ProviderError, match="failed") as error:
        gateway.chat("model", "prompt")

    assert error.value.kind == "timeout"
    assert len(transport.requests) == 1
    assert transport.requests[0]["timeout"] == pytest.approx(0.5)


def test_gateway_observer_records_success_test_substage_boundaries():
    events = []
    gateway = HttpGateway(QueueTransport([Response(200, {"ok": True})]))

    with gateway.operation_context(observer=events.append):
        with gateway.operation_context(
            substage="writing_tests.generate", run_id="run-trace"
        ):
            gateway.chat("model", "private prompt", run_id="run-trace")

    assert events[0] == {
        "event": "start",
        "operation": "writing_tests.generate",
        "run_id": "run-trace",
    }
    assert events[1]["operation"] == "writing_tests.generate.chat"
    assert events[1]["event"] == "start"
    assert events[2]["operation"] == "route_model"
    assert events[2]["event"] == "start"
    assert events[3]["operation"] == "route_model"
    assert events[3]["event"] == "end"
    assert events[4]["event"] == "end"
    assert events[5]["event"] == "end"
    assert all("private prompt" not in repr(event) for event in events)


def test_model_catalog_routing_is_inside_gateway_deadline_and_trace():
    catalog_entered = threading.Event()
    release_catalog = threading.Event()
    finished = threading.Event()
    result = []
    events = []

    class BlockingCatalog:
        def fetch(self, *, force=False):
            del force
            catalog_entered.set()
            release_catalog.wait(2)
            return CatalogSnapshot(go=(ModelInfo(id="model", provider="go"),))

    class RecordingTransport:
        requests = []

        def request(self, url, **kwargs):
            self.requests.append({"url": url, **kwargs})
            return Response(200, {"ok": True})

    transport = RecordingTransport()
    gateway = HttpGateway(
        transport,
        catalog=BlockingCatalog(),
        config=GatewayConfig(operation_timeout_s=0.05, max_retries=0),
    )

    def request():
        try:
            with gateway.operation_context(observer=events.append):
                gateway.chat("model", "prompt")
        except BaseException as exc:
            result.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert catalog_entered.wait(1)
        assert finished.wait(0.5), "catalog routing exceeded its caller deadline"
    finally:
        release_catalog.set()
        worker.join(1)

    assert len(result) == 1
    assert isinstance(result[0], ProviderError)
    assert result[0].kind == "timeout"
    assert transport.requests == []
    assert any(
        event.get("operation") == "route_model" and event.get("event") == "start"
        for event in events
    )


def test_cancellation_during_catalog_routing_stays_pending_then_cancels():
    from prompt_enhancer.failures import RunCancelled

    catalog_entered = threading.Event()
    release_catalog = threading.Event()
    cancelled = threading.Event()
    finished = threading.Event()
    result = []
    events = []

    class BlockingCatalog:
        def fetch(self, *, force=False):
            del force
            catalog_entered.set()
            release_catalog.wait(2)
            return CatalogSnapshot(go=(ModelInfo(id="model", provider="go"),))

    gateway = HttpGateway(
        QueueTransport([Response(200, {})]),
        catalog=BlockingCatalog(),
        config=GatewayConfig(operation_timeout_s=0.05, max_retries=0),
    )

    def request():
        try:
            with gateway.operation_context(
                cancel_check=cancelled.is_set, observer=events.append
            ):
                gateway.chat("model", "prompt", run_id="catalog-cancel")
        except BaseException as exc:
            result.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert catalog_entered.wait(1)
        cancelled.set()
        assert finished.wait(0.5), "catalog cancellation was not deadline bounded"
    finally:
        release_catalog.set()
        worker.join(1)

    assert len(result) == 1
    assert isinstance(result[0], RunCancelled)
    assert any(event.get("event") == "cancel_pending" for event in events)


@pytest.mark.parametrize("operation", ["decide", "decide_batch"])
def test_decision_model_catalog_routing_is_inside_gateway_deadline_and_trace(
    operation,
):
    catalog_entered = threading.Event()
    release_catalog = threading.Event()
    finished = threading.Event()
    result = []
    events = []

    class BlockingCatalog:
        def fetch(self, *, force=False):
            del force
            catalog_entered.set()
            release_catalog.wait(2)
            return CatalogSnapshot(go=(ModelInfo(id="custom-jev", provider="go"),))

    transport = QueueTransport([Response(200, {"model": "custom-jev", "answers": {}})])
    gateway = HttpGateway(
        transport,
        catalog=BlockingCatalog(),
        config=GatewayConfig(
            jev_model="custom-jev", operation_timeout_s=0.05, max_retries=0
        ),
    )

    def request():
        try:
            with gateway.operation_context(observer=events.append):
                if operation == "decide":
                    gateway.decide({"state": "prompt", "instructions": "judge"})
                else:
                    gateway.decide_batch(
                        [{"key": "k", "type": "noul", "query": "q", "state": "prompt"}]
                    )
        except BaseException as exc:
            result.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert catalog_entered.wait(1)
        assert finished.wait(0.5), "decision catalog routing exceeded caller deadline"
    finally:
        release_catalog.set()
        worker.join(1)

    assert len(result) == 1
    assert isinstance(result[0], ProviderError)
    assert result[0].kind == "timeout"
    assert transport.requests == []
    assert any(
        event.get("operation") == "route_model" and event.get("event") == "start"
        for event in events
    )


@pytest.mark.parametrize("operation", ["decide", "decide_batch"])
def test_default_jev_pricing_catalog_lookup_obeys_operation_deadline(operation):
    catalog_entered = threading.Event()
    release_catalog = threading.Event()
    finished = threading.Event()
    result = []
    events = []

    class BlockingCatalog:
        def fetch(self, *, force=False):
            del force
            catalog_entered.set()
            release_catalog.wait(5)
            return CatalogSnapshot(go=(), openrouter=())

    transport = QueueTransport(
        [Response(200, {"model": JEV_MODEL, "answers": {"k": True}})]
    )
    gateway = HttpGateway(
        transport,
        catalog=BlockingCatalog(),
        config=GatewayConfig(operation_timeout_s=0.05, max_retries=0),
    )

    def request():
        try:
            with gateway.operation_context(observer=events.append):
                payload = {"key": "k", "type": "noul", "query": "q", "state": "seed"}
                if operation == "decide":
                    gateway.decide(payload, run_id="pricing-deadline")
                else:
                    gateway.decide_batch([payload], run_id="pricing-deadline")
        except BaseException as exc:
            result.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert catalog_entered.wait(5)
        assert finished.wait(1), "pricing catalog lookup escaped the Gateway deadline"
    finally:
        release_catalog.set()
        worker.join(5)
    assert not worker.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], ProviderError)
    assert result[0].kind == "timeout"
    assert transport.requests == []
    assert any(
        event.get("operation") == "route_model"
        and event.get("event") == "error"
        and event.get("error_kind") == "deadline_exceeded"
        for event in events
    )


def test_stalled_transport_returns_deadline_failure_and_aborts_when_supported():
    started = threading.Event()
    release = threading.Event()
    late_response_done = threading.Event()

    class StalledTransport:
        supports_request_id = True
        aborted = False

        def request(self, _url, **_kwargs):
            started.set()
            release.wait()
            late_response_done.set()
            return Response(200, {"usage": {"cost": 123.0}, "late": True})

        def abort_request(self, _request_id):
            self.aborted = True
            release.set()

    transport = StalledTransport()
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(timeout=10.0, operation_timeout_s=0.03, max_retries=0),
    )
    began = time.monotonic()

    with pytest.raises(ProviderError) as error:
        gateway.chat("model", "prompt")

    assert started.is_set()
    assert error.value.kind == "timeout"
    assert time.monotonic() - began < 0.5
    assert transport.aborted is True
    assert release.is_set()
    assert late_response_done.wait(0.5)
    assert gateway.usage_report()["total"] == 0


def test_http_transport_closes_a_stalled_response_body_at_deadline():
    release = threading.Event()

    class Socket:
        closed = False

        def settimeout(self, _timeout):
            pass

        def close(self):
            self.closed = True
            release.set()

    sock = Socket()

    class Raw:
        _sock = sock

    class FilePointer:
        raw = Raw()

    class Response:
        status = 200
        headers = {}
        fp = FilePointer()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read1(self, _size):
            release.wait()
            return b"{}"

    gateway = HttpGateway(
        HttpTransport(opener=lambda *_args, **_kwargs: Response()),
        config=GatewayConfig(timeout=5.0, operation_timeout_s=0.03, max_retries=0),
    )
    began = time.monotonic()

    with pytest.raises(ProviderError) as error:
        gateway.chat("model", "prompt")

    assert error.value.kind == "timeout"
    assert time.monotonic() - began < 0.5
    assert sock.closed is True


def test_unresponsive_transport_workers_are_capped_across_gateway_calls():
    release = threading.Event()
    all_started = threading.Event()
    started_count = [0]
    started_lock = threading.Lock()
    operation_time = [0.0]
    transport_workers = []

    class NeverCompletingTransport:
        def request(self, _url, **_kwargs):
            with started_lock:
                transport_workers.append(threading.current_thread())
                started_count[0] += 1
                if started_count[0] == 8:
                    all_started.set()
            release.wait()
            return Response(200, {})

    gateway = HttpGateway(
        NeverCompletingTransport(),
        config=GatewayConfig(operation_timeout_s=0.03, max_retries=0),
        monotonic=lambda: operation_time[0],
    )
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        futures = [pool.submit(gateway.chat, "model", "prompt") for _ in range(8)]
        assert all_started.wait(5), "transport workers did not all start"
        # Hold logical time fixed until all eight workers are actually blocked
        # in transport, then expire each operation deterministically.
        operation_time[0] = 1.0
        for future in futures:
            with pytest.raises(ProviderError) as error:
                future.result(timeout=5)
            assert error.value.kind == "timeout"

        # Use a moving clock for the capacity wait; expired workers retain
        # all eight slots and must not allow a ninth worker to start.
        gateway._monotonic = time.monotonic
        with pytest.raises(ProviderError) as saturated:
            gateway.chat("model", "prompt")
        assert saturated.value.kind == "transport_busy"
    finally:
        release.set()
        pool.shutdown(wait=True)
        with started_lock:
            workers = list(transport_workers)
        join_deadline = time.monotonic() + 5
        for worker in workers:
            worker.join(timeout=max(0, join_deadline - time.monotonic()))
        assert not any(worker.is_alive() for worker in workers)


def test_gateway_operation_deadline_is_shared_by_retries_and_retry_delay():
    now = [0.0]

    class TimedTransport:
        requests = []

        def request(self, url, **kwargs):
            self.requests.append(kwargs)
            now[0] += 0.4
            return {"status_code": 503, "headers": {}}

    def sleep(seconds):
        now[0] += seconds

    transport = TimedTransport()
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(
            timeout=4.0,
            operation_timeout_s=1.0,
            max_retries=3,
            backoff=0.2,
        ),
        sleep=sleep,
        monotonic=lambda: now[0],
    )

    with pytest.raises(ProviderError) as error:
        gateway.chat("model", "prompt")

    assert error.value.kind == "timeout"
    assert len(transport.requests) == 2
    assert [item["timeout"] for item in transport.requests] == pytest.approx([1.0, 0.4])


def test_gateway_cancellation_before_request_sends_no_transport_call():
    from prompt_enhancer.failures import RunCancelled

    transport = QueueTransport([Response(200, {"ok": True})])
    gateway = HttpGateway(transport)

    with gateway.operation_context(cancel_check=lambda: True):
        with pytest.raises(RunCancelled):
            gateway.chat("model", "prompt", run_id="run-2")

    assert transport.requests == []


def test_gateway_cancellation_after_active_response_discards_response():
    from prompt_enhancer.failures import RunCancelled

    cancelled = [False]

    class CancelOnResponse:
        requests = []

        def request(self, url, **kwargs):
            self.requests.append(kwargs)
            cancelled[0] = True
            return Response(200, {"ok": True})

    transport = CancelOnResponse()
    gateway = HttpGateway(transport)

    with gateway.operation_context(cancel_check=lambda: cancelled[0]):
        with pytest.raises(RunCancelled):
            gateway.chat("model", "prompt", run_id="run-3")

    assert len(transport.requests) == 1


def test_cancel_during_stalled_transport_stays_pending_then_aborts_as_cancelled():
    from prompt_enhancer.failures import RunCancelled

    cancelled = [False]
    release = threading.Event()
    events = []

    class CancellableTransport:
        supports_request_id = True
        aborted = False

        def request(self, _url, **_kwargs):
            cancelled[0] = True
            release.wait()
            return Response(200, {})

        def abort_request(self, _request_id):
            self.aborted = True
            release.set()

    transport = CancellableTransport()
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(operation_timeout_s=0.03, max_retries=0),
    )

    with gateway.operation_context(
        cancel_check=lambda: cancelled[0], observer=events.append
    ):
        with pytest.raises(RunCancelled):
            gateway.chat("model", "prompt", run_id="run-cancel")

    assert transport.aborted is True
    assert any(event["event"] == "cancel_pending" for event in events)
    assert release.is_set()


@pytest.mark.parametrize("max_retries", [0, 2])
def test_cancellation_aborts_active_transport_before_deadline_without_retry(
    max_retries,
):
    from prompt_enhancer.failures import RunCancelled

    cancelled = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    aborted = threading.Event()
    request_count = 0
    events = []
    results = []

    class AbortRaisesTransport:
        supports_request_id = True

        def request(self, _url, **_kwargs):
            nonlocal request_count
            request_count += 1
            entered.set()
            assert release.wait(5)
            raise TimeoutError("socket closed by cancellation")

        def abort_request(self, _request_id):
            aborted.set()
            release.set()

    gateway = HttpGateway(
        AbortRaisesTransport(),
        config=GatewayConfig(operation_timeout_s=30, max_retries=max_retries),
    )

    def run_request():
        try:
            gateway.chat("model", "prompt", run_id="run-cancel-immediate")
        except BaseException as exc:  # capture worker outcome for the test thread
            results.append(exc)

    with gateway.operation_context(
        cancel_check=cancelled.is_set, observer=events.append
    ):
        request_context = contextvars.copy_context()
        request_thread = threading.Thread(
            target=lambda: request_context.run(run_request)
        )
        request_thread.start()
        try:
            assert entered.wait(2)
            cancelled.set()
            abort_observed = aborted.wait(2)
        finally:
            # Release the test transport even when the regression fails.
            release.set()
            request_thread.join(2)

    assert not request_thread.is_alive()
    assert abort_observed
    assert request_count == 1
    assert len(results) == 1 and isinstance(results[0], RunCancelled)
    assert any(event["event"] == "cancel_abort_requested" for event in events)
    assert not any(event.get("error_kind") == "deadline_exceeded" for event in events)


def test_retry_after_cannot_extend_gateway_operation_deadline():
    now = [0.0]
    delays = []
    transport = QueueTransport([{"status_code": 429, "headers": {"Retry-After": "10"}}])

    def sleep(seconds):
        delays.append(seconds)
        now[0] += seconds

    gateway = HttpGateway(
        transport,
        config=GatewayConfig(operation_timeout_s=1.0, max_retries=2),
        sleep=sleep,
        monotonic=lambda: now[0],
    )

    with pytest.raises(ProviderError) as error:
        gateway.chat("model", "prompt")

    assert error.value.kind == "timeout"
    assert len(transport.requests) == 1
    assert delays == []


def test_network_retry_backoff_must_fit_inside_gateway_operation_deadline():
    class NetworkFailureTransport:
        requests = []

        def request(self, url, **kwargs):
            self.requests.append({"url": url, **kwargs})
            raise TimeoutError("temporary network failure")

    transport = NetworkFailureTransport()
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(operation_timeout_s=0.5, max_retries=2, backoff=1.0),
        sleep=lambda _seconds: pytest.fail("backoff must not exceed remaining budget"),
    )

    with pytest.raises(ProviderError) as error:
        gateway.chat("model", "prompt")

    assert error.value.kind == "timeout"
    assert len(transport.requests) == 1


def test_gateway_cancellation_interrupts_retry_wait_and_records_safe_events():
    from prompt_enhancer.failures import RunCancelled

    cancelled = [False]
    events = []
    transport = QueueTransport([{"status_code": 503, "headers": {"Retry-After": "5"}}])

    def sleep(_seconds):
        cancelled[0] = True

    gateway = HttpGateway(
        transport,
        config=GatewayConfig(max_retries=2),
        sleep=sleep,
    )
    with gateway.operation_context(
        cancel_check=lambda: cancelled[0], observer=events.append
    ):
        with pytest.raises(RunCancelled):
            gateway.chat("model", "private prompt", run_id="run-1")

    assert len(transport.requests) == 1
    assert events[0]["event"] == "start"
    assert any(event["event"] == "cancel_pending" for event in events)
    assert events[-1]["error_kind"] == "cancelled"
    assert "private prompt" not in repr(events)


def test_configured_jev_pin_is_sent_to_decisions_api():
    pin = "typesafe/jev-1.13-20261001"
    transport = QueueTransport(
        [Response(200, {"model": pin, "answers": {"decision": True}})]
    )
    gateway = HttpGateway(transport, config=GatewayConfig(jev_model=pin, max_retries=0))
    assert gateway.decide({"state": "prompt", "instructions": "judge"}) is True
    assert transport.requests[0]["json"]["model"] == pin
    assert gateway.decision_log[0]["answered_by"] == pin


@pytest.mark.parametrize("phase", ["route", "transport"])
def test_gateway_waits_for_transient_worker_capacity_within_deadline(phase):
    acquired = threading.Event()
    semaphore = threading.BoundedSemaphore(1)
    assert semaphore.acquire(blocking=False)

    class ObservedCapacity:
        def acquire(self, *args, **kwargs):
            acquired.set()
            return semaphore.acquire(*args, **kwargs)

        def release(self):
            semaphore.release()

    gateway = HttpGateway(
        QueueTransport([Response(200, {"ok": True})]),
        config=GatewayConfig(operation_timeout_s=2, max_retries=0),
    )
    gateway._transport_workers = ObservedCapacity()
    if phase == "transport":
        gateway._bounded_catalog_route = lambda factory, **kwargs: factory()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gateway.chat, "model", "prompt")
        assert acquired.wait(1)
        semaphore.release()
        assert future.result(timeout=2) == {"ok": True}


def test_cancellation_while_waiting_for_worker_capacity_starts_no_request():
    from prompt_enhancer.failures import RunCancelled

    events = []
    cancelled = threading.Event()
    waiting = threading.Event()
    semaphore = threading.BoundedSemaphore(1)
    assert semaphore.acquire(blocking=False)

    class ObservedCapacity:
        def acquire(self, *args, **kwargs):
            waiting.set()
            return semaphore.acquire(*args, **kwargs)

        def release(self):
            semaphore.release()

    transport = QueueTransport([Response(200, {})])
    gateway = HttpGateway(transport, config=GatewayConfig(operation_timeout_s=2))
    gateway._transport_workers = ObservedCapacity()

    def call():
        with gateway.operation_context(
            cancel_check=cancelled.is_set, observer=events.append
        ):
            return gateway.chat("model", "prompt")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call)
        assert waiting.wait(1)
        cancelled.set()
        with pytest.raises(RunCancelled):
            future.result(timeout=0.5)
    assert transport.requests == []
    routing = [event for event in events if event.get("operation") == "route_model"]
    assert [event["event"] for event in routing] == ["start", "error"]
    assert routing[-1]["error_kind"] == "cancelled"
    semaphore.release()


def test_completed_paid_response_is_accounted_before_cancellation():
    from prompt_enhancer.failures import RunCancelled

    cancelled = threading.Event()

    class PaidResponse:
        def request(self, _url, **_kwargs):
            cancelled.set()
            return Response(
                200,
                {
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "cost": 0.03,
                    },
                    "answer": "paid",
                },
            )

    gateway = HttpGateway(PaidResponse())
    with gateway.operation_context(cancel_check=cancelled.is_set):
        with pytest.raises(RunCancelled):
            gateway.chat("model", "prompt")
    usage = gateway.usage_report()
    assert usage["calls"] == 1
    assert usage["tokens"]["total_tokens"] == 18
    assert usage["total"] == 0.03


def test_http_abort_interrupts_real_socket_body_read():
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    release_server = threading.Event()
    reading = threading.Event()
    finished = threading.Event()
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"{")
            self.wfile.flush()
            release_server.wait(5)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    transport = HttpTransport()
    read_body = transport._read_body

    def observed_read(response, timeout):
        reading.set()
        return read_body(response, timeout)

    transport._read_body = observed_read

    def download():
        try:
            transport.request(
                f"http://127.0.0.1:{server.server_port}/", timeout=4, request_id="body"
            )
        except (OSError, ValueError) as exc:
            errors.append(type(exc).__name__)
        finally:
            finished.set()

    worker = threading.Thread(target=download)
    worker.start()
    try:
        assert reading.wait(2)
        with transport._active_lock:
            response = transport._active_responses["body"]
            sock = response.fp.raw._sock
        transport.abort_request("body")
        assert finished.wait(0.5), "abort left the makefile body read blocked"
    finally:
        if not finished.is_set():
            sock.shutdown(socket.SHUT_RDWR)
        release_server.set()
        worker.join(5)
        server.shutdown()
        server.server_close()
        serving.join(2)
    assert not worker.is_alive()


def test_eight_way_panel_survives_one_lingering_worker_slot():
    release = threading.Event()
    seven_started = threading.Event()
    lock = threading.Lock()
    started = 0

    class PanelTransport:
        def request(self, _url, **_kwargs):
            nonlocal started
            with lock:
                started += 1
                if started == 7:
                    seven_started.set()
            assert release.wait(5)
            return Response(200, {"ok": True})

    gateway = HttpGateway(
        PanelTransport(), config=GatewayConfig(operation_timeout_s=3, max_retries=0)
    )
    assert gateway._transport_workers.acquire(blocking=False)
    with ThreadPoolExecutor(max_workers=8) as panel:
        futures = [panel.submit(gateway.chat, "model", "prompt") for _ in range(8)]
        try:
            assert seven_started.wait(2)
        finally:
            gateway._transport_workers.release()
            release.set()
        assert [future.result(timeout=3) for future in futures] == [{"ok": True}] * 8
    assert gateway.usage_report()["calls"] == 8
