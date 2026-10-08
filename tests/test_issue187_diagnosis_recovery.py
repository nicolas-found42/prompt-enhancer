"""Public optimization control for source-preserving diagnosis recovery."""

import json

from test_issue46_diagnosis_fanout import BatchGateway

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.catalog import JEV_MODEL, ModelInfo, StaticModelCatalog
from prompt_enhancer.jev import batch_decision_payload


def test_single_oversized_question_retains_all_source_windows_and_stays_incomplete():
    prompt = "Write a brief note using this source: " + "お" * 1500
    gateway = BatchGateway()
    gateway.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=3000),)
    )
    activity = []
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        prompt, progress=lambda stage, facts: activity.append((stage, facts))
    )
    assert result["report"]["outcome"] == "failed_operational"
    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert evidence["complete"] is False
    questions = evidence["questions"]
    root = next(q for q in questions if q["key"] == "task_type")
    assert root["required"] is True
    assert root["status"] == "source_windows_only"
    assert root["source_span"] == {
        "start": 0,
        "end": len(prompt),
        "unit": "unicode_codepoints",
    }
    windows = root["windows"]
    assert len(windows) > 1
    assert "".join(w["source"] for w in windows) == prompt
    assert windows[0]["source_span"]["start"] == 0
    assert windows[-1]["source_span"]["end"] == len(prompt)
    assert any(w["raw_answer"] is not None for w in windows)
    assert all(w["scope"] == "partial_original_prompt" for w in windows)
    for batch in gateway.batches:
        _, envelope = batch_decision_payload(batch, model=JEV_MODEL)
        assert len(json.dumps(envelope, ensure_ascii=False).encode()) <= 1976
    assert evidence["provider_requests"] <= 8
    events = [facts for stage, facts in activity if stage == "activity"]
    assert events[0]["kind"] == "repair"
    assert any(
        item["kind"] == "blocked" and "partial checks" in item["summary"]
        for item in events
    )
    assert questions and all(
        q["status"]
        in {
            "source_windows_only",
            "held",
            "completed",
            "unused",
            "invalid_answer",
            "provider_error",
        }
        for q in questions
    )


def test_malformed_required_answer_is_retained_and_cannot_mark_diagnosis_complete():
    class MalformedRoot(BatchGateway):
        def _answer(self, request, **params):
            if request.get("key") == "task_type":
                return {"type": "not-a-decision", "text": "raw malformed answer"}
            return super()._answer(request, **params)

    result = PromptOptimizer(
        gateway=MalformedRoot(), store=RunStore(":memory:")
    ).optimize("Write a brief note.")
    assert result["status"] == "failed"
    assert result["report"]["outcome"] == "failed_operational"
    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert evidence["complete"] is False
    root = next(item for item in evidence["questions"] if item["key"] == "task_type")
    assert root["required"] is True
    assert root["status"] == "invalid_answer"
    assert root["raw_answer"] == {
        "type": "not-a-decision",
        "text": "raw malformed answer",
    }


def test_partial_provider_batch_preserves_received_answer_and_integrity_error():
    from prompt_enhancer.gateway import GatewayConfig, HttpGateway

    raw = {
        "type": "choice",
        "choice": "general",
        "probabilities": {"general": 1.0},
        "confidence": 1.0,
    }

    class PartialTransport:
        def request(self, url, **params):
            assert len(params["json"]["questions"]) > 1
            return {
                "status_code": 200,
                "headers": {},
                "json": {"model": JEV_MODEL, "answers": {"task_type": raw}},
            }

    gateway = HttpGateway(
        PartialTransport(),
        config=GatewayConfig(max_retries=0),
        catalog=StaticModelCatalog(
            (), (ModelInfo(JEV_MODEL, "openrouter", context_window=32000),)
        ),
    )
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        "Write a brief note."
    )
    assert result["report"]["outcome"] == "failed_operational"
    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert evidence["complete"] is False
    assert evidence["provider_requests"] == 1
    root = next(item for item in evidence["questions"] if item["key"] == "task_type")
    assert root["raw_answer"] == raw
    assert root["status"] == "provider_error"
    assert root["attempts"][0]["error"]["kind"] == "invalid_response"


def test_source_window_count_and_request_caps_preserve_held_remainder():
    prompt = "Write a brief note using this source: " + "お" * 18500
    gateway = BatchGateway()
    gateway.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=3000),)
    )
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        prompt
    )
    assert result["report"]["outcome"] == "failed_operational"
    evidence = result["report"]["diagnosis"]["request_evidence"]
    root = next(item for item in evidence["questions"] if item["key"] == "task_type")
    assert len(root["windows"]) == 16
    assert root["windows"][-1]["status"] == "held"
    assert root["windows"][-1]["reason"] == "source_window_count_cap"
    assert "".join(item["source"] for item in root["windows"]) == prompt
    assert evidence["provider_requests"] <= 8


def test_context_subdivision_counts_each_physical_request_and_latency_once():
    from prompt_enhancer.gateway import GatewayConfig, HttpGateway

    requests = []

    class SmallContextTransport:
        def request(self, url, **params):
            payload = params["json"]
            requests.append(payload)
            if len(json.dumps(payload, ensure_ascii=False).encode()) > 2000:
                return {
                    "status_code": 400,
                    "json": {"error": {"code": "max_tokens_exceeded"}},
                    "headers": {},
                }
            answers = {}
            for key, question in payload["questions"].items():
                if question["type"] == "choice":
                    value = "general" if key == "task_type" else "none"
                    answers[key] = {
                        "type": "choice",
                        "choice": value,
                        "probabilities": {value: 1.0},
                        "confidence": 1.0,
                    }
                else:
                    answers[key] = {
                        "type": "noul",
                        "probability_true": 0.01,
                        "confidence": 1.0,
                    }
            return {
                "status_code": 200,
                "json": {"model": JEV_MODEL, "answers": answers},
                "headers": {},
            }

    gateway = HttpGateway(
        SmallContextTransport(),
        config=GatewayConfig(max_retries=0),
        catalog=StaticModelCatalog(
            (), (ModelInfo(JEV_MODEL, "openrouter", context_window=32000),)
        ),
    )
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        "Write a brief note."
    )
    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert result["report"]["outcome"] == "failed_operational"
    assert evidence["complete"] is False
    assert evidence["provider_requests"] == len(requests) <= 8
    assert len(evidence["request_latencies_ms"]) == len(requests)
    assert any(
        item["attempts"] and item["attempts"][0]["error"]["kind"] == "context_length"
        for item in evidence["questions"]
    )


def test_deadline_during_windows_preserves_received_answers_without_global_success():
    from active_clock import TickingClock

    clock = TickingClock()

    class SlowWindows(BatchGateway):
        def decide_batch(self, requests, **params):
            if requests[0].get("state", {}).get("source_window"):
                clock.now += 60
            return super().decide_batch(requests, **params)

    prompt = "Write a brief note using this source: " + "お" * 1500
    gateway = SlowWindows()
    gateway.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=3000),)
    )
    result = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), clock=clock
    ).optimize(prompt)
    assert result["report"]["control_state"] == "deadline_reached"
    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == prompt
    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert evidence["complete"] is False
    root = next(item for item in evidence["questions"] if item["key"] == "task_type")
    assert root["status"] == "source_windows_only"
    assert len(root["windows"]) == 2
    assert all(item["raw_answer"] is not None for item in root["windows"])
