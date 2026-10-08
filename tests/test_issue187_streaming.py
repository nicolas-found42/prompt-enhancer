"""Public HTTP stream controls, to move into tests after the current commit."""

import json
import threading

import pytest
from active_clock import TickingClock

from prompt_enhancer.catalog import StaticModelCatalog
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    HttpTransport,
    completion_text,
)


def frame(value):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n\n"


class TimedResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, clock, chunks):
        self.clock, self.chunks = clock, iter(chunks)

    def __enter__(self):
        self.clock.now = 0.01
        return self

    def __exit__(self, *args):
        pass

    def read1(self, size):
        at, chunk = next(self.chunks, (0.8, b""))
        self.clock.now = at
        return chunk


def test_stream_profile_times_visible_content_and_retains_raw_frames_and_usage():
    clock = TickingClock()
    metadata = {
        "id": "gen-controlled",
        "model": "old-llama",
        "provider": "Novita",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
    }
    reasoning = {"choices": [{"index": 0, "delta": {"reasoning": "private reasoning"}}]}
    first = {"choices": [{"index": 0, "delta": {"content": "H"}}]}
    last = {
        "choices": [{"index": 0, "delta": {"content": "é"}, "finish_reason": "stop"}]
    }
    usage = {
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 1},
            "cost": 0.001,
        },
    }
    encoded_last = frame(last)
    split = encoded_last.index("é".encode()) + 1
    chunks = [
        (0.05, b": OPENROUTER PROCESSING\n\n"),
        (0.1, frame(metadata)),
        (0.2, frame(reasoning)),
        (0.3, frame(first)),
        (0.35, encoded_last[:split]),
        (0.4, encoded_last[split:]),
        (0.6, frame(usage)),
        (0.7, b"data: [DONE]\n\n"),
    ]
    response = TimedResponse(clock, chunks)
    transport = HttpTransport(
        opener=lambda *args, **kwargs: response, profile_streams=True
    )
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(max_retries=0),
        catalog=StaticModelCatalog(()),
        monotonic=clock,
        profile_requests=True,
    )
    raw = gateway.chat("old-llama", [], role="weak", stream=True)
    assert raw["protocol"] == "raw-chat-sse-1"
    assert b"".join(chunk for _, chunk in chunks) == raw["raw_body"].encode("utf-8")
    assert raw["events"][0] == metadata
    assert raw["events"][1] == reasoning
    assert raw["events"][2] == first
    assert completion_text(raw) == "Hé"
    sidecar = gateway.profiling_report()["requests"][0]
    assert sidecar["headers_ms"] == pytest.approx(10)
    assert sidecar["first_byte_ms"] == pytest.approx(50)
    assert sidecar["ttft_ms"] == pytest.approx(300)
    assert sidecar["visible_generation_interval_ms"] == pytest.approx(100)
    assert sidecar["served_provider"] == "novita"
    assert sidecar["reasoning_tokens"] == 1
    assert sidecar["visible_output_chars"] == 2
    assert sidecar["response_bytes"] == sum(len(chunk) for _, chunk in chunks)
    assert sidecar["output_tokens_per_second"] == pytest.approx(20)
    assert "private reasoning" not in json.dumps(sidecar)
    role_usage = gateway.usage.for_role("weak")[0]
    assert role_usage.input_tokens == 10
    assert role_usage.output_tokens == 3
    assert role_usage.cost == pytest.approx(0.001)


def test_reasoning_only_stream_never_reports_visible_ttft():
    clock = TickingClock()
    chunks = [
        (0.1, frame({"choices": [{"index": 0, "delta": {"reasoning": "thinking"}}]})),
        (0.2, b"data: [DONE]\n\n"),
    ]
    transport = HttpTransport(
        opener=lambda *args, **kwargs: TimedResponse(clock, chunks),
        profile_streams=True,
    )
    gateway = HttpGateway(
        transport,
        catalog=StaticModelCatalog(()),
        monotonic=clock,
        profile_requests=True,
    )
    raw = gateway.chat("old-llama", [], stream=True)
    sidecar = gateway.profiling_report()["requests"][0]
    assert sidecar["ttft_ms"] is None
    assert sidecar["output_tokens_per_second"] is None
    with pytest.raises(ValueError):
        completion_text(raw)


@pytest.mark.parametrize(
    "body",
    [
        frame({"choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        b"data: invalid JSON\n\ndata: [DONE]\n\n",
        frame({"error": {"message": "private service diagnostic"}})
        + b"data: [DONE]\n\n",
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "partial"},
                        "finish_reason": "length",
                    }
                ]
            }
        )
        + b"data: [DONE]\n\n",
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "partial"},
                        "finish_reason": "error",
                    }
                ]
            }
        )
        + b"data: [DONE]\n\n",
        frame({"choices": [{"index": 0, "delta": {"content": "partial"}}]})
        + b"data: 42\n\ndata: [DONE]\n\n",
        frame({"choices": [{"index": 0, "delta": {"content": "partial"}}]})
        + frame({"choices": [{"index": 0, "delta": {"content": ["unread output"]}}]})
        + b"data: [DONE]\n\n",
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "partial"},
                        "finish_reason": {"invalid": "shape"},
                    }
                ]
            }
        )
        + b"data: [DONE]\n\n",
        b'data: {"choices":[{"index":0,"delta":{"content":"bad \xff"}}]}\n\ndata: [DONE]\n\n',
    ],
)
def test_incomplete_malformed_or_failed_stream_is_retained_but_not_a_usable_answer(
    body,
):
    clock = TickingClock()
    transport = HttpTransport(
        opener=lambda *args, **kwargs: TimedResponse(clock, [(0.1, body)]),
        profile_streams=True,
    )
    gateway = HttpGateway(
        transport,
        catalog=StaticModelCatalog(()),
        monotonic=clock,
        profile_requests=True,
    )
    raw = gateway.chat("old-llama", [], stream=True)
    import base64

    assert base64.b64decode(raw["raw_body_base64"]) == body
    with pytest.raises(ValueError):
        completion_text(raw)
    assert "private service diagnostic" not in json.dumps(gateway.profiling_report())


def test_stream_opt_in_enables_streaming_for_chat_requests():
    clock = TickingClock()
    requests = []
    body = (
        frame({"choices": [{"index": 0, "delta": {"content": "visible"}}]})
        + b"data: [DONE]\n\n"
    )

    def opener(request, **kwargs):
        requests.append(json.loads(request.data))
        return TimedResponse(clock, [(0.1, body)])

    gateway = HttpGateway(
        HttpTransport(opener=opener, profile_streams=True),
        catalog=StaticModelCatalog(()),
        monotonic=clock,
        profile_requests=True,
        stream_chat_for_profiling=True,
    )
    assert completion_text(gateway.chat("old-llama", [])) == "visible"
    assert requests[0]["stream"] is True
    assert gateway.profiling_report()["requests"][0]["streaming_requested"] is True


def test_stream_opt_in_requires_profile_and_a_capable_transport_before_calls():
    from test_gateway import QueueTransport

    with pytest.raises(ValueError, match="request profiling"):
        HttpGateway(stream_chat_for_profiling=True)
    transport = QueueTransport([])
    with pytest.raises(ValueError, match="stream transport"):
        HttpGateway(transport, profile_requests=True, stream_chat_for_profiling=True)
    assert transport.requests == []


@pytest.mark.parametrize("served", ["Novita", "Groq"])
def test_streamed_comparison_preserves_and_enforces_the_served_provider_identity(
    served,
):
    from prompt_enhancer.config import Settings
    from prompt_enhancer.runner import ComparisonFailure, run_candidates
    from prompt_enhancer.tuning_profile import WEAK_MODEL, apply_tuning_profile

    clock = TickingClock()
    body = (
        frame(
            {
                "model": WEAK_MODEL,
                "provider": served,
                "id": "gen-control",
                "choices": [
                    {"index": 0, "delta": {"content": "PING"}, "finish_reason": "stop"}
                ],
            }
        )
        + b"data: [DONE]\n\n"
    )
    gateway = HttpGateway(
        HttpTransport(
            opener=lambda *args, **kwargs: TimedResponse(clock, [(0.1, body)]),
            profile_streams=True,
        ),
        catalog=StaticModelCatalog(()),
        profile_requests=True,
        stream_chat_for_profiling=True,
    )
    settings = apply_tuning_profile(Settings(), "llama-tuning-v1")
    if served == "Groq":
        with pytest.raises(ComparisonFailure) as failed:
            run_candidates(
                [], None, gateway, original="Reply with PING.", settings=settings
            )
        assert (
            failed.value.comparison["attempts"][0]["error"]["kind"]
            == "provider_identity_mismatch"
        )
    else:
        panel = run_candidates(
            [], None, gateway, original="Reply with PING.", settings=settings
        )
        assert len(panel.results) == 3
        assert all(item.output == "PING" for item in panel.results)
        assert all(
            item.response_details["served_provider"] == "novita"
            for item in panel.results
        )


def test_conflicting_stream_identity_frames_cannot_hide_a_wrong_served_provider():
    from prompt_enhancer.config import Settings
    from prompt_enhancer.runner import ComparisonFailure, run_candidates
    from prompt_enhancer.tuning_profile import WEAK_MODEL, apply_tuning_profile

    clock = TickingClock()
    body = (
        frame(
            {
                "provider": "Groq",
                "model": WEAK_MODEL,
                "choices": [{"index": 0, "delta": {"content": "PING"}}],
            }
        )
        + frame(
            {
                "provider": "Novita",
                "model": WEAK_MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + b"data: [DONE]\n\n"
    )
    gateway = HttpGateway(
        HttpTransport(
            opener=lambda *args, **kwargs: TimedResponse(clock, [(0.1, body)]),
            profile_streams=True,
        ),
        catalog=StaticModelCatalog(()),
        profile_requests=True,
        stream_chat_for_profiling=True,
    )
    with pytest.raises(ComparisonFailure) as failed:
        run_candidates(
            [],
            None,
            gateway,
            original="Reply with PING.",
            settings=apply_tuning_profile(Settings(), "llama-tuning-v1"),
        )
    assert failed.value.kind == "provider_identity_mismatch"
    assert all(
        item["served_provider"] is None
        for item in gateway.profiling_report()["requests"]
    )


def test_stream_worker_finishing_after_timeout_cannot_add_visible_timing():
    from prompt_enhancer.gateway import ProviderError

    release = threading.Event()
    finished = threading.Event()
    body = (
        frame({"choices": [{"index": 0, "delta": {"content": "late"}}]})
        + b"data: [DONE]\n\n"
    )

    class BlockingResponse:
        status = 200
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            finished.set()

        def close(self):
            pass

        def read1(self, size):
            release.wait(1)
            return body

    gateway = HttpGateway(
        HttpTransport(
            opener=lambda *args, **kwargs: BlockingResponse(), profile_streams=True
        ),
        catalog=StaticModelCatalog(()),
        config=GatewayConfig(max_retries=0, operation_timeout_s=0.03),
        profile_requests=True,
    )
    try:
        with pytest.raises(ProviderError) as failed:
            gateway.chat("old-llama", [], stream=True)
        assert failed.value.kind == "timeout"
        before = gateway.profiling_report()
        assert before["requests"][0]["ttft_ms"] is None
        assert before["requests"][0]["status"] == "failed"
    finally:
        release.set()
    assert finished.wait(1)
    assert gateway.profiling_report() == before


def test_public_optimization_reads_streamed_writer_strong_and_weak_answers():
    from test_always_attempt import _gateway

    from prompt_enhancer import PromptOptimizer, RunStore

    original = "Reply with exactly PING and nothing else."
    rewrite = "Respond with exactly PING and nothing else."
    gateway = _gateway(candidate_text=rewrite, weak_output="PING")
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        text = handler(model, messages, **params)
        event = {
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": "stop"}
            ]
        }
        return {
            "protocol": "raw-chat-sse-1",
            "events": [event],
            "raw_body": (frame(event) + b"data: [DONE]\n\n").decode(),
            "complete": True,
            "errors": [],
        }

    gateway.chat_handler = chat
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        original
    )
    assert result["report"]["outcome"] == "converged"
    assert result["final_prompt"] == rewrite


def test_public_tuning_run_keeps_truncated_stream_output_and_rejects_it():
    from active_clock import advancing_chat
    from test_always_attempt import _gateway

    from prompt_enhancer import PromptOptimizer, RunStore

    original = "Reply with exactly PING and nothing else."
    gateway = _gateway(candidate_text="Respond with exactly PING and nothing else.")
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] != "weak":
            return handler(model, messages, **params)
        event = {
            "model": model,
            "provider": "Novita",
            "choices": [
                {"index": 0, "delta": {"content": "PI"}, "finish_reason": "length"}
            ],
        }
        return {
            "protocol": "raw-chat-sse-1",
            "events": [event],
            "raw_body": (frame(event) + b"data: [DONE]\n\n").decode(),
            "complete": True,
            "errors": [],
        }

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    result = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), clock=clock
    ).optimize(original, {"evaluation_profile": "llama-tuning-v1"})
    assert result["report"]["outcome"] == "failed_operational"
    assert result["final_prompt"] == original
    attempts = result["report"]["failure"]["comparison"]["attempts"]
    assert len(attempts) == 1
    assert all(
        item["error"]["response_details"]["partial_output"] == "PI"
        for item in attempts[0]["requests"]
    )


@pytest.mark.parametrize("newline", [b"\r\n", b"\r"])
def test_stream_framing_handles_bom_comments_multiline_data_and_other_choices(newline):
    clock = TickingClock()
    body = b"\xef\xbb\xbf: keep alive" + newline + newline
    body += b'data: {"choices":' + newline
    body += (
        b'data: [{"index":1,"delta":{"content":"unused"}},{"index":0,"delta":{"content":"visible"}}]}'
        + newline
        + newline
    )
    body += b"data: [DONE]" + newline + newline
    # Split both the UTF-8 BOM and CRLF line endings across transport reads.
    boundary = body.index(newline, body.index(b"data:")) + 1
    chunks = [
        (0.05, body[:1]),
        (0.1, body[1:2]),
        (0.15, body[2:boundary]),
        (0.2, body[boundary:]),
    ]
    gateway = HttpGateway(
        HttpTransport(
            opener=lambda *args, **kwargs: TimedResponse(clock, chunks),
            profile_streams=True,
        ),
        catalog=StaticModelCatalog(()),
        monotonic=clock,
        profile_requests=True,
    )
    raw = gateway.chat("old-llama", [], stream=True)
    assert raw["raw_body"].encode() == body
    assert completion_text(raw) == "visible"
    sidecar = gateway.profiling_report()["requests"][0]
    assert sidecar["first_byte_ms"] == pytest.approx(50)
    assert sidecar["ttft_ms"] == pytest.approx(200)
    assert sidecar["output_tokens_per_second"] is None


def test_visible_frames_in_one_transport_read_share_the_arrival_time():
    clock = TickingClock()
    body = (
        frame({"choices": [{"index": 0, "delta": {"content": "first"}}]})
        + frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "second"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
        + b"data: [DONE]\n\n"
    )

    def advancing_monotonic():
        clock.now += 0.001
        return clock.now

    gateway = HttpGateway(
        HttpTransport(
            opener=lambda *args, **kwargs: TimedResponse(clock, [(0.1, body)]),
            profile_streams=True,
        ),
        catalog=StaticModelCatalog(()),
        monotonic=advancing_monotonic,
        profile_requests=True,
    )
    assert completion_text(gateway.chat("old-llama", [], stream=True)) == "firstsecond"
    record = gateway.profiling_report()["requests"][0]
    assert record["visible_content_frames"] == 2
    assert record["visible_generation_interval_ms"] == 0
    assert record["output_tokens_per_second"] is None


def test_later_accounting_frame_cannot_erase_a_truncated_first_choice():
    clock = TickingClock()
    body = (
        frame(
            {
                "choices": [
                    {
                        "index": 1,
                        "delta": {"content": "unused"},
                        "finish_reason": "stop",
                    },
                    {
                        "index": 0,
                        "delta": {"content": "partial"},
                        "finish_reason": "length",
                    },
                ]
            }
        )
        + frame(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "completion_tokens": 3,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
        + b"data: [DONE]\n\n"
    )
    gateway = HttpGateway(
        HttpTransport(
            opener=lambda *args, **kwargs: TimedResponse(clock, [(0.1, body)]),
            profile_streams=True,
        ),
        catalog=StaticModelCatalog(()),
        monotonic=clock,
        profile_requests=True,
    )
    raw = gateway.chat("old-llama", [], stream=True)
    assert raw["events"][0]["choices"][1]["finish_reason"] == "length"
    with pytest.raises(ValueError, match="usable completion"):
        completion_text(raw)
    assert gateway.profiling_report()["requests"][0]["stream_complete"] is False


def test_public_optimizer_rejects_complete_json_when_the_writer_stream_is_truncated():
    from active_clock import advancing_chat
    from test_always_attempt import _gateway

    from prompt_enhancer import PromptOptimizer, RunStore

    original = "Reply with exactly PING and nothing else."
    gateway = _gateway(
        candidate_text="Respond with exactly PING and nothing else.", weak_output="PING"
    )
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        text = handler(model, messages, **params)
        if params["role"] != "writer":
            return text
        event = {
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": "length"}
            ]
        }
        return {
            "protocol": "raw-chat-sse-1",
            "events": [event],
            "raw_body": (frame(event) + b"data: [DONE]\n\n").decode(),
            "complete": True,
            "errors": [],
        }

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 10)
    result = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] == "failed_operational"
    assert result["final_prompt"] == original
