"""Public Gateway profiling controls for #187; no live inference."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from active_clock import TickingClock
from test_gateway import QueueTransport, Response

from prompt_enhancer.catalog import StaticModelCatalog
from prompt_enhancer.gateway import GatewayConfig, HttpGateway, ProviderError


def test_profile_sidecar_retains_attempts_without_changing_raw_answer_or_usage():
    raw = {
        "id": "gen-test",
        "provider": "Novita",
        "model": "old-llama",
        "choices": [{"message": {"content": "private output"}}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }
    transport = QueueTransport(
        [
            Response(503, {"error": {"message": "private diagnostic"}}),
            Response(200, raw),
        ]
    )
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(max_retries=1),
        catalog=StaticModelCatalog(()),
        profile_requests=True,
    )
    answer = gateway.chat(
        "old-llama",
        [{"role": "user", "content": "private prompt"}],
        role="weak",
        run_id="run-profile",
        seed=7,
        temperature=0.7,
        max_tokens=4096,
        provider={"only": ["novita"], "allow_fallbacks": False},
    )
    assert answer is raw
    report = gateway.profiling_report()
    assert report["enabled"] is True
    requests = report["requests"]
    assert len(requests) == 2
    assert [r["http_status"] for r in requests] == [503, 200]
    assert [r["status"] for r in requests] == ["http_error", "completed"]
    assert requests[1]["requested_model"] == "old-llama"
    assert requests[1]["served_model"] == "old-llama"
    assert requests[1]["requested_provider"] == "novita"
    assert requests[1]["served_provider"] == "novita"
    assert requests[1]["request_bytes"] > 0
    assert requests[1]["max_output_tokens"] == 4096
    assert requests[1]["output_tokens"] == 3
    assert requests[1]["reasoning_tokens"] == 2
    assert requests[1]["ttft_ms"] is None
    assert requests[1]["first_byte_ms"] is None
    assert requests[1]["output_tokens_per_second"] is None
    assert requests[1]["sampling"]["effective"] is None
    text = json.dumps(report)
    assert "private prompt" not in text
    assert "private output" not in text
    assert "private diagnostic" not in text
    assert gateway.usage.for_role("weak")[0].output_tokens == 3


def test_profiling_is_opt_in_and_does_not_invent_missing_response_metadata():
    raw = {"choices": [{"message": {"content": "answer"}}]}
    ordinary = HttpGateway(
        QueueTransport([Response(200, raw)]), catalog=StaticModelCatalog(())
    )
    assert ordinary.chat("old-llama", []) is raw
    assert ordinary.profiling_report()["enabled"] is False
    assert ordinary.profiling_report()["requests"] == []
    profiled = HttpGateway(
        QueueTransport([Response(200, raw)]),
        catalog=StaticModelCatalog(()),
        profile_requests=True,
    )
    assert profiled.chat("old-llama", []) is raw
    record = profiled.profiling_report()["requests"][0]
    assert all(
        record[key] is None
        for key in (
            "served_model",
            "served_provider",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "reported_cost",
            "generation_id",
        )
    )
    assert record["requested_provider"] is None


def test_attempt_duration_measures_the_transport_interval_without_using_it_as_ttft():
    clock = TickingClock()

    class TimedTransport:
        def request(self, url, **params):
            clock.now += 0.25
            return {
                "status_code": 200,
                "json": {
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {"completion_tokens": 10},
                },
            }

    gateway = HttpGateway(
        TimedTransport(),
        monotonic=clock,
        catalog=StaticModelCatalog(()),
        profile_requests=True,
    )
    gateway.chat("old-llama", [], reasoning={"effort": "low", "max_tokens": 200})
    record = gateway.profiling_report()["requests"][0]
    assert record["adapter_attempt_ms"] == 250
    assert record["queue_ms"] == 0
    assert record["transport_ms"] == 250
    assert record["finished_monotonic_s"] - record["dispatched_monotonic_s"] == 0.25
    assert record["ttft_ms"] is None
    assert record["output_tokens_per_second"] is None
    assert record["reasoning"]["requested_effort"] == "low"
    assert record["reasoning"]["requested_allowance_tokens"] == 200
    assert record["reasoning"]["effective"] is None


def test_timeout_profiles_stay_failed_when_the_abandoned_worker_later_completes():
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()

    class DelayedTransport:
        def request(self, url, **params):
            entered.set()
            assert release.wait(2)
            exited.set()
            return {
                "status_code": 200,
                "json": {"model": "late-model", "usage": {"completion_tokens": 100}},
            }

    gateway = HttpGateway(
        DelayedTransport(),
        config=GatewayConfig(operation_timeout_s=0.05, max_retries=0),
        catalog=StaticModelCatalog(()),
        profile_requests=True,
    )
    try:
        with pytest.raises(ProviderError, match="deadline"):
            gateway.chat("old-llama", [])
        assert entered.is_set()
        before = gateway.profiling_report()
        assert len(before["requests"]) == 1
        record = before["requests"][0]
        assert record["status"] == "failed"
        assert record["served_model"] is None
        assert record["output_tokens"] is None
        release.set()
        assert exited.wait(1)
        assert gateway.profiling_report() == before
        before["requests"][0]["status"] = "completed"
        assert gateway.profiling_report()["requests"][0]["status"] == "failed"
    finally:
        release.set()


def test_new_run_resets_sidecars_and_invalid_identity_metadata_remains_unknown():
    raw = {
        "model": "<script>private</script>",
        "provider": "private diagnostic",
        "id": "private-secret",
        "usage": {"completion_tokens": True, "cost": float("inf")},
    }
    gateway = HttpGateway(
        QueueTransport([Response(200, raw)]),
        catalog=StaticModelCatalog(()),
        profile_requests=True,
    )
    gateway.chat("old-llama", [])
    record = gateway.profiling_report()["requests"][0]
    assert record["served_model"] is None
    assert record["served_provider"] is None
    assert record["generation_id"] is None
    assert record["output_tokens"] is None
    assert record["reported_cost"] is None
    gateway.new_run("next-run")
    assert gateway.profiling_report()["requests"] == []


def test_expired_run_cannot_create_a_profiled_model_attempt():
    from prompt_enhancer.active_budget import (
        ActiveBudget,
        ActiveDeadlineExceeded,
        budget_scope,
    )

    clock = TickingClock()
    transport = QueueTransport([])
    gateway = HttpGateway(
        transport,
        monotonic=clock,
        catalog=StaticModelCatalog(()),
        profile_requests=True,
    )
    with budget_scope(ActiveBudget(clock, 0)):
        clock.now = 150
        with pytest.raises(ActiveDeadlineExceeded):
            gateway.chat("old-llama", [])
    assert transport.requests == []
    assert gateway.profiling_report()["requests"] == []


def test_reset_drops_an_earlier_run_attempt_that_finishes_after_the_new_run():
    entered, release = threading.Event(), threading.Event()
    raw = {"choices": [{"message": {"content": "answer"}}]}

    class OverlappingTransport:
        def request(self, url, **params):
            if params["json"]["messages"][0]["content"] == "old":
                entered.set()
                assert release.wait(2)
            return {"status_code": 200, "json": raw}

    gateway = HttpGateway(
        OverlappingTransport(), catalog=StaticModelCatalog(()), profile_requests=True
    )
    gateway.new_run("old-run")
    with ThreadPoolExecutor(max_workers=1) as executor:
        earlier = executor.submit(
            gateway.chat,
            "old-llama",
            [{"role": "user", "content": "old"}],
            run_id="old-run",
        )
        try:
            assert entered.wait(1)
            gateway.new_run("new-run")
            assert (
                gateway.chat(
                    "old-llama", [{"role": "user", "content": "new"}], run_id="new-run"
                )
                is raw
            )
        finally:
            release.set()
        assert earlier.result(timeout=1) is raw
    records = gateway.profiling_report()["requests"]
    assert len(records) == 1
    assert records[0]["run_id"] == "new-run"
