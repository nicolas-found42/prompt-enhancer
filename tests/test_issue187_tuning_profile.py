"""Public-flow red control for the opt-in #187 tuning profile."""

import pytest
from active_clock import TickingClock, advancing_chat
from test_always_attempt import _gateway

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.gateway import ProviderError


def test_failed_comparison_retains_completed_and_failed_samples_from_each_service():
    gateway = _gateway(candidate_text="Respond with exactly PING and nothing else.")
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] == "weak":
            # The same seed fails on both services; its two parallel siblings succeed.
            from prompt_enhancer.runner import _stable_seed

            if params["seed"] == _stable_seed(0, "matched_comparison", model, 1):
                raise ProviderError("openrouter", model, 503, role="weak")
            return "PING"
        return handler(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(
        "Reply with exactly PING and nothing else.",
        {"evaluation_profile": "llama-tuning-v1", "seed": 0},
    )

    assert result["report"]["outcome"] == "failed_operational"
    comparison = result["report"]["failure"]["comparison"]
    assert [item["requested_provider"] for item in comparison["attempts"]] == [
        "novita",
        "groq",
    ]
    for attempt in comparison["attempts"]:
        assert len(attempt["outputs"]) == 2
        assert [item["sample"] for item in attempt["requests"]] == [0, 1, 2]
        assert [item["status"] for item in attempt["requests"]] == [
            "completed",
            "failed",
            "completed",
        ]
        assert attempt["requests"][1]["error"]["http_status"] == 503


def test_tuning_request_uses_openrouter_even_if_go_catalog_lists_llama():
    from test_gateway import QueueTransport, Response

    from prompt_enhancer.catalog import StaticModelCatalog
    from prompt_enhancer.config import Settings
    from prompt_enhancer.gateway import GatewayConfig, HttpGateway
    from prompt_enhancer.runner import run_candidates
    from prompt_enhancer.tuning_profile import WEAK_MODEL, apply_tuning_profile

    raw = {"choices": [{"message": {"content": "PING"}, "finish_reason": "stop"}]}
    transport = QueueTransport([Response(200, raw) for _ in range(3)])
    gateway = HttpGateway(
        transport, config=GatewayConfig(), catalog=StaticModelCatalog([WEAK_MODEL])
    )
    panel = run_candidates(
        [],
        None,
        gateway,
        original="Reply with PING.",
        settings=apply_tuning_profile(Settings(), "llama-tuning-v1"),
    )
    assert len(panel.results) == 3
    assert all(
        item["url"].startswith("https://openrouter.ai/") for item in transport.requests
    )
    assert all("route_provider" not in item["json"] for item in transport.requests)
    assert all(
        item["json"]["provider"]["only"] == ["novita"] for item in transport.requests
    )


def test_truncated_profile_answers_retain_visible_partial_text_without_qualifying():
    gateway = _gateway(candidate_text="Respond with exactly PING and nothing else.")
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] == "weak":
            return {
                "choices": [{"message": {"content": "PI"}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 4096},
            }
        return handler(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(
        "Reply with exactly PING and nothing else.",
        {"evaluation_profile": "llama-tuning-v1"},
    )
    assert result["report"]["outcome"] == "failed_operational"
    attempts = result["report"]["failure"]["comparison"]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["outputs"] == []
    assert all(
        item["error"]["response_details"]["partial_output"] == "PI"
        for item in attempts[0]["requests"]
    )


def test_profile_configuration_pins_jev_and_rejects_substitution_before_calls():
    from prompt_enhancer.catalog import JEV_MODEL
    from prompt_enhancer.config import Settings

    gateway = _gateway()
    optimizer = PromptOptimizer(
        config=Settings(judge_model="incorrect-judge"),
        store=RunStore(":memory:"),
        gateway=gateway,
    )
    configuration = optimizer.run_configuration(
        {"evaluation_profile": "llama-tuning-v1"}
    )
    assert configuration["models"]["judge"] == JEV_MODEL
    assert configuration["weak_samples"] == 3
    assert configuration["provider_policy"]["status"] == "provisional"
    with pytest.raises(ValueError, match="cannot be overridden"):
        optimizer.optimize(
            "Reply with PING.",
            {
                "evaluation_profile": "llama-tuning-v1",
                "model_overrides": {"weak": ["other"]},
            },
        )
    assert gateway.calls == []


def test_pilot_can_request_groq_first_and_rejects_other_services_before_calls():
    calls = []
    gateway = _gateway(
        candidate_text="Respond with exactly PING and nothing else.", weak_output="PING"
    )
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] == "weak":
            calls.append(params["provider"]["only"][0])
        return handler(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    )
    result = optimizer.optimize(
        "Reply with exactly PING and nothing else.",
        {"evaluation_profile": "llama-tuning-v1", "evaluation_provider": "groq"},
    )
    assert result["report"]["outcome"] == "converged"
    assert calls and set(calls) == {"groq"}
    assert result["report"]["configuration"]["provider_policy"]["primary"] == "groq"
    before = len(gateway.calls)
    with pytest.raises(ValueError, match="only Novita or Groq"):
        optimizer.optimize(
            "Reply with PING.",
            {
                "evaluation_profile": "llama-tuning-v1",
                "evaluation_provider": "deepinfra",
            },
        )
    assert len(gateway.calls) == before


def test_tuning_profile_uses_qwen_and_three_matched_llama_samples_on_novita():
    calls = []
    gateway = _gateway(
        candidate_text="Respond with exactly PING and nothing else.", weak_output="PING"
    )
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        calls.append({"model": model, "prompt": messages[-1]["content"], **params})
        return handler(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(
        "Reply with exactly PING and nothing else.",
        {"evaluation_profile": "llama-tuning-v1"},
    )
    assert result["report"]["outcome"] == "converged"
    assert result["report"]["models"]["writer"] == "qwen/qwen3.7-flash"
    assert result["report"]["models"]["strong"] == "qwen/qwen3.7-flash"
    assert result["report"]["models"]["weak"] == ["meta-llama/llama-3.1-8b-instruct"]
    weak = [item for item in calls if item["role"] == "weak"]
    assert weak and all(
        item["provider"]
        == {
            "only": ["novita"],
            "order": ["novita"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        for item in weak
    )
    original = [
        item
        for item in weak
        if item["prompt"] == "Reply with exactly PING and nothing else."
    ]
    rewrite = [
        item
        for item in weak
        if item["prompt"] == "Respond with exactly PING and nothing else."
    ]
    assert len(original) == 3
    assert {item["seed"] for item in original} == {item["seed"] for item in rewrite}
    outputs = result["report"]["per_model"]["panel"]["outputs"]
    assert all(item["response_details"]["served_provider"] is None for item in outputs)
    assert all(item["response_details"]["served_model"] is None for item in outputs)
    assert all(item["response_details"]["generation_id"] is None for item in outputs)


@pytest.mark.parametrize("status", [401, 429])
def test_permanent_or_unclassified_quota_errors_do_not_switch_services(status):
    calls = []
    gateway = _gateway(candidate_text="Respond with exactly PING and nothing else.")
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] == "weak":
            calls.append(params["provider"]["only"][0])
            raise ProviderError(
                "openrouter",
                model,
                status,
                role="weak",
                response_details={"provider_error_code": "insufficient_quota"},
            )
        return handler(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(
        "Reply with exactly PING and nothing else.",
        {"evaluation_profile": "llama-tuning-v1"},
    )

    assert result["report"]["outcome"] == "failed_operational"
    assert calls and set(calls) == {"novita"}


def test_service_fallback_reestablishes_the_baseline_and_all_rewrite_samples():
    original_prompt = "Reply with exactly PING and nothing else."
    rewrite_prompt = "Respond with exactly PING and nothing else."
    calls = []
    gateway = _gateway(candidate_text=rewrite_prompt, weak_output="PING")
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] == "weak":
            provider = params["provider"]["only"][0]
            prompt = messages[-1]["content"]
            calls.append(
                {"provider": provider, "prompt": prompt, "seed": params["seed"]}
            )
            if provider == "novita" and prompt == rewrite_prompt:
                raise ProviderError("openrouter", model, 503, role="weak")
        return handler(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(
        original_prompt,
        {"evaluation_profile": "llama-tuning-v1"},
    )

    assert result["report"]["outcome"] == "converged"
    groq = [item for item in calls if item["provider"] == "groq"]
    baseline = [item for item in groq if item["prompt"] == original_prompt]
    rewrite = [item for item in groq if item["prompt"] == rewrite_prompt]
    assert len(baseline) == 3
    assert {item["seed"] for item in baseline} == {item["seed"] for item in rewrite}
    panel = result["report"]["per_model"]["panel"]
    assert panel["comparison"]["requested_provider"] == "groq"
    assert panel["comparison"]["attempts"][0]["status"] == "incomplete"
    assert all(
        item["response_details"]["requested_provider"] == "groq"
        for item in panel["outputs"]
    )


@pytest.mark.parametrize("served_provider", ["Novita", "Groq"])
def test_served_identity_is_retained_and_mismatched_routing_cannot_qualify(
    served_provider,
):
    gateway = _gateway(
        candidate_text="Respond with exactly PING and nothing else.", weak_output="PING"
    )
    handler = gateway.chat_handler

    def chat(model, messages, **params):
        if params["role"] == "weak":
            return {
                "id": "gen-controlled-identity",
                "provider": served_provider,
                "model": model,
                "choices": [{"message": {"content": "PING"}, "finish_reason": "stop"}],
            }
        return handler(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(chat, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(
        "Reply with exactly PING and nothing else.",
        {"evaluation_profile": "llama-tuning-v1"},
    )
    if served_provider == "Groq":
        assert result["report"]["outcome"] == "failed_operational"
        assert result["final_prompt"] == "Reply with exactly PING and nothing else."
    else:
        outputs = result["report"]["per_model"]["panel"]["outputs"]
        assert all(
            item["response_details"]["served_provider"] == "novita" for item in outputs
        )
        assert all(
            item["response_details"]["served_model"]
            == "meta-llama/llama-3.1-8b-instruct"
            for item in outputs
        )
        assert all(
            item["response_details"]["generation_id"] == "gen-controlled-identity"
            for item in outputs
        )
