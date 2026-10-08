import json

import pytest
from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.config import Settings
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore
from prompt_enhancer.styles import (
    COMMON_STYLES,
    DEFAULT_STYLE,
    IMPROVEMENT_STYLES,
    MORE_STYLES,
    parse_improvement_style,
)


def _decide(request, **_kwargs):
    if request.get("type") == "choice":
        if str(request.get("key", "")).startswith("evaluate:compare:") and str(
            request.get("key", "")
        ).endswith(":verbosity_direction"):
            choice = "same"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        choice = "general" if request.get("key") == "task_type" else "none"
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: 1.0},
            "confidence": 1.0,
        }
    key = str(request.get("key", ""))
    probability = 1.0 if key.startswith(("score:", "evaluate:")) else 0.01
    return {"type": "noul", "probability_true": probability, "confidence": 1.0}


def _echoing_chat(_model, messages, *, role, **_kwargs):
    if role == "writer":
        state = json.loads(messages[1]["content"])
        if "strategies" in state:
            return json.dumps(
                {item["name"]: state["prompt"] for item in state["strategies"]}
            )
        return '{"tests":[]}'
    return "pass"


def _client(store=None):
    from prompt_enhancer.gateway import ScriptedGateway

    store = store or RunStore(":memory:")
    gateway = ScriptedGateway(chat=_echoing_chat, decision=_decide)
    app = create_app(optimizer=PromptOptimizer(store=store, gateway=gateway))
    return TestClient(app), app.state.jobs, store


def test_style_catalog_has_auto_six_common_and_fifteen_more() -> None:
    assert DEFAULT_STYLE == "auto"
    assert len(COMMON_STYLES) == 6
    assert len(MORE_STYLES) == 15
    assert len(IMPROVEMENT_STYLES) == 1 + 6 + 15
    assert len(set(IMPROVEMENT_STYLES)) == len(IMPROVEMENT_STYLES)


def test_parse_accepts_missing_as_auto_and_every_named_style() -> None:
    assert parse_improvement_style(None) == "auto"
    assert parse_improvement_style("") == "auto"
    for style in IMPROVEMENT_STYLES:
        assert parse_improvement_style(style) == style
        assert parse_improvement_style(f" {style.upper()} ") == style


def test_parse_rejects_unknown_style_with_valid_values() -> None:
    try:
        parse_improvement_style("make-it-pop")
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - the assertion below always runs on success
        raise AssertionError("expected ValueError for unknown style")
    assert "improvement_style must be one of" in message
    assert "auto" in message and "clearer" in message


def test_legacy_tier_option_does_not_enter_the_run_contract() -> None:
    store = RunStore(":memory:")
    client, _, _ = _client(store)
    response = client.post(
        "/api/optimize",
        json={
            "prompt": "Explain recursion to a beginner in one paragraph.",
            "tier": "fast",
            "improvement_style": "shorter",
        },
    )

    assert response.status_code == 200
    result = response.json()
    saved = store.get_run(result["run_id"])
    assert saved is not None
    assert saved["options"]["improvement_style"] == "shorter"
    assert "tier" not in saved and "tier" not in saved["options"]
    assert result["report"]["applied_style"] == "shorter"
    assert result["report"]["outcome"] is None
    assert result["report"]["control_state"] == "deadline_reached"
    assert "No changed prompt qualified" in result["report"]["summary"]
    last_round = result["report"]["history"][-1]
    assert last_round["convergence"]["passed"] is False
    assert (
        last_round["evidence"]["evaluation_evidence"]["candidates"]["original"][
            "accept"
        ]["accepted"]
        is True
    )


def test_each_named_style_is_accepted_and_recorded(deterministic_active_clock) -> None:
    deterministic_active_clock.step_per_round_s = 80.0
    for style in IMPROVEMENT_STYLES:
        store = RunStore(":memory:")
        client, jobs, _ = _client(store)
        response = client.post(
            "/api/jobs/optimize",
            json={"prompt": "Explain recursion.", "improvement_style": style},
        )
        assert response.status_code == 202, style
        run_id = response.json()["run_id"]
        jobs.wait(run_id)
        record = store.get_run(run_id)
        assert record is not None, style
        assert record["options"]["improvement_style"] == style, style
        assert "tier" not in record and "tier" not in record["options"], style


def test_unknown_style_is_rejected_with_a_clear_error() -> None:
    client, _, _ = _client()
    for path in ("/api/optimize", "/api/jobs/optimize"):
        response = client.post(
            path, json={"prompt": "Explain recursion.", "improvement_style": "zany"}
        )
        assert response.status_code == 422, path
        assert "improvement_style must be one of" in str(response.json()["detail"])


def test_five_explicit_weak_models_keep_order_and_do_not_change_defaults() -> None:
    from prompt_enhancer.gateway import ScriptedGateway

    picks = ["alpha-weak", "beta-weak", "gamma-weak", "delta-weak", "epsilon-weak"]
    settings = Settings()
    defaults = settings.weak_models
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=ScriptedGateway(chat=_echoing_chat, decision=_decide),
        config=settings,
    )
    result = optimizer.optimize(
        "Explain recursion to a beginner in one paragraph.",
        {
            "improvement_style": "auto",
            "model_overrides": {"weak": picks},
        },
    )

    assert result["report"]["models"]["weak"] == picks
    assert settings.weak_models == defaults


@pytest.mark.parametrize("count", [3, 4])
def test_short_explicit_weak_panel_is_rejected_before_gateway_calls(count: int) -> None:
    from prompt_enhancer.gateway import ScriptedGateway

    calls: list[str] = []
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: calls.append("chat") or '{"tests":[]}',
        decision=lambda *_args, **_kwargs: (
            calls.append("decision")
            or {
                "type": "noul",
                "probability_true": 0.0,
                "confidence": 1.0,
            }
        ),
    )
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway)
    picks = [f"weak-{index}" for index in range(count)]

    with pytest.raises(ValueError, match="weak panel requires 5 distinct models"):
        optimizer.optimize("Explain recursion.", {"model_overrides": {"weak": picks}})
    assert calls == []
