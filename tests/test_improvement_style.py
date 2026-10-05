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
        choice = "general" if request.get("key") == "task_type" else "none"
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: 1.0},
            "confidence": 1.0,
        }
    return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}


def _client(store=None):
    from prompt_enhancer.gateway import ScriptedGateway

    store = store or RunStore(":memory:")
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=_decide
    )
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


def test_legacy_tier_input_is_ignored_and_deep_workload_runs() -> None:
    client, _, store = _client()
    response = client.post(
        "/api/optimize",
        json={
            "prompt": "Explain recursion to a beginner in one paragraph.",
            # Legacy tier input, top-level and smuggled via options: ignored.
            "tier": "fast",
            "options": {"tier": "fast"},
            "improvement_style": "shorter",
        },
    )
    assert response.status_code == 200
    result = response.json()
    saved = store.get_run(result["run_id"])
    assert saved is not None
    assert saved["tier"] == "deep"
    assert saved["options"]["tier"] == "deep"
    assert saved["options"]["improvement_style"] == "shorter"
    assert result["report"]["status"] in {
        "no_qualified_candidate",
        "improved_unverified",
        "improved",
    }


def test_each_named_style_is_accepted_and_recorded() -> None:
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
        assert record["tier"] == "deep", style


def test_unknown_style_is_rejected_with_a_clear_error() -> None:
    client, _, _ = _client()
    for path in ("/api/optimize", "/api/jobs/optimize"):
        response = client.post(
            path, json={"prompt": "Explain recursion.", "improvement_style": "zany"}
        )
        assert response.status_code == 422, path
        assert "improvement_style must be one of" in str(response.json()["detail"])


def test_deep_run_tops_up_a_short_explicit_weak_panel_from_defaults() -> None:
    client, _, _ = _client()
    picks = ["alpha-weak", "beta-weak", "gamma-weak"]
    response = client.post(
        "/api/optimize",
        json={
            "prompt": "Explain recursion to a beginner in one paragraph.",
            "improvement_style": "auto",
            "model_overrides": {"weak": picks},
        },
    )
    assert response.status_code == 200
    models = response.json()["report"]["models"]
    assert models["weak"][:3] == picks
    assert len(models["weak"]) == 5
    assert len(set(models["weak"])) == 5
    assert set(Settings().weak_models) | set(picks) >= set(models["weak"])
