from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.catalog import ModelInfo, StaticModelCatalog
from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def test_optimize_endpoint_returns_result_and_lists_local_run() -> None:
    store = RunStore(":memory:")
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {"type": "choice", "choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0}
        return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}
    gateway = ScriptedGateway(chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide)
    client = TestClient(create_app(optimizer=PromptOptimizer(store=store, gateway=gateway)))

    response = client.post(
        "/api/optimize",
        json={
            "prompt": "Explain recursion to a beginner in one paragraph.",
            "tier": "standard",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["original_kept"] is True
    assert result["report"]["status"] == "unverified"
    assert result["cost"]["total"] == 0.0

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert runs.json()[0]["run_id"] == result["run_id"]


def test_settings_endpoint_never_returns_credentials() -> None:
    settings = Settings(openrouter_api_key="secret", opencode_go_key="secret")
    client = TestClient(create_app(store=RunStore(":memory:"), settings=settings))

    response = client.get("/api/settings")

    assert response.status_code == 200
    assert "secret" not in response.text
    assert "openrouter_api_key" not in response.json()


def test_feedback_preserves_run_options_and_missing_run_returns_404() -> None:
    store = RunStore(":memory:")
    client = TestClient(create_app(store=store, settings=Settings()))
    result = client.post(
        "/api/optimize",
        json={
            "prompt": "Explain recursion to a beginner.",
            "options": {"model_overrides": {"writer": "writer-choice"}},
        },
    ).json()

    saved = client.post(
        f"/api/runs/{result['run_id']}/feedback", json={"decision": "accept"}
    )

    assert saved.status_code == 200
    assert saved.json()["feedback"] == "accept"
    assert store.get_run(result["run_id"])["options"]["model_overrides"] == {
        "writer": "writer-choice"
    }
    missing = client.post("/api/runs/missing/feedback", json={"decision": "accept"})
    assert missing.status_code == 404


def test_configured_app_uses_live_gateway_and_persists_model_defaults(tmp_path) -> None:
    database = tmp_path / "runs.sqlite3"
    settings = Settings(
        database_path=str(database),
        openrouter_api_key="openrouter-secret",
        opencode_go_key="go-secret",
    )
    client = TestClient(create_app(settings=settings))

    updated = client.put("/api/settings", json={"writer_model": "writer-v2"})
    assert updated.status_code == 200
    assert updated.json()["writer_model"] == "writer-v2"
    assert "secret" not in updated.text

    reopened = TestClient(create_app(settings=Settings(database_path=str(database))))
    assert reopened.get("/api/settings").json()["writer_model"] == "writer-v2"


def test_catalog_exposes_gateway_models_without_credentials() -> None:
    gateway = ScriptedGateway(catalog=StaticModelCatalog(
        [ModelInfo("go-one", "go")], [ModelInfo("or-one", "openrouter")]
    ))
    optimizer = PromptOptimizer(gateway=gateway, store=RunStore(":memory:"))
    client = TestClient(create_app(optimizer=optimizer))

    catalog = client.get("/api/catalog")

    assert catalog.status_code == 200
    assert [item["id"] for item in catalog.json()["providers"]["go"]] == ["go-one"]
    assert [item["id"] for item in catalog.json()["providers"]["openrouter"]] == ["or-one"]
    assert catalog.json()["judge"]["id"] == "typesafe/jev-1.13"


def test_resume_and_deep_return_client_errors_for_invalid_run_state() -> None:
    client = TestClient(create_app(store=RunStore(":memory:"), settings=Settings()))

    missing = client.post("/api/runs/missing/resume", json={"answers": {}})
    assert missing.status_code == 404
    missing_deep = client.post("/api/runs/missing/deep")
    assert missing_deep.status_code == 404

    completed = client.post("/api/optimize", json={"prompt": "Explain recursion."}).json()
    repeated = client.post(f"/api/runs/{completed['run_id']}/resume", json={"answers": {}})
    assert repeated.status_code == 409
