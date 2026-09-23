from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.config import Settings
from prompt_enhancer.store import RunStore


def test_optimize_endpoint_returns_result_and_lists_local_run() -> None:
    store = RunStore(":memory:")
    client = TestClient(create_app(store=store, settings=Settings()))

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
    assert result["report"]["status"] == "no_change"
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
