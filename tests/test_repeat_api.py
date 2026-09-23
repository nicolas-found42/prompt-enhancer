from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app


class MemoryStore:
    def __init__(self, run: dict[str, Any]) -> None:
        self.run = deepcopy(run)
        self.saved: list[dict[str, Any]] = []

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return deepcopy(self.run) if run_id == self.run["run_id"] else None

    def save_run(self, run: dict[str, Any]) -> None:
        self.saved.append(deepcopy(run))
        self.run = deepcopy(run)


class DeepOptimizer:
    def __init__(self) -> None:
        self.started: list[str] = []

    def start_deep_pass(self, run_id: str) -> dict[str, Any]:
        self.started.append(run_id)
        return {
            "run_id": run_id,
            "tier": "deep",
            "final_prompt": "Original prompt",
            "original_kept": True,
            "report": {
                "status": "no_change",
                "history": [
                    {
                        "round_number": 1,
                        "tier_round": 1,
                        "tier": "standard",
                        "max_rounds": 2,
                        "original_kept": True,
                        "candidate_failures": [],
                    },
                    {
                        "round_number": 2,
                        "tier_round": 2,
                        "tier": "standard",
                        "max_rounds": 2,
                        "original_kept": True,
                        "candidate_failures": [],
                    },
                    {
                        "round_number": 3,
                        "tier_round": 1,
                        "tier": "deep",
                        "max_rounds": 3,
                        "original_kept": True,
                        "candidate_failures": [],
                    },
                ],
                "offer_deep": None,
                "escalation": {
                    "status": "completed",
                    "run_id": run_id,
                    "source_tier": "standard",
                    "target_tier": "deep",
                    "started_after_round": 2,
                    "completed_through_round": 3,
                    "final_original_kept": True,
                },
            },
        }


def _store() -> MemoryStore:
    return MemoryStore(
        {
            "run_id": "run-api",
            "original_prompt": "Original prompt",
            "tier": "standard",
            "original_kept": True,
            "report": {
                "status": "no_change",
                "history": [],
                "offer_deep": {
                    "run_id": "run-api",
                    "from_tier": "standard",
                    "to_tier": "deep",
                    "state": "offered",
                },
            },
        }
    )


def test_deep_endpoint_keeps_run_id_and_serializes_round_escalation() -> None:
    store = _store()
    optimizer = DeepOptimizer()
    client = TestClient(create_app(optimizer=optimizer, store=store))

    response = client.post("/api/runs/run-api/deep")

    assert response.status_code == 200
    payload = response.json()
    assert optimizer.started == ["run-api"]
    assert payload["run_id"] == "run-api"
    assert payload["tier"] == "deep"
    assert [round_["tier"] for round_ in payload["report"]["history"]] == [
        "standard",
        "standard",
        "deep",
    ]
    assert payload["report"]["escalation"]["status"] == "completed"
    assert "api_key" not in response.text


def test_deep_endpoint_rejects_provider_credentials() -> None:
    store = _store()
    optimizer = DeepOptimizer()
    client = TestClient(create_app(optimizer=optimizer, store=store))

    response = client.post(
        "/api/runs/run-api/deep",
        json={"openrouter_api_key": "browser-must-not-supply-this"},
    )

    assert response.status_code == 422
    assert optimizer.started == []
