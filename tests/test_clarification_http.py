"""HTTP contract tests for clarification resume and skip."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from prompt_enhancer.clarification import (
    ClarificationService,
    GapAssessment,
    InMemoryClarificationRepository,
    build_plan,
)
from prompt_enhancer.clarification_api import register_clarification_routes


class FakeOptimizer:
    def __init__(self) -> None:
        self.service = ClarificationService(InMemoryClarificationRepository())
        self.service.start(
            "http-run",
            "write a report",
            build_plan(
                [
                    GapAssessment(
                        id="format",
                        label="output format",
                        impact="high",
                        present=None,
                        question="Which format?",
                        options=({"value": "json", "label": "JSON", "preselected": True},),
                    )
                ]
            ),
        )

    def resume(self, run_id, answers):
        return self.service.resume(run_id, answers)

    def skip(self, run_id):
        return self.service.skip(run_id)


def test_resume_route_maps_answers_to_same_run() -> None:
    app = FastAPI()
    register_clarification_routes(app, FakeOptimizer())
    response = TestClient(app).post("/api/optimize/resume/http-run", json={"answers": {"format": "json"}})

    assert response.status_code == 200
    assert response.json()["run_id"] == "http-run"
    assert response.json()["status"] == "completed"


def test_skip_route_records_default_assumption() -> None:
    app = FastAPI()
    register_clarification_routes(app, FakeOptimizer())
    response = TestClient(app).post("/api/optimize/skip/http-run")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["assumptions"][0]["source"] == "skipped_clarification"
