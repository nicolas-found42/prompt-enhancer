"""Run control: cancel, budget pause, and approval resume, tested at the API boundary.

Covers ticket #167: a running optimization can be cancelled (completed
rounds preserved); a time/spend limit pauses the run at a Round boundary
into an awaiting-approval state that resumes only on approval (continue) or
stops permanently (stop); progress snapshots carry elapsed time and
accumulated cost; a paused run survives a backend restart via the RunStore.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.models import Tier
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.run_control import RoundTracker
from prompt_enhancer.store import RunStore

PromptOptimizer = partial(PromptOptimizer, writer_instruction_version=4)

IMPROVED = "What is 2 + 2?"


def _gateway(
    *,
    candidate_text: str = IMPROVED,
    support: str = "supported_by_original",
    meaning_probability: float = 0.99,
    recheck_probability: float = 0.99,
    cost_per_call: float = 0.0,
) -> ScriptedGateway:
    """Scripted gateway for the no-gaps direct-answer prompt.

    With the default support the rewrite passes fidelity and the run
    improves in one round; with ``support="new_requirement"`` every
    candidate is rejected and Deep runs all three rounds.
    """

    class CostlyGateway(ScriptedGateway):
        def chat(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().chat(*args, **kwargs)
            finally:
                self.usage.record(
                    role=str(kwargs.get("role", "writer")),
                    provider="scripted",
                    model="scripted",
                    cost=cost_per_call,
                )

        def decide(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().decide(*args, **kwargs)
            finally:
                self.usage.record(
                    role=str(kwargs.get("role", "judge")),
                    provider="scripted",
                    model="scripted",
                    cost=cost_per_call,
                )

    def chat(_model: str, messages: Any, *, role: str, **_kwargs: Any) -> Any:
        if role == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            return json.dumps(
                {item["name"]: candidate_text for item in state["strategies"]}
            )
        if role == "writer":
            return '{"tests":[]}'
        return "4"

    baseline_score_attempts = 0

    def decide(request: Any, **_kwargs: Any) -> Any:
        nonlocal baseline_score_attempts
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "task_type":
                selected, probability = "general", 1.0
            elif key == "strategy_choice":
                selected, probability = "add_missing_context", 1.0
            elif key.startswith("fidelity:sentence:"):
                selected, probability = (
                    support,
                    (0.99 if support == "supported_by_original" else 1.0),
                )
                return {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": {
                        "supported_by_original": 0.01,
                        "supported_by_assumption": 0.0,
                        "new_requirement": 0.99,
                        "unknown": 0.0,
                    }
                    if selected != "supported_by_original"
                    else {selected: probability},
                    "confidence": 0.99,
                }
            else:
                selected, probability = "none", 1.0
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": {selected: probability},
                "confidence": 1.0,
            }
        if key == "fidelity:meaning":
            probability = meaning_probability
        elif key.startswith("strategy_recheck:"):
            probability = recheck_probability
        elif key.startswith("faithful:"):
            probability = 0.01
        elif key.startswith("score:"):
            state = request["state"]
            if state["candidate_prompt"] == state["original_prompt"]:
                baseline_score_attempts += 1
                # The first baseline is deliberately below floor so control
                # tests reach their round boundary; the next baseline passes
                # and terminates a continued run without an artificial cap.
                probability = 0.99 if baseline_score_attempts > 5 else 0.01
            else:
                probability = 0.01
        elif key.startswith("grade_"):
            probability = float(request["state"]["output"] == "pass")
        else:
            probability = 0.01
        return {
            "type": "noul",
            "probability_true": probability,
            "confidence": 1.0,
        }

    return CostlyGateway(chat=chat, decision=decide)


def _rejected_gateway(**kwargs: Any) -> ScriptedGateway:
    return _gateway(support="new_requirement", meaning_probability=0.01, **kwargs)


def _client(
    gateway: ScriptedGateway, store: RunStore | None = None
) -> tuple[TestClient, Any]:
    optimizer = PromptOptimizer(store=store or RunStore(":memory:"), gateway=gateway)
    app = create_app(optimizer=optimizer)
    return TestClient(app), app.state.jobs


def _start(client: TestClient, prompt: str = "whats 2 plus 2", **limits: Any) -> str:
    response = client.post("/api/jobs/optimize", json={"prompt": prompt, **limits})
    assert response.status_code == 202, response.text
    return str(response.json()["run_id"])


def _paused_run_id(client: TestClient, jobs: Any, **limits: Any) -> str:
    run_id = _start(client, **limits)
    result = jobs.wait(run_id, timeout=30)["result"]
    assert result["status"] == "needs_input", result
    assert result["report"]["status"] == "awaiting_approval", result["report"]
    return run_id


def test_spend_limit_pauses_run_at_round_boundary() -> None:
    client, jobs = _client(_rejected_gateway(cost_per_call=1.0))

    run_id = _paused_run_id(client, jobs, spend_limit_usd=0.0)

    job = client.get(f"/api/jobs/{run_id}").json()
    assert job["state"] == "done"
    result = job["result"]
    pause = result["report"]["pause"]
    assert pause["reason"] == "spend_limit"
    assert pause["spent_usd"] > 0.0
    history = result["report"]["history"]
    assert len(history) == 1
    assert history[0]["status"] == "no_qualified_candidate"

    stored = client.get(f"/api/runs/{run_id}").json()
    assert stored["status"] == "needs_input"
    assert stored["result"]["report"]["status"] == "awaiting_approval"

    # A budget pause is not a clarification pause: answers cannot resume it.
    assert (
        client.post(f"/api/jobs/{run_id}/resume", json={"answers": {}}).status_code
        == 409
    )


def test_time_limit_pause_then_stop_then_continue_rejected() -> None:
    client, jobs = _client(_rejected_gateway())

    run_id = _paused_run_id(client, jobs, time_limit_s=0.0)
    paused = jobs.wait(run_id)["result"]
    assert paused["report"]["pause"]["reason"] == "time_limit"
    assert len(paused["report"]["history"]) == 1

    stopped = client.post(f"/api/runs/{run_id}/stop").json()
    assert stopped["report"]["status"] == "stopped"
    assert len(stopped["report"]["history"]) == 1
    assert stopped["report"]["history"][0]["status"] == "no_qualified_candidate"

    # Stopping is permanent: approval to continue is gone.
    continued = client.post(f"/api/jobs/{run_id}/continue", json={})
    assert continued.status_code == 409

    listed = {run["run_id"]: run for run in client.get("/api/runs").json()}
    assert listed[run_id]["status"] == "failed"


def test_approval_continue_resumes_from_pause_boundary() -> None:
    client, jobs = _client(_rejected_gateway(cost_per_call=1.0))

    run_id = _paused_run_id(client, jobs, spend_limit_usd=0.0)
    paused = client.get(f"/api/jobs/{run_id}").json()["result"]
    paused_spent = float(paused["report"]["pause"]["spent_usd"])

    # Approval continues without the spent limit, so the run must finish
    # instead of pausing again at the next boundary.
    started = client.post(f"/api/jobs/{run_id}/continue", json={})
    assert started.status_code == 202
    finished = jobs.wait(run_id, timeout=30)["result"]

    assert finished["run_id"] == run_id
    assert finished["status"] == "completed"
    assert finished["report"]["status"] == "converged"
    assert len(finished["report"]["history"]) >= 2
    assert finished["report"]["history"][0] == paused["report"]["history"][0]
    assert float(finished["cost"]["total"]) >= paused_spent


def test_paused_run_survives_backend_restart(tmp_path: Any) -> None:
    import time as _time

    database = str(tmp_path / "runs.sqlite3")
    optimizer = PromptOptimizer(
        store=RunStore(database), gateway=_rejected_gateway(cost_per_call=1.0)
    )
    client = TestClient(create_app(optimizer=optimizer))
    run_id = client.post(
        "/api/jobs/optimize",
        json={"prompt": "whats 2 plus 2", "spend_limit_usd": 0.0},
    ).json()["run_id"]
    # Poll the job endpoint until the worker finishes the paused run.
    deadline = _time.time() + 30
    while True:
        job = client.get(f"/api/jobs/{run_id}").json()
        if job["state"] == "done" or _time.time() >= deadline:
            break
        _time.sleep(0.01)
    assert job["result"]["report"]["status"] == "awaiting_approval"

    restarted = TestClient(
        create_app(
            optimizer=PromptOptimizer(
                store=RunStore(database), gateway=_rejected_gateway(cost_per_call=1.0)
            )
        )
    )
    # The restarted backend never saw the job, but the paused run is resumable.
    assert restarted.get(f"/api/jobs/{run_id}").status_code == 404
    assert restarted.post(f"/api/jobs/{run_id}/continue", json={}).status_code == 202

    deadline = _time.time() + 30
    while _time.time() < deadline:
        stored = restarted.get(f"/api/runs/{run_id}").json()
        if stored["result"]["report"]["status"] != "awaiting_approval":
            break
        _time.sleep(0.01)
    finished = restarted.get(f"/api/runs/{run_id}").json()["result"]
    assert finished["report"]["status"] == "converged"
    assert len(finished["report"]["history"]) >= 2


def test_cancel_mid_run_preserves_completed_rounds() -> None:
    entered = threading.Event()
    release = threading.Event()
    seen = {"strategy_choices": 0}
    gateway = _rejected_gateway()

    base_decide = gateway.decision_handler

    def decide(request: Any, **kwargs: Any) -> Any:
        if request.get("key") == "strategy_choice":
            seen["strategy_choices"] += 1
            if seen["strategy_choices"] >= 2:
                entered.set()
                assert release.wait(10)
        assert base_decide is not None
        return base_decide(request, **kwargs)

    gateway.decision_handler = decide
    client, jobs = _client(gateway)

    run_id = _start(client)
    assert entered.wait(10)

    assert client.post(f"/api/jobs/{run_id}/cancel").json()["cancel_requested"] is True
    release.set()
    result = jobs.wait(run_id, timeout=30)["result"]

    assert result["status"] == "failed"
    assert result["report"]["status"] == "cancelled"
    assert result["report"]["failure"]["kind"] == "cancelled"
    history = result["report"]["history"]
    assert len(history) >= 1
    assert all(
        entry["status"] in {"no_qualified_candidate", "converged"} for entry in history
    )

    stored = client.get(f"/api/runs/{run_id}").json()
    assert stored["outcome"] == "cancelled"
    assert stored["result"]["report"]["history"] == history


def test_round_tracker_keeps_best_selected_prompt_across_regression_and_resume() -> (
    None
):
    tracker = RoundTracker()
    prompts = ("first", "best", "terminal regression")
    scores = (0.8, 0.95, 0.9)
    for index, (prompt, score) in enumerate(zip(prompts, scores, strict=True), start=1):
        evidence = {
            "selection_evidence": {
                "selected_candidate": {
                    "candidate_id": f"candidate-{index}",
                    "text": prompt,
                }
            }
        }
        outcome = SimpleNamespace(
            status="completed",
            final_prompt=prompt,
            original_kept=False,
            selected_candidate_id=f"candidate-{index}",
            selected_strategy="specify_output_format",
            failures=(),
            cost={},
            timing={},
            convergence={
                "selected": True,
                "passed": True,
                "scores": {"clarity": score},
            },
            continue_rounds=True,
            evidence=lambda evidence=evidence: evidence,
        )
        tracker.record(
            SimpleNamespace(round_number=index, tier_round=index, tier=Tier.FAST),
            outcome,
        )

    assert tracker.final_prompt == "best"
    resumed = RoundTracker.preload(tracker.history, "terminal regression", False)
    assert resumed.final_prompt == "best"


def test_progress_snapshot_carries_elapsed_time_and_cost() -> None:
    entered = threading.Event()
    release = threading.Event()
    gateway = _rejected_gateway(cost_per_call=1.0)
    base_decide = gateway.decision_handler

    def decide(request: Any, **kwargs: Any) -> Any:
        if request.get("key") == "strategy_choice":
            entered.set()
            assert release.wait(10)
        assert base_decide is not None
        return base_decide(request, **kwargs)

    gateway.decision_handler = decide
    client, jobs = _client(gateway)

    run_id = _start(client)
    assert entered.wait(10)
    snapshot = client.get(f"/api/jobs/{run_id}").json()

    assert snapshot["state"] == "running"
    assert snapshot["elapsed_ms"] >= 0
    assert snapshot["cost_total"] > 0.0

    release.set()
    finished = jobs.wait(run_id, timeout=30)["result"]
    assert finished["status"] == "completed"


def test_invalid_run_limits_rejected() -> None:
    client, _jobs = _client(_rejected_gateway())

    assert (
        client.post(
            "/api/jobs/optimize",
            json={"prompt": "whats 2 plus 2", "spend_limit_usd": -1},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/jobs/optimize", json={"prompt": "whats 2 plus 2", "time_limit_s": -5}
        ).status_code
        == 422
    )


def test_continue_rejects_runs_not_awaiting_approval() -> None:
    client, jobs = _client(_rejected_gateway())

    assert client.post("/api/jobs/unknown/continue", json={}).status_code == 404

    # A finished run is not paused for approval.
    run_id = _start(client)
    jobs.wait(run_id, timeout=30)
    assert client.post(f"/api/jobs/{run_id}/continue", json={}).status_code == 409
    assert client.post(f"/api/runs/{run_id}/stop").status_code == 409
    assert client.post("/api/runs/unknown/stop").status_code == 404
