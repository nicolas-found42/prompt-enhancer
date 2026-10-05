"""Quality score vector: six judged dimensions with max-gate floors.

Every candidate in a round carries a recorded six-dimension vector in the
run report; any dimension below its floor rejects the candidate outright,
and the existing fidelity checks remain the fidelity dimension's hard gate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.history import RunHistory
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.rounds import RoundPlan, run_round
from prompt_enhancer.score_vector import SCORE_DIMENSIONS, ScoreVector
from prompt_enhancer.store import RunStore

PROMPT = "Summarize the report."
GAPS = {
    "confirmed_gaps": [{"key": "output_format", "label": "output format"}],
    "problem_sentences": [
        {
            "sentence_id": "s0001",
            "sentence": {
                "id": "s0001",
                "text": PROMPT,
                "start": 0,
                "end": len(PROMPT),
            },
        }
    ],
}
TESTS = (
    '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}]}'
)
WEAK = ("weak-a", "weak-b", "weak-c", "weak-d", "weak-e")


def _gateway(
    *,
    weak_passes: Callable[[str, str], bool] = lambda _model, prompt: prompt != PROMPT,
    unfaithful: Collection[str] = (),
    score_for: Callable[[str, str], float] | None = None,
    tests: str = TESTS,
) -> ScriptedGateway:
    """Scripted gateway with per-dimension score control.

    ``score_for`` maps ``(candidate_prompt, dimension)`` to a 0-1 score;
    the default answers 0.99 for every judged dimension.
    """

    def chat(model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                return json.dumps(
                    {
                        item["name"]: "Rewrite " + item["name"]
                        for item in state["strategies"]
                    }
                )
            return tests
        prompt = messages[0]["content"]
        passed = weak_passes(model, prompt)
        return {"choices": [{"message": {"content": "pass" if passed else "fail"}}]}

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        state = request.get("state", {})
        if request.get("type") == "choice":
            if key.startswith("fidelity:sentence:"):
                return {
                    "type": "choice",
                    "choice": "supported_by_original",
                    "probabilities": {
                        "supported_by_original": 0.99,
                        "supported_by_assumption": 0.0,
                        "new_requirement": 0.0,
                        "unknown": 0.01,
                    },
                    "confidence": 0.99,
                }
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        if key.startswith("strategy_recheck:"):
            probability = 1.0
        elif key.startswith("grade_"):
            probability = float(state["output"] == "pass")
        elif key == "fidelity:meaning":
            probability = 0.0 if state["candidate_prompt"] in unfaithful else 1.0
        elif key.startswith("score:"):
            dimension = key.removeprefix("score:")
            candidate = state["candidate_prompt"]
            probability = (
                score_for(candidate, dimension) if score_for is not None else 0.99
            )
        elif "candidate_prompt" in state:
            probability = 0.0 if state["candidate_prompt"] in unfaithful else 1.0
        else:
            probability = 1.0
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(chat=chat, decision=decide)


def _plan(*, settings: Settings | None = None) -> RoundPlan:
    return RoundPlan(
        prompt=PROMPT,
        working_prompt=PROMPT,
        run_id="run",
        tier="fast",
        seed=17,
        diagnosis=GAPS,
        assumptions=(),
        settings=settings or Settings(weak_models=WEAK),
        faithfulness_threshold=0.8,
        writer_instruction_version=2,
        applied_style="clearer",
    )


def _vectors(outcome) -> Mapping[str, dict]:
    assert outcome.ranking is not None
    return {
        item.candidate.candidate_id: item.candidate.metadata["score_vector"]
        for item in outcome.ranking.ranked
    }


def test_every_candidate_carries_a_six_dimension_vector() -> None:
    outcome = run_round(_gateway(), _plan())

    assert outcome.ranking is not None and outcome.ranking.selected is not None
    vectors = _vectors(outcome)
    assert len(vectors) >= 2
    for vector in vectors.values():
        assert list(vector["scores"]) == list(SCORE_DIMENSIONS)
        assert list(vector["floors"]) == list(SCORE_DIMENSIONS)
        assert all(0.0 <= score <= 1.0 for score in vector["scores"].values())
        assert vector["breaches"] == []
        assert vector["passed"] is True


def test_later_round_reads_an_adjusted_settings_floor() -> None:
    settings = Settings()
    settings.score_floor_clarity = 0.72

    outcome = run_round(_gateway(), _plan(settings=settings))

    vectors = _vectors(outcome)
    assert vectors
    assert all(vector["floors"]["clarity"] == 0.72 for vector in vectors.values())


def test_optimizer_run_after_recalibration_uses_persisted_floors(
    tmp_path: Path,
) -> None:
    database = tmp_path / "calibrated-runs.sqlite3"
    store = RunStore(database)
    history = RunHistory(store)
    for index in range(12):
        decision = "accept" if index % 2 == 0 else "reject"
        score = 0.95 if decision == "accept" else 0.45
        history.save_run(
            {
                "run_id": f"calibration-{index}",
                "prompt": PROMPT,
                "result": {"status": "completed", "final_prompt": "Saved result"},
                "feedback": decision,
                "feedback_labels": {
                    "decision": decision,
                    "status": "linked",
                    "candidate_id": f"candidate-{index}",
                    "score_vector": {
                        dimension: score for dimension in SCORE_DIMENSIONS
                    },
                    "weak_dimensions": [],
                },
            }
        )
    optimizer = PromptOptimizer(
        store=store,
        gateway=_gateway(),
        config=Settings(database_path=str(database)),
    )
    client = TestClient(create_app(optimizer=optimizer, store=store))

    recalibration = client.post("/api/quality/floors/recalibrate")
    assert recalibration.status_code == 200
    assert recalibration.json()["adjusted_floors"]["clarity"] == 0.7

    after_restart = PromptOptimizer(
        store=RunStore(database),
        gateway=_gateway(),
        config=Settings(database_path=str(database)),
    )
    result = after_restart.optimize(PROMPT, {"clarification_allowed": False})

    assert result["report"].get("candidates"), result["report"]
    assert all(
        candidate["metadata"]["score_vector"]["floors"]["clarity"] == 0.7
        for candidate in result["report"]["candidates"]
    )


def test_floor_breach_rejects_and_names_the_dimension() -> None:
    def score_for(candidate: str, dimension: str) -> float:
        if "add_done_criteria" in candidate and dimension == "clarity":
            return 0.10
        return 0.99

    outcome = run_round(_gateway(score_for=score_for), _plan())

    assert outcome.ranking is not None
    vectors = _vectors(outcome)
    by_text = {
        item.candidate.text: (item.candidate.candidate_id, item)
        for item in outcome.ranking.ranked
    }
    candidate_id, ranked = by_text["Rewrite add_done_criteria"]
    assert ranked.selected is False
    vector = vectors[candidate_id]
    assert vector["passed"] is False
    assert vector["breaches"] == ["clarity"]
    assert vector["scores"]["specificity"] == 0.99
    reasons = outcome.ranking.rejection_reasons[candidate_id]
    assert any("clarity" in reason for reason in reasons)


def test_fidelity_failure_rejects_regardless_of_other_scores() -> None:
    outcome = run_round(
        _gateway(unfaithful=("Rewrite specify_output_format",)), _plan()
    )

    assert outcome.ranking is not None
    vectors = _vectors(outcome)
    by_text = {
        item.candidate.text: (item.candidate.candidate_id, item)
        for item in outcome.ranking.ranked
    }
    candidate_id, ranked = by_text["Rewrite specify_output_format"]
    assert ranked.selected is False
    assert vectors[candidate_id]["scores"]["fidelity"] == 0.0
    assert vectors[candidate_id]["breaches"] == ["fidelity"]
    reasons = outcome.ranking.rejection_reasons[candidate_id]
    assert any("fidelity" in reason for reason in reasons)
    assert not any(
        reason.startswith("score floor breached: fidelity") for reason in reasons
    )


def test_floors_are_read_from_settings() -> None:
    def score_for(_candidate: str, dimension: str) -> float:
        return 0.10 if dimension == "clarity" else 0.99

    relaxed = Settings(
        weak_models=WEAK,
        score_floor_fidelity=0.8,
        score_floor_style_fit=0.05,
        score_floor_clarity=0.05,
        score_floor_specificity=0.05,
        score_floor_coherence=0.05,
        score_floor_safety=0.05,
    )
    outcome = run_round(_gateway(score_for=score_for), _plan(settings=relaxed))

    assert outcome.ranking is not None and outcome.ranking.selected is not None
    vectors = _vectors(outcome)
    assert vectors[outcome.ranking.selected.candidate_id]["floors"]["clarity"] == 0.05
    assert vectors[outcome.ranking.selected.candidate_id]["passed"] is True


def test_malformed_score_answer_fails_closed() -> None:
    base = _gateway()
    base_decide = base.decision_handler

    def decide(request, **kwargs):
        if str(request.get("key", "")).startswith("score:"):
            return {"type": "noul"}
        return base_decide(request, **kwargs)

    gateway = ScriptedGateway(chat=base.chat_handler, decision=decide)
    outcome = run_round(gateway, _plan())

    assert outcome.ranking is not None
    assert outcome.ranking.selected is None
    for vector in _vectors(outcome).values():
        assert vector["passed"] is False
        assert set(vector["breaches"]) == set(SCORE_DIMENSIONS) - {"fidelity"}


def test_default_floors_are_documented_policy_constants() -> None:
    settings = Settings(weak_models=WEAK)
    floors = settings.score_floors
    assert set(floors) == set(SCORE_DIMENSIONS)
    assert floors["fidelity"] == 0.8
    assert floors["safety"] == 0.8
    assert all(0.0 < floor < 1.0 for floor in floors.values())


def test_score_vector_clamps_and_reports_breaches() -> None:
    vector = ScoreVector.from_probabilities(
        {
            "style_fit": 1.7,
            "clarity": -0.2,
            "specificity": 0.9,
            "coherence": 0.9,
            "safety": 0.9,
        },
        fidelity_passed=True,
        floors={dimension: 0.6 for dimension in SCORE_DIMENSIONS},
    )
    assert vector.scores["style_fit"] == 1.0
    assert vector.scores["clarity"] == 0.0
    assert vector.passed is False
    assert vector.breaches == ("clarity",)
    assert any("clarity" in reason for reason in vector.breach_reasons)
