"""Always-attempt engine: missing success tests never force a no-op.

Covers ticket #164–#172/165: a simple prompt with no faithful success
tests still gets candidates written and fidelity-gated; a passing rewrite
can converge while remaining unverified, total rejection keeps retrying
until a user budget pauses it, and the tested-path selection floor holds.
"""

import json
from functools import partial

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.gateway import ScriptedGateway

PromptOptimizer = partial(PromptOptimizer, writer_instruction_version=4)

IMPROVED = "What is 2 + 2?"


def _gateway(
    *,
    candidate_text: str = IMPROVED,
    support: str = "supported_by_original",
    meaning_probability: float = 0.99,
    recheck_probability: float = 0.99,
    score_probability: float = 0.99,
    baseline_score_probability: float | None = None,
    weak_output: str = "4",
    with_test: bool = False,
):
    """Scripted gateway for the no-gaps direct-answer prompt."""

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            return json.dumps(
                {item["name"]: candidate_text for item in state["strategies"]}
            )
        if role == "writer" and with_test:
            return (
                '{"tests":[{"question":"Does the answer give the sum?",'
                '"kind":"noul","expected":"yes"}]}'
            )
        if role == "writer":
            return '{"tests":[]}'
        return weak_output

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "task_type":
                selected, probability = "general", 1.0
            elif key == "strategy_choice":
                selected, probability = "add_missing_context", 1.0
            elif key.startswith("evaluate:compare:") and key.endswith(
                ":verbosity_direction"
            ):
                selected, probability = "same", 1.0
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
            probability = 0.99 if with_test else 0.01
        elif key.startswith("grade_"):
            probability = float(request["state"]["output"] == "pass")
        elif key.startswith("score:"):
            state = request["state"]
            probability = (
                baseline_score_probability
                if baseline_score_probability is not None
                and state["candidate_prompt"] == state["original_prompt"]
                else score_probability
            )
        elif key.startswith("evaluate:"):
            probability = 0.99
        else:
            probability = 0.01
        return {
            "type": "noul",
            "probability_true": probability,
            "confidence": 1.0,
        }

    return ScriptedGateway(chat=chat, decision=decide)


def test_simple_prompt_without_tests_converges_with_unverified_evidence() -> None:
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(store=store, gateway=_gateway())

    result = optimizer.optimize("whats 2 plus 2")

    assert result["status"] == "completed"
    assert result["original_kept"] is False
    assert result["final_prompt"] == IMPROVED
    report = result["report"]
    assert report["status"] == "converged"
    assert report["convergence"]["verification"] == "unverified"
    assert report["strong_check"] is None
    assert report["tests"] == []
    evidence = report["selection_evidence"]
    assert evidence["selected_candidate_id"] is not None
    ranked = evidence["ranking"]
    assert any(item["selected"] for item in ranked)
    assert any(item["text"] == IMPROVED for item in ranked if item["selected"])
    record = store.get_run(result["run_id"])
    assert record is not None
    assert record["result"]["report"]["status"] == "converged"


def test_rejected_candidates_and_below_floor_baseline_retry_until_budget_pause() -> (
    None
):
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(
        store=store,
        gateway=_gateway(
            support="new_requirement",
            meaning_probability=0.01,
            baseline_score_probability=0.1,
        ),
    )

    result = optimizer.optimize("whats 2 plus 2", {"time_limit_s": 0})

    assert result["status"] == "needs_input"
    assert result["original_kept"] is True
    assert result["final_prompt"] == "whats 2 plus 2"
    report = result["report"]["history"][0]["evidence"]
    assert result["report"]["history"][0]["status"] == "no_qualified_candidate"
    evidence = report["selection_evidence"]
    assert evidence["selected_candidate_id"] is None
    ranked = evidence["ranking"]
    assert len(ranked) > 0
    assert not any(item["selected"] for item in ranked)


def test_zero_pass_rates_with_tests_still_keep_the_original() -> None:
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(
        store=store, gateway=_gateway(with_test=True, weak_output="fail")
    )

    result = optimizer.optimize("whats 2 plus 2", {"time_limit_s": 0})

    assert result["status"] == "needs_input"
    assert result["original_kept"] is True
    round_report = result["report"]["history"][0]
    assert round_report["status"] == "improvement_not_verified"
    assert round_report["evidence"]["tests"] != []


def test_clarified_baseline_without_changed_candidates_converges_after_acceptance() -> (
    None
):
    from prompt_enhancer.config import Settings
    from prompt_enhancer.rounds import RoundPlan, run_round

    working = "whats 2 plus 2. Goal: summarize."
    plan = RoundPlan(
        prompt="whats 2 plus 2",
        working_prompt=working,
        run_id="clarified-run",
        seed=1,
        diagnosis={"confirmed_gaps": []},
        assumptions=(),
        settings=Settings(),
        faithfulness_threshold=0.8,
        writer_instruction_version=4,
    )
    outcome = run_round(_gateway(recheck_probability=0.01), plan)

    assert outcome.status == "converged"
    assert outcome.original_kept is False
    assert outcome.final_prompt == working
    assert outcome.reported_failure is None
    assert outcome.convergence["passed"] is True
    assert outcome.convergence["selected"] is True
    assert outcome.ranking.selected.candidate_id == "original"
    assert (
        outcome.evaluation_evidence["candidates"]["original"]["accept"]["accepted"]
        is True
    )
