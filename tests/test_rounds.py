"""Behaviour of one Round, exercised through run_round with a scripted gateway."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection

from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.rounds import RoundPlan, run_round

PROMPT = "Summarize the report."
GAPS = {"confirmed_gaps": [{"key": "output_format", "label": "output format"}], "problem_sentences": []}
TESTS = '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}]}'
WEAK = ("weak-a", "weak-b", "weak-c", "weak-d", "weak-e")


def _gateway(
    *,
    weak_passes: Callable[[str, str], bool] = lambda _model, prompt: prompt != PROMPT,
    strong_passes: Callable[[str], bool] = lambda _prompt: True,
    unfaithful: Collection[str] = (),
    blocked_strategies: Collection[str] = (),
    tests: str = TESTS,
) -> ScriptedGateway:
    def chat(model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                return json.dumps({item["name"]: f"Rewrite {item['name']}" for item in state["strategies"]})
            return tests
        prompt = messages[0]["content"]
        passed = strong_passes(prompt) if role == "strong_check" else weak_passes(model, prompt)
        # Weak and strong models answer in the chat-completions shape.
        return {"choices": [{"message": {"content": "pass" if passed else "fail"}}]}

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        state = request.get("state", {})
        if request.get("type") == "choice":
            return {"type": "choice", "choice": "none", "probabilities": {"none": 1.0}, "confidence": 1.0}
        if key.startswith("strategy_recheck:"):
            probability = 0.0 if key.removeprefix("strategy_recheck:") in blocked_strategies else 1.0
        elif key.startswith("grade_"):
            probability = float(state["output"] == "pass")
            if key.endswith("_second"):
                probability = 1.0 - probability
        elif "candidate_prompt" in state:
            probability = 0.0 if state["candidate_prompt"] in unfaithful else 1.0
        else:
            probability = 1.0
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(chat=chat, decision=decide)


def _plan(*, tier="fast", diagnosis=GAPS, prior_failures=()) -> RoundPlan:
    return RoundPlan(
        prompt=PROMPT,
        working_prompt=PROMPT,
        run_id="run",
        tier=tier,
        seed=17,
        diagnosis=diagnosis,
        assumptions=(),
        settings=Settings(weak_models=WEAK),
        faithfulness_threshold=0.8,
        writer_instruction_version=2,
        prior_failures=prior_failures,
    )


def _candidates(outcome) -> dict[str, dict]:
    return {candidate["candidate_id"]: candidate for candidate in outcome.candidates}


def test_round_without_faithful_tests_is_unverified() -> None:
    outcome = run_round(_gateway(tests='{"tests":[]}'), _plan())

    assert outcome.status == "unverified"
    assert outcome.original_kept is True
    assert outcome.ranking is None
    assert outcome.report()["offer_deep"] is True


def test_round_without_confirmed_gaps_keeps_the_prompt_and_declines_deep() -> None:
    outcome = run_round(_gateway(), _plan(diagnosis={"confirmed_gaps": [], "problem_sentences": []}))

    assert outcome.status == "no_change"
    assert "No confirmed gaps" in outcome.summary
    assert outcome.report()["offer_deep"] is False


def test_round_with_no_eligible_strategy_keeps_the_prompt() -> None:
    all_strategies = {
        "add_missing_context", "specify_output_format", "add_done_criteria", "remove_contradictions",
        "add_example", "split_into_steps", "role_play", "repeated_emphasis",
    }
    outcome = run_round(_gateway(blocked_strategies=all_strategies), _plan())

    assert outcome.status == "no_change"
    assert outcome.summary == "No candidate strategy was selected."
    assert outcome.failures == ()


def test_round_picks_a_verified_winner_and_reports_the_losers() -> None:
    outcome = run_round(_gateway(), _plan())

    assert outcome.status == "improved"
    assert outcome.original_kept is False
    assert outcome.final_prompt.startswith("Rewrite ")
    assert outcome.selected_candidate_id is not None
    assert outcome.selected_strategy == _candidates(outcome)[outcome.selected_candidate_id]["strategy"]
    assert {failure.candidate_id for failure in outcome.failures} == set(_candidates(outcome)) - {outcome.selected_candidate_id}
    assert outcome.continue_rounds is False


def test_round_rejects_a_candidate_that_fails_fidelity() -> None:
    outcome = run_round(_gateway(unfaithful={"Rewrite specify_output_format"}), _plan())
    candidate = next(c for c in outcome.candidates if c["strategy"] == "specify_output_format")

    assert candidate["selected"] is False
    assert "candidate failed fidelity checks" in candidate["rejection_reasons"]
    assert outcome.final_prompt != "Rewrite specify_output_format"


def test_round_rejects_a_weak_panel_winner_that_regresses_on_the_strong_model() -> None:
    outcome = run_round(_gateway(strong_passes=lambda prompt: prompt == PROMPT), _plan())

    assert outcome.original_kept is True
    assert outcome.status == "no_change"
    assert all(
        any(reason.startswith("strong_check_regression") for reason in candidate["rejection_reasons"])
        for candidate in outcome.candidates
    )
    assert outcome.continue_rounds is True


def test_round_marks_and_blocks_a_crutch_strategy_that_regresses() -> None:
    crutches = {"add_example", "split_into_steps", "role_play", "repeated_emphasis"}
    outcome = run_round(
        _gateway(strong_passes=lambda prompt: prompt.removeprefix("Rewrite ") not in crutches),
        _plan(tier="deep"),
    )
    strong = outcome.report()["strong_check"]["candidates"]
    crutch_outcomes = [entry for entry in strong if entry["strategy"] in crutches]

    assert crutch_outcomes
    assert all(entry["crutch"] is True and entry["eligible"] is False for entry in crutch_outcomes)
    assert outcome.final_prompt.removeprefix("Rewrite ") not in crutches


def test_round_keeps_the_prompt_when_no_candidate_beats_it() -> None:
    outcome = run_round(_gateway(weak_passes=lambda _model, prompt: prompt == PROMPT), _plan())

    assert outcome.original_kept is True
    assert outcome.status == "no_change"
    assert outcome.selected_strategy is None
    assert outcome.failures and outcome.continue_rounds is True
    assert all("weak pass rates" in failure.summary for failure in outcome.failures)


def test_round_grades_each_weak_model_separately() -> None:
    outcome = run_round(_gateway(weak_passes=lambda model, prompt: prompt != PROMPT and model == "weak-a"), _plan())
    grade = next(iter(_candidates(outcome).values()))["grade"]

    assert grade["per_model"] == {"weak-a": 1.0, "weak-b": 0.0}
    assert grade["worst"] == 0.0
    assert grade["mean"] == 0.5


def test_round_is_reproducible_and_seeds_every_sample_differently() -> None:
    first = run_round(_gateway(), _plan(tier="standard"))
    second = run_round(_gateway(), _plan(tier="standard"))
    outputs = first.report()["per_model"]["panel"]["outputs"]
    rewrite = [output for output in outputs if output["candidate_id"] != "original"]

    assert first.report() == second.report()
    assert outputs[0]["candidate_id"] == "original"
    assert len({output["seed"] for output in rewrite}) == len(rewrite)
    assert {output["output"] for output in outputs} <= {"pass", "fail"}
