"""Behaviour of one Round, exercised through run_round with a scripted gateway."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from dataclasses import replace

import pytest

from prompt_enhancer.candidate_evaluation import evaluate_candidate_packages
from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.rounds import RoundPlan, run_round
from prompt_enhancer.selector import RankingCandidate

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
    strong_passes: Callable[[str], bool] = lambda _prompt: True,
    unfaithful: Collection[str] = (),
    blocked_strategies: Collection[str] = (),
    tests: str = TESTS,
    writer_echo: bool = False,
    candidate_text: str | None = None,
    writer_outputs: Mapping[str, str] | None = None,
    rerank_scores: Mapping[str, float] | None = None,
    accept_scores: Mapping[str, float] | None = None,
    audit_scores: Mapping[str, float] | None = None,
    structure_probability: float = 0.01,
    verbosity_choice: str = "same",
) -> ScriptedGateway:
    def chat(model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                if writer_echo:
                    return json.dumps(
                        {item["name"]: state["prompt"] for item in state["strategies"]}
                    )
                if candidate_text is not None:
                    return json.dumps(
                        {item["name"]: candidate_text for item in state["strategies"]}
                    )
                if writer_outputs is not None:
                    return json.dumps(
                        {
                            item["name"]: writer_outputs.get(
                                item["name"], f"Rewrite {item['name']}"
                            )
                            for item in state["strategies"]
                        }
                    )
                return json.dumps(
                    {
                        item["name"]: f"Rewrite {item['name']}"
                        for item in state["strategies"]
                    }
                )
            return tests
        prompt = messages[0]["content"]
        passed = (
            strong_passes(prompt)
            if role == "strong_check"
            else weak_passes(model, prompt)
        )
        # Weak and strong models answer in the chat-completions shape.
        return {"choices": [{"message": {"content": "pass" if passed else "fail"}}]}

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        state = request.get("state", {})
        if request.get("type") == "choice":
            if key.startswith("evaluate:compare:") and key.endswith(
                ":verbosity_direction"
            ):
                choice = verbosity_choice
                return {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": {choice: 1.0},
                    "confidence": 1.0,
                }
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
            probability = (
                0.0
                if key.removeprefix("strategy_recheck:") in blocked_strategies
                else 1.0
            )
        elif key.startswith("grade_"):
            probability = float(state["output"] == "pass")
            if key.endswith("_second"):
                probability = 1.0 - probability
        elif key == "fidelity:meaning":
            probability = 0.0 if state["candidate_prompt"] in unfaithful else 1.0
        elif key.startswith("evaluate:rerank:"):
            probability = (rerank_scores or {}).get(state["candidate_id"], 1.0)
        elif key.startswith("evaluate:accept:"):
            probability = (accept_scores or {}).get(state["candidate_id"], 1.0)
        elif key.startswith("evaluate:audit:"):
            probability = (audit_scores or {}).get(state["candidate_id"], 1.0)
        elif (
            key.startswith("evaluate:compare:")
            and state.get("aspect") == "structure_added"
        ):
            probability = structure_probability
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


def test_round_without_faithful_tests_converges_with_unverified_evidence() -> None:
    outcome = run_round(_gateway(tests='{"tests":[]}'), _plan())

    assert outcome.status == "converged"
    assert outcome.convergence["verification"] == "unverified"
    assert outcome.original_kept is False
    assert outcome.final_prompt != PROMPT
    assert outcome.ranking is not None
    assert outcome.ranking.selected is not None
    assert outcome.report()["offer_deep"] is False


def test_requirement_paraphrase_is_valid_but_exact_output_literal_is_binding() -> None:
    paraphrased = "Keep the answer within 20 words. Summarize the report."
    semantic = run_round(
        _gateway(
            tests='{"tests":[]}',
            candidate_text=paraphrased,
        ),
        replace(
            _plan(diagnosis={"confirmed_gaps": [], "problem_sentences": []}),
            hard_constraints=("Use no more than 20 words",),
        ),
    )

    assert semantic.status == "converged"
    assert semantic.final_prompt == paraphrased

    exact_prompt = 'Reply with exactly "OK".'
    exact = run_round(
        _gateway(
            tests='{"tests":[]}',
            candidate_text=paraphrased,
        ),
        replace(
            _plan(diagnosis={"confirmed_gaps": [], "problem_sentences": []}),
            prompt=exact_prompt,
            working_prompt=exact_prompt,
            hard_constraints=("OK",),
            exact_output=True,
        ),
    )
    assert exact.original_kept is True
    assert exact.final_prompt == exact_prompt


def test_round_without_confirmed_gaps_still_rewrites_the_prompt() -> None:
    outcome = run_round(
        _gateway(), _plan(diagnosis={"confirmed_gaps": [], "problem_sentences": []})
    )

    assert outcome.status == "converged"
    assert outcome.original_kept is False
    assert outcome.selected_candidate_id is not None
    assert outcome.report()["offer_deep"] is False


def test_round_rejects_a_whole_prompt_mirror_candidate_on_a_clear_input() -> None:
    outcome = run_round(
        _gateway(writer_echo=True),
        _plan(diagnosis={"confirmed_gaps": [], "problem_sentences": []}),
    )

    assert outcome.status == "improvement_not_verified"
    assert outcome.original_kept is True
    assert outcome.report()["failure"]["kind"] == "improvement_not_verified"
    assert outcome.continue_rounds is True


def test_round_with_no_eligible_strategy_reports_an_unverified_improvement() -> None:
    all_strategies = {
        "add_missing_context",
        "specify_output_format",
        "add_done_criteria",
        "remove_contradictions",
        "add_example",
        "split_into_steps",
        "role_play",
        "repeated_emphasis",
    }
    outcome = run_round(_gateway(blocked_strategies=all_strategies), _plan())

    assert outcome.status == "improvement_not_verified"
    assert "could not produce a verified changed prompt" in outcome.summary
    assert outcome.failures == ()
    assert outcome.report()["failure"]["kind"] == "improvement_not_verified"


def test_round_picks_a_verified_winner_and_reports_the_losers() -> None:
    outcome = run_round(_gateway(), _plan())

    assert outcome.status == "converged"
    assert outcome.original_kept is False
    assert outcome.final_prompt.startswith("Rewrite ")
    assert outcome.selected_candidate_id is not None
    assert (
        outcome.selected_strategy
        == _candidates(outcome)[outcome.selected_candidate_id]["strategy"]
    )
    assert {failure.candidate_id for failure in outcome.failures} == set(
        _candidates(outcome)
    ) - {outcome.selected_candidate_id}
    assert outcome.continue_rounds is False


def test_round_accept_gate_rejects_semantic_leader_and_promotes_survivor() -> None:
    gateway = _gateway(
        rerank_scores={
            "candidate-3-add_done_criteria": 0.99,
            "candidate-1-add_missing_context": 0.8,
            "candidate-2-specify_output_format": 0.7,
        },
        accept_scores={
            "candidate-3-add_done_criteria": 0.1,
            "candidate-1-add_missing_context": 0.95,
        },
    )

    outcome = run_round(gateway, _plan())

    assert outcome.selected_strategy == "add_missing_context"
    evidence = outcome.report()["evaluation_evidence"]["candidates"]
    assert evidence["candidate-3-add_done_criteria"]["accept"]["accepted"] is False
    assert evidence["candidate-3-add_done_criteria"]["eligible"] is False
    assert evidence["candidate-1-add_missing_context"]["accept"]["accepted"] is True
    assert evidence["candidate-1-add_missing_context"]["eligible"] is True
    assert any(
        item["candidate_id"] == "candidate-3-add_done_criteria"
        and item["stage"] == "accept"
        and item["capability"] == "gate"
        for item in outcome.report()["judgment_provenance"]
    )


def test_round_accept_bundle_contains_full_evidence_and_unverified_status() -> None:
    gateway = _gateway(tests='{"tests":[]}')

    outcome = run_round(gateway, _plan())

    gate_entry = next(
        entry
        for entry in gateway.decision_log
        if entry["question"]["key"].startswith("evaluate:accept:")
    )
    package = gate_entry["question"]["state"]["complete_candidate_evidence"]
    assert package["fidelity"]["passed"] is True
    assert package["score_vector"]["passed"] is True
    assert set(package["comparison"]) == {
        "task_preserved",
        "no_invented_detail",
        "structure_added",
        "verbosity_direction",
    }
    assert package["downstream_verification"] == "unverified"
    assert package["success_tests"] == []
    assert package["success_test_outputs"]
    assert outcome.status == "converged"
    assert outcome.report()["evaluation_evidence"]["candidates"]


def test_accept_request_snapshot_matches_what_gateway_received() -> None:
    gateway = _gateway(tests='{"tests":[]}')
    respond = gateway.decision_handler
    assert respond is not None
    received: list[dict] = []

    def capture(request, **kwargs):
        if str(request.get("key", "")).startswith("evaluate:accept:"):
            received.append(deepcopy(request))
        return respond(request, **kwargs)

    gateway.decision_handler = capture
    candidate = RankingCandidate("candidate-a", "A clearer summary.")
    evaluation = evaluate_candidate_packages(
        gateway,
        PROMPT,
        [candidate],
        constraints=(),
        improvement_style="clearer",
        style_bundle=("clarify",),
        success_tests=(),
        candidate_outputs={},
        strong_evidence={},
        judge_model="test-judge",
        run_id="accept-snapshot",
        round_number=1,
    )

    logged = next(
        entry["question"]
        for entry in gateway.decision_log
        if entry["question"]["key"].startswith("evaluate:accept:")
    )
    assert len(received) == 1
    assert received[0] == logged
    assert "accept" not in logged["state"]["complete_candidate_evidence"]
    assert "eligible" not in logged["state"]["complete_candidate_evidence"]
    assert evaluation.candidates["candidate-a"]["accept"]["accepted"] is True
    assert evaluation.candidates["candidate-a"]["eligible"] is True


def test_non_probe_noul_count_matches_actual_score_vector_answers() -> None:
    gateway = _gateway()

    outcome = run_round(gateway, _plan())

    actual = sum(
        entry["question"]["key"].startswith("score:")
        and entry["question"].get("type") == "noul"
        for entry in gateway.decision_log
    )
    reported = sum(
        item["capability"] == "noul" and item["stage"] == "score_vector"
        for item in outcome.report()["judgment_provenance"]
    )
    assert actual > 0
    assert reported == actual


def test_report_provenance_normalizes_usable_choice_without_changing_raw_answer() -> (
    None
):
    outcome = run_round(_gateway(), _plan())

    record = next(
        item
        for item in outcome.report()["judgment_provenance"]
        if item["question_key"].startswith("evaluate:compare:")
        and item["question_key"].endswith(":verbosity_direction")
    )
    raw = record["raw_answer"]
    assert raw["type"] == "choice"
    assert record["usable"] is True
    assert record["selected"] == raw["choice"]
    assert record["probability"] == raw["probabilities"][raw["choice"]]


def test_report_provenance_does_not_normalize_wrong_primitive_choice_answer() -> None:
    gateway = _gateway()
    respond = gateway.decision_handler
    assert respond is not None

    def wrong_primitive(request, **kwargs):
        if str(request.get("key", "")).endswith(":verbosity_direction"):
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        return respond(request, **kwargs)

    gateway.decision_handler = wrong_primitive
    outcome = run_round(gateway, _plan())

    record = next(
        item
        for item in outcome.report()["judgment_provenance"]
        if item["question_key"].startswith("evaluate:compare:")
        and item["question_key"].endswith(":verbosity_direction")
    )
    assert record["raw_answer"] == {
        "type": "noul",
        "probability_true": 0.99,
        "confidence": 1.0,
    }
    assert record["usable"] is False
    assert "selected" not in record
    assert "probability" not in record


@pytest.mark.parametrize("aspect", ["structure_added", "verbosity_direction"])
def test_malformed_descriptive_comparison_blocks_acceptance_without_hiding_raw(
    aspect: str,
) -> None:
    base = _gateway()
    respond = base.decision_handler
    assert respond is not None

    def wrong_shapes(request, **kwargs):
        key = str(request.get("key", ""))
        if key.endswith(":" + aspect) and aspect == "structure_added":
            return {
                "type": "choice",
                "choice": "added",
                "probabilities": {"added": 1.0},
                "confidence": 1.0,
            }
        if key.endswith(":" + aspect) and aspect == "verbosity_direction":
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        return respond(request, **kwargs)

    base.decision_handler = wrong_shapes
    candidate = RankingCandidate("candidate-a", "A clearer summary.")
    result = evaluate_candidate_packages(
        base,
        PROMPT,
        [candidate],
        constraints=(),
        improvement_style="clearer",
        style_bundle=("clarify",),
        success_tests=(),
        candidate_outputs={},
        strong_evidence={},
        judge_model="test-judge",
        run_id=f"wrong-shapes-{aspect}",
        round_number=1,
    )

    evidence = result.candidates["candidate-a"]
    malformed = evidence["comparison"][aspect]
    assert malformed["usable"] is False
    matching_record = next(
        item for item in result.provenance if item["question_key"].endswith(aspect)
    )
    assert matching_record["usable"] is False
    assert "selected" not in matching_record
    assert "probability" not in matching_record
    assert evidence["comparison"]["task_preserved"]["usable"] is True
    assert evidence["comparison"]["no_invented_detail"]["usable"] is True
    assert evidence["eligible"] is False
    assert any(
        "comparison incomplete" in reason for reason in evidence["rejection_reasons"]
    )


def test_descriptive_structure_and_shorter_choice_do_not_reject_proofread_candidate() -> (
    None
):
    gateway = _gateway(
        structure_probability=0.01,
        verbosity_choice="shorter",
    )

    outcome = run_round(gateway, _plan())

    assert outcome.selected_candidate_id is not None
    evidence = outcome.report()["evaluation_evidence"]["candidates"][
        outcome.selected_candidate_id
    ]
    assert evidence["comparison"]["structure_added"]["probability"] == 0.01
    assert evidence["comparison"]["verbosity_direction"]["selected"] == "shorter"
    assert evidence["eligible"] is True


def test_original_baseline_must_pass_its_own_accept_gate() -> None:
    gateway = _gateway(
        weak_passes=lambda _model, _prompt: True,
        accept_scores={
            "original": 0.01,
            "candidate-1-add_missing_context": 0.01,
            "candidate-2-specify_output_format": 0.01,
            "candidate-3-add_done_criteria": 0.01,
        },
    )

    outcome = run_round(gateway, _plan())

    evidence = outcome.report()["evaluation_evidence"]["candidates"]["original"]
    assert evidence["accept"]["accepted"] is False
    assert outcome.converged is False
    assert outcome.selected_candidate_id is None


def test_round_rejects_a_candidate_that_fails_fidelity() -> None:
    outcome = run_round(_gateway(unfaithful={"Rewrite specify_output_format"}), _plan())
    candidate = next(
        c for c in outcome.candidates if c["strategy"] == "specify_output_format"
    )

    assert candidate["selected"] is False
    assert "candidate failed fidelity checks" in candidate["rejection_reasons"]
    assert outcome.final_prompt != "Rewrite specify_output_format"


def test_round_rejects_a_weak_panel_winner_that_regresses_on_the_strong_model() -> None:
    outcome = run_round(
        _gateway(strong_passes=lambda prompt: prompt == PROMPT), _plan()
    )

    assert outcome.original_kept is True
    assert outcome.status == "improvement_not_verified"
    assert all(
        any(
            reason.startswith("strong_check_regression")
            for reason in candidate["rejection_reasons"]
        )
        for candidate in outcome.candidates
    )
    assert outcome.continue_rounds is True


def test_round_marks_and_blocks_a_crutch_strategy_that_regresses() -> None:
    crutches = {"add_example", "split_into_steps", "role_play", "repeated_emphasis"}
    outcome = run_round(
        _gateway(
            strong_passes=lambda prompt: prompt.removeprefix("Rewrite ") not in crutches
        ),
        _plan(tier="deep"),
    )
    strong = outcome.report()["strong_check"]["candidates"]
    crutch_outcomes = [entry for entry in strong if entry["strategy"] in crutches]

    assert crutch_outcomes
    assert all(
        entry["crutch"] is True and entry["eligible"] is False
        for entry in crutch_outcomes
    )
    assert outcome.final_prompt.removeprefix("Rewrite ") not in crutches


def test_round_converges_on_its_measured_baseline_when_rewrites_are_worse() -> None:
    outcome = run_round(
        _gateway(weak_passes=lambda _model, prompt: prompt == PROMPT), _plan()
    )

    assert outcome.original_kept is True
    assert outcome.status == "converged"
    assert outcome.selected_strategy == "original"
    assert outcome.failures and outcome.continue_rounds is False
    assert all("weak pass rates" in failure.summary for failure in outcome.failures)
    report = outcome.report()
    assert "failure" not in report
    assert report["convergence"]["source"] == "original_baseline"
    assert report["selection_evidence"]["selected_candidate_id"] == "original"


def test_round_grades_each_weak_model_separately() -> None:
    outcome = run_round(
        _gateway(
            weak_passes=lambda model, prompt: prompt != PROMPT and model == "weak-a"
        ),
        _plan(),
    )
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
