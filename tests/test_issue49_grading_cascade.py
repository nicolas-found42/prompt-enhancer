"""Uncertain weak grades resolve through a bounded Jev cascade."""

from __future__ import annotations

import json
from pathlib import Path

from test_issue44_screen_and_grade import _run_screened_round

from prompt_enhancer import jev_questions
from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.config import Settings
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    DecisionPolicy,
    runtime_question_identity,
)
from prompt_enhancer.evaluation.harness import default_engine_factory
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.grading import grade_panel_with_jev
from prompt_enhancer.grading_cascade import CascadeBudget
from prompt_enhancer.runner import PanelResult
from prompt_enhancer.store import RunStore


def test_decisive_jev_confirmation_resolves_borderline_grade_without_strong_fallback() -> (
    None
):
    result, batches = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.95, 0.05),
        generated_test_count=1,
    )

    assert result["original_kept"] is False
    cascade = result["report"]["grading_cascade"]
    assert cascade["confirmed_pass_count"] > 0
    assert cascade["escalation_count"] == 0
    assert cascade["unresolved_count"] == 0
    assert any(
        batch and str(batch[0]["key"]).startswith("grade-confirm:") for batch in batches
    )


def test_fast_retains_original_with_unresolved_grade_and_no_cascade_requests() -> None:
    result, batches = _run_screened_round(
        writer_instruction_version=7,
        tier="fast",
        grade_pass_probability=0.5,
        generated_test_count=1,
    )

    assert result["original_kept"] is True
    cascade = result["report"]["grading_cascade"]
    assert cascade["confirmation_count"] == 0
    assert cascade["unresolved_count"] > 0
    assert all(item["reason"] == "pair_budget_exhausted" for item in cascade["pairs"])
    assert not any(
        batch and str(batch[0]["key"]).startswith("grade-confirm:") for batch in batches
    )


def test_unsupported_strong_fallback_quote_cannot_resolve_uncertain_grade() -> None:
    result, batches = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.5, 0.5),
        generated_test_count=1,
        priced_catalog=True,
        decision_policy=_verification_policy("Does the answer satisfy criterion 0?"),
        strong_evidence={
            "suggested_verdict": "pass",
            "prompt_quote": "text absent from prompt",
            "output_quote": "pass",
            "rationale": "The answer is complete.",
        },
    )

    assert result["original_kept"] is True
    cascade = result["report"]["grading_cascade"]
    assert cascade["escalation_count"] > 0
    assert cascade["verification_count"] == 0
    assert any(item["reason"] == "invalid_evidence_quote" for item in cascade["pairs"])
    assert any(
        batch and str(batch[0]["key"]).startswith("grade-confirm:") for batch in batches
    )


def _verification_policy(criterion: str) -> DecisionPolicy:
    questions = {}
    for name, question in jev_questions.GRADING_VERIFY_QUESTIONS.items():
        question_id = f"grade-verify:{name}"
        identity = runtime_question_identity(
            question_id,
            {
                "type": "noul",
                "question": question,
                "question_schema": {
                    "criterion": criterion,
                    "suggested_verdict": "pass",
                },
            },
            family="grading_verification",
            rubric_version="issue-49-v1",
            snapshot=JEV_MODEL,
        )
        questions[question_id] = {
            "identity": identity.to_dict(),
            "verdict": "gate",
            "threshold": 0.8,
        }
    return DecisionPolicy.from_artifact(
        CalibrationArtifact.from_dict(
            {
                "kind": "calibration-artifact",
                "name": "known-verification-policy",
                "input_digest": "test-fixture",
                "questions": questions,
            }
        )
    )


def test_supported_strong_evidence_resolves_only_after_consistent_jev_verification() -> (
    None
):
    criterion = "Does the answer satisfy the request?"
    options = {
        "writer_instruction_version": 7,
        "tier": "standard",
        "grade_pass_probability": 0.5,
        "confirmation_answers": (0.95, 0.5, 0.5),
        "generated_test_count": 1,
        "priced_catalog": True,
        "criterion_text": criterion,
        "strong_evidence": {
            "suggested_verdict": "pass",
            "prompt_quote": "Summarize the report.",
            "output_quote": "pass",
            "rationale": "The cited output answers the requested task.",
        },
        "decision_policy": _verification_policy(criterion),
    }
    verified, batches = _run_screened_round(
        **options,
        verification_answers=(0.95, 0.95, 0.05, 0.95),
    )

    assert verified["original_kept"] is False
    cascade = verified["report"]["grading_cascade"]
    assert cascade["escalation_count"] > 0
    assert cascade["verification_count"] > 0
    assert cascade["unresolved_count"] == 0
    assert any(
        batch and str(batch[0]["key"]).startswith("grade-verify:") for batch in batches
    )

    contradicted, _ = _run_screened_round(
        **options,
        verification_answers=(0.95, 0.05, 0.95, 0.95),
    )
    assert contradicted["original_kept"] is True
    assert contradicted["report"]["grading_cascade"]["unresolved_count"] > 0


def test_rank_only_calibration_cannot_make_extreme_raw_grade_authoritative() -> None:
    initial, batches = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        generated_test_count=1,
    )
    assert initial["original_kept"] is False
    request = next(
        item for batch in batches for item in batch if item["key"] == "grade_0_0_first"
    )
    identity = runtime_question_identity(
        "grade:t0",
        request,
        family="grading",
        rubric_version="issue-49-v1",
        snapshot=JEV_MODEL,
    )
    policy = DecisionPolicy.from_artifact(
        CalibrationArtifact.from_dict(
            {
                "kind": "calibration-artifact",
                "name": "rank-only",
                "input_digest": "fixture",
                "questions": {
                    identity.question_id: {
                        "identity": identity.to_dict(),
                        "verdict": "ranker",
                    }
                },
            }
        )
    )

    held, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        generated_test_count=1,
        decision_policy=policy,
    )

    assert held["original_kept"] is True
    assert any(
        item["reason"] == "calibration_not_gate_capable"
        for item in held["report"]["grading_cascade"]["pairs"]
    )


def test_confident_choice_and_score_failures_do_not_trigger_confirmation() -> None:
    def decide(request: dict, **_kwargs: object) -> dict:
        if request["type"] == "choice":
            return {
                "type": "choice",
                "choice": "fail",
                "probabilities": {"pass": 0.05, "fail": 0.95},
            }
        criteria = list(request["criteria"])
        return {
            "type": "score",
            "score": 0.1,
            "probabilities": {
                str(index): 0.1 if level == "good" else 0.9
                for index, level in enumerate(criteria)
            },
            "legend": {str(index): level for index, level in enumerate(criteria)},
        }

    gateway = ScriptedGateway(decision=decide)
    grades, _ = grade_panel_with_jev(
        [PanelResult("candidate", "weak", 0, 7, "answer", "prompt")],
        [
            {
                "id": "choice",
                "question": "Which answer?",
                "kind": "choice",
                "expected": "pass",
                "options": ["pass", "fail"],
            },
            {
                "id": "score",
                "question": "How complete?",
                "kind": "score",
                "expected": "good",
                "levels": ["bad", "good"],
            },
        ],
        gateway,
        judge_model=gateway.jev_model,
        run_id="confident-failure",
        shared_state=True,
        cascade_budget=CascadeBudget.for_tier("standard"),
        cascade_strong_model="strong",
    )

    assert grades["candidate"].worst == 0.0
    assert grades["candidate"].unresolved_grade_outputs == 0
    assert not any(call["role"] == "judge_confirmation" for call in gateway.calls)


def test_cascade_pair_and_dollar_caps_hold_remaining_grades_unresolved() -> None:
    standard, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.95, 0.05),
        generated_test_count=3,
    )
    assert standard["original_kept"] is True
    assert standard["report"]["grading_cascade"]["confirmation_count"] == 10
    assert any(
        item["reason"] == "pair_budget_exhausted"
        for item in standard["report"]["grading_cascade"]["pairs"]
    )

    capped, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.95, 0.05),
        generated_test_count=1,
        settings=Settings(grading_cascade_dollar_cap=0.001),
    )
    assert capped["original_kept"] is True
    assert capped["report"]["grading_cascade"]["confirmation_count"] == 1
    assert any(
        item["reason"] == "dollar_budget_exhausted"
        for item in capped["report"]["grading_cascade"]["pairs"]
    )


def test_provider_failure_and_unpriced_fallback_stay_unresolved() -> None:
    failed, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        generated_test_count=1,
        confirmation_provider_error=True,
    )
    assert failed["original_kept"] is True
    assert any(
        item["reason"].startswith("confirmation_provider_")
        for item in failed["report"]["grading_cascade"]["pairs"]
    )

    unpriced, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        generated_test_count=1,
        confirmation_answers=(0.95, 0.5, 0.5),
        decision_policy=_verification_policy("Does the answer satisfy criterion 0?"),
    )
    assert unpriced["original_kept"] is True
    assert unpriced["report"]["grading_cascade"]["escalation_count"] == 0
    assert any(
        item["reason"] == "missing_trustworthy_fallback_pricing"
        for item in unpriced["report"]["grading_cascade"]["pairs"]
    )

    uncalibrated, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        generated_test_count=1,
        criterion_text="Does the answer satisfy the request?",
        confirmation_answers=(0.95, 0.5, 0.5),
        priced_catalog=True,
        strong_evidence={
            "suggested_verdict": "pass",
            "prompt_quote": "Summarize the report.",
            "output_quote": "pass",
            "rationale": "The output answers the request.",
        },
        verification_answers=(0.95, 0.95, 0.05, 0.95),
    )
    assert uncalibrated["original_kept"] is True
    assert uncalibrated["report"]["grading_cascade"]["verification_count"] == 0
    assert uncalibrated["report"]["grading_cascade"]["escalation_count"] == 0
    assert all(
        "strong_answer" not in pair
        for pair in uncalibrated["report"]["grading_cascade"]["pairs"]
    )
    assert any(
        item["reason"] == "missing_gate_calibration"
        for item in uncalibrated["report"]["grading_cascade"]["pairs"]
    )


def test_detected_output_cannot_be_rescued_and_unresolved_screen_skips_confirmation() -> (
    None
):
    detected, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        generated_test_count=1,
        output_screen={"pass": 0.99},
        confirmation_answers=(0.95, 0.95, 0.05),
    )
    assert detected["original_kept"] is True
    assert detected["report"]["grading_cascade"]["confirmation_count"] == 0
    assert all(
        item["status"] == "steering_detected"
        for item in detected["report"]["output_screen"]
        if item["candidate_id"] != "original"
    )

    unresolved, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        generated_test_count=1,
        output_screen={"pass": None},
    )
    assert unresolved["original_kept"] is True
    assert unresolved["report"]["grading_cascade"]["confirmation_count"] == 0
    assert any(
        item["reason"] == "output_screen_unresolved"
        for item in unresolved["report"]["grading_cascade"]["pairs"]
    )


def test_current_cascade_recording_replays_initial_and_confirmation_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cascade.json"
    original, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.95, 0.05),
        generated_test_count=1,
        record_path=path,
    )
    assert json.loads(path.read_text())["writer_instruction_version"] == 7
    engine = default_engine_factory(path)
    engine.store = RunStore(":memory:")

    replayed = engine.optimize(
        "Read the background notes. Summarize the report.",
        {"tier": "standard", "clarification_allowed": False},
    )

    assert replayed["final_prompt"] == original["final_prompt"]
    assert (
        replayed["report"]["grading_cascade"] == original["report"]["grading_cascade"]
    )


def test_calibrated_fallback_recording_replays_all_three_stages(tmp_path: Path) -> None:
    path = tmp_path / "fallback.json"
    criterion = "Does the answer satisfy the request?"
    original, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.5, 0.5),
        generated_test_count=1,
        priced_catalog=True,
        criterion_text=criterion,
        strong_evidence={
            "suggested_verdict": "pass",
            "prompt_quote": "Summarize the report.",
            "output_quote": "pass",
            "rationale": "The cited output answers the requested task.",
        },
        verification_answers=(0.95, 0.95, 0.05, 0.95),
        decision_policy=_verification_policy(criterion),
        record_path=path,
    )
    assert original["original_kept"] is False
    engine = default_engine_factory(path)
    engine.store = RunStore(":memory:")

    replayed = engine.optimize(
        "Read the background notes. Summarize the report.",
        {"tier": "standard", "clarification_allowed": False},
    )

    assert replayed["final_prompt"] == original["final_prompt"]
    assert (
        replayed["report"]["grading_cascade"] == original["report"]["grading_cascade"]
    )


def test_recorded_cascade_budget_overrides_replay_without_extra_calls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capped.json"
    original, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.95, 0.05),
        generated_test_count=1,
        settings=Settings(grading_cascade_pair_cap=1, grading_cascade_dollar_cap=0.001),
        record_path=path,
    )
    assert original["report"]["grading_cascade"]["confirmation_count"] == 1
    engine = default_engine_factory(path)
    engine.store = RunStore(":memory:")

    replayed = engine.optimize(
        "Read the background notes. Summarize the report.",
        {"tier": "standard", "clarification_allowed": False},
    )

    assert (
        replayed["report"]["grading_cascade"] == original["report"]["grading_cascade"]
    )


def test_deep_pair_cap_is_thirty_even_when_more_outputs_are_borderline() -> None:
    def decide(request: dict, **_kwargs: object) -> dict:
        key = str(request["key"])
        probability = (
            0.95
            if key.endswith(("sufficient", "meets"))
            else 0.05
            if key.endswith("violation")
            else 0.5
        )
        return {"type": "noul", "probability_true": probability}

    gateway = ScriptedGateway(decision=decide)
    observation: dict = {}
    grades, _ = grade_panel_with_jev(
        [
            PanelResult("candidate", "weak", index, index, "answer", "prompt")
            for index in range(31)
        ],
        [
            {
                "id": "t0",
                "question": "Does it answer?",
                "kind": "noul",
                "expected": "yes",
            }
        ],
        gateway,
        judge_model=gateway.jev_model,
        run_id="deep-cap",
        shared_state=True,
        cascade_budget=CascadeBudget.for_tier("deep"),
        cascade_observation=observation,
        cascade_strong_model="strong",
    )

    assert observation["confirmation_count"] == 30
    assert observation["unresolved_count"] == 1
    assert grades["candidate"].unresolved_grade_outputs == 1


def test_inclusive_uncertainty_boundaries_and_confirmation_decision_table() -> None:
    failed, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        generated_test_count=1,
        grade_pass_probability=0.3,
        confirmation_answers=(0.8, 0.2, 0.8),
    )
    assert failed["original_kept"] is True
    assert failed["report"]["grading_cascade"]["confirmed_fail_count"] > 0

    passed, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        generated_test_count=1,
        grade_pass_probability=0.7,
        confirmation_answers=(0.8, 0.8, 0.2),
    )
    assert passed["original_kept"] is False
    assert passed["report"]["grading_cascade"]["confirmed_pass_count"] > 0


def test_exact_json_failure_blocks_semantic_confirmation_and_fallback_claim() -> None:
    result, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        generated_test_count=1,
        criterion_text="Is the output valid JSON?",
        decision_policy=_verification_policy("Is the output valid JSON?"),
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.95, 0.05),
        priced_catalog=True,
        strong_evidence={
            "suggested_verdict": "pass",
            "prompt_quote": "Summarize the report.",
            "output_quote": "pass",
            "rationale": "The output appears valid.",
        },
    )

    assert result["original_kept"] is True
    cascade = result["report"]["grading_cascade"]
    assert cascade["confirmed_pass_count"] == 0
    assert any(
        item["reason"] == "deterministic_evidence_conflict" for item in cascade["pairs"]
    )


def test_unimplemented_numeric_check_cannot_be_confirmed_by_prose() -> None:
    result, _ = _run_screened_round(
        writer_instruction_version=7,
        tier="standard",
        generated_test_count=1,
        criterion_text="Does the answer include 12 citations?",
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.95, 0.05),
    )

    assert result["original_kept"] is True
    assert any(
        item["reason"] == "deterministic_check_unavailable"
        for item in result["report"]["grading_cascade"]["pairs"]
    )
