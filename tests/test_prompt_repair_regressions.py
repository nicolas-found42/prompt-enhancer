"""Regression seams from the three live prompt failures.

Calibration fixtures here are synthetic known-answer policies. They verify
runtime loading and identity enforcement, not empirical calibration accuracy.
"""

import json
from pathlib import Path

import pytest
from test_issue44_screen_and_grade import _run_screened_round
from test_issue49_grading_cascade import _verification_policy

from prompt_enhancer.api import create_app
from prompt_enhancer.candidate_evaluation import evaluate_candidate_packages
from prompt_enhancer.config import Settings
from prompt_enhancer.failures import describe_failure
from prompt_enhancer.fidelity import check_candidate_fidelity
from prompt_enhancer.gateway import ProviderError, ScriptedGateway
from prompt_enhancer.grading import grade_candidate
from prompt_enhancer.rounds import RoundPlan, run_round
from prompt_enhancer.runner import PanelResult, run_candidates
from prompt_enhancer.selector import RankingCandidate
from prompt_enhancer.store import RunStore
from prompt_enhancer.styles import style_authorization_for


def test_app_loads_matching_calibration_and_opens_verified_fallback(tmp_path: Path):
    criterion = "Does the answer satisfy the request?"
    artifact = tmp_path / "calibration.json"
    artifact.write_text(json.dumps(_verification_policy(criterion).artifact_dicts[0]))
    app = create_app(
        store=RunStore(), settings=Settings(decision_policy_path=str(artifact))
    )
    policy = app.state.optimizer.decision_policy
    assert policy is not None
    assert policy.has_candidate("grade-verify:support")
    result, _ = _run_screened_round(
        writer_instruction_version=7,
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.5, 0.5),
        generated_test_count=1,
        priced_catalog=True,
        criterion_text=criterion,
        decision_policy=policy,
        strong_evidence={
            "suggested_verdict": "pass",
            "prompt_quote": "Summarize the report.",
            "output_quote": "pass",
            "rationale": "The output satisfies the task.",
        },
        verification_answers=(0.95, 0.95, 0.05, 0.95),
    )
    assert result["report"]["grading_cascade"]["verification_count"] > 0
    assert result["report"]["grading_cascade"]["unresolved_count"] == 0


def test_configured_calibration_fails_closed_on_changed_criterion(tmp_path: Path):
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(_verification_policy("Original criterion").artifact_dicts[0])
    )
    app = create_app(
        store=RunStore(), settings=Settings(decision_policy_path=str(artifact))
    )
    result, _ = _run_screened_round(
        writer_instruction_version=7,
        grade_pass_probability=0.5,
        confirmation_answers=(0.95, 0.5, 0.5),
        generated_test_count=1,
        priced_catalog=True,
        criterion_text="Different criterion",
        decision_policy=app.state.optimizer.decision_policy,
        strong_evidence={
            "suggested_verdict": "pass",
            "prompt_quote": "Summarize the report.",
            "output_quote": "pass",
            "rationale": "The output satisfies the task.",
        },
        pause_after_round=True,
    )
    assert result["report"]["grading_cascade"]["verification_count"] == 0
    assert result["report"]["grading_cascade"]["unresolved_count"] > 0


def test_timeout_description_names_deadline_without_network_diagnosis():
    failure = describe_failure(ProviderError("go", "mimo", None, kind="timeout"))
    assert "deadline" in failure["headline"]
    assert "internet" not in failure["hint"]


@pytest.mark.parametrize("prior_vector", [None, {"scores": {}, "passed": False}])
def test_current_round_reports_absent_grades_and_retains_unverified_status(
    prior_vector,
):
    from test_score_vector import WEAK, _gateway

    prompt = "Explain how rainbows form to a 10-year-old in no more than 120 words."
    outcome = run_round(
        _gateway(tests='{"tests":[]}'),
        RoundPlan(
            prompt,
            prompt,
            "absent-tests",
            42,
            {},
            (),
            Settings(weak_models=WEAK),
            0.8,
            15,
            applied_style="audience_fit",
            route_strategies=(),
            prior_vector=prior_vector,
            round_number=4 if prior_vector is not None else 1,
        ),
    )
    report = outcome.report()
    assert outcome.converged
    assert report["selection_evidence"]["original_score"] is None
    assert report["selection_evidence"]["winner_score"] is None
    assert report["convergence"]["verification"] == "unverified"
    assert report["convergence"]["gain"] is None
    if prior_vector is not None:
        assert "first round" not in outcome.summary


@pytest.mark.parametrize(
    "authorized,original_mass,style_mass,passed",
    [
        (True, 0.58, 0.41, True),
        (False, 0.58, 0.41, False),
        (True, 0.30, 0.30, False),
        (True, 0.10, 0.10, False),
    ],
)
def test_fidelity_combines_only_authorized_support_mass(
    authorized, original_mass, style_mass, passed
):
    def decide(q, **_kwargs):
        if q["type"] == "choice":
            return {
                "type": "choice",
                "choice": "supported_by_original",
                "probabilities": {
                    "supported_by_original": original_mass,
                    "authorized_style_presentation": style_mass,
                    "new_requirement": 1 - original_mass - style_mass,
                },
            }
        return {"type": "noul", "noul": 0.99}

    r = check_candidate_fidelity(
        ScriptedGateway(decision=decide),
        "Include learning goals.",
        "1. Learning goals",
        {},
        "specify_output_format",
        run_id="split",
        judge_model="judge",
        applied_style="structured",
        style_authorization=style_authorization_for("structured")
        if authorized
        else None,
    )
    assert r.passed is passed


def test_absent_success_tests_do_not_send_zero_grade_to_acceptance():
    requests = []

    def decide(q, **_kwargs):
        requests.append(q)
        if q["type"] == "choice":
            return {"type": "choice", "choice": "same", "probabilities": {"same": 1.0}}
        return {"type": "noul", "noul": 0.99}

    p = "Explain rainbows to a 10-year-old in no more than 120 words."
    grade = grade_candidate(
        "original", [PanelResult("original", "weak", 0, 1, "Answer", p)], lambda _run: 0
    )
    result = evaluate_candidate_packages(
        ScriptedGateway(decision=decide),
        p,
        [RankingCandidate("original", p, grade=grade)],
        constraints=(),
        improvement_style="audience_fit",
        style_bundle=(),
        success_tests=(),
        candidate_outputs={"original": [{"output": "unverified panel answer"}]},
        strong_evidence={},
        judge_model="judge",
        run_id="empty-tests",
        round_number=1,
    )
    accept = next(q for q in requests if q["key"].startswith("evaluate:accept:"))
    evidence = accept["state"]["complete_candidate_evidence"]
    assert evidence["success_test_grade"] is None
    assert evidence["success_test_outputs"] == []
    assert result.candidates["original"]["success_test_outputs"] == [
        {"output": "unverified panel answer"}
    ]
    assert evidence["downstream_verification"] == "unverified"


@pytest.mark.parametrize("failed_check", [None, "verify", "audit"])
def test_baseline_acceptance_preserves_failed_applicable_checks(failed_check):
    requests = []

    def decide(q, **_kwargs):
        requests.append(q)
        if q["type"] == "choice":
            return {"type": "choice", "choice": "same", "probabilities": {"same": 1.0}}
        failed = failed_check and q["key"].startswith(f"evaluate:{failed_check}:")
        return {"type": "noul", "noul": 0.01 if failed else 0.99}

    prompt = "Make a plan using a single heading."
    result = evaluate_candidate_packages(
        ScriptedGateway(decision=decide),
        prompt,
        [RankingCandidate("original", prompt)],
        constraints=("Use a single heading.",),
        improvement_style="structured",
        style_bundle=("specify_output_format",),
        success_tests=(),
        candidate_outputs={},
        strong_evidence={},
        judge_model="judge",
        run_id="applicable-checks",
        round_number=1,
    )
    assert result.candidates["original"]["accept"]["accepted"]
    assert result.candidates["original"]["eligible"] is (failed_check is None)
    accept = next(q for q in requests if q["key"].startswith("evaluate:accept:"))
    assert "only when those checks are applicable" in accept["query"]
    assert "An unchanged original may" in accept["query"]


def test_original_baseline_retains_measured_strong_check_evidence():
    from test_score_vector import WEAK, _gateway

    base = _gateway()

    def decide(q, **kwargs):
        if q["key"].startswith("output-screen:") or (
            q["key"].startswith("success-test-screen:")
            and q.get("dimension") == "evaluator_instructions"
        ):
            return {"type": "noul", "noul": 0.01}
        return base.decide(q, **kwargs)

    outcome = run_round(
        ScriptedGateway(chat=base.chat, decision=decide),
        RoundPlan(
            "Explain rainbows.",
            "Explain rainbows.",
            "baseline-strong",
            42,
            {},
            (),
            Settings(weak_models=WEAK),
            0.8,
            15,
            applied_style="audience_fit",
        ),
    )
    report = outcome.report()
    strong = report["evaluation_evidence"]["candidates"]["original"]["strong_check"]
    assert strong["source"] == "original_baseline"
    assert strong["score"] == report["strong_check"]["original_score"]
    assert strong["model"] == report["strong_check"]["model"]


@pytest.mark.parametrize(
    "reply,kind",
    [
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "choices": [{"message": {"content": ""}}],
            },
            "incomplete_response",
        ),
        (
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "partial"}}
                ]
            },
            "incomplete_response",
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            "empty_response",
        ),
        (
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": None, "reasoning": "private"},
                    }
                ]
            },
            "empty_response",
        ),
        ({"stop_reason": "max_tokens", "content": "partial"}, "incomplete_response"),
    ],
)
def test_unusable_panel_reply_is_operational_not_a_quality_grade(reply, kind):
    g = ScriptedGateway(chat=lambda *_args, **_kwargs: reply)
    with pytest.raises(ProviderError) as error:
        run_candidates(
            [],
            ["weak"],
            g,
            original="Make a plan.",
            settings=Settings(weak_model_count=1, weak_samples=1),
        )
    assert error.value.kind == kind
    assert error.value.to_dict()["response_details"]["visible_chars"] in (0, 7)
