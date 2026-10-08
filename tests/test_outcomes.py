"""Canonical outcomes require persisted decision evidence."""

from __future__ import annotations

import pytest

from prompt_enhancer.outcomes import apply_outcome_fields
from prompt_enhancer.score_vector import SCORE_DIMENSIONS

ORIGINAL = "Write a concise report."
FINAL = "Write a concise report with a clear structure."
FLOORS = {dimension: 0.6 for dimension in SCORE_DIMENSIONS}
SCORES = {dimension: 0.8 for dimension in SCORE_DIMENSIONS}


def _accepted_report(*, verification: str) -> dict:
    grade = {
        "worst": 1.0,
        "threshold": 0.5,
        "ungradable_outputs": 0,
        "detected_outputs": 0,
        "unresolved_screen_outputs": 0,
        "unresolved_grade_outputs": 0,
    }
    candidate = {
        "candidate_id": "candidate-1",
        "text": FINAL,
        "metadata": {
            "fidelity": {"passed": True},
            "score_vector": {"passed": True},
            "evaluation": {
                "eligible": True,
                "accept": {"accepted": True},
                "downstream_verification": verification,
                "success_tests": (
                    [{"question": "Does it answer?"}]
                    if verification == "verified"
                    else []
                ),
                "success_test_outputs": (
                    [{"output": "A concise report."}]
                    if verification == "verified"
                    else []
                ),
                "success_test_grade": grade if verification == "verified" else None,
            },
        },
    }
    return {
        "status": "stopped",
        "applied_style": "clearer",
        "history": [
            {
                "status": "improved",
                "selected_candidate_id": "candidate-1",
                "selection_evidence": {
                    "selected_candidate_id": "candidate-1",
                    "selected_candidate": candidate,
                },
                "evidence": {
                    "strong_check": {
                        "candidates": [{"candidate_id": "candidate-1", "passed": True}]
                    }
                },
            }
        ],
    }


def test_converged_requires_all_measured_dimensions_and_gain_evidence() -> None:
    report = apply_outcome_fields(
        {
            **_accepted_report(verification="unverified"),
            "status": "converged",
            "applied_style": "clearer",
            "convergence": {
                "status": "converged",
                "passed": True,
                "scores": SCORES,
                "floors": FLOORS,
                "gain": 0.0,
                "epsilon": 0.01,
            },
        },
        original_prompt=ORIGINAL,
        final_prompt=FINAL,
    )

    assert report["outcome"] == "converged"
    assert "measured gain" in report["outcome_reason"]
    assert report["applied_style"] == "clearer"

    baseline = apply_outcome_fields(
        report, original_prompt=ORIGINAL, final_prompt=ORIGINAL
    )
    assert baseline["outcome"] is None


def test_converged_does_not_survive_a_dimension_floor_breach() -> None:
    scores = {**SCORES, "specificity": 0.2}
    report = apply_outcome_fields(
        {
            "status": "converged",
            "convergence": {
                "status": "converged",
                "passed": True,
                "scores": scores,
                "floors": FLOORS,
                "gain": None,
                "epsilon": 0.01,
            },
        },
        original_prompt=ORIGINAL,
        final_prompt=ORIGINAL,
    )

    assert report["outcome"] is None


def test_improved_tested_requires_accept_fidelity_score_tests_and_strong_pass() -> None:
    report = apply_outcome_fields(
        _accepted_report(verification="verified"),
        original_prompt=ORIGINAL,
        final_prompt=FINAL,
        control_state="stopped",
    )

    assert report["outcome"] == "improved_tested"
    assert report["control_state"] == "stopped"
    assert report["applied_style"] == "clearer"

    failed = _accepted_report(verification="verified")
    failed["history"][0]["evidence"]["strong_check"]["candidates"][0]["passed"] = False
    rejected = apply_outcome_fields(
        failed, original_prompt=ORIGINAL, final_prompt=FINAL
    )
    assert rejected["outcome"] is None


def test_improved_unverified_requires_explicit_accepted_candidate_evidence() -> None:
    report = apply_outcome_fields(
        _accepted_report(verification="unverified"),
        original_prompt=ORIGINAL,
        final_prompt=FINAL,
    )
    assert report["outcome"] == "improved_unverified"

    label_only = apply_outcome_fields(
        {"status": "improved_unverified", "applied_style": "clearer"},
        original_prompt=ORIGINAL,
        final_prompt=FINAL,
    )
    assert label_only["outcome"] is None


@pytest.mark.parametrize(
    ("status", "failure", "expected"),
    [
        (
            "impossible",
            {"kind": "impossible", "message": "Cannot satisfy exact format."},
            "impossible",
        ),
        (
            "failed",
            {"kind": "provider_error", "message": "Provider unavailable."},
            "failed_operational",
        ),
        ("failed", {"kind": "cancelled", "message": "Stopped."}, None),
    ],
)
def test_impossible_and_operational_outcomes_follow_failure_evidence(
    status: str, failure: dict, expected: str | None
) -> None:
    report = apply_outcome_fields(
        {"status": status, "failure": failure, "applied_style": "clearer"},
        original_prompt=ORIGINAL,
        final_prompt=ORIGINAL,
    )
    assert report["outcome"] == expected


def test_verified_but_incomplete_test_evidence_is_not_relabelled_unverified() -> None:
    report = _accepted_report(verification="verified")
    report["history"][0]["evidence"]["strong_check"]["candidates"] = []
    report = apply_outcome_fields(report, original_prompt=ORIGINAL, final_prompt=FINAL)
    assert report["outcome"] is None
