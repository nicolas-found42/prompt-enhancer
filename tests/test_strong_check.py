"""Behavior tests for the strong-reference safety seam."""

from dataclasses import dataclass

from prompt_enhancer.strong_check import (
    StrongCheckPolicy,
    eligible_candidates,
    retain_original,
)


@dataclass
class Candidate:
    id: str
    prompt: str
    strategy: str
    crutch: bool = False
    weak_score: float = 0.0


def test_candidate_that_does_not_regress_is_reported_as_eligible():
    candidate = Candidate("candidate-1", "Be concise.", "output-format")
    calls = []

    def evaluate(prompt, tests):
        calls.append((prompt, tests))
        return {"Explain this.": 0.70, "Be concise.": 0.80}[prompt]

    report = StrongCheckPolicy("glm-5.3-flash").check(
        "Explain this.", [candidate], ("test-a",), evaluate
    )

    outcome = report.outcome_for(candidate.id)
    assert outcome.passed is True
    assert outcome.eligible is True
    assert outcome.regression is False
    assert "strong_check_passed" in outcome.reason
    assert report.original_retained is False
    assert calls == [
        ("Explain this.", ("test-a",)),
        ("Be concise.", ("test-a",)),
    ]


def test_weak_gain_cannot_override_a_strong_regression():
    candidate = Candidate("candidate-1", "Add five steps.", "steps", weak_score=0.99)

    def evaluate(prompt, tests):
        return 0.80 if prompt == "Explain this." else 0.60

    report = StrongCheckPolicy().check(
        "Explain this.", [candidate], ("same-tests",), evaluate
    )

    outcome = report.outcome_for(candidate.id)
    assert candidate.weak_score > 0.80
    assert outcome.passed is False
    assert outcome.eligible is False
    assert "below original" in outcome.reason
    assert report.original_retained is True
    assert report.fallback_reason == (
        "original_retained: no candidate passed the strong check"
    )


def test_crutch_strategy_is_rejected_when_it_regresses():
    candidate = Candidate(
        "candidate-1", "Role-play an expert and think step by step.", "steps", True
    )

    def evaluate(prompt, tests):
        return {"Explain this.": 0.90, candidate.prompt: 0.89}[prompt]

    report = StrongCheckPolicy().check(
        "Explain this.", [candidate], ("same-tests",), evaluate
    )

    outcome = report.outcome_for(candidate.id)
    assert outcome.crutch is True
    assert outcome.passed is False
    assert outcome.eligible is False
    assert "strong_check_regression" in outcome.reason
    assert eligible_candidates([candidate], report) == []


def test_safe_fallback_keeps_original_and_exposes_a_reason():
    candidates = [
        Candidate("candidate-1", "Add a forced role-play.", "role-play", True),
        Candidate("candidate-2", "Add a hard step-by-step scaffold.", "steps", True),
    ]

    def evaluate(prompt, tests):
        return 0.90 if prompt == "Explain this." else 0.70

    report = StrongCheckPolicy().check(
        "Explain this.", candidates, ("same-tests",), evaluate
    )

    assert report.original_retained is True
    assert report.passed_candidates == ()
    assert report.to_dict()["fallback_reason"] == (
        "original_retained: no candidate passed the strong check"
    )
    assert [entry["reason"] for entry in report.to_dict()["candidates"]]
    assert all(entry["eligible"] is False for entry in report.to_dict()["candidates"])

    annotated = retain_original(report, "original_retained: weak gains were unsafe")
    assert annotated.fallback_reason == "original_retained: weak gains were unsafe"
