"""Public behavior tests for strategy search, weak-panel evaluation, and ranking."""

from prompt_enhancer.grading import GradeReport
from prompt_enhancer.selector import RankingCandidate, rank_candidates
from prompt_enhancer.strategies import search_strategies
from prompt_enhancer.strong_check import StrongCheckOutcome, StrongCheckReport


def test_search_strategies_returns_multiple_named_candidates_and_tier_budgets():
    calls = []

    def writer(request):
        calls.append(request)
        return {
            strategy.name: f"rewrite for {strategy.name}"
            for strategy in request.strategies
        }

    fast = search_strategies("Explain the result", tier="fast", writer=writer)
    deep = search_strategies("Explain the result", tier="deep", writer=writer)

    assert len(fast.candidates) == 3
    assert len(deep.candidates) == 6
    assert len(calls) == 2
    assert len({candidate.strategy_kind for candidate in deep.candidates}) == 2
    assert any(candidate.is_crutch for candidate in deep.candidates)
    assert all(
        candidate.candidate_id.startswith("candidate-") for candidate in deep.candidates
    )
    assert deep.budget.models == 5
    assert deep.budget.samples == 3
    assert deep.budget.max_rounds == 3


def test_search_strategies_uses_previous_failures_to_target_a_strategy():
    result = search_strategies(
        "Make a plan",
        diagnosis={"gaps": ["planning"]},
        tier="deep",
        previous_round_failures=["split_into_steps did not help"],
    )

    assert result.candidates[0].strategy.name == "split_into_steps"
    assert result.previous_failures == ("split_into_steps did not help",)


def test_candidate_writer_receives_confirmed_diagnosis():
    captured = []
    diagnosis = {
        "confirmed_gaps": [{"key": "context", "label": "relevant context"}],
        "problem_sentences": [
            {"sentence": {"text": "Plan the trip"}, "kind": "vagueness"}
        ],
    }

    def writer(request):
        captured.append(request.to_dict())
        return {strategy.name: request.prompt for strategy in request.strategies}

    search_strategies("Plan the trip", diagnosis=diagnosis, tier="fast", writer=writer)

    assert captured[0]["diagnosis"] == diagnosis


def _grade(worst: float, mean: float = 0.0, spread: float = 0.0) -> GradeReport:
    return GradeReport(
        "graded", {"weak": worst}, {"weak": (worst,)}, worst, mean, spread
    )


def _candidate(candidate_id, text, grade, *, eligible=True, reasons=(), metadata=None):
    return RankingCandidate(
        candidate_id,
        text,
        "specify_output_format",
        "safe",
        grade,
        eligible=eligible,
        rejection_reasons=tuple(reasons),
        metadata=metadata or {},
    )


def _original(text, grade):
    return RankingCandidate("original", text, "original", "baseline", grade)


def test_selector_orders_worst_mean_spread_then_length_and_reports_reasons():
    baseline = _original("x" * 10, _grade(0.4, 0.4, 0.1))
    candidates = [
        _candidate("long", "x" * 20, _grade(0.8, 0.6, 0.1)),
        _candidate("short", "x" * 5, _grade(0.8, 0.6, 0.1)),
        _candidate("mean", "x" * 30, _grade(0.8, 0.7, 0.2)),
        _candidate("worst", "x" * 30, _grade(0.9, 0.1, 0.4)),
        _candidate(
            "blocked", "x" * 5, _grade(1.0, 1.0, 0.0), reasons=["fidelity failed"]
        ),
    ]
    strong = StrongCheckReport(
        "strong",
        1.0,
        tuple(
            StrongCheckOutcome(
                candidate.candidate_id,
                candidate.strategy,
                1.0,
                1.0,
                passed=True,
                reason="passed",
            )
            if candidate.candidate_id != "blocked"
            else StrongCheckOutcome(
                "blocked",
                candidate.strategy,
                0.5,
                1.0,
                passed=False,
                reason="strong check failed",
            )
            for candidate in candidates
        ),
    )

    result = rank_candidates(baseline, candidates, strong_check=strong)

    assert result.selected_candidate_id == "worst"
    assert [item.candidate.candidate_id for item in result.ranked if item.selected] == [
        "worst"
    ]
    assert result.rejection_reasons["blocked"] == (
        "fidelity failed",
        "strong check failed",
    )
    assert result.to_dict()["original_score"]["worst"] == 0.4
    assert result.to_dict()["winner_score"]["worst"] == 0.9


def test_selector_does_not_claim_an_unrun_strong_check_failed():
    candidate = _candidate(
        "blocked",
        "Rewrite",
        _grade(0.9),
        eligible=False,
        reasons=["candidate failed fidelity checks"],
        metadata={"fidelity": {"passed": False}},
    )

    result = rank_candidates(
        _original("Original", _grade(0.2)),
        [candidate],
        strong_check=StrongCheckReport("strong", 1.0, ()),
    )

    assert result.rejection_reasons["blocked"] == ("candidate failed fidelity checks",)
    assert result.ranked[0].to_dict()["metadata"]["fidelity"] == {"passed": False}


def test_selector_keeps_original_when_no_candidate_beats_it():
    result = rank_candidates(
        _original("original", _grade(0.8, 0.8)),
        [_candidate("candidate", "a longer candidate", _grade(0.8, 0.8))],
    )

    assert result.original_kept
    assert result.selected_candidate_id is None
    assert result.final_prompt == "original"
    assert "does not beat the original" in result.rejection_reasons["candidate"][0]


def test_selector_prefers_shorter_prompt_when_grades_are_equal():
    result = rank_candidates(
        _original("an original prompt", _grade(0.8, 0.8)),
        [_candidate("candidate", "shorter", _grade(0.8, 0.8))],
    )

    assert result.selected_candidate_id == "candidate"
