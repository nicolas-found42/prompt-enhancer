"""Public behavior tests for strategy search, weak-panel evaluation, and ranking."""

import pytest

from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.grading import GradeReport
from prompt_enhancer.runner import run_candidates
from prompt_enhancer.selector import RankingCandidate, rank_candidates
from prompt_enhancer.strategies import search_strategies
from prompt_enhancer.strong_check import StrongCheckOutcome, StrongCheckReport


def test_search_strategies_uses_settings_backed_fixed_candidate_workload():
    calls = []

    def writer(request):
        calls.append(request)
        return {
            strategy.name: f"rewrite for {strategy.name}"
            for strategy in request.strategies
        }

    default = search_strategies("Explain the result", writer=writer)
    smaller = search_strategies(
        "Explain the result", settings=Settings(candidate_count=3), writer=writer
    )
    oversized = search_strategies(
        "Explain the result", settings=Settings(candidate_count=20), writer=writer
    )

    assert len(default.candidates) == 6
    assert len(smaller.candidates) == 3
    assert len(oversized.candidates) == 6
    assert len(calls) == 3
    assert len(Settings().weak_models) == 5
    assert len(set(Settings().weak_models)) == 5
    assert "muse-spark-1.3-contributor" not in Settings().weak_models
    assert len({candidate.strategy_kind for candidate in default.candidates}) == 2
    assert any(candidate.is_crutch for candidate in default.candidates)
    assert all(
        candidate.candidate_id.startswith("candidate-")
        for candidate in default.candidates
    )
    assert default.budget.candidates == 6
    assert default.budget.models == 5
    assert default.budget.samples == 3


def test_search_strategies_uses_previous_failures_to_target_a_strategy():
    result = search_strategies(
        "Make a plan",
        diagnosis={"gaps": ["planning"]},
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

    search_strategies("Plan the trip", diagnosis=diagnosis, writer=writer)

    assert captured[0]["diagnosis"] == diagnosis


def test_run_candidates_uses_five_settings_models_and_three_samples_by_default():
    calls = []
    gateway = ScriptedGateway(
        chat=lambda _model, messages, *, role, **_kwargs: (
            calls.append((role, messages[0]["content"])) or "answer"
        )
    )
    search = search_strategies("Explain the result")

    result = run_candidates(
        search.candidates[:1], Settings().weak_models, gateway, original="original"
    )

    assert result.models == Settings().weak_models[:5]
    assert len(result.models) == 5
    assert result.samples == 3
    assert len(result.results) == 30  # Original plus one candidate, 5 × 3 each.
    assert len(calls) == 30

    overridden = run_candidates(
        search.candidates[:1],
        Settings().weak_models,
        gateway,
        original="original",
        samples=2,
    )
    assert overridden.samples == 2
    assert len(overridden.results) == 20


def test_run_candidates_rejects_short_or_duplicate_panels_before_gateway_calls():
    calls = []
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: calls.append("called") or "answer"
    )
    candidates = search_strategies("Explain the result").candidates[:1]

    for models in (
        ("one", "two", "three", "four"),
        ("one", "two", "two", "four", "five"),
    ):
        with pytest.raises(ValueError):
            run_candidates(candidates, models, gateway)

    assert calls == []


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


def test_selector_selects_an_equally_good_changed_candidate():
    result = rank_candidates(
        _original("original", _grade(0.8, 0.8)),
        [_candidate("candidate", "a longer candidate", _grade(0.8, 0.8))],
    )

    assert not result.original_kept
    assert result.selected_candidate_id == "candidate"
    assert result.final_prompt == "a longer candidate"
    assert result.rejection_reasons == {}


def test_selector_rejects_a_candidate_that_never_changed_the_prompt():
    result = rank_candidates(
        _original("same", _grade(0.8, 0.8)),
        [_candidate("mirror", "Same.", _grade(1.0, 1.0, 0.0))],
    )

    assert result.original_kept
    assert result.selected_candidate_id is None
    assert result.final_prompt == "same"
    assert "always-improve policy" in result.rejection_reasons["mirror"][0]


def test_selector_keeps_original_when_every_candidate_is_strictly_worse():
    result = rank_candidates(
        _original("original", _grade(0.9, 0.9)),
        [_candidate("candidate", "a candidate", _grade(0.5, 0.5))],
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


def test_semantic_order_applies_only_after_binding_eligibility_gates():
    baseline = _original("original prompt", _grade(0.0, 0.0))
    zero_pass = _candidate("zero", "rewrite with zero passes", _grade(0.0, 0.0))
    fidelity_rejected = _candidate(
        "fidelity",
        "rewrite failing fidelity",
        _grade(0.9, 0.9),
        eligible=False,
        reasons=("candidate failed fidelity checks",),
    )
    semantic_winner = _candidate(
        "semantic", "eligible semantic winner", _grade(0.6, 0.6)
    )
    robust_winner = _candidate("robust", "eligible robust winner", _grade(0.8, 0.7))

    result = rank_candidates(
        baseline,
        [zero_pass, fidelity_rejected, semantic_winner, robust_winner],
        candidate_order={
            "zero": 1.0,
            "fidelity": 0.99,
            "semantic": 0.95,
            "robust": 0.8,
        },
    )

    assert result.selected_candidate_id == "semantic"
    assert [item.candidate.candidate_id for item in result.ranked if item.rank] == [
        "semantic",
        "robust",
    ]
    assert "zero pass" in result.rejection_reasons["zero"][0]
    assert result.rejection_reasons["fidelity"] == ("candidate failed fidelity checks",)


def test_semantic_order_ties_use_existing_robust_then_stable_id_order():
    baseline = _original("original prompt", _grade(0.2, 0.2))
    candidates = [
        _candidate("z-short", "short rewrite", _grade(0.8, 0.7)),
        _candidate("a-long", "a much longer rewrite", _grade(0.9, 0.7)),
        _candidate("m-long", "another longer rewrite", _grade(0.9, 0.7)),
    ]

    result = rank_candidates(
        baseline,
        candidates,
        candidate_order={candidate.candidate_id: 0.8 for candidate in candidates},
    )

    assert [item.candidate.candidate_id for item in result.ranked if item.rank] == [
        "a-long",
        "m-long",
        "z-short",
    ]
