"""Public behavior tests for strategy search, weak-panel evaluation, and ranking."""

from prompt_enhancer.grading import grade_candidate
from prompt_enhancer.runner import PanelResult, run_candidate_panel, run_candidates
from prompt_enhancer.selector import rank_candidates
from prompt_enhancer.strategies import (
    CandidateDraft,
    RewriteStrategy,
    search_strategies,
)


def test_search_strategies_returns_multiple_named_candidates_and_tier_budgets():
    calls = []

    def writer(request):
        calls.append(request)
        return {strategy.name: f"rewrite for {strategy.name}" for strategy in request.strategies}

    fast = search_strategies("Explain the result", tier="fast", writer=writer)
    deep = search_strategies("Explain the result", tier="deep", writer=writer)

    assert len(fast.candidates) == 3
    assert len(deep.candidates) == 6
    assert len(calls) == 2
    assert len({candidate.strategy_kind for candidate in deep.candidates}) == 2
    assert any(candidate.is_crutch for candidate in deep.candidates)
    assert all(candidate.candidate_id.startswith("candidate-") for candidate in deep.candidates)
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
        "problem_sentences": [{"sentence": {"text": "Plan the trip"}, "kind": "vagueness"}],
    }

    def writer(request):
        captured.append(request.to_dict())
        return {strategy.name: request.prompt for strategy in request.strategies}

    search_strategies("Plan the trip", diagnosis=diagnosis, tier="fast", writer=writer)

    assert captured[0]["diagnosis"] == diagnosis


def test_runner_is_parallel_order_stable_and_reproducible():
    seen = []

    def execute(request):
        seen.append(request.to_dict())
        return f"{request.model}:{request.sample}:{request.seed}"

    first = run_candidates(
        [{"id": "rewrite", "prompt": "Rewrite this"}],
        ["weak-a", "weak-b"],
        execute,
        original="Original",
        samples=2,
        run_seed=17,
        max_workers=4,
    )
    second = run_candidates(
        [{"id": "rewrite", "prompt": "Rewrite this"}],
        ["weak-a", "weak-b"],
        execute,
        original="Original",
        samples=2,
        run_seed=17,
        max_workers=4,
    )

    assert first.to_dict() == second.to_dict()
    assert first.candidate_ids == ("original", "rewrite")
    assert len(first.results) == 8
    assert all(len({result.seed for result in first.by_candidate("rewrite")}) == 4 for _ in [0])
    assert [result.sample for result in first.by_candidate("rewrite")] == [0, 1, 0, 1]
    assert all(result.seed != 0 for result in first.results)
    assert len(seen) == 16


def test_runner_reads_normalized_responses_text_before_raw_output():
    class Gateway:
        def chat(self, model, messages, **kwargs):
            return {
                "output": [{"type": "reasoning", "summary": []}],
                "choices": [{"message": {"content": "OK"}}],
            }

    result = run_candidate_panel("Return OK", ["muse-spark-1.3-contributor"], Gateway())

    assert result.results[0].output == "OK"


def test_grading_reports_per_model_worst_mean_and_sample_spread():
    candidate = CandidateDraft(
        "rewrite",
        "A prompt",
        RewriteStrategy("add_context", "safe", "context"),
    )
    panel = [
        PanelResult("rewrite", "weak-a", 0, 1, "pass"),
        PanelResult("rewrite", "weak-a", 1, 2, "fail"),
        PanelResult("rewrite", "weak-b", 0, 3, "pass"),
        PanelResult("rewrite", "weak-b", 1, 4, "pass"),
    ]

    report = grade_candidate(
        candidate,
        panel,
        lambda request: {"passed": request.output == "pass"},
        threshold=0.5,
    )

    assert report.per_model_pass_rates == {"weak-a": 0.5, "weak-b": 1.0}
    assert report.worst_model_pass_rate == 0.5
    assert report.mean_pass_rate == 0.75
    assert report.sample_spread == 0.5
    assert report.to_dict()["per_model_pass_rates"] == report.per_model


def test_selector_orders_worst_mean_spread_then_length_and_reports_reasons():
    baseline = {"id": "original", "text": "x" * 10, "grade": {"worst": 0.4, "mean": 0.4, "spread": 0.1}}
    candidates = [
        {"id": "long", "text": "x" * 20, "grade": {"worst": 0.8, "mean": 0.6, "spread": 0.1}},
        {"id": "short", "text": "x" * 5, "grade": {"worst": 0.8, "mean": 0.6, "spread": 0.1}},
        {"id": "mean", "text": "x" * 30, "grade": {"worst": 0.8, "mean": 0.7, "spread": 0.2}},
        {"id": "worst", "text": "x" * 30, "grade": {"worst": 0.9, "mean": 0.1, "spread": 0.4}},
        {"id": "blocked", "text": "x" * 5, "grade": {"worst": 1.0, "mean": 1.0, "spread": 0.0}, "eligible": False, "rejection_reasons": ["fidelity failed"]},
    ]

    result = rank_candidates(
        baseline,
        candidates,
        strong_check={"outcomes": {"blocked": {"passed": False, "reason": "strong check failed"}}},
    )

    assert result.selected_candidate_id == "worst"
    assert [item.candidate.candidate_id for item in result.ranked if item.selected] == ["worst"]
    assert result.rejection_reasons["blocked"] == ("fidelity failed", "strong check failed")
    assert result.to_dict()["original_score"]["worst"] == 0.4
    assert result.to_dict()["winner_score"]["worst"] == 0.9


def test_selector_does_not_claim_an_unrun_strong_check_failed():
    baseline = {"id": "original", "text": "Original", "grade": {"worst": 0.2}}
    candidate = {
        "id": "blocked", "text": "Rewrite", "grade": {"worst": 0.9},
        "eligible": False, "rejection_reasons": ["candidate failed fidelity checks"],
        "metadata": {"fidelity": {"passed": False}},
    }

    result = rank_candidates(baseline, [candidate], strong_check={"passed_candidates": ()})

    assert result.rejection_reasons["blocked"] == ("candidate failed fidelity checks",)
    assert result.ranked[0].to_dict()["metadata"]["fidelity"] == {"passed": False}


def test_selector_keeps_original_when_no_candidate_beats_it():
    result = rank_candidates(
        {"id": "original", "text": "original", "grade": {"worst": 0.8, "mean": 0.8, "spread": 0.0}},
        [{"id": "candidate", "text": "a longer candidate", "grade": {"worst": 0.8, "mean": 0.8, "spread": 0.0}}],
    )

    assert result.original_kept
    assert result.selected_candidate_id is None
    assert result.final_prompt == "original"
    assert "does not beat the original" in result.rejection_reasons["candidate"][0]


def test_selector_prefers_shorter_prompt_when_grades_are_equal():
    result = rank_candidates(
        {"id": "original", "text": "an original prompt", "grade": {"worst": 0.8, "mean": 0.8, "spread": 0.0}},
        [{"id": "candidate", "text": "shorter", "grade": {"worst": 0.8, "mean": 0.8, "spread": 0.0}}],
    )

    assert result.selected_candidate_id == "candidate"
