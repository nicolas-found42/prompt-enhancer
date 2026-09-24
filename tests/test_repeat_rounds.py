from __future__ import annotations

import pytest

from prompt_enhancer.config import Settings
from prompt_enhancer.repeat import RepeatCoordinator, Tier
from prompt_enhancer.rounds import CandidateFailure, RoundOutcome, RoundPlan

GAP = {"confirmed_gaps": [{"key": "goal"}], "problem_sentences": []}


def _outcome(
    request,
    *,
    failures: tuple[CandidateFailure, ...] = (),
    final_prompt: str = "original",
    diagnosis: dict = GAP,
) -> RoundOutcome:
    plan = RoundPlan(
        prompt="original",
        working_prompt="original",
        run_id=request.run_id,
        tier=request.tier.value,
        seed=0,
        diagnosis=diagnosis,
        assumptions=(),
        settings=Settings(),
        faithfulness_threshold=0.8,
        writer_instruction_version=2,
        prior_failures=request.prior_failures,
    )
    return RoundOutcome(
        plan=plan,
        status="no_change" if final_prompt == "original" else "improved",
        summary="",
        final_prompt=final_prompt,
        original_kept=final_prompt == "original",
        cost={"total": 0.001},
        timing={"elapsed_ms": 12},
        failures=failures,
    )


def _failure(reason: str = "candidate-1 missed the required output format") -> CandidateFailure:
    return CandidateFailure(
        candidate_id="candidate-1",
        strategy="specify_output_format",
        reasons=(reason,),
        weak_pass_rates={"weak-a": 0.25},
        strong_pass_rate=1.0,
    )


def _run(tier: Tier, execute_round, run_id: str = "run"):
    return RepeatCoordinator().run(run_id=run_id, prompt="original", tier=tier, execute_round=execute_round)


@pytest.mark.parametrize(("tier", "expected_rounds"), [(Tier.FAST, 1), (Tier.STANDARD, 2), (Tier.DEEP, 3)])
def test_each_tier_honors_its_max_round_limit(tier: Tier, expected_rounds: int) -> None:
    requests = []

    def execute_round(request):
        requests.append(request)
        return _outcome(request, failures=(_failure(),))

    result = _run(tier, execute_round)

    assert tier.max_rounds == expected_rounds
    assert len(requests) == expected_rounds
    assert [request.tier_round for request in requests] == list(range(1, expected_rounds + 1))
    assert all(request.max_rounds == expected_rounds for request in requests)
    assert len(result.history) == expected_rounds


def test_later_round_receives_previous_candidate_failures() -> None:
    requests = []

    def execute_round(request):
        requests.append(request)
        if request.round_number == 1:
            return _outcome(request, failures=(_failure("candidate-2 lost on the smallest weak model"),))
        return _outcome(request)

    result = _run(Tier.STANDARD, execute_round)

    assert requests[0].prior_failures == ()
    assert requests[1].prior_round_failures[0].candidate_id == "candidate-1"
    assert "candidate-2 lost on the smallest weak model" in requests[1].prior_failures[0]
    assert requests[1].prior_failures == (requests[1].prior_round_failures[0].summary,)
    assert result.history[0].candidate_failures[0].weak_pass_rates == {"weak-a": 0.25}
    assert result.history[0].cost == {"total": 0.001}
    assert result.history[0].timing == {"elapsed_ms": 12}


def test_a_round_without_failures_ends_the_tier_early() -> None:
    requests = []

    def execute_round(request):
        requests.append(request)
        return _outcome(request)

    _run(Tier.DEEP, execute_round)

    assert len(requests) == 1


def test_lower_tier_no_change_result_offers_a_more_expensive_deep_pass() -> None:
    payload = _run(Tier.FAST, lambda request: _outcome(request), run_id="run-offer").as_payload()
    offer = payload["report"]["offer_deep"]

    assert payload["final_prompt"] == "original"
    assert payload["original_kept"] is True
    assert offer["from_tier"] == "fast"
    assert offer["to_tier"] == "deep"
    assert offer["target_max_rounds"] == 3
    assert "Up to 3 Deep rounds" in offer["expected_effort"]
    assert "Higher expected model cost" in offer["expected_cost_change"]
    assert offer["expected_evaluation_multiplier"] > 1
    assert offer["state"] == "offered"
    assert payload["report"]["history"][0]["round_number"] == 1


def test_round_without_confirmed_gaps_is_not_offered_deep() -> None:
    no_gaps = {"confirmed_gaps": [], "problem_sentences": []}
    result = _run(Tier.FAST, lambda request: _outcome(request, diagnosis=no_gaps))

    assert result.as_payload()["report"]["offer_deep"] is None


def test_improved_result_does_not_offer_deep() -> None:
    result = _run(Tier.FAST, lambda request: _outcome(request, final_prompt="Summarize in three bullets."))

    assert result.as_payload()["report"]["offer_deep"] is None


def test_deep_pass_keeps_run_id_and_appends_tiered_round_history() -> None:
    coordinator = RepeatCoordinator()
    lower = coordinator.run(
        run_id="run-same",
        prompt="Repair the failing test.",
        tier=Tier.STANDARD,
        execute_round=lambda request: _outcome(request, failures=(_failure(),)),
        questions=["Which suite?"],
        answers={"q1": "unit"},
        assumptions=["Use the default formatter"],
    )
    persisted_run = {**lower.as_payload(), "original_prompt": "Repair the failing test."}
    deep_requests = []

    def deep_round(request):
        deep_requests.append(request)
        return _outcome(request, failures=(_failure(f"deep candidate {request.tier_round} still failed"),))

    payload = coordinator.deep_pass(persisted_run, deep_round).as_payload()

    assert payload["run_id"] == "run-same"
    assert payload["tier"] == "deep"
    assert len(deep_requests) == 3
    assert all(request.workflow.run_id == "run-same" for request in deep_requests)
    assert all(request.workflow.original_prompt == "Repair the failing test." for request in deep_requests)
    assert deep_requests[0].workflow.answers == {"q1": "unit"}
    assert [
        (entry["tier"], entry["tier_round"], entry["round_number"])
        for entry in payload["report"]["history"]
    ] == [
        ("standard", 1, 1),
        ("standard", 2, 2),
        ("deep", 1, 3),
        ("deep", 2, 4),
        ("deep", 3, 5),
    ]
    assert payload["report"]["offer_deep"] is None
    assert payload["report"]["escalation"]["status"] == "completed"
    assert payload["report"]["escalation"]["run_id"] == "run-same"
    assert payload["report"]["escalation"]["source_tier"] == "standard"
    assert payload["report"]["escalation"]["target_tier"] == "deep"


def test_deep_pass_from_a_stored_run_tells_the_writer_each_failure_once() -> None:
    coordinator = RepeatCoordinator()
    lower = coordinator.run(
        run_id="run-stored",
        prompt="original",
        tier=Tier.FAST,
        execute_round=lambda request: _outcome(request, failures=(_failure(),)),
    )
    deep_requests = []

    def deep_round(request):
        deep_requests.append(request)
        return _outcome(request)

    coordinator.deep_pass(lower.as_payload(), deep_round)

    assert deep_requests[0].prior_failures == (_failure().summary,)
