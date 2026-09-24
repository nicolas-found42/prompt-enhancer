from __future__ import annotations

from typing import Any

import pytest

from prompt_enhancer.repeat import RepeatCoordinator, Tier


def _failed_round(
    failure: str = "candidate-1 missed the required output format",
    *,
    continue_rounds: bool = True,
) -> dict[str, Any]:
    return {
        "final_prompt": "original",
        "original_kept": True,
        "continue_rounds": continue_rounds,
        "candidate_failures": [
            {
                "candidate_id": "candidate-1",
                "strategy": "specify_output_format",
                "reasons": [failure],
                "weak_pass_rates": {"weak-a": 0.25},
                "strong_pass_rate": 1.0,
            }
        ],
        "report": {"status": "no_change"},
        "cost": {"total": 0.001},
        "timing": {"elapsed_ms": 12},
    }


@pytest.mark.parametrize(
    ("tier", "expected_rounds"),
    [
        (Tier.FAST, 1),
        (Tier.STANDARD, 2),
        (Tier.DEEP, 3),
    ],
)
def test_each_tier_honors_its_max_round_limit(
    tier: Tier,
    expected_rounds: int,
) -> None:
    requests = []

    def execute_round(request):
        requests.append(request)
        return _failed_round()

    result = RepeatCoordinator().optimize(
        "Return the release status.",
        {"run_id": "run-limits", "tier": tier},
        execute_round,
    )

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
            return _failed_round("candidate-2 lost on the smallest weak model")
        return _failed_round(continue_rounds=False)

    result = RepeatCoordinator().optimize(
        "Explain the incident.",
        {"run_id": "run-failures", "tier": Tier.STANDARD},
        execute_round,
    )

    assert requests[0].prior_failures == ()
    assert requests[1].prior_round_failures[0].candidate_id == "candidate-1"
    assert "candidate-2 lost on the smallest weak model" in requests[1].prior_failures[0]
    assert requests[1].prior_failures == (requests[1].prior_round_failures[0].summary,)
    assert result.history[0].candidate_failures[0].weak_pass_rates == {"weak-a": 0.25}
    assert result.history[0].cost == {"total": 0.001}
    assert result.history[0].timing == {"elapsed_ms": 12}


def test_lower_tier_no_change_result_offers_a_more_expensive_deep_pass() -> None:
    result = RepeatCoordinator().optimize(
        "Summarize the weekly report.",
        {"run_id": "run-offer", "tier": Tier.FAST},
        lambda request: _failed_round(continue_rounds=False),
    )

    payload = result.as_payload()
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


def test_round_that_declines_deep_is_not_offered_it() -> None:
    def no_gap_round(_request: Any) -> dict[str, Any]:
        outcome = _failed_round(continue_rounds=False)
        return {**outcome, "report": {"status": "no_change", "offer_deep": False}}

    result = RepeatCoordinator().optimize(
        "Summarize the weekly report.",
        {"run_id": "run-no-gaps", "tier": Tier.FAST},
        no_gap_round,
    )

    assert result.as_payload()["report"]["offer_deep"] is None


def test_improved_result_does_not_offer_deep() -> None:
    result = RepeatCoordinator().optimize(
        "Summarize the report.",
        {"run_id": "run-improved", "tier": Tier.FAST},
        lambda request: {
            "final_prompt": "Summarize the report in three bullets.",
            "original_kept": False,
            "continue_rounds": False,
            "candidate_failures": [],
            "report": {"status": "improved"},
        },
    )

    assert result.as_payload()["report"]["offer_deep"] is None


def test_deep_pass_keeps_run_id_and_appends_tiered_round_history() -> None:
    coordinator = RepeatCoordinator()
    lower_result = coordinator.optimize(
        "Repair the failing test.",
        {
            "run_id": "run-same",
            "tier": Tier.STANDARD,
            "questions": ["Which suite?"],
            "answers": {"q1": "unit"},
            "assumptions": ["Use the default formatter"],
        },
        lambda request: _failed_round(),
    )
    persisted_run = {
        **lower_result.as_payload(),
        "original_prompt": "Repair the failing test.",
    }
    deep_requests = []

    def deep_round(request):
        deep_requests.append(request)
        return _failed_round(f"deep candidate {request.tier_round} still failed")

    deep_result = coordinator.deep_pass(persisted_run, deep_round)
    payload = deep_result.as_payload()

    assert payload["run_id"] == "run-same"
    assert payload["tier"] == "deep"
    assert len(deep_requests) == 3
    assert all(request.workflow.run_id == "run-same" for request in deep_requests)
    assert all(
        request.workflow.original_prompt == "Repair the failing test."
        for request in deep_requests
    )
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


def test_facade_rejects_provider_credentials_before_running() -> None:
    def should_not_run(_request):
        raise AssertionError("credential-bearing options must be rejected before a round")

    with pytest.raises(ValueError, match="Provider credentials"):
        RepeatCoordinator().optimize(
            "Write a release note.",
            {"run_id": "run-secret", "tier": Tier.FAST, "openrouter_api_key": "do-not-accept"},
            should_not_run,
        )
