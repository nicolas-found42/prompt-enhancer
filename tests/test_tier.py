import json

import pytest

from prompt_enhancer import Tier


def test_tier_reads_user_input_and_defaults_to_standard() -> None:
    assert Tier.parse(" Deep ") is Tier.DEEP
    assert Tier.parse(Tier.FAST) is Tier.FAST
    assert Tier.parse(None) is Tier.STANDARD
    with pytest.raises(ValueError, match="tier must be one of: fast, standard, deep"):
        Tier.parse("turbo")


def test_tier_behaves_as_its_string_in_reports() -> None:
    assert Tier.DEEP == "deep"
    assert json.dumps({"tier": Tier.FAST}) == '{"tier": "fast"}'


@pytest.mark.parametrize(
    ("tier", "candidates", "models", "samples", "rounds"),
    [(Tier.FAST, 3, 2, 1, 1), (Tier.STANDARD, 4, 3, 2, 2), (Tier.DEEP, 6, 5, 3, 3)],
)
def test_each_tier_owns_its_budget(
    tier: Tier, candidates: int, models: int, samples: int, rounds: int
) -> None:
    budget = tier.budget

    assert (budget.candidates, budget.models, budget.samples, budget.max_rounds) == (
        candidates,
        models,
        samples,
        rounds,
    )
    assert tier.max_rounds == rounds
    assert tier.weak_model_evaluations == candidates * models * samples * rounds
