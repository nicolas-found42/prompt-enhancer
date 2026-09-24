"""Properties of round evidence persisted between optimization rounds."""

from hypothesis import given
from hypothesis import strategies as st

from prompt_enhancer.rounds import CandidateFailure

pass_rate = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
optional_pass_rate = st.one_of(st.none(), pass_rate)
optional_text = st.one_of(st.none(), st.text())


@given(
    candidate_id=st.text(min_size=1),
    strategy=optional_text,
    reasons=st.lists(st.text(min_size=1), max_size=5).map(tuple),
    weak_pass_rates=st.dictionaries(st.text(min_size=1), pass_rate, max_size=5),
    strong_pass_rate=optional_pass_rate,
    mean_pass_rate=optional_pass_rate,
    worst_pass_rate=optional_pass_rate,
    sample_spread=optional_pass_rate,
    candidate_prompt=optional_text,
)
def test_candidate_failure_round_trips_through_report(
    candidate_id: str,
    strategy: str | None,
    reasons: tuple[str, ...],
    weak_pass_rates: dict[str, float],
    strong_pass_rate: float | None,
    mean_pass_rate: float | None,
    worst_pass_rate: float | None,
    sample_spread: float | None,
    candidate_prompt: str | None,
) -> None:
    failure = CandidateFailure(
        candidate_id=candidate_id,
        strategy=strategy,
        reasons=reasons,
        weak_pass_rates=weak_pass_rates,
        strong_pass_rate=strong_pass_rate,
        mean_pass_rate=mean_pass_rate,
        worst_pass_rate=worst_pass_rate,
        sample_spread=sample_spread,
        candidate_prompt=candidate_prompt,
    )

    restored = CandidateFailure.from_dict(failure.to_dict())

    assert restored == failure
    assert restored.summary == failure.summary
