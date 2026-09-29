"""The labelled criterion set and the offline half of reading criteria with a model."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_criterion_reading import (
    Reading,
    load_cases,
    number_candidates,
    outcome,
    regex_check,
    resolve,
    score,
)

OPERATORS = {"at most", "under", "at least", "more than", "exactly", "between"}


def _reading(**overrides: object) -> Reading:
    fields: dict[str, object] = {
        "kind": "word_count",
        "op": "under",
        "bound": "100",
        "low": "none",
        "high": "none",
        "noul": {
            "partial": 0.1,
            "conditional": 0.1,
            "negated": 0.1,
            "approximate": 0.1,
        },
    }
    return Reading(**{**fields, **overrides})  # type: ignore[arg-type]


def test_fixture_has_50_development_and_100_heldout_unique_criteria() -> None:
    cases = load_cases("all")

    assert Counter(case["split"] for case in cases) == {
        "development": 50,
        "heldout": 100,
    }
    assert len({case["criterion"] for case in cases}) == len(cases)
    assert len({case["id"] for case in cases}) == len(cases)


def test_every_label_is_a_supported_check_or_null() -> None:
    for case in load_cases("all"):
        expected = case["expected"]
        if expected is None:
            continue
        if expected["kind"] == "valid_json":
            assert set(expected) == {"kind", "negated"}
            continue
        assert expected["kind"] in {"word_count", "sentence_count"}
        assert expected["operator"] in OPERATORS
        assert (expected["operator"] == "between") == (
            expected["upper_bound"] is not None
        )
        if expected["upper_bound"] is not None:
            assert expected["bound"] < expected["upper_bound"]


def test_regex_baseline_reproduces_the_issue_on_the_development_split() -> None:
    cases = load_cases("development")

    tally = score(cases, {c["id"]: regex_check(c["criterion"]) for c in cases})

    assert (
        tally["correct_check"],
        tally["correct_abstain"],
        tally["missed"],
        tally["wrong_confident"],
    ) == (8, 11, 24, 7)


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("at most 1,000 words", {"1,000": 1000}),
        ("under one hundred words", {"one hundred": 100}),
        ("twenty-five words or fewer", {"twenty-five": 25}),
        ("one hundred and fifty words", {"one hundred and fifty": 150}),
        (
            "between two hundred and three hundred words",
            {"two hundred": 200, "three hundred": 300},
        ),
        ("between three and six sentences", {"three": 3, "six": 6}),
        ("the 5-paragraph essay in 500 words", {"5": 5, "500": 500}),
        ("no one is someone", {"one": 1}),
    ],
)
def test_number_candidates_are_valued_in_code(text: str, spans: dict[str, int]) -> None:
    assert number_candidates(text) == spans


def test_resolve_builds_a_check_from_the_selected_span_and_operator() -> None:
    check = resolve(
        _reading(op="at most", bound="two hundred"),
        {"two hundred": 200},
        (0.5, 0.5),
    )

    assert check == {
        "kind": "word_count",
        "operator": "at most",
        "bound": 200,
        "upper_bound": None,
    }


def test_resolve_orders_a_range_and_needs_two_distinct_ends() -> None:
    candidates = {"3": 3, "5": 5}

    swapped = resolve(_reading(op="between", low="5", high="3"), candidates, (0.5, 0.5))
    same = resolve(_reading(op="between", low="5", high="5"), candidates, (0.5, 0.5))

    assert swapped is not None
    assert (swapped["bound"], swapped["upper_bound"]) == (3, 5)
    assert same is None


@pytest.mark.parametrize("judgment", ["partial", "conditional", "approximate"])
def test_resolve_leaves_a_flagged_criterion_unresolved(judgment: str) -> None:
    noul = {"partial": 0.1, "conditional": 0.1, "negated": 0.1, "approximate": 0.1}
    reading = _reading(noul={**noul, judgment: 0.5})

    assert resolve(reading, {"100": 100}, (0.5, 0.5)) is None
    assert resolve(reading, {"100": 100}, (0.6, 0.6)) is not None


def test_resolve_never_invents_a_bound_the_criterion_does_not_contain() -> None:
    assert resolve(_reading(bound="none"), {"100": 100}, (0.5, 0.5)) is None
    assert resolve(_reading(bound="250"), {"100": 100}, (0.5, 0.5)) is None
    assert resolve(_reading(kind="other"), {"100": 100}, (0.5, 0.5)) is None


def test_resolve_negates_json_and_abstains_between_the_band_edges() -> None:
    noul = {"partial": 0.0, "conditional": 0.0, "approximate": 0.0}
    reading = _reading(kind="valid_json", op="none", bound="none")

    def with_negation(probability: float) -> Reading:
        return _reading(
            kind=reading.kind,
            op=reading.op,
            bound=reading.bound,
            noul={**noul, "negated": probability},
        )

    band = (0.2, 0.8)
    assert resolve(with_negation(0.1), {}, band) == {
        "kind": "valid_json",
        "negated": False,
    }
    assert resolve(with_negation(0.48), {}, band) is None
    assert resolve(with_negation(0.9), {}, band) == {
        "kind": "valid_json",
        "negated": True,
    }


def test_outcome_separates_wrong_from_missed_and_correctly_unchecked() -> None:
    expected = {"kind": "valid_json", "negated": False}

    assert outcome(expected, expected) == "correct_check"
    assert outcome(None, None) == "correct_abstain"
    assert outcome(None, expected) == "missed"
    assert outcome(expected, None) == "wrong_confident"
    assert outcome({"kind": "valid_json", "negated": True}, expected) == (
        "wrong_confident"
    )
