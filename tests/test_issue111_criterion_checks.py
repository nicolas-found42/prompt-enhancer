"""Deterministic criterion checks agree on what they can and cannot verify."""

from __future__ import annotations

import pytest

from prompt_enhancer.criterion_checks import check_criterion


def _words(count: int) -> str:
    return " ".join(["word"] * count)


@pytest.mark.parametrize(
    ("criterion", "count", "passed"),
    [
        ("The answer is under 100 words", 99, True),
        ("The answer is under 100 words", 100, False),
        ("Uses fewer than 10 words", 9, True),
        ("Uses fewer than 10 words", 10, False),
        ("The answer has at most 1,000 words", 1000, True),
        ("The answer has at most 1,000 words", 1001, False),
        ("Stays within 20 words", 20, True),
        ("Stays within 20 words", 21, False),
        ("Has at least 3 words", 3, True),
        ("Has at least 3 words", 2, False),
        ("Has more than 3 words", 3, False),
        ("Has more than 3 words", 4, True),
        ("Has exactly 5 words", 5, True),
        ("Has exactly 5 words", 6, False),
        ("Has between 3 and 5 words", 2, False),
        ("Has between 3 and 5 words", 3, True),
        ("Has between 3 and 5 words", 5, True),
        ("Has between 3 and 5 words", 6, False),
    ],
)
def test_word_bounds_pass_and_fail_at_their_boundaries(
    criterion: str, count: int, passed: bool
) -> None:
    check = check_criterion(criterion, _words(count))

    assert check.unsupported == ()
    assert check.exact is not None
    assert check.exact["kind"] == "word_count"
    assert check.exact["observed"] == count
    assert check.exact["passed"] is passed


def test_word_bound_records_operator_and_numeric_bound() -> None:
    check = check_criterion("The answer has at most 1,000 words", _words(3))

    assert check.exact is not None
    assert check.exact["operator"] == "at most"
    assert check.exact["bound"] == 1000


def test_valid_json_is_checked() -> None:
    good = check_criterion("The response is valid JSON", '{"a": 1}')
    bad = check_criterion("The response is valid JSON", "not json")

    assert good.exact == {"kind": "valid_json", "passed": True}
    assert bad.exact == {"kind": "valid_json", "passed": False}


def test_two_constraints_are_both_evaluated() -> None:
    criterion = "Response is at most 50 words and valid JSON"

    both = check_criterion(criterion, '{"a": 1}')
    bad_json = check_criterion(criterion, "one two three")
    too_long = check_criterion(criterion, '{"a": "' + _words(60) + '"}')

    assert both.exact is not None and both.exact["passed"] is True
    assert bad_json.exact is not None and bad_json.exact["passed"] is False
    assert too_long.exact is not None and too_long.exact["passed"] is False
    assert [item["kind"] for item in both.exact["checks"]] == [
        "valid_json",
        "word_count",
    ]
    assert too_long.exact["checks"][0]["passed"] is True
    assert too_long.exact["checks"][1]["passed"] is False


@pytest.mark.parametrize(
    ("criterion", "unit"),
    [
        ("Lists at least 3 sources", "sources"),
        ("Does the answer include 12 citations?", "citations"),
        ("Has 2 examples", "examples"),
        ("Uses no more than 5 steps", "steps"),
        ("Has a 50 words summary", "words"),
    ],
)
def test_uncountable_units_stay_unsupported_and_name_the_unit(
    criterion: str, unit: str
) -> None:
    check = check_criterion(criterion, "x y z")

    assert check.exact is None
    assert check.unsupported == (unit,)


def test_unsupported_unit_beside_a_supported_check_is_still_reported() -> None:
    check = check_criterion("At most 50 words and lists 3 sources", "x y z")

    assert check.exact is not None and check.exact["kind"] == "word_count"
    assert check.unsupported == ("sources",)


@pytest.mark.parametrize(
    "criterion",
    [
        "The code should compile without errors",
        "Does the code compile?",
        "The snippet executes successfully",
        "The syntax is correct",
        "Includes a passing unit test",
        "Shows how to run code",
    ],
)
def test_code_execution_criteria_are_unsupported(criterion: str) -> None:
    check = check_criterion(criterion, "x")

    assert check.exact is None
    assert check.unsupported == ("code_execution",)


@pytest.mark.parametrize(
    "criterion",
    [
        "Compile a list of the sources used",
        "The answer compiles the key points",
        "The answer is friendly",
    ],
)
def test_ordinary_wording_does_not_require_a_deterministic_check(
    criterion: str,
) -> None:
    check = check_criterion(criterion, "x")

    assert check.exact is None
    assert check.unsupported == ()
