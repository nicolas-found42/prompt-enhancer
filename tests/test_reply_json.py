from __future__ import annotations

import json

import pytest

from prompt_enhancer.reply_json import parse_reply_json


def test_prose_before_the_json_is_ignored() -> None:
    reply = 'Here are the success tests:\n\n{"tests": [{"question": "Is it short?"}]}'

    assert parse_reply_json(reply) == {"tests": [{"question": "Is it short?"}]}


def test_braces_in_the_prose_before_the_json_do_not_hide_it() -> None:
    reply = 'Fill each {placeholder} as asked, then use this: {"ok": true}'

    assert parse_reply_json(reply) == {"ok": True}


@pytest.mark.parametrize(
    "reply",
    [
        'Sure: {"tests": [{"question": "a"}, {"question": "b"',
        'Sure: {"tests": [{"question": "a"}, {"question": "b',
        'Sure: {"tests": [{"question": "a"}, ',
    ],
    ids=["ends-after-value", "ends-inside-string", "ends-after-comma"],
)
def test_truncated_json_is_not_mistaken_for_its_complete_inner_object(
    reply: str,
) -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_reply_json(reply)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('  {"a": 1}\n', {"a": 1}),
        ("[1, 2]", [1, 2]),
        ('"just a string"', "just a string"),
        ("42", 42),
    ],
    ids=["object", "padded", "array", "string", "number"],
)
def test_a_reply_that_is_entirely_json_parses_as_json_loads_would(
    reply: str, expected: object
) -> None:
    assert parse_reply_json(reply) == expected


def test_a_fenced_block_with_prose_on_both_sides_is_read() -> None:
    reply = 'Sure!\n```json\n{"gaps": {"tone": []}}\n```\nLet me know if you need more.'

    assert parse_reply_json(reply) == {"gaps": {"tone": []}}


def test_an_array_after_prose_is_read() -> None:
    assert parse_reply_json("The options are: [1, 2, 3]. Enjoy.") == [1, 2, 3]


def test_braces_inside_string_values_do_not_end_the_value_early() -> None:
    reply = 'Result: {"text": "use {name} and [index] as shown"} (done)'

    assert parse_reply_json(reply) == {"text": "use {name} and [index] as shown"}


def test_accept_skips_values_of_the_wrong_shape() -> None:
    reply = 'Example: {"example": true}\nAnswer: {"tests": []}'

    value = parse_reply_json(
        reply, accept=lambda v: isinstance(v, dict) and "tests" in v
    )

    assert value == {"tests": []}


@pytest.mark.parametrize(
    "reply", ["", "   ", "I cannot help with that.", "Done {not json}"]
)
def test_a_reply_without_json_raises_json_decode_error(reply: str) -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_reply_json(reply)


def test_a_reply_with_only_unaccepted_values_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_reply_json('Here: {"a": 1}', accept=lambda v: False)


def test_a_reply_that_is_entirely_json_is_returned_whatever_its_shape() -> None:
    assert parse_reply_json("[1, 2]", accept=lambda v: isinstance(v, dict)) == [1, 2]


def test_repair_closes_an_unfinished_value_that_follows_prose() -> None:
    reply = 'Here you go: {"tests": [{"question": "a"}'

    def closing(unfinished: str) -> str:
        assert unfinished.startswith('{"tests"')
        return "]}"

    assert parse_reply_json(reply, repair=closing) == {"tests": [{"question": "a"}]}


def test_repair_that_adds_nothing_leaves_truncation_an_error() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_reply_json('Here: {"tests": [{"question": "a"}', repair=lambda _text: "")


@pytest.mark.parametrize(
    "stray",
    ["}", ")", ".", "}}", " ]"],
    ids=["brace", "paren", "dot", "two-braces", "bracket"],
)
def test_complete_json_followed_by_stray_characters_is_read(stray: str) -> None:
    reply = '{"tests": [{"question": "Is it short?"}]}' + stray

    assert parse_reply_json(reply) == {"tests": [{"question": "Is it short?"}]}
