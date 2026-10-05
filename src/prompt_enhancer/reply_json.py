"""Read a model reply as JSON when the model wraps it in prose."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


def parse_reply_json(
    text: str,
    *,
    accept: Callable[[Any], bool] | None = None,
    repair: Callable[[str], str] | None = None,
) -> Any:
    """Return the JSON value in a model reply, ignoring prose around it.

    A reply that is entirely JSON parses exactly as ``json.loads`` would, whatever
    its shape, so callers keep rejecting wrong shapes themselves. Otherwise the
    first complete top-level object or array is returned, skipping braces that
    belong to prose. A reply that stops partway through a value raises instead of
    returning one of the complete values nested inside it. ``accept`` skips values
    of the wrong shape found among prose. ``repair`` receives the text from the
    start of an unfinished value and returns the closing characters it lacks.

    Raises ``json.JSONDecodeError`` when no acceptable value is found.
    """

    whole = _whole_reply(text)
    if whole is not _MISSING:
        return whole

    decoder = json.JSONDecoder()
    repair_text = re.sub(r"\s*```\s*$", "", text).rstrip()
    end_of_text = len(repair_text)
    first_error: json.JSONDecodeError | None = None
    position = 0
    while (start := _next_container(text, position)) is not None:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as error:
            first_error = first_error or error
            if repair is not None and error.pos >= end_of_text - 1:
                suffix = repair(repair_text[start:])
                if suffix:
                    try:
                        value = json.loads(repair_text[start:] + suffix)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if accept is None or accept(value):
                            return value
            position = _container_end(text, start)
            continue
        if accept is None or accept(value):
            return value
        position = end
    raise first_error or json.JSONDecodeError(
        "no JSON object or array in reply", text, 0
    )


_MISSING = object()


def _whole_reply(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _MISSING


def _next_container(text: str, position: int) -> int | None:
    candidates = [
        index
        for index in (text.find("{", position), text.find("[", position))
        if index >= 0
    ]
    return min(candidates) if candidates else None


def _container_end(text: str, start: int) -> int:
    """Skip a failed outer container, including its nested values and strings."""

    stack = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or stack[-1] != {"}": "{", "]": "["}[character]:
                # A mismatched delimiter gives no trustworthy outer boundary.
                return len(text)
            stack.pop()
            if not stack:
                return index + 1
    return len(text)
