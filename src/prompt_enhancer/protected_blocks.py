"""Conservative literal source scopes for explicitly protected fenced blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE = re.compile(
    r"(?m)^(?P<indent> {0,3})(?P<marker>`{3,}|~{3,})(?P<info>[^\r\n]*)\r?\n"
)
_PROTECT = re.compile(
    r"(?im)^Do not change any character in the supplied (?:code|data):[ \t]*\r?\n"
)


@dataclass(frozen=True)
class FencedSource:
    start: int
    end: int
    body: str
    indentation_uncertain: bool = False


def fenced_sources(prompt: str) -> tuple[FencedSource, ...]:
    """Read top-level fences; avoid treating fenced data as new instructions."""
    blocks = []
    position = 0
    while (opening := _FENCE.search(prompt, position)) is not None:
        marker = opening["marker"]
        if marker[0] == "`" and "`" in opening["info"]:
            position = opening.end()
            continue
        closing = re.compile(
            rf"(?m)^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*\r?$"
        ).search(prompt, opening.end())
        end = closing.end() if closing is not None else len(prompt)
        body = prompt[opening.end() : closing.start() if closing is not None else end]
        indent = len(opening["indent"])
        uncertain = indent > 0 and re.search(r"(?m)^ *\t", body) is not None
        if indent:
            body = re.sub(rf"(?m)^ {{0,{indent}}}", "", body)
        blocks.append(FencedSource(opening.start(), end, body, uncertain))
        position = end
    return tuple(blocks)


def protected_sources(prompt: str) -> tuple[tuple[int, int, str, bool], ...]:
    blocks = fenced_sources(prompt)
    protected = []
    for directive in _PROTECT.finditer(prompt):
        if any(block.start <= directive.start() < block.end for block in blocks):
            continue
        block = next(
            (block for block in blocks if block.start == directive.end()), None
        )
        if block is not None and block.body:
            protected.append(
                (directive.start(), block.end, block.body, block.indentation_uncertain)
            )
    return tuple(protected)


def protected_block_status(
    candidate: str, expected: str, *, source_indentation_uncertain: bool = False
) -> str:
    blocks = fenced_sources(candidate)

    def without_indentation(text: str) -> str:
        return re.sub(r"(?m)^[ \t]+", "", text)

    if source_indentation_uncertain:
        return (
            "untestable"
            if expected in candidate
            or any(
                without_indentation(block.body) == without_indentation(expected)
                for block in blocks
            )
            else "failed"
        )
    if any(
        block.body == expected and not block.indentation_uncertain for block in blocks
    ):
        return "tested"
    outside = candidate
    for block in reversed(blocks):
        outside = outside[: block.start] + outside[block.end :]

    if expected in outside or any(
        block.indentation_uncertain
        and without_indentation(block.body) == without_indentation(expected)
        for block in blocks
    ):
        return "untestable"
    return "failed"
