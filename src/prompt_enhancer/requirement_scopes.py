"""Explicit named-section count grammar and conservative output boundaries."""

import re
from dataclasses import dataclass

from .criterion_reading import number_candidates
from .protected_blocks import source_data_spans

_NUMBER = r"\d{1,9}|one|two|three|four|five|six|seven|eight|nine|ten"
_COUNT = rf"(?:exactly\s+)?(?P<count>{_NUMBER})\s+(?P<unit>words?|sentences?|lines?|bullets?)"
_AFTER = re.compile(
    _COUNT
    + r"\s+(?:under\s+(?:the\s+)?(?:section\s+)?|in\s+(?:the\s+)?section\s+)(?P<name>[A-Za-z][A-Za-z0-9 _-]{0,80}?)(?:\s+(?:section|heading|list))?(?=\s+and\s+|[.;\n]|$)",
    re.I,
)
_BEFORE = re.compile(
    r"(?:Give|Write|Return|Provide)\s+(?P<name>Section\s+[A-Za-z0-9_-]+)\s+"
    + _COUNT
    + r"(?:\s+and\s+"
    + _COUNT.replace("?P<count>", "?P<next_count>").replace("?P<unit>", "?P<next_unit>")
    + r")?",
    re.I,
)


@dataclass(frozen=True)
class ScopedCount:
    start: int
    end: int
    scope: str
    kind: str
    expected: str


def scoped_counts(prompt: str) -> tuple[ScopedCount, ...]:
    data_spans = source_data_spans(prompt)
    found = []
    for pattern in (_AFTER, _BEFORE):
        prior_end = None
        for match in pattern.finditer(prompt):
            prefix = prompt[: match.start()]
            if pattern is _BEFORE:
                if prefix.strip() and not re.search(r"[.!?]\s+$", prefix):
                    continue
            elif not re.fullmatch(r"\s*(?:Give|Write|Return|Provide)\s+", prefix, re.I):
                if prior_end is None or not re.fullmatch(
                    r"\s+and\s+", prompt[prior_end : match.start()], re.I
                ):
                    continue
            prior_end = match.end()
            if any(a < match.end() and match.start() < b for a, b in data_spans):
                continue
            name = match["name"].strip()
            if (
                pattern is _AFTER
                and re.search(
                    r"\bunder\s+(?:the\s+)?$",
                    prompt[match.start() : match.start("name")],
                    re.I,
                )
                and not name[0].isupper()
                and not re.search(r"\b(?:section|heading|list)$", match[0], re.I)
            ):
                # Bare lower-case phrases such as "under pressure" do not name
                # a section. Semantic extraction retains any actual instruction.
                continue
            for prefix in ("", "next_"):
                if prefix and "next_count" not in match.groupdict():
                    continue
                raw = match[prefix + "count"]
                if raw is None:
                    continue
                value = number_candidates(raw.casefold()).get(raw.casefold())
                if value is None:
                    continue
                unit = match[prefix + "unit"].casefold().removesuffix("s")
                found.append(
                    ScopedCount(
                        match.start(prefix + "count")
                        if pattern is _BEFORE
                        else match.start(),
                        match.end(prefix + "unit")
                        if pattern is _BEFORE
                        else match.end(),
                        "section:" + name,
                        unit + "_count",
                        str(value),
                    )
                )
    return tuple(found)


def section_text(output: str, scope: str) -> tuple[str | None, str | None]:
    """Require one explicit heading and an unambiguous following boundary."""
    name = scope.removeprefix("section:")
    lines = output.splitlines()
    headings = []
    for index, line in enumerate(lines):
        markdown = re.fullmatch(r"(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*", line)
        plain = re.fullmatch(r"([A-Za-z][A-Za-z0-9 _-]{0,100}):[ \t]*", line)
        if markdown:
            headings.append((index, len(markdown[1]), markdown[2]))
        elif plain:
            headings.append((index, 1, plain[1]))
    targets = [
        item
        for item in headings
        if item[2].strip().casefold() == name.strip().casefold()
    ]
    if not targets:
        if name.casefold() in output.casefold():
            return None, "The named section has no supported explicit heading boundary."
        return None, None  # Explicit section is missing: a known failure.
    if len(targets) != 1 or "```" in output or "~~~" in output:
        return None, "Repeated or fenced section boundaries need a scope decision."
    start, level, _ = targets[0]
    end = next(
        (index for index, depth, _ in headings if index > start and depth <= level),
        len(lines),
    )
    if any(start < index < end for index, _, _ in headings):
        return None, "Nested section boundaries need a scope decision."
    return "".join(line + "\n" for line in lines[start + 1 : end]), None
