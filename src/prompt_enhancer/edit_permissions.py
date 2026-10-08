"""Mechanical edits authorized explicitly by the source, without normalization."""

import json
import re
import unicodedata
from collections import Counter
from typing import Any

_DIRECTIVE = re.compile(
    r"(?is)^(?:improve|correct|fix|change)\s+"
    r"(?P<permissions>(?:punctuation|capitalization|case|word order)"
    r"(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)(?:punctuation|capitalization|case|word order)){0,3})"
    r"\s+only,?\s+(?P<preserve>preserving\s+every\s+word(?:[^:\n]*))\s*:\s*(?P<text>.+)$"
)


def edit_contract(prompt: str) -> tuple[str, str] | None:
    match = _DIRECTIVE.fullmatch(prompt)
    if match is None:
        return None
    text = match["text"]
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1]
    permissions = re.findall(
        r"punctuation|capitalization|case|word order", match["permissions"].casefold()
    )
    preserve = match["preserve"].casefold()
    # Explicit preservation narrows the authorized operation; no operation is
    # inferred merely from an evaluator asking whether words were preserved.
    if "case" in preserve and any(p in permissions for p in ("capitalization", "case")):
        return (
            json.dumps({"text": text, "permissions": permissions}),
            "The source both permits and preserves case; its interpretation needs clarification.",
        )
    if "order" in preserve and "word order" in permissions:
        return (
            json.dumps({"text": text, "permissions": permissions}),
            "The source both permits and preserves order; its interpretation needs clarification.",
        )
    return json.dumps({"text": text, "permissions": permissions}), ""


def check_edit(expected: str, output: Any, uncertainty: str | None) -> tuple[str, str]:
    contract = json.loads(expected)
    source = contract["text"]
    permissions = contract["permissions"]
    if uncertainty:
        return "untestable", uncertainty
    if not isinstance(output, str):
        return "failed", "The output must be text for the source-declared edit."
    # Complex word boundaries and case expansions require an explicit convention.
    if any(ord(ch) > 127 for ch in source + output) or re.search(
        r"\w['./-]\w", source + "\n" + output
    ):
        return (
            "untestable",
            "This tokenization or case convention is unsupported; the source edit remains unresolved.",
        )

    def normalize(text: str) -> str:
        if "punctuation" in permissions:
            text = "".join(
                ch for ch in text if not unicodedata.category(ch).startswith("P")
            )
        if "capitalization" in permissions or "case" in permissions:
            text = text.lower()
        return text

    original, changed = normalize(source), normalize(output)
    if "word order" in permissions:
        original_words = re.findall(r"\w+", original)
        changed_words = re.findall(r"\w+", changed)
        passed = Counter(original_words) == Counter(changed_words) and re.sub(
            r"\w+", "", original
        ) == re.sub(r"\w+", "", changed)
    else:
        passed = original == changed
    return (
        "tested" if passed else "failed",
        "Only the source-declared edits may change; unauthorized word, case or order changes fail independently of Jev judgments.",
    )
