"""Deterministic checks for the parts of a success criterion code can verify.

One parse of the criterion answers both questions the grading cascade asks:
which constraints code can evaluate, and which constraints it cannot, so the
two can never disagree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_NUMBER = r"\d{1,3}(?:,\d{3})+|\d+"

_BOUND_OPERATORS = {
    "at most": ("at most", "no more than", "up to", "within", "maximum of", "max of"),
    "under": ("under", "fewer than", "less than", "below"),
    "at least": ("at least", "no fewer than", "no less than", "minimum of", "min of"),
    "more than": ("more than", "over", "above"),
    "exactly": ("exactly",),
}
_OPERATOR_NAMES = {
    phrase: name for name, phrases in _BOUND_OPERATORS.items() for phrase in phrases
}
_WORD_BOUND = re.compile(
    rf"\b(?P<operator>{'|'.join(sorted(map(re.escape, _OPERATOR_NAMES), key=len, reverse=True))})"
    rf"\s+(?P<bound>{_NUMBER})\s+words?\b"
)
_WORD_RANGE = re.compile(
    rf"\bbetween\s+(?P<low>{_NUMBER})\s+and\s+(?P<high>{_NUMBER})\s+words?\b"
)
_COUNTED_UNIT = re.compile(
    rf"\b(?:{_NUMBER})\s+(?P<unit>word|citation|item|sentence|character|step|example|source)s?\b"
)
_EXECUTION = re.compile(
    r"\b(?:execut(?:e|es|ed|ing|ion)|syntax|unit tests?"
    r"|run(?:s|ning)? (?:the |this |your )?code)\b"
    r"|\bcompil(?:e|es|ed|ing)\b(?!\s+(?:a|an|the|your|all|some|any|this|these)\b)"
)


@dataclass(frozen=True, slots=True)
class CriterionCheck:
    """What code verified about an output, and what it could not verify."""

    exact: dict[str, Any] | None
    unsupported: tuple[str, ...]


def _number(text: str) -> int:
    return int(text.replace(",", ""))


def _word_check(
    operator: str, bound: int, count: int, *, upper: int | None = None
) -> dict[str, Any]:
    passed = {
        "at most": lambda: count <= bound,
        "under": lambda: count < bound,
        "at least": lambda: count >= bound,
        "more than": lambda: count > bound,
        "exactly": lambda: count == bound,
        "between": lambda: bound <= count <= (upper if upper is not None else bound),
    }[operator]()
    check: dict[str, Any] = {
        "kind": "word_count",
        "operator": operator,
        "bound": bound,
        "observed": count,
        "passed": passed,
    }
    if upper is not None:
        check["upper_bound"] = upper
    return check


def check_criterion(criterion: str, output: str) -> CriterionCheck:
    """Evaluate every supported constraint in ``criterion`` against ``output``."""
    lowered = criterion.casefold()
    word_count = len(re.findall(r"\b\w+\b", output))
    checks: list[dict[str, Any]] = []
    if "valid json" in lowered:
        try:
            json.loads(output)
        except json.JSONDecodeError:
            passed = False
        else:
            passed = True
        checks.append({"kind": "valid_json", "passed": passed})

    remainder = lowered
    for match in _WORD_RANGE.finditer(lowered):
        checks.append(
            _word_check(
                "between",
                _number(match["low"]),
                word_count,
                upper=_number(match["high"]),
            )
        )
        remainder = remainder.replace(match.group(), " ")
    for match in _WORD_BOUND.finditer(lowered):
        checks.append(
            _word_check(
                _OPERATOR_NAMES[match["operator"]],
                _number(match["bound"]),
                word_count,
            )
        )
        remainder = remainder.replace(match.group(), " ")

    unsupported = [
        f"{unit}s"
        for unit in dict.fromkeys(
            match["unit"] for match in _COUNTED_UNIT.finditer(remainder)
        )
    ]
    if _EXECUTION.search(lowered):
        unsupported.append("code_execution")

    if not checks:
        exact = None
    elif len(checks) == 1:
        exact = checks[0]
    else:
        exact = {
            "kind": "all_of",
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
        }
    return CriterionCheck(exact=exact, unsupported=tuple(unsupported))
