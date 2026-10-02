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
_NEGATION = re.compile(r"(?:\bnot|\bnever|\bcannot|n't)\s+(?:\w+\s+){0,3}$")
_SCOPE = re.compile(r"\b(?:per|each|every)\b")
# "compile" only means running a compiler when it is not an instruction to
# gather things ("compile a list of ...").
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


def count_words(output: str) -> int:
    r"""`\b\w+\b` matches, the counting rule both the regex and the reader use."""
    return len(re.findall(r"\b\w+\b", output))


def parses_as_json(output: str) -> bool:
    """Whether ``output`` is JSON that parses, without saying what that means."""
    try:
        json.loads(output)
    except json.JSONDecodeError:
        return False
    return True


def valid_json_check(output: str, *, negated: bool = False) -> dict[str, Any]:
    """A ``valid_json`` check dict, optionally requiring the opposite."""
    valid = parses_as_json(output)
    return {"kind": "valid_json", "passed": (not valid) if negated else valid}


def _word_check(
    operator: str, bound: int, count: int, *, upper: int | None = None
) -> dict[str, Any]:
    passed = {
        "at most": count <= bound,
        "under": count < bound,
        "at least": count >= bound,
        "more than": count > bound,
        "exactly": count == bound,
        "between": upper is not None and bound <= count <= upper,
    }[operator]
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
    word_count = count_words(output)
    checks: list[dict[str, Any]] = []
    if "valid json" in lowered:
        checks.append(valid_json_check(output))

    # A bound that is negated or applies to part of the output would be graded
    # against the whole output, so it is reported instead of checked.
    scoped = _SCOPE.search(lowered) is not None
    unsupported: list[str] = []
    remainder = lowered
    for match in (*_WORD_RANGE.finditer(lowered), *_WORD_BOUND.finditer(lowered)):
        remainder = remainder.replace(match.group(), " ")
        if _NEGATION.search(lowered[: match.start()]):
            unsupported.append("negated_word_bound")
        elif scoped:
            unsupported.append("scoped_word_bound")
        elif match.re is _WORD_RANGE:
            checks.append(
                _word_check(
                    "between",
                    _number(match["low"]),
                    word_count,
                    upper=_number(match["high"]),
                )
            )
        else:
            checks.append(
                _word_check(
                    _OPERATOR_NAMES[match["operator"]],
                    _number(match["bound"]),
                    word_count,
                )
            )

    unsupported.extend(
        f"{unit}s"
        for unit in dict.fromkeys(
            match["unit"] for match in _COUNTED_UNIT.finditer(remainder)
        )
    )
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
    return CriterionCheck(exact=exact, unsupported=tuple(dict.fromkeys(unsupported)))
