"""Source-backed requirements and mechanical evidence for prompt enhancement."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .clarification import (
    ClarificationOption,
    ClarificationPlan,
    ClarificationQuestion,
)
from .criterion_checks import count_words
from .criterion_reading import number_candidates
from .requirement_formats import (
    CSV_DIRECTIVE,
    JSON_DIRECTIVE,
    JSON_SCHEMA_DIRECTIVE,
    check_format,
    schema_from_source,
)

_EXACT_REPLY = re.compile(
    r"(?im)^(?>[ \t]*)(?:reply|respond|output|return|print)(?:[ \t]+with)?[ \t]+"
    r"(?:exactly|only)(?:[ \t]*:[ \t]*|[ \t]+)"
    r'(?P<literal>"[^"\n]+"|\x27[^\x27\n]+\x27|`[^`\n]+`|[\w-]+)'
    r"(?P<tail>(?:[ \t]+(?:and[ \t]+nothing[ \t]+else|nothing[ \t]+else))?"
    r"(?>[ \t]*)[.!]?[ \t]*)(?=$|\n)"
)
_FORMAT_NAMES = frozenset(
    {"json", "csv", "text", "markdown", "code", "html", "xml", "yaml"}
)
_PUNCTUATION_ONLY = re.compile(
    r"(?is)^[ \t]*(?:improve|correct|fix|change)\s+punctuation\s+only,?\s+"
    r"preserving\s+every\s+word\s+and\s+(?:its|their)\s+order\s*:\s*"
    r'(?P<literal>"[^"\n]*"|\x27[^\x27\n]*\x27)(?>\s*)[.!]?[ \t]*$'
)
_PLACEHOLDER = re.compile(
    r"\{\{[ \t]*[A-Za-z_][A-Za-z0-9_.]*[ \t]*\}\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}"
)
_COUNT_FRAGMENT = (
    r"exactly[ \t]+(?P<bound>\d{1,9}|[a-z-]{1,30}(?:[ \t]+[a-z-]{1,30}){0,5}?)"
    r"[ \t]+(?P<unit>words?|sentences?|lines?|bullets?)"
)
RequirementKind = Literal[
    "exact_output",
    "punctuation_only",
    "protected_value",
    "word_count",
    "sentence_count",
    "line_count",
    "bullet_count",
    "json_format",
    "json_schema",
    "csv_shape",
]
_COUNT_KINDS: dict[str, RequirementKind] = {
    "word": "word_count",
    "sentence": "sentence_count",
    "line": "line_count",
    "bullet": "bullet_count",
}
_COUNT_CLAUSE = re.compile(_COUNT_FRAGMENT, re.IGNORECASE)
_COUNT_DIRECTIVE = re.compile(
    r"[ \t]*(?:write|reply|respond|return|output)(?:[ \t]+(?:with|using))?[ \t]+"
    + _COUNT_FRAGMENT
    + r"(?:[ \t]+and[ \t]+"
    + _COUNT_FRAGMENT.replace("?P<bound>", "?P<next_bound>").replace(
        "?P<unit>", "?P<next_unit>"
    )
    + r")?(?>[ \t]*)[.!]?[ \t]*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Requirement:
    """One obligation, with its exact source and the oracle it supports."""

    id: str
    source: str
    start: int
    end: int
    kind: RequirementKind
    scope: str
    expected: str
    source_kind: str = "original_prompt"

    @property
    def protected_values(self) -> tuple[str, ...]:
        if self.kind == "json_schema":
            schema = json.loads(self.expected)
            return tuple(dict.fromkeys((*schema, *schema.values())))
        if self.kind == "csv_shape":
            return tuple(json.loads(self.expected))
        return (
            (self.expected,)
            if self.kind in {"exact_output", "punctuation_only", "protected_value"}
            else ()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "source_kind": self.source_kind,
            "source_span": {
                "start": self.start,
                "end": self.end,
                "unit": "unicode_codepoints",
            },
            "kind": self.kind,
            "scope": self.scope,
            "protected_values": list(self.protected_values),
            "oracle": {"kind": self.kind, "expected": self.expected},
        }


def extract_requirements(prompt: str) -> tuple[Requirement, ...]:
    """Recognize supported literals, input slots, punctuation edits and counts.

    A choice of literals, scoped counts, or additional requested results
    need further interpretation beyond this conservative grammar.
    Those source spans still need semantic extraction and uncertainty coverage.
    """
    found: list[Requirement] = []
    json_format = JSON_DIRECTIVE.fullmatch(prompt)
    if json_format is not None:
        found.append(
            Requirement(
                id=f"requirement:{json_format.start()}:{json_format.end()}",
                source=json_format.group(),
                start=json_format.start(),
                end=json_format.end(),
                kind="json_format",
                scope="whole_output",
                expected="valid_json",
            )
        )
    shape = JSON_SCHEMA_DIRECTIVE.fullmatch(prompt)
    if (
        shape is not None
        and (schema := schema_from_source(shape["schema"])) is not None
    ):
        found.append(
            Requirement(
                id=f"requirement:{shape.start()}:{shape.end()}",
                source=shape.group(),
                start=shape.start(),
                end=shape.end(),
                kind="json_schema",
                scope="whole_output",
                expected=schema,
            )
        )
    csv_format = CSV_DIRECTIVE.fullmatch(prompt)
    if csv_format is not None:
        found.append(
            Requirement(
                id=f"requirement:{csv_format.start()}:{csv_format.end()}",
                source=csv_format.group(),
                start=csv_format.start(),
                end=csv_format.end(),
                kind="csv_shape",
                scope="whole_output",
                expected=json.dumps(
                    [column.strip() for column in csv_format["columns"].split(",")]
                ),
            )
        )
    counts = _COUNT_DIRECTIVE.fullmatch(prompt)
    if counts is not None:
        for clause in _COUNT_CLAUSE.finditer(prompt):
            bound = clause["bound"].casefold()
            values = number_candidates(bound)
            if bound not in values or len(values) != 1:
                continue
            unit = clause["unit"].casefold().removesuffix("s")
            found.append(
                Requirement(
                    id=f"requirement:{clause.start()}:{clause.end()}",
                    source=clause.group(),
                    start=clause.start(),
                    end=clause.end(),
                    kind=_COUNT_KINDS[unit],
                    scope="whole_output",
                    expected=str(values[bound]),
                )
            )
    for placeholder in _PLACEHOLDER.finditer(prompt):
        found.append(
            Requirement(
                id=f"requirement:{placeholder.start()}:{placeholder.end()}",
                source=placeholder.group(),
                start=placeholder.start(),
                end=placeholder.end(),
                kind="protected_value",
                scope="candidate_prompt",
                expected=placeholder.group(),
            )
        )
    punctuation = _PUNCTUATION_ONLY.fullmatch(prompt)
    if punctuation is not None:
        found.append(
            Requirement(
                id=f"requirement:{punctuation.start()}:{punctuation.end()}",
                source=punctuation.group(),
                start=punctuation.start(),
                end=punctuation.end(),
                kind="punctuation_only",
                scope="whole_output",
                expected=punctuation["literal"][1:-1],
            )
        )
    for match in _EXACT_REPLY.finditer(prompt):
        # A directive introduced as source material needs a semantic scope
        # decision; this conservative recognizer handles leading instructions.
        if prompt[: match.start()].strip():
            continue
        literal = match["literal"]
        if literal[0] not in "\"'`" and literal.casefold() in _FORMAT_NAMES:
            # A bare format name can describe a format rather than its spelling.
            # Quoting it explicitly still supports an exact literal requirement.
            continue
        expected = literal[1:-1] if literal[0] in "\"'`" else literal
        found.append(
            Requirement(
                id=f"requirement:{match.start()}:{match.end()}",
                source=match.group(),
                start=match.start(),
                end=match.end(),
                kind="exact_output",
                scope="whole_output",
                expected=expected,
            )
        )
    return tuple(found)


def _selected_count(kind: str, assumptions: Sequence[Mapping[str, Any]]) -> str | None:
    for assumption in assumptions:
        if (
            assumption.get("source") != "answer"
            or assumption.get("key") != f"conflict:{kind}"
        ):
            continue
        answer = str(assumption.get("value", ""))
        unit = kind.removesuffix("_count")
        match = re.fullmatch(rf"Use exactly (\d{{1,9}}) {unit}s\.", answer)
        if match is not None:
            return match[1]
    return None


def effective_requirements(
    prompt: str,
    assumptions: Sequence[Mapping[str, Any]] = (),
) -> tuple[Requirement, ...]:
    """An explicit user choice can supersede a conflicting source count."""
    original = extract_requirements(prompt)
    retained = tuple(
        item
        for item in original
        if _selected_count(item.kind, assumptions) in {None, item.expected}
    )
    additions = []
    for kind in _COUNT_KINDS.values():
        selected = _selected_count(kind, assumptions)
        if selected is None or any(
            item.kind == kind and item.expected == selected for item in original
        ):
            continue
        source = f"Use exactly {selected} {kind.removesuffix('_count')}s."
        additions.append(
            Requirement(
                id=f"user_answer:conflict:{kind}",
                source=source,
                start=0,
                end=len(source),
                kind=kind,
                scope="whole_output",
                expected=selected,
                source_kind="user_answer",
            )
        )
    return retained + tuple(additions)


def resolved_prompt(prompt: str, assumptions: Sequence[Mapping[str, Any]]) -> str:
    """Apply a confirmed count choice to the supported whole-answer directive."""
    ledger = requirement_ledger(prompt, assumptions)
    if not ledger["contradictions"] or any(
        item["status"] != "resolved_by_user" for item in ledger["contradictions"]
    ):
        return prompt
    requirements = effective_requirements(prompt, assumptions)
    if not requirements:
        return prompt
    counts = dict.fromkeys(
        f"exactly {item.expected} {item.kind.removesuffix('_count')}s"
        for item in requirements
    )
    return "Write " + " and ".join(counts) + "."


def requirement_ledger(
    prompt: str,
    assumptions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Keep recognized obligations and contradictions without implying full coverage."""
    requirements = extract_requirements(prompt)
    contradictions = []
    for kind in ("word_count", "sentence_count", "line_count", "bullet_count"):
        counts = [item for item in requirements if item.kind == kind]
        if len({item.expected for item in counts}) > 1:
            selected = _selected_count(kind, assumptions)
            contradictions.append(
                {
                    "id": f"conflict:{kind}",
                    "requirement_ids": [item.id for item in counts],
                    "status": "resolved_by_user"
                    if selected is not None
                    else "unresolved",
                    **({"selected_count": selected} if selected is not None else {}),
                    "reason": "The same complete answer cannot have both exact counts.",
                }
            )
    return {
        "version": "source-requirements-1",
        "requirements": [item.to_dict() for item in requirements]
        + [
            item.to_dict()
            for item in effective_requirements(prompt, assumptions)
            if item.source_kind == "user_answer"
        ],
        "contradictions": contradictions,
        "coverage": "partial",
        "reason": "Conservative source-backed recognition; other obligations still need coverage.",
    }


def conflict_plan(prompt: str) -> ClarificationPlan | None:
    ledger = requirement_ledger(prompt)
    if not ledger["contradictions"]:
        return None
    questions = []
    for conflict in ledger["contradictions"]:
        kind = conflict["id"].removeprefix("conflict:")
        unit = kind.removesuffix("_count")
        values = dict.fromkeys(
            item.expected for item in extract_requirements(prompt) if item.kind == kind
        )
        questions.append(
            ClarificationQuestion(
                id=conflict["id"],
                prompt=f"The prompt asks for incompatible {unit} counts. Which count should the answer use?",
                options=tuple(
                    ClarificationOption(value, f"Use exactly {value} {unit}s.")
                    for value in values
                )
                + (ClarificationOption("other", "Other", other=True),),
                default_answer="",
                label=f"Resolve the {unit} count conflict",
                required_answer=True,
            )
        )
    return ClarificationPlan(tuple(questions), ())


def output_findings(
    requirements: Sequence[Requirement], outputs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Check exact visible text, keeping each candidate/model/sample identity."""
    return tuple(
        {
            "requirement_id": item.id,
            "source": item.source,
            "scope": item.scope,
            "candidate_id": output.get("candidate_id"),
            "model": output.get("model"),
            "sample": output.get("sample"),
            "status": _finding_status(item, output.get("output")),
            "check": item.kind,
            "expected": item.expected,
            "observed_chars": len(str(output.get("output", ""))),
            "reason": _finding_reason(item, output.get("output")),
        }
        for item in requirements
        if item.scope == "whole_output"
        for output in outputs
    )


def _ambiguous_word_count(requirement: Requirement, output: Any) -> bool:
    return (
        requirement.kind == "word_count"
        and isinstance(output, str)
        and re.search(r"\w['’\-‐‑./]\w", output) is not None
    )


def _finding_status(requirement: Requirement, output: Any) -> str:
    if requirement.kind in {"json_format", "json_schema", "csv_shape"}:
        return check_format(requirement.kind, output, requirement.expected)[0]
    if requirement.kind == "sentence_count":
        return "untestable"
    if _uncertainty(requirement, output):
        return "untestable"
    return "tested" if _satisfies(requirement, output) else "failed"


def _satisfies(requirement: Requirement, output: Any) -> bool:
    if not isinstance(output, str):
        return False
    if requirement.kind == "exact_output":
        return output == requirement.expected
    if requirement.kind == "word_count":
        return count_words(output) == int(requirement.expected)
    if requirement.kind == "line_count":
        return len(output.splitlines()) == int(requirement.expected)
    if requirement.kind == "bullet_count":
        return len(re.findall(r"(?m)^[-*+][ \t]+\S", output)) == int(
            requirement.expected
        )
    if requirement.kind != "punctuation_only":
        return False

    def non_punctuation(text: str) -> str:
        return "".join(
            ch for ch in text if not unicodedata.category(ch).startswith("P")
        )

    return non_punctuation(output) == non_punctuation(requirement.expected)


def _uncertainty(requirement: Requirement, output: Any) -> str | None:
    if requirement.kind == "sentence_count":
        return "Sentence boundaries need interpretation; no deterministic sentence convention was specified."
    if _ambiguous_word_count(requirement, output):
        return "Ordinary word boundaries are ambiguous in this answer; no counting convention was specified."
    if isinstance(output, str):
        if requirement.kind == "bullet_count" and (
            re.search(r"(?m)^[ \t]+[-*+][ \t]|^[ \t]*\d+[.)][ \t]", output)
            or "```" in output
            or "~~~" in output
        ):
            return "Nested, numbered or fenced lists need a scope decision before bullets can be counted."
        if requirement.kind == "line_count" and re.search(
            r"[\v\f\x1c-\x1e\x85\u2028\u2029]", output
        ):
            return "Unusual line separators need an explicit line-count convention."
    return None


def _finding_reason(requirement: Requirement, output: Any) -> str:
    if requirement.kind in {"json_format", "json_schema", "csv_shape"}:
        return check_format(requirement.kind, output, requirement.expected)[1]
    uncertainty = _uncertainty(requirement, output)
    if uncertainty is not None:
        return uncertainty
    if requirement.kind == "exact_output":
        return "The answer must equal the required literal with no extra text."
    if requirement.kind == "punctuation_only":
        return "Only punctuation may change; letters, case, spacing and order must remain intact."
    unit = requirement.kind.removesuffix("_count")
    return f"The complete answer must contain exactly {requirement.expected} {unit}s."


def preserves_literal(candidate: str, literal: str) -> bool:
    """A protected word cannot survive only inside a different token."""
    return re.search(rf"(?<!\w){re.escape(literal)}(?!\w)", candidate) is not None


def prompt_findings(
    requirements: Sequence[Requirement], candidate_id: str, candidate: str
) -> tuple[dict[str, Any], ...]:
    """Check protected source values independently of downstream answers."""
    findings = tuple(
        {
            "requirement_id": item.id,
            "source": item.source,
            "scope": "candidate_prompt",
            "candidate_id": candidate_id,
            "status": "tested" if preserves_literal(candidate, value) else "failed",
            "check": "protected_value",
            "expected": value,
            "reason": "The protected source value must remain verbatim in the prompt.",
        }
        for item in requirements
        for value in item.protected_values
    )
    for item in requirements:
        if item.kind != "json_schema":
            continue
        match = JSON_SCHEMA_DIRECTIVE.fullmatch(candidate)
        actual = schema_from_source(match["schema"]) if match is not None else None
        findings += (
            {
                "requirement_id": item.id,
                "source": item.source,
                "scope": "candidate_prompt",
                "candidate_id": candidate_id,
                "status": "untestable"
                if actual is None
                else "tested"
                if actual == item.expected
                else "failed",
                "check": "json_type_bindings",
                "expected": item.expected,
                "reason": "The rewritten JSON declaration must retain each source key's type; an unsupported declaration needs semantic coverage.",
            },
        )
    return findings


def check_summaries(
    findings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Typed per-obligation facts for a candidate's chronological activity."""
    summaries: dict[str, dict[str, Any]] = {}
    for finding in findings:
        key = str(finding["requirement_id"])
        summary = summaries.setdefault(
            key,
            {
                "requirement_id": key,
                "source": finding["source"],
                "tested": 0,
                "failed": 0,
                "untestable": 0,
                "reasons": [],
            },
        )
        status = str(finding["status"])
        summary[status] += 1
        if status != "tested" and finding["reason"] not in summary["reasons"]:
            summary["reasons"].append(finding["reason"])
    return tuple(summaries.values())
