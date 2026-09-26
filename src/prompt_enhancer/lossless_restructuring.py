"""Code-owned, provably lossless prompt restructuring.

Jev assigns each stable source unit to one of a closed set of roles. This
module owns segmentation, exact preservation checks, and rendering; it never
asks a model to generate or revise source text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import jev_questions
from .diagnosis import Sentence, split_sentences
from .gateway import Gateway, ProviderError
from .jev import ChoiceDecision, JevResponseError, parse_decision

ROLE_ORDER = ("context", "task", "constraint", "output_format", "example", "other")
ROLE_LABELS = (*ROLE_ORDER, "unknown")
ROLE_CONFIDENCE_THRESHOLD = 0.8
MAX_RESTRUCTURE_UNITS = 40
PROOF_KIND = "lossless-restructure-v1"

ROLE_DESCRIPTIONS = jev_questions.RESTRUCTURE_ROLE_DESCRIPTIONS

ROLE_HEADINGS = {
    "context": "Context",
    "task": "Task",
    "constraint": "Constraints",
    "output_format": "Output format",
    "example": "Examples",
    "other": "Other",
}

_LIST_ITEM = re.compile(r"^ {0,3}(?:[-*+]\s+|\d+[.)]\s+)")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class SourceUnit:
    id: str
    text: str
    start: int
    end: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    unit_id: str
    selected_role: str | None
    role: str
    confidence: float | None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "selected_role": self.selected_role,
            "role": self.role,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProofCheck:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LosslessBuild:
    text: str | None
    proof: Mapping[str, Any] | None
    evidence: Mapping[str, Any]
    decline_reason: str | None = None


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _line_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    spans: list[tuple[int, int, str]] = []
    position = 0
    for line in text.splitlines(keepends=True):
        spans.append((position, position + len(line), line))
        position += len(line)
    if position < len(text):
        spans.append((position, len(text), text[position:]))
    return tuple(spans)


def _protected_markdown_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return fenced-code and list-block ranges that must not be split."""
    lines = _line_spans(text)
    spans: list[tuple[int, int]] = []

    index = 0
    while index < len(lines):
        start, end, line = lines[index]
        opening = _FENCE.match(line.rstrip("\r\n"))
        if opening is not None:
            fence = opening.group(1)
            marker = fence[0]
            closing = re.compile(rf"^ {{0,3}}{re.escape(marker)}{{{len(fence)},}}\s*$")
            finish = end
            index += 1
            while index < len(lines):
                line_start, line_end, candidate = lines[index]
                finish = line_end
                index += 1
                if closing.match(candidate.rstrip("\r\n")):
                    break
            spans.append((start, finish))
            continue
        index += 1

    # A list's exact markers, continuations, and internal punctuation remain
    # an indivisible unit. Treat the containing nonblank paragraph as the block
    # so conservative boundary detection cannot bisect a lazy continuation.
    paragraph_start: int | None = None
    paragraph_end = 0
    paragraph_lines: list[str] = []
    for start, end, line in (*lines, (len(text), len(text), "")):
        if line.strip():
            if paragraph_start is None:
                paragraph_start = start
            paragraph_end = end
            paragraph_lines.append(line)
            continue
        if paragraph_start is not None:
            if any(_LIST_ITEM.match(item.rstrip("\r\n")) for item in paragraph_lines):
                spans.append((paragraph_start, paragraph_end))
            paragraph_start = None
            paragraph_lines = []

    return tuple(sorted(spans))


def segment_source_units(prompt: str) -> tuple[SourceUnit, ...]:
    """Partition the prompt into exact sentence units, protecting Markdown blocks.

    The established sentence splitter supplies safe sentence boundaries. Each
    unit owns the exact source slice up to its boundary, including its original
    whitespace; the resulting ranges therefore cover the prompt exactly.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return ()
    sentences: tuple[Sentence, ...] = split_sentences(prompt)
    if not sentences:
        return ()
    protected = _protected_markdown_spans(prompt)
    sentence_boundaries = [sentence.end for sentence in sentences[:-1]]
    boundaries = [
        boundary
        for boundary in sentence_boundaries
        if not any(start < boundary < end for start, end in protected)
    ]
    for start, end in protected:
        if any(start < boundary < end for boundary in sentence_boundaries):
            if end < len(prompt):
                boundaries.append(end)
    boundaries = sorted(set(boundaries))
    offsets = [0, *boundaries, len(prompt)]
    units: list[SourceUnit] = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if end <= start:
            continue
        content = prompt[start:end]
        units.append(
            SourceUnit(
                id=f"u{len(units) + 1:04d}",
                text=content,
                start=start,
                end=end,
                sha256=_hash(content),
            )
        )
    if not units or "".join(unit.text for unit in units) != prompt:
        return ()
    return tuple(units)


def _normalise_assignments(
    units: Sequence[SourceUnit], assignments: Sequence[RoleAssignment]
) -> tuple[RoleAssignment, ...] | None:
    by_id: dict[str, RoleAssignment] = {}
    unit_ids = {unit.id for unit in units}
    for assignment in assignments:
        if assignment.unit_id not in unit_ids or assignment.unit_id in by_id:
            return None
        selected = assignment.selected_role
        if selected is not None and selected not in ROLE_LABELS:
            return None
        expected_role = (
            selected
            if selected in ROLE_ORDER
            and assignment.confidence is not None
            and assignment.confidence >= ROLE_CONFIDENCE_THRESHOLD
            else "other"
        )
        if assignment.role != expected_role:
            return None
        by_id[assignment.unit_id] = assignment

    result: list[RoleAssignment] = []
    for unit in units:
        assignment = by_id.get(unit.id)
        if assignment is None:
            assignment = RoleAssignment(
                unit.id, None, "other", None, {}, "missing_role"
            )
        result.append(assignment)
    return tuple(result)


def _render(
    units: Sequence[SourceUnit], assignments: Sequence[RoleAssignment]
) -> tuple[str, tuple[str, ...]]:
    role_by_id = {assignment.unit_id: assignment.role for assignment in assignments}
    grouped = {
        role: [unit.text for unit in units if role_by_id[unit.id] == role]
        for role in ROLE_ORDER
    }
    active = tuple(role for role in ROLE_ORDER if grouped[role])
    sections = [
        f"### {ROLE_HEADINGS[role]}\n\n{''.join(grouped[role])}" for role in active
    ]
    return "\n\n".join(sections), active


def _units_match_source(prompt: str, units: Sequence[SourceUnit]) -> bool:
    expected = segment_source_units(prompt)
    return (
        tuple(unit.to_dict() for unit in units)
        == tuple(unit.to_dict() for unit in expected)
        and "".join(unit.text for unit in units) == prompt
    )


def render_lossless_candidate(
    prompt: str,
    units: Sequence[SourceUnit],
    assignments: Sequence[RoleAssignment],
) -> LosslessBuild:
    """Render the fixed sections and create a replayable exact-source proof."""
    if not _units_match_source(prompt, units):
        return _declined("source segmentation did not cover the prompt exactly")
    normalized = _normalise_assignments(units, assignments)
    if normalized is None:
        return _declined("role assignments contained invalid or duplicate unit IDs")
    text, active_roles = _render(units, normalized)
    if text == prompt:
        return _declined("restructured rendering is identical to the source")
    if len(active_roles) < 2:
        return _declined("roles did not produce meaningful structure")

    proof = {
        "kind": PROOF_KIND,
        "source_sha256": _hash(prompt),
        "candidate_sha256": _hash(text),
        "units": [unit.to_dict() for unit in units],
        "assignments": [assignment.to_dict() for assignment in normalized],
    }
    proof_check = verify_lossless_proof(prompt, text, proof)
    if not proof_check.passed:
        return _declined(
            "deterministic source-preservation proof failed",
            reasons=list(proof_check.reasons),
        )
    role_evidence = [
        {
            "unit_id": assignment.unit_id,
            "source_sha256": next(
                unit.sha256 for unit in units if unit.id == assignment.unit_id
            ),
            "selected_role": assignment.selected_role,
            "role": assignment.role,
            "confidence": assignment.confidence,
            "probabilities": dict(assignment.probabilities),
            "reason": assignment.reason,
        }
        for assignment in normalized
    ]
    unknowns = [
        assignment.unit_id
        for assignment in normalized
        if assignment.role == "other"
        and (
            assignment.selected_role in {None, "unknown"}
            or assignment.confidence is None
            or assignment.confidence < ROLE_CONFIDENCE_THRESHOLD
        )
    ]
    evidence = {
        "outcome": "candidate_built",
        "source_preservation": {
            "status": "passed",
            "proof_kind": PROOF_KIND,
            "unit_count": len(units),
            "source_sha256": _hash(prompt),
            "unit_ids": [unit.id for unit in units],
            "unit_sha256": {unit.id: unit.sha256 for unit in units},
        },
        "roles": role_evidence,
        "unknowns": unknowns,
        "active_sections": list(active_roles),
    }
    return LosslessBuild(text, proof, evidence)


def verify_lossless_proof(
    original_prompt: str, candidate_prompt: str, proof: Mapping[str, Any] | None
) -> ProofCheck:
    """Rebuild the body from source ranges and role labels; never strip headings."""
    reasons: list[str] = []
    if not isinstance(proof, Mapping) or proof.get("kind") != PROOF_KIND:
        return ProofCheck(
            False, ("lossless proof is missing or has an unknown version",)
        )
    if proof.get("source_sha256") != _hash(original_prompt):
        reasons.append("source prompt hash does not match")
    if proof.get("candidate_sha256") != _hash(candidate_prompt):
        reasons.append("rendered candidate hash does not match")

    units = segment_source_units(original_prompt)
    proof_units = proof.get("units")
    if not isinstance(proof_units, list) or proof_units != [
        unit.to_dict() for unit in units
    ]:
        reasons.append("source-unit identities or hashes do not match")

    raw_assignments = proof.get("assignments")
    assignments: list[RoleAssignment] = []
    if not isinstance(raw_assignments, list):
        reasons.append("role assignments are missing")
    else:
        for item in raw_assignments:
            if not isinstance(item, Mapping):
                reasons.append("role assignment is malformed")
                continue
            selected = item.get("selected_role")
            confidence = item.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
            ):
                reasons.append("role confidence is invalid")
                continue
            role = item.get("role")
            if not isinstance(item.get("unit_id"), str) or not isinstance(role, str):
                reasons.append("role assignment identity is invalid")
                continue
            probabilities = item.get("probabilities")
            if not isinstance(probabilities, Mapping):
                probabilities = {}
            assignments.append(
                RoleAssignment(
                    str(item["unit_id"]),
                    str(selected) if selected is not None else None,
                    role,
                    float(confidence) if confidence is not None else None,
                    {
                        str(key): float(value)
                        for key, value in probabilities.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    },
                    str(item["reason"]) if item.get("reason") is not None else None,
                )
            )
    normalized = _normalise_assignments(units, assignments)
    if normalized is None:
        reasons.append("role assignment IDs or labels are invalid")
    elif len(assignments) != len(units):
        reasons.append("role assignment multiplicity does not match source units")
    else:
        rendered, _roles = _render(units, normalized)
        if rendered != candidate_prompt:
            reasons.append("candidate does not reconstruct from source-unit roles")
    return ProofCheck(not reasons, tuple(dict.fromkeys(reasons)))


def _declined(reason: str, *, reasons: Sequence[str] = ()) -> LosslessBuild:
    return LosslessBuild(
        None,
        None,
        {
            "outcome": "declined",
            "decline_reason": reason,
            "proof_reasons": list(reasons),
            "source_preservation": {"status": "not_proven"},
            "roles": [],
            "unknowns": [],
        },
        reason,
    )


def _usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_calls = before.get("calls")
    after_calls = after.get("calls")
    if not isinstance(before_calls, int) or not isinstance(after_calls, int):
        return {"status": "unavailable"}
    call_delta = after_calls - before_calls
    before_roles = before.get("cost_by_role", {})
    after_roles = after.get("cost_by_role", {})
    if not isinstance(before_roles, Mapping) or not isinstance(after_roles, Mapping):
        return {"status": "unavailable"}
    role_costs = {
        str(role): max(0.0, float(value) - float(before_roles.get(role, 0.0)))
        for role, value in after_roles.items()
        if isinstance(value, (int, float))
    }
    before_tokens = before.get("tokens", {})
    after_tokens = after.get("tokens", {})
    token_delta: dict[str, int] = {}
    if isinstance(before_tokens, Mapping) and isinstance(after_tokens, Mapping):
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            left, right = before_tokens.get(key, 0), after_tokens.get(key, 0)
            if isinstance(left, int) and isinstance(right, int):
                token_delta[key] = max(0, right - left)
    return {
        "status": "reported" if call_delta > 0 else "unavailable",
        "calls": max(0, call_delta),
        "cost_by_role": role_costs,
        "tokens": token_delta,
    }


def build_lossless_candidate(
    prompt: str,
    gateway: Gateway,
    *,
    judge_model: str,
    run_id: str,
) -> LosslessBuild:
    """Classify source units, render them, and return preservation evidence."""
    before = gateway.usage_report()
    units = segment_source_units(prompt)
    if not units:
        return _declined("source text could not be segmented losslessly")
    if len(units) > MAX_RESTRUCTURE_UNITS:
        return _declined(
            f"source has {len(units)} units; role-assignment limit is {MAX_RESTRUCTURE_UNITS}"
        )
    requests = [
        {
            "model": judge_model,
            "key": f"restructure_lossless:role:{unit.id}",
            "type": "choice",
            "query": jev_questions.RESTRUCTURE_ROLE_QUESTION,
            "criteria": ROLE_DESCRIPTIONS,
            "state": {
                "source_prompt": prompt,
                "source_units": [
                    {"id": source_unit.id, "text": source_unit.text}
                    for source_unit in units
                ],
                "target_unit_id": unit.id,
                "target_unit_text": unit.text,
            },
        }
        for unit in units
    ]
    try:
        raw_answers = gateway.decide_batch(requests, role="judge", run_id=run_id)
    except (ProviderError, JevResponseError, TypeError, ValueError) as exc:
        result = _declined(
            f"role-assignment response was unusable ({type(exc).__name__})"
        )
        return _with_cost(
            result, _usage_delta(before, gateway.usage_report()), len(requests)
        )
    if not isinstance(raw_answers, list) or len(raw_answers) != len(requests):
        result = _declined("role-assignment response did not match requested unit IDs")
        return _with_cost(
            result, _usage_delta(before, gateway.usage_report()), len(requests)
        )

    assignments: list[RoleAssignment] = []
    for unit, raw in zip(units, raw_answers, strict=True):
        if isinstance(raw, Mapping):
            response_id = raw.get("unit_id", raw.get("id"))
            if response_id is not None and response_id != unit.id:
                result = _declined(
                    "role-assignment response contained an invalid unit ID"
                )
                return _with_cost(
                    result, _usage_delta(before, gateway.usage_report()), len(requests)
                )
        try:
            decision = parse_decision(raw)
        except (JevResponseError, TypeError, ValueError):
            decision = None
        if not isinstance(decision, ChoiceDecision):
            assignments.append(
                RoleAssignment(unit.id, None, "other", None, {}, "missing_role")
            )
            continue
        if decision.selected not in ROLE_LABELS:
            result = _declined(
                f"role-assignment response selected invalid role for {unit.id}"
            )
            return _with_cost(
                result, _usage_delta(before, gateway.usage_report()), len(requests)
            )
        confidence = decision.confidence
        if decision.selected == "unknown":
            assigned_role = "other"
            reason = "unknown_role"
        elif confidence < ROLE_CONFIDENCE_THRESHOLD:
            assigned_role = "other"
            reason = "low_confidence"
        elif decision.selected == "other":
            assigned_role = "other"
            reason = "other_role"
        else:
            assigned_role = decision.selected
            reason = None
        assignments.append(
            RoleAssignment(
                unit.id,
                decision.selected,
                assigned_role,
                confidence,
                decision.probabilities,
                reason,
            )
        )

    result = render_lossless_candidate(prompt, units, assignments)
    return _with_cost(
        result, _usage_delta(before, gateway.usage_report()), len(requests)
    )


def _with_cost(
    result: LosslessBuild, cost: Mapping[str, Any], request_count: int
) -> LosslessBuild:
    return LosslessBuild(
        result.text,
        result.proof,
        {
            **dict(result.evidence),
            "cost": dict(cost),
            "role_assignment_requests": request_count,
        },
        result.decline_reason,
    )


__all__ = [
    "MAX_RESTRUCTURE_UNITS",
    "PROOF_KIND",
    "ROLE_CONFIDENCE_THRESHOLD",
    "ROLE_DESCRIPTIONS",
    "ROLE_LABELS",
    "ROLE_ORDER",
    "LosslessBuild",
    "ProofCheck",
    "RoleAssignment",
    "SourceUnit",
    "build_lossless_candidate",
    "render_lossless_candidate",
    "segment_source_units",
    "verify_lossless_proof",
]
