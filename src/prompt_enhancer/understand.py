"""The Understand stage: screen, classify, extract, audit, and probe a run.

On submission the engine screens any embedded/pasted content (treating it as
data, never as instructions), classifies the request (task type plus, for
Auto, the best-fit improvement style with a conservative Clearer fallback),
extracts explicit literal requirements, and asks bounded ambiguity/conflict
probes. Extraction is audited against the original text before its values are
used as hard-gate evidence. Each capability runs only when its inputs exist;
independent questions go out in one batch, and provenance is recorded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import jev_questions
from .jev import ChoiceDecision, NoulDecision, parse_decision
from .styles import DEFAULT_STYLE, IMPROVEMENT_STYLES

#: Confidence below which Auto ignores the inferred style and falls back.
CLASSIFY_FALLBACK_CONFIDENCE = 0.5

#: The conservative style Auto uses when the inference is uncertain.
FALLBACK_STYLE = "clearer"

#: An audited literal needs this probability to become hard-gate evidence.
AUDIT_KEEP_PROBABILITY = 0.8

#: Screen fires only when the prompt looks like it carries embedded content.
_EMBEDDED_MARKERS = ("```", "paste", "pasted", "attached", "embedded", "following:")

#: Words that suggest the request may be ambiguous or conflicted.
_AMBIGUITY_MARKERS = ("maybe", "etc", "something", "stuff", "whatever", "?")
_CONFLICT_MARKERS = (" but ", "however", "instead", "except", "although")

#: Words that mark a literal as exact-output the response must reproduce.
_EXACT_MARKERS = re.compile(
    r"\b(exactly|verbatim|word for word|respond with|reply with|"
    r"output only|only output|must read|must say)\b",
    re.IGNORECASE,
)

_DOUBLE_QUOTED = re.compile(r'"([^"\n]{1,200})"')
_SINGLE_QUOTED = re.compile(r"'([^'\n]*\s[^'\n]*)'")

#: An unquoted literal after ``exactly:`` (e.g. ``Reply with exactly: OK``).
_BARE_EXACT = re.compile(r"\bexactly\s*:\s*(.+)", re.IGNORECASE)

#: At most this many extracted candidates are sent to audit.
MAX_AUDIT_ITEMS = 5

#: At most this many ambiguity/conflict probes are asked per run.
MAX_PROBES = 2


def looks_embedded(prompt: str) -> bool:
    """True when the prompt may carry pasted content to screen as data."""
    lowered = prompt.lower()
    return '"' in prompt or any(
        marker in lowered or marker in prompt for marker in _EMBEDDED_MARKERS
    )


def extract_literal_candidates(prompt: str) -> tuple[str, ...]:
    """Deterministically collect quoted (or bare-exact) literal candidates."""
    found: list[str] = []
    for match in _DOUBLE_QUOTED.findall(prompt):
        text = match.strip()
        if text and text not in found:
            found.append(text)
    for match in _SINGLE_QUOTED.findall(prompt):
        text = match.strip().strip(",.;:!?")
        if text and text not in found:
            found.append(text)
    bare = _BARE_EXACT.search(prompt)
    if bare:
        text = bare.group(1).strip().strip("\"'").strip()
        first_line = text.splitlines()[0].strip() if text else ""
        if first_line and first_line not in found:
            found.append(first_line[:120])
    return tuple(found[:MAX_AUDIT_ITEMS])


def is_exact_output(prompt: str, literals: Sequence[str]) -> bool:
    """True when the prompt demands an exact literal response."""
    return bool(literals) and _EXACT_MARKERS.search(prompt) is not None


@dataclass(frozen=True)
class UnderstandResult:
    """What the Understand stage decided, with evidence for the report."""

    requested_style: str
    applied_style: str
    inferred_style: str | None
    task_type: str
    hard_constraints: tuple[str, ...] = ()
    exact_output: bool = False
    screen_embedded: bool | None = None
    probes: tuple[dict[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_style": self.requested_style,
            "applied_style": self.applied_style,
            "inferred_style": self.inferred_style,
            "task_type": self.task_type,
            "hard_constraints": list(self.hard_constraints),
            "exact_output": self.exact_output,
            "screen_embedded": self.screen_embedded,
            "probes": [dict(item) for item in self.probes],
            "provenance": {
                key: dict(value) if isinstance(value, Mapping) else value
                for key, value in self.provenance.items()
            },
        }


def _choice_answer(gateway: Any, request: Mapping[str, Any], run_id: str) -> Any:
    return gateway.decide(request, role="judge", run_id=run_id)


def run_understand(
    gateway: Any,
    prompt: str,
    *,
    requested_style: str = DEFAULT_STYLE,
    diagnosis: Mapping[str, Any] | None = None,
    judge_model: str = "",
    run_id: str | None = None,
) -> UnderstandResult:
    """Run the Understand stage and return its audited result."""
    style = str(requested_style or DEFAULT_STYLE).strip().lower()
    if style not in IMPROVEMENT_STYLES:
        style = DEFAULT_STYLE
    provenance: dict[str, Any] = {}
    task_type = "general"
    if isinstance(diagnosis, Mapping):
        task_type = str(
            diagnosis.get("task_type") or diagnosis.get("task") or "general"
        ).lower()

    lead_requests: list[dict[str, Any]] = []
    if looks_embedded(prompt):
        lead_requests.append(
            {
                "model": judge_model,
                "key": "understand:screen",
                "type": "noul",
                "query": jev_questions.UNDERSTAND_SCREEN_QUESTION,
                "state": {"prompt": prompt},
            }
        )
    classify_request: dict[str, Any] | None = None
    if style == DEFAULT_STYLE:
        classify_request = {
            "model": judge_model,
            "key": "understand:classify:style",
            "type": "choice",
            "query": jev_questions.UNDERSTAND_STYLE_QUESTION,
            "criteria": {
                name: label
                for name, label in IMPROVEMENT_STYLES.items()
                if name != DEFAULT_STYLE
            },
            "state": {"prompt": prompt, "task_type": task_type},
        }
        lead_requests.append(classify_request)

    lead_answers: list[Any] = []
    if lead_requests:
        lead_answers = gateway.decide_batch(lead_requests, role="judge", run_id=run_id)
    answers = dict(
        zip([item["key"] for item in lead_requests], lead_answers, strict=True)
    )

    screen_embedded: bool | None = None
    if "understand:screen" in answers:
        decision = parse_decision(answers["understand:screen"])
        screen_embedded = (
            isinstance(decision, NoulDecision) and decision.probability >= 0.8
        )
        provenance["screen"] = {
            "fired": True,
            "key": "understand:screen",
            "embedded_content": screen_embedded,
        }
    else:
        provenance["screen"] = {"fired": False, "reason": "no embedded markers"}

    inferred_style: str | None = None
    applied_style = style
    if classify_request is not None:
        decision = parse_decision(answers["understand:classify:style"])
        selected = decision.selected if isinstance(decision, ChoiceDecision) else None
        confidence = (
            decision.confidence if isinstance(decision, ChoiceDecision) else 0.0
        )
        entry: dict[str, Any] = {
            "fired": True,
            "key": "understand:classify:style",
            "selected": selected,
            "confidence": confidence,
        }
        if (
            selected not in IMPROVEMENT_STYLES
            or selected == DEFAULT_STYLE
            or confidence < CLASSIFY_FALLBACK_CONFIDENCE
        ):
            entry["fallback"] = FALLBACK_STYLE
            inferred_style = FALLBACK_STYLE
        else:
            inferred_style = selected
        applied_style = inferred_style
        provenance["classify"] = entry
    else:
        provenance["classify"] = {
            "fired": False,
            "reason": "explicit style requested",
            "selected": style,
        }

    candidates = extract_literal_candidates(prompt)
    hard_constraints: list[str] = []
    if candidates:
        audit_requests = [
            {
                "model": judge_model,
                "key": f"understand:audit:{index}",
                "type": "noul",
                "query": jev_questions.understand_audit_question(candidate),
                "state": {"prompt": prompt, "extracted": candidate},
            }
            for index, candidate in enumerate(candidates)
        ]
        audit_answers = gateway.decide_batch(
            audit_requests, role="judge", run_id=run_id
        )
        dropped: list[str] = []
        for candidate, raw in zip(candidates, audit_answers, strict=True):
            decision = parse_decision(raw)
            probability = (
                decision.probability if isinstance(decision, NoulDecision) else 0.0
            )
            if probability >= AUDIT_KEEP_PROBABILITY:
                hard_constraints.append(candidate)
            else:
                dropped.append(candidate)
        provenance["extract"] = {
            "fired": True,
            "candidates": list(candidates),
            "key_prefix": "understand:audit",
        }
        provenance["audit"] = {
            "fired": True,
            "keys": [item["key"] for item in audit_requests],
            "kept": list(hard_constraints),
            "dropped": dropped,
        }
    else:
        provenance["extract"] = {"fired": False, "reason": "no quoted literals"}
        provenance["audit"] = {
            "fired": False,
            "reason": "nothing extracted to audit",
        }

    probe_requests: list[dict[str, Any]] = []
    lowered = f" {prompt.lower()} "
    if any(marker in lowered for marker in _CONFLICT_MARKERS):
        probe_requests.append(
            {
                "model": judge_model,
                "key": "understand:probe:conflict",
                "type": "noul",
                "query": jev_questions.UNDERSTAND_PROBE_CONFLICT_QUESTION,
                "state": {"prompt": prompt},
            }
        )
    if any(marker in lowered for marker in _AMBIGUITY_MARKERS):
        probe_requests.append(
            {
                "model": judge_model,
                "key": "understand:probe:ambiguity",
                "type": "noul",
                "query": jev_questions.UNDERSTAND_PROBE_AMBIGUITY_QUESTION,
                "state": {"prompt": prompt},
            }
        )
    probe_requests = probe_requests[:MAX_PROBES]
    probes: list[dict[str, Any]] = []
    if probe_requests:
        probe_answers = gateway.decide_batch(
            probe_requests, role="judge", run_id=run_id
        )
        for request, raw in zip(probe_requests, probe_answers, strict=True):
            decision = parse_decision(raw)
            probes.append(
                {
                    "key": request["key"],
                    "probability": decision.probability
                    if isinstance(decision, NoulDecision)
                    else 0.0,
                }
            )
    provenance["probe"] = (
        {"fired": True, "keys": [item["key"] for item in probe_requests]}
        if probe_requests
        else {"fired": False, "reason": "no ambiguity or conflict markers"}
    )

    return UnderstandResult(
        requested_style=style,
        applied_style=applied_style,
        inferred_style=inferred_style,
        task_type=task_type,
        hard_constraints=tuple(hard_constraints),
        exact_output=is_exact_output(prompt, hard_constraints),
        screen_embedded=screen_embedded,
        probes=tuple(probes),
        provenance=provenance,
    )


__all__ = [
    "AUDIT_KEEP_PROBABILITY",
    "CLASSIFY_FALLBACK_CONFIDENCE",
    "FALLBACK_STYLE",
    "UnderstandResult",
    "extract_literal_candidates",
    "is_exact_output",
    "looks_embedded",
    "run_understand",
]
