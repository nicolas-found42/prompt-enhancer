"""Improvement styles: the user-visible goal of a run.

The composer offers ``auto`` plus a fixed catalog of named styles.  The
catalog lives here so the API validation and (later) the style-to-strategy
bundles share one source of truth; the web composer mirrors the values and
labels in ``web/src/styles.ts``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_STYLE = "auto"

#: (value, label) pairs always visible beside the prompt box.
COMMON_STYLES: tuple[tuple[str, str], ...] = (
    ("clearer", "Clearer"),
    ("shorter", "Shorter"),
    ("more_specific", "More specific"),
    ("add_detail", "Add useful detail"),
    ("structured", "Structured/actionable"),
    ("creative", "Creative"),
)

#: (value, label) pairs grouped under "More styles".
MORE_STYLES: tuple[tuple[str, str], ...] = (
    ("decision_ready", "Decision-ready"),
    ("research_ready", "Research-ready"),
    ("code_ready", "Code-ready"),
    ("teach_me", "Teach me"),
    ("audience_fit", "Audience-fit"),
    ("tone_voice", "Tone/voice"),
    ("persuasive", "Persuasive"),
    ("faithful_transform", "Faithful transform"),
    ("exact_format", "Exact format"),
    ("proofread_only", "Proofread only"),
    ("ask_me_first", "Ask-me-first"),
    ("safety_aware", "Safety-aware"),
    ("challenge_it", "Challenge it"),
    ("red_team", "Red-team"),
    ("surprise_me", "Surprise me"),
)

IMPROVEMENT_STYLES: dict[str, str] = {
    DEFAULT_STYLE: "Auto",
    **dict(COMMON_STYLES),
    **dict(MORE_STYLES),
}

# These permissions describe presentation, never additional task substance.
STYLE_PRESENTATION = {
    "clearer": "Use direct wording and make existing relationships explicit.",
    "shorter": "Remove redundant wording while retaining every requirement.",
    "more_specific": "Name and disambiguate details already supported by the request.",
    "add_detail": "Explain existing instructions without adding tasks or facts.",
    "structured": "Group and order existing content with useful headings.",
    "creative": "Use imaginative wording and framing for the existing content.",
    "decision_ready": "Present existing options and reasoning so a decision is easier.",
    "research_ready": "Distinguish supplied facts from assumptions needing confirmation.",
    "code_ready": "Present existing technical requirements as an implementable contract.",
    "teach_me": "Explain existing content step by step in accessible language.",
    "audience_fit": "Adapt language for the stated audience without inventing audience facts.",
    "tone_voice": "Adjust prose tone and voice, including warm conversational phrasing.",
    "persuasive": "Make existing supported reasons compelling without invented promises.",
    "faithful_transform": "Transform wording naturally while retaining the information flow.",
    "exact_format": "Specify presentation of existing deliverables without adding content.",
    "proofread_only": "Correct spelling, grammar, and punctuation only.",
    "ask_me_first": "Ask for essential missing information instead of guessing its value.",
    "safety_aware": "Make existing risks and uncertainty explicit without asserting unknown facts.",
    "challenge_it": "Question supported assumptions without inventing objections as facts.",
    "red_team": "Frame scrutiny of existing requirements and risks without adding deliverables.",
    "surprise_me": "Use an unexpected usable presentation of the same content.",
}
STYLE_BOUNDARY = (
    "Presentation permission only: preserve task facts, scope, deliverables, "
    "success criteria, exact output, hard literals, and all stated constraints. "
    "New substantive details require the original prompt or a confirmed user answer."
)


def style_authorization_for(applied_style: str) -> dict[str, str]:
    """Return bounded catalog permission after Auto has resolved to a style."""
    if applied_style not in STYLE_PRESENTATION:
        return {}
    return {
        "applied_style": applied_style,
        "presentation": STYLE_PRESENTATION[applied_style],
        "boundary": STYLE_BOUNDARY,
    }


def validated_style_authorization(
    applied_style: str | None, authorization: Mapping[str, Any] | None
) -> dict[str, str]:
    """Reject absent, forged, mismatched, or unresolved style permissions."""
    canonical = style_authorization_for(applied_style or "")
    return canonical if canonical and authorization == canonical else {}


def parse_improvement_style(value: object) -> str:
    """Read an improvement style from user input; missing means Auto."""
    style = str(value or DEFAULT_STYLE).strip().lower()
    if style in IMPROVEMENT_STYLES:
        return style
    valid = ", ".join(IMPROVEMENT_STYLES)
    raise ValueError(f"improvement_style must be one of: {valid}")
