"""Improvement styles: the user-visible goal of a run.

The composer offers ``auto`` plus a fixed catalog of named styles.  The
catalog lives here so the API validation and (later) the style-to-strategy
bundles share one source of truth; the web composer mirrors the values and
labels in ``web/src/styles.ts``.
"""

from __future__ import annotations

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


def parse_improvement_style(value: object) -> str:
    """Read an improvement style from user input; missing means Auto."""
    style = str(value or DEFAULT_STYLE).strip().lower()
    if style in IMPROVEMENT_STYLES:
        return style
    valid = ", ".join(IMPROVEMENT_STYLES)
    raise ValueError(f"improvement_style must be one of: {valid}")
