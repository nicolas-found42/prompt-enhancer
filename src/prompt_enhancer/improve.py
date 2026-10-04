"""The always-improve policy: a verified changed prompt or a reported failure.

Returning the user's input unchanged is no longer a success outcome. When the
bounded rounds cannot produce a verified changed candidate, the run carries an
explicit ``improvement_not_verified`` failure instead of a silent keep.
"""

from __future__ import annotations

from collections.abc import Sequence

ALWAYS_IMPROVE_POLICY_VERSION = 1

_CARRIER_STRIP = str.maketrans("", "", " \t\r\n")
_CARRIER_PUNCTUATION = tuple(".,;:!?…—–()[]{}\"'‘’“”`")


def _carrier(text: str) -> str:
    """The text without case, whitespace, or punctuation carriers."""
    stripped = text.translate(_CARRIER_STRIP).casefold()
    while stripped and stripped[-1] in _CARRIER_PUNCTUATION:
        stripped = stripped[:-1]
    return stripped


def prompts_differ_meaningfully(original: str, candidate: str) -> bool:
    """True when the candidate is not a cosmetic mirror of the original.

    Case, whitespace, and punctuation carriers alone do not count as a change.
    """
    return _carrier(original) != _carrier(candidate)


def identical_candidates(original: str, candidates: Sequence[str]) -> list[int]:
    """Indices of candidates that would return the input rather than change it.

    Includes cosmetic-only mirrors, so a writer echo or punctuation-only
    "rewrite" is rejected as unchanged rather than selected.
    """
    carrier = _carrier(original)
    return [
        index
        for index, candidate in enumerate(candidates)
        if not candidate.strip() or _carrier(candidate) == carrier
    ]


def no_candidate_changed_reason() -> str:
    """The report reason when no candidate differed from the input."""
    return (
        "improvement_not_verified: no candidate changed the prompt; "
        "the original is returned"
    )


def no_confirmed_improvement_reason() -> str:
    return (
        "No candidate beat the original under robust ranking, so the original "
        "prompt is returned unchanged this time."
    )


def improvement_unverified_summary() -> str:
    return (
        "The rounds could not produce a verified changed prompt. The user's "
        "original prompt is returned unchanged this time; retrying may still "
        "produce an improvement."
    )


def failure_unverified() -> dict[str, str]:
    """The failure entry when verification left no changed candidate."""
    return {
        "kind": "improvement_not_verified",
        "headline": "No verified improvement was found",
        "hint": (
            "Retrying gives the rewrite models another chance to produce a "
            "verified changed prompt."
        ),
        "message": (
            "No candidate passed verification or changed the prompt, so the "
            "original prompt is returned unchanged."
        ),
    }


def failure_no_confirmed_improvement() -> dict[str, str]:
    """The failure entry when verified candidates did not beat the input."""
    return {
        "kind": "improvement_not_verified",
        "headline": "No verified improvement was found",
        "hint": (
            "Retrying gives the rewrite models another chance to produce a "
            "verified changed prompt."
        ),
        "message": (
            "Verified candidates were produced but did not beat the original "
            "under robust ranking, so the original prompt is returned unchanged."
        ),
    }


__all__ = [
    "ALWAYS_IMPROVE_POLICY_VERSION",
    "failure_no_confirmed_improvement",
    "failure_unverified",
    "identical_candidates",
    "improvement_unverified_summary",
    "no_candidate_changed_reason",
    "no_confirmed_improvement_reason",
    "prompts_differ_meaningfully",
]
