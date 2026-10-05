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


def improved_unverified_summary(strategy: str | None = None) -> str:
    """The report summary for a fidelity-gated rewrite with no success tests."""
    gated = (
        f" The winning rewrite (strategy: {strategy}) passed the meaning and "
        "safety checks."
        if strategy
        else " The winning rewrite passed the meaning and safety checks."
    )
    return (
        "No faithful success tests were available, so the improvement is "
        f"unproven.{gated} What the model answers to the improved prompt was "
        "not verified — treat answer quality as untested."
    )


def no_qualified_candidate_summary(attempts: int) -> str:
    """The report summary when every written candidate failed its gates."""
    return (
        f"{attempts} candidate{'s' if attempts != 1 else ''} "
        "were written and checked, but none passed the meaning and safety "
        "gates, so the original prompt is returned unchanged. The attempts "
        "and their rejection reasons are listed below; retrying gives the "
        "rewrite models another chance."
    )


def no_candidate_written_summary(rejections: int) -> str:
    """The report summary when strategy search wrote no candidate at all."""
    return (
        "No rewrite candidate could be written"
        + (
            f" ({rejections} strategies were considered and rejected)"
            if rejections
            else ""
        )
        + ", so the original prompt is returned unchanged."
    )


def failure_no_qualified_candidate(attempts: int) -> dict[str, str]:
    """The failure entry when gates rejected every written candidate.

    The kind stays ``improvement_not_verified``: like the other
    always-improve policy failures it is a valid evaluation observation,
    not an operational error. The report status and headline carry the
    no-qualified-candidate distinction.
    """
    return {
        "kind": "improvement_not_verified",
        "headline": "No rewrite passed its checks",
        "hint": (
            "Retrying gives the rewrite models another chance to produce a "
            "rewrite that keeps your meaning."
        ),
        "message": no_qualified_candidate_summary(attempts),
    }


def failure_no_candidate_written(rejections: int) -> dict[str, str]:
    """The failure entry when strategy search wrote no candidate at all."""
    return {
        "kind": "improvement_not_verified",
        "headline": "No rewrite passed its checks",
        "hint": (
            "Retrying gives the rewrite models another chance to produce a "
            "rewrite that keeps your meaning."
        ),
        "message": no_candidate_written_summary(rejections),
    }


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


def impossible_summary(style: str, constraints: Sequence[str]) -> str:
    """The report summary for a style/constraint pairing with no way through."""
    literals = " and ".join(repr(item) for item in constraints)
    return (
        f"The {style} style cannot apply to this prompt: every strategy in "
        f"its bundle would rewrite the exact literal {literals} the prompt "
        "requires verbatim, so the original prompt is returned unchanged and "
        "no violation was emitted. Try a compatible style such as Exact "
        "format or Proofread only."
    )


def failure_impossible(style: str, constraints: Sequence[str]) -> dict[str, str]:
    """The failure entry for a proven style/constraint impossibility."""
    return {
        "kind": "impossible",
        "headline": "Style and requirements cannot both be satisfied",
        "hint": (
            "Resubmit with a compatible style, or relax the exact-output requirement."
        ),
        "message": impossible_summary(style, constraints),
    }


__all__ = [
    "ALWAYS_IMPROVE_POLICY_VERSION",
    "failure_no_candidate_written",
    "failure_no_confirmed_improvement",
    "failure_no_qualified_candidate",
    "failure_impossible",
    "failure_unverified",
    "identical_candidates",
    "impossible_summary",
    "improved_unverified_summary",
    "improvement_unverified_summary",
    "no_candidate_changed_reason",
    "no_candidate_written_summary",
    "no_confirmed_improvement_reason",
    "no_qualified_candidate_summary",
    "prompts_differ_meaningfully",
]
