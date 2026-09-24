"""Strong-reference safety gate for candidate prompts.

The weak panel tells us whether a rewrite helps small models.  That is not
enough to ship a rewrite: strict instructions often help weak models while
hurting a stronger model.  This module keeps the strong check behind a small,
dependency-injected seam so the optimizer can use a gateway, a replay, or a
scripted test double without coupling this policy to a transport.

``run_strong_checks`` is intentionally transport-agnostic.  The injected
``evaluate_prompt`` callback is called once for the original and once for each
candidate with the *same* success-test object.  The callback may execute a
strong model and grade its result; returning one numeric score keeps the policy
easy to exercise in tests and prevents the implementation from accidentally
using a different rubric for the original and candidates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .selector import RankingCandidate

EvaluatePrompt = Callable[[str, Any], float]


@dataclass(frozen=True)
class StrongCheckOutcome:
    """The strong-reference result for one candidate.

    ``passed`` is deliberately based on the strong score alone.  Weak-panel
    gains and candidate ranking belong to the selector; this gate only answers
    whether a candidate regressed on the reference model.
    """

    candidate_id: str
    strategy: str | None
    candidate_score: float
    original_score: float
    crutch: bool = False
    passed: bool = False
    reason: str = ""

    @property
    def eligible(self) -> bool:
        """Whether this candidate may enter the selector's ranking."""

        if self.crutch and not self.passed:
            return False
        return self.passed

    @property
    def regression(self) -> bool:
        """Whether the candidate scored below the original."""

        return self.candidate_score < self.original_score

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report entry."""

        return {
            "candidate_id": self.candidate_id,
            "strategy": self.strategy,
            "candidate_score": self.candidate_score,
            "original_score": self.original_score,
            "crutch": self.crutch,
            "passed": self.passed,
            "eligible": self.eligible,
            "regression": self.regression,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StrongCheckReport:
    """Strong-check evidence exposed to the selector and run report.

    ``fallback_reason`` is set when the original is retained because no
    candidate passed this gate.  A selector that later retains the original
    for weak-panel ranking can call :func:`retain_original` to replace this
    with the selector's more specific reason.
    """

    model: str
    original_score: float
    outcomes: tuple[StrongCheckOutcome, ...]
    fallback_reason: str | None = None

    @property
    def passed_candidates(self) -> tuple[StrongCheckOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.eligible)

    @property
    def original_retained(self) -> bool:
        return not self.passed_candidates

    def outcome_for(self, candidate_id: str) -> StrongCheckOutcome:
        for outcome in self.outcomes:
            if outcome.candidate_id == candidate_id:
                return outcome
        raise KeyError(candidate_id)

    def with_fallback_reason(self, reason: str) -> StrongCheckReport:
        return replace(self, fallback_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        """Serialize strong-check evidence for the public run report."""

        return {
            "model": self.model,
            "original_score": self.original_score,
            "candidates": [outcome.to_dict() for outcome in self.outcomes],
            "original_retained": self.original_retained,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class StrongCheckPolicy:
    """Small policy object suitable for dependency injection into a facade."""

    model: str = "strong-reference"

    def check(
        self,
        original_prompt: str,
        candidates: Sequence[RankingCandidate],
        tests: Any,
        evaluate_prompt: EvaluatePrompt,
    ) -> StrongCheckReport:
        return run_strong_checks(
            original_prompt,
            candidates,
            tests,
            evaluate_prompt,
            model=self.model,
        )


def run_strong_checks(
    original_prompt: str,
    candidates: Sequence[RankingCandidate],
    tests: Any,
    evaluate_prompt: EvaluatePrompt,
    *,
    model: str = "strong-reference",
) -> StrongCheckReport:
    """Evaluate the original and candidates against one strong reference.

    ``candidates`` should contain the promising candidates only.  The callback
    is invoked with ``(prompt, tests)`` and must return a finite numeric
    score.  The same ``tests`` object is passed for every invocation.
    """

    original_score = _score(evaluate_prompt(original_prompt, tests), "original")
    outcomes: list[StrongCheckOutcome] = []
    seen_ids: set[str] = set()

    for candidate in candidates:
        candidate_id, prompt, strategy = candidate.candidate_id, candidate.text, candidate.strategy
        crutch = candidate.is_crutch
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate strong-check candidate id: {candidate_id}")
        seen_ids.add(candidate_id)
        candidate_score = _score(
            evaluate_prompt(prompt, tests), f"candidate {candidate_id}"
        )
        passed = candidate_score >= original_score
        if not passed:
            reason = (
                "strong_check_regression: "
                f"score {candidate_score:g} is below original {original_score:g}"
            )
        elif crutch:
            reason = (
                "crutch_allowed_after_strong_check: "
                f"score {candidate_score:g} meets original {original_score:g}"
            )
        else:
            reason = (
                "strong_check_passed: "
                f"score {candidate_score:g} meets original {original_score:g}"
            )
        outcomes.append(
            StrongCheckOutcome(
                candidate_id=candidate_id,
                strategy=strategy,
                candidate_score=candidate_score,
                original_score=original_score,
                crutch=crutch,
                passed=passed,
                reason=reason,
            )
        )

    if any(outcome.eligible for outcome in outcomes):
        fallback_reason = None
    else:
        fallback_reason = "original_retained: no candidate passed the strong check"
    return StrongCheckReport(
        model=model,
        original_score=original_score,
        outcomes=tuple(outcomes),
        fallback_reason=fallback_reason,
    )


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} strong score must be numeric, not bool")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} strong score must be numeric") from exc
    if not isfinite(score):
        raise ValueError(f"{label} strong score must be finite")
    return score


__all__ = [
    "EvaluatePrompt",
    "StrongCheckOutcome",
    "StrongCheckPolicy",
    "StrongCheckReport",
    "run_strong_checks",
]
