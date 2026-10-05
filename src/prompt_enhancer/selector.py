"""Strict robust candidate selection and rejection reporting.

The selector only orders already-graded candidates.  Hard checks and the
strong reference check are injected as eligibility decisions (the strong-check
module owns those decisions), so this module never duplicates either policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .grading import GradeReport
from .improve import prompts_differ_meaningfully
from .strong_check import StrongCheckReport


@dataclass(frozen=True)
class RankingCandidate:
    """A candidate carrying its grade and an external eligibility decision."""

    candidate_id: str
    text: str = ""
    strategy: str = ""
    strategy_kind: str = "safe"
    grade: GradeReport | None = None
    eligible: bool = True
    rejection_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.text

    @property
    def is_crutch(self) -> bool:
        return self.strategy_kind == "crutch" or self.strategy in {
            "split_into_steps",
            "add_example",
            "role_play",
            "repeated_emphasis",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "prompt": self.text,
            "strategy": self.strategy,
            "strategy_kind": self.strategy_kind,
            "is_crutch": self.is_crutch,
            "grade": _serialize_grade(self.grade),
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RankedCandidate:
    candidate: RankingCandidate
    rank: int
    metrics: tuple[float, float, float, int]
    selected: bool
    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = self.candidate.to_dict()
        payload.update(
            {
                "rank": self.rank,
                "metrics": {
                    "worst_model_pass_rate": self.metrics[0],
                    "mean_pass_rate": self.metrics[1],
                    "sample_spread": self.metrics[2],
                    "length": self.metrics[3],
                },
                "selected": self.selected,
                "rejection_reasons": list(self.rejection_reasons),
            }
        )
        return payload


@dataclass(frozen=True)
class RankingResult:
    selected: RankingCandidate | None
    original: RankingCandidate
    original_kept: bool
    ranked: tuple[RankedCandidate, ...]
    rejection_reasons: Mapping[str, tuple[str, ...]]

    @property
    def selected_candidate_id(self) -> str | None:
        return self.selected.candidate_id if self.selected is not None else None

    @property
    def winner(self) -> RankingCandidate | None:
        return self.selected

    @property
    def final_prompt(self) -> str:
        return (self.selected or self.original).text

    def to_dict(self) -> dict[str, Any]:
        selected_grade = _serialize_grade((self.selected or self.original).grade)
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate": self.selected.to_dict() if self.selected else None,
            "winner_score": selected_grade,
            "original_kept": self.original_kept,
            "original_score": _serialize_grade(self.original.grade),
            "ranking": [item.to_dict() for item in self.ranked],
            "rejected_candidates": [
                item.to_dict() for item in self.ranked if not item.selected
            ],
            "rejection_reasons": {
                candidate_id: list(reasons)
                for candidate_id, reasons in self.rejection_reasons.items()
            },
        }

    @property
    def report(self) -> dict[str, Any]:
        return self.to_dict()


def _serialize_grade(grade: GradeReport | None) -> dict[str, Any] | None:
    return grade.to_dict() if grade is not None else None


def _strong_decision(
    candidate: RankingCandidate, strong_check: StrongCheckReport | None
) -> tuple[bool | None, str | None]:
    """The strong check's verdict on a candidate: passed, failed, or not run."""
    if strong_check is None:
        return None, None
    try:
        outcome = strong_check.outcome_for(candidate.candidate_id)
    except KeyError:
        # Upstream fidelity can exclude a candidate before any strong-model run;
        # an eligible candidate the strong check never ran has not passed it.
        return (
            (None, None)
            if not candidate.eligible
            else (False, "strong check did not pass")
        )
    if outcome.passed:
        return True, None
    return False, outcome.reason or "strong check failed"


def _key(candidate: RankingCandidate) -> tuple[float, float, float, int]:
    grade = candidate.grade
    if grade is None:
        return (-0.0, -0.0, 0.0, len(candidate.text))
    return (-grade.worst, -grade.mean, grade.spread, len(candidate.text))


def _worse_than_baseline(
    candidate: RankingCandidate, baseline: RankingCandidate
) -> bool:
    """True only when the candidate is strictly worse than the original.

    Comparison follows the robust ranking order: worst pass rate, then mean
    pass rate, then sample spread. Length never makes a candidate worse; at
    equality a shorter candidate wins in the final ordering.
    """
    if baseline.grade is None or candidate.grade is None:
        return False
    if baseline.grade.worst > candidate.grade.worst:
        return True
    if candidate.grade.worst > baseline.grade.worst:
        return False
    if baseline.grade.mean > candidate.grade.mean:
        return True
    if candidate.grade.mean > baseline.grade.mean:
        return False
    return candidate.grade.spread > baseline.grade.spread


def rank_candidates(
    original: RankingCandidate,
    candidates: Sequence[RankingCandidate],
    *,
    strong_check: StrongCheckReport | None = None,
    allow_unverified_selection: bool = False,
) -> RankingResult:
    """Rank changed candidates by worst, mean, spread, then prompt length.

    The always-improve policy: an eligible candidate that changed the prompt
    beats the unchanged original when it is at least as good on worst and
    mean (ties break toward the shorter prompt). A candidate that is strictly
    worse than the original is never selected; candidates that do not
    meaningfully differ from the original are rejected as unchanged; and a
    zero-pass-rate candidate never displaces the original, because a rewrite
    that passes nothing is not an improvement.
    ``strong_check`` is the strong check's report; this function only consumes
    each candidate's verdict and never repeats the strong-model comparison.
    ``allow_unverified_selection`` covers runs with no success tests: pass
    rates do not exist (they are not zero), so the best fidelity-eligible
    changed candidate wins and the unverified evidence status — recorded by
    the caller — carries the uncertainty instead of the selection floor.
    """

    baseline = original
    parsed = list(candidates)
    eligible: list[RankingCandidate] = []
    reasons: dict[str, list[str]] = {}
    for candidate in parsed:
        candidate_reasons = list(candidate.rejection_reasons)
        if not candidate.eligible and not candidate_reasons:
            candidate_reasons.append("candidate marked ineligible by an upstream check")
        if not prompts_differ_meaningfully(baseline.text, candidate.text):
            candidate_reasons.append(
                "candidate did not change the prompt (always-improve policy)"
            )
        strong_passed, strong_reason = _strong_decision(candidate, strong_check)
        if strong_passed is False:
            candidate_reasons.append(strong_reason or "strong check did not pass")
        if not candidate_reasons and candidate.grade is None:
            candidate_reasons.append("candidate has no grading report")
        if not candidate_reasons and _worse_than_baseline(candidate, baseline):
            candidate_reasons.append("does not beat the original under robust ranking")
        if candidate_reasons:
            reasons[candidate.candidate_id] = candidate_reasons
        else:
            eligible.append(candidate)

    eligible.sort(key=lambda candidate: (_key(candidate), candidate.candidate_id))
    # A changed candidate wins ties with the original only when it actually
    # passes something: a zero-pass-rate rewrite is indistinguishable from a
    # broken prompt, so it never displaces the original. Runs without success
    # tests have no pass rates at all, so the floor cannot apply there; the
    # caller opts into selecting the best eligible candidate instead.
    best = eligible[0] if eligible else None
    selected = (
        best
        if best is not None
        and (
            allow_unverified_selection
            or best.grade is None
            or (best.grade.worst > 0 or baseline.grade is None)
        )
        else None
    )
    selected_id = selected.candidate_id if selected else None

    ranked: list[RankedCandidate] = []
    for index, candidate in enumerate(eligible, start=1):
        candidate_reasons = list(reasons.get(candidate.candidate_id, ()))
        if selected is None:
            candidate_reasons.append("does not beat the original under robust ranking")
        elif candidate.candidate_id != selected_id:
            candidate_reasons.append(f"ranked below selected candidate {selected_id}")
        if candidate_reasons:
            reasons[candidate.candidate_id] = candidate_reasons
        ranked.append(
            RankedCandidate(
                candidate,
                index,
                _key(candidate),
                selected is not None and candidate.candidate_id == selected_id,
                tuple(candidate_reasons),
            )
        )
    # Ineligible candidates still need a place in the report, even though they
    # are not part of the performance ordering.
    for candidate in parsed:
        if candidate in eligible:
            continue
        ranked.append(
            RankedCandidate(
                candidate,
                0,
                _key(candidate),
                False,
                tuple(reasons.get(candidate.candidate_id, ("candidate rejected",))),
            )
        )
    ranked.sort(
        key=lambda item: (item.rank == 0, item.rank, item.candidate.candidate_id)
    )

    return RankingResult(
        selected=selected,
        original=baseline,
        original_kept=selected is None,
        ranked=tuple(ranked),
        rejection_reasons={key: tuple(value) for key, value in reasons.items()},
    )


__all__ = ["RankedCandidate", "RankingCandidate", "RankingResult", "rank_candidates"]
