"""Strict robust candidate selection and rejection reporting.

The selector only orders already-graded candidates.  Hard checks and the
strong reference check are injected as eligibility decisions (the strong-check
module owns those decisions), so this module never duplicates either policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RankingCandidate:
    """A candidate carrying its grade and an external eligibility decision."""

    candidate_id: str
    text: str = ""
    strategy: str = ""
    strategy_kind: str = "safe"
    grade: Any = None
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
        selected_grade = (
            _serialize_grade(self.selected.grade) if self.selected else None
        )
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


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping):
            if name in value:
                return value[name]
        elif hasattr(value, name):
            return getattr(value, name)
    return default


def _serialize_grade(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "worst": _metric(value, "worst"),
        "mean": _metric(value, "mean"),
        "spread": _metric(value, "spread"),
    }


def _metric(grade: Any, name: str) -> float:
    aliases = {
        "worst": ("worst", "worst_model", "worst_model_pass_rate", "worst_rate"),
        "mean": ("mean", "mean_pass_rate", "average", "average_pass_rate"),
        "spread": ("spread", "sample_spread", "sample_variation", "variation"),
    }[name]
    value = _field(grade, *aliases, default=None)
    if value is None and name == "worst":
        rates = _field(grade, "per_model", "per_model_pass_rates", default={})
        if isinstance(rates, Mapping) and rates:
            numeric = [float(item) for item in rates.values()]
            if numeric:
                return min(numeric)
    if value is None and name == "mean":
        rates = _field(grade, "per_model", "per_model_pass_rates", default={})
        if isinstance(rates, Mapping) and rates:
            numeric = [float(item) for item in rates.values()]
            if numeric:
                return sum(numeric) / len(numeric)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _length(candidate: Any) -> int:
    value = _field(candidate, "length", "prompt_length", default=None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    text = _field(candidate, "text", "prompt", default="")
    return len(str(text))


def _strategy_kind(candidate: Any) -> str:
    value = _field(candidate, "strategy_kind", "kind", default=None)
    if value is not None:
        return str(value)
    strategy = _field(candidate, "strategy", default="")
    if isinstance(strategy, Mapping):
        value = strategy.get("kind", strategy.get("type"))
        if value is not None:
            return str(value)
    if str(strategy) in {
        "split_into_steps",
        "add_example",
        "role_play",
        "repeated_emphasis",
    }:
        return "crutch"
    return "safe"


def _candidate(value: Any, *, default_id: str = "candidate") -> RankingCandidate:
    if isinstance(value, RankingCandidate):
        return value
    candidate_id = str(_field(value, "candidate_id", "id", default=default_id))
    text = str(_field(value, "text", "prompt", default="") or "")
    strategy = _field(value, "strategy", default="")
    if isinstance(strategy, Mapping):
        strategy_name = str(strategy.get("name", strategy.get("id", "")))
    else:
        strategy_name = str(strategy or "")
    grade = _field(value, "grade", "metrics", "score", default=None)
    eligible_value = _field(value, "eligible", "passed", default=True)
    reasons_value = _field(value, "rejection_reasons", "reasons", default=())
    if isinstance(reasons_value, str):
        reasons = (reasons_value,)
    else:
        reasons = tuple(str(reason) for reason in (reasons_value or ()))
    metadata = _field(value, "metadata", default={})
    return RankingCandidate(
        candidate_id=candidate_id,
        text=text,
        strategy=strategy_name,
        strategy_kind=_strategy_kind(value),
        grade=grade,
        eligible=bool(eligible_value),
        rejection_reasons=reasons,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _strong_decision(
    candidate: RankingCandidate, strong_check: Any
) -> tuple[bool | None, str | None]:
    # A candidate-level strong outcome is enough when no report object exists.
    if strong_check is None:
        return None, None
    value = strong_check
    if isinstance(strong_check, Mapping) and "outcomes" not in strong_check:
        value = strong_check
    else:
        outcome_for = _field(strong_check, "outcome_for", default=None)
        if callable(outcome_for):
            try:
                value = outcome_for(candidate.candidate_id)
            except (AttributeError, KeyError, TypeError, ValueError):
                value = None
        elif isinstance(strong_check, Mapping):
            outcomes = strong_check.get("outcomes", {})
            if isinstance(outcomes, Mapping):
                value = outcomes.get(candidate.candidate_id)
        else:
            value = _field(strong_check, "outcomes", default=None)
            if isinstance(value, Mapping):
                value = value.get(candidate.candidate_id)
    if value is None:
        passed_candidates = _field(strong_check, "passed_candidates", default=None)
        if passed_candidates is not None:
            if isinstance(passed_candidates, Mapping):
                passed = candidate.candidate_id in passed_candidates
            else:
                passed = candidate.candidate_id in tuple(passed_candidates)
            return passed, (None if passed else "strong check did not pass")
        return None, None
    passed = _field(value, "passed", "eligible", "success", default=None)
    if passed is None:
        return None, None
    reason = _field(value, "reason", "rejection_reason", default=None)
    if bool(passed):
        return True, None
    return False, str(reason or "strong check failed")


def _key(candidate: RankingCandidate) -> tuple[float, float, float, int]:
    return (
        -_metric(candidate.grade, "worst"),
        -_metric(candidate.grade, "mean"),
        _metric(candidate.grade, "spread"),
        _length(candidate),
    )


def _original(value: Any, original_grade: Any = None) -> RankingCandidate:
    if isinstance(value, RankingCandidate) and original_grade is None:
        return value
    if isinstance(value, str):
        return RankingCandidate(
            "original", value, "original", "baseline", original_grade
        )
    if isinstance(value, Mapping):
        grade = original_grade or value.get("grade", value.get("metrics"))
        return RankingCandidate(
            str(value.get("candidate_id", value.get("id", "original"))),
            str(value.get("text", value.get("prompt", "")) or ""),
            "original",
            "baseline",
            grade,
        )
    grade = original_grade or _field(value, "grade", "metrics", default=None)
    return RankingCandidate(
        str(_field(value, "candidate_id", "id", default="original")),
        str(_field(value, "text", "prompt", default="") or ""),
        "original",
        "baseline",
        grade,
    )


def rank_candidates(
    original: Any,
    candidates: Sequence[Any],
    *,
    original_grade: Any = None,
    strong_check: Any = None,
) -> RankingResult:
    """Rank eligible candidates by worst, mean, spread, then prompt length.

    The original is retained unless the best eligible candidate is strictly
    better on worst, mean, spread, or length. ``strong_check`` is an external report or
    outcome map; this function only consumes its ``passed``/``eligible``
    decision and never repeats the strong-model comparison.
    """

    baseline = _original(original, original_grade)
    parsed = [
        _candidate(value, default_id=f"candidate-{index + 1}")
        for index, value in enumerate(candidates)
    ]
    eligible: list[RankingCandidate] = []
    reasons: dict[str, list[str]] = {}
    for candidate in parsed:
        candidate_reasons = list(candidate.rejection_reasons)
        if not candidate.eligible and not candidate_reasons:
            candidate_reasons.append("candidate marked ineligible by an upstream check")
        strong_passed, strong_reason = _strong_decision(candidate, strong_check)
        if strong_passed is False:
            candidate_reasons.append(strong_reason or "strong check did not pass")
        if not candidate_reasons and candidate.grade is None:
            candidate_reasons.append("candidate has no grading report")
        if candidate_reasons:
            reasons[candidate.candidate_id] = candidate_reasons
        else:
            eligible.append(candidate)

    eligible.sort(key=lambda candidate: (_key(candidate), candidate.candidate_id))
    best = eligible[0] if eligible else None
    selected = (
        best
        if best is not None and _key(best) < _key(baseline)
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
