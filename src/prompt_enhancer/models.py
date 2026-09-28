"""Small, serializable contracts shared by the engine and HTTP surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4


@dataclass(frozen=True)
class TierBudget:
    """The hard search budget for one tier: per round, and how many rounds."""

    candidates: int
    models: int
    samples: int
    max_rounds: int
    name: str = "standard"
    grading_confirmation_pairs: int = 0
    grading_cascade_dollars: float = 0.0
    attribution_pairs: int = 0
    attribution_dollars: float = 0.0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "tier": self.name,
            "candidates": self.candidates,
            "models": self.models,
            "samples": self.samples,
            "max_rounds": self.max_rounds,
        }


class Tier(StrEnum):
    """How much effort a run spends: Fast, Standard, or Deep."""

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"

    @classmethod
    def parse(cls, value: Any) -> Tier:
        """Read a tier from user input; a missing tier means Standard."""
        tier = str(value or "standard").strip().lower()
        try:
            return cls(tier)
        except ValueError:
            raise ValueError("tier must be one of: fast, standard, deep") from None

    @property
    def budget(self) -> TierBudget:
        return _BUDGETS[self]

    @property
    def max_rounds(self) -> int:
        return self.budget.max_rounds

    @property
    def weak_model_evaluations(self) -> int:
        """Weak-model outputs the tier may run across all of its rounds."""
        budget = self.budget
        return budget.candidates * budget.models * budget.samples * budget.max_rounds


_BUDGETS: dict[Tier, TierBudget] = {
    Tier.FAST: TierBudget(candidates=3, models=2, samples=1, max_rounds=1, name="fast"),
    Tier.STANDARD: TierBudget(
        candidates=4,
        models=3,
        samples=2,
        max_rounds=2,
        name="standard",
        grading_confirmation_pairs=10,
        grading_cascade_dollars=0.02,
        attribution_pairs=10,
        attribution_dollars=0.01,
    ),
    Tier.DEEP: TierBudget(
        candidates=6,
        models=5,
        samples=3,
        max_rounds=3,
        name="deep",
        grading_confirmation_pairs=30,
        grading_cascade_dollars=0.05,
        attribution_pairs=30,
        attribution_dollars=0.03,
    ),
}

RunStatus = Literal["completed", "needs_input", "failed"]


class CostBreakdown(TypedDict, total=False):
    total: float
    by_role: dict[str, float]


class Timing(TypedDict, total=False):
    total_ms: int
    started_at: str
    finished_at: str


class OptimizeResult(TypedDict):
    status: RunStatus
    run_id: str
    report: dict[str, Any]
    cost: CostBreakdown
    timing: Timing
    final_prompt: NotRequired[str]
    original_kept: NotRequired[bool]
    questions: NotRequired[list[dict[str, Any]]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    return uuid4().hex
