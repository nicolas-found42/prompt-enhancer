"""Small, serializable contracts shared by the engine and HTTP surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class WorkloadBudget:
    """Fixed per-round candidate and weak-panel workload."""

    candidates: int
    models: int
    samples: int

    def to_dict(self) -> dict[str, int]:
        return {
            "candidates": self.candidates,
            "models": self.models,
            "samples": self.samples,
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
