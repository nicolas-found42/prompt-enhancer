"""Small, serializable contracts shared by the engine and HTTP surfaces."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

Tier = Literal["fast", "standard", "deep"]
RunStatus = Literal["completed", "needs_input"]


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


def normalize_tier(value: Any) -> str:
    tier = str(value or "standard").strip().lower()
    if tier in {"fast", "standard", "deep"}:
        return tier
    raise ValueError("tier must be one of: fast, standard, deep")
