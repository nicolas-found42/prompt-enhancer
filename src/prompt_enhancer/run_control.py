"""Run control: cancel, budget pause, and approval resume for long loops.

This module owns the run-control seam on the optimizer's public interface:
optional time/spend limits pause a run at a Round boundary into an
awaiting-approval state, cancel ends a run while preserving completed
rounds, and approval either continues from the paused boundary or stops
permanently. Progress payloads carry elapsed time and accumulated cost.

Persistence reuses :class:`prompt_enhancer.store.RunStore`: the paused
result is saved like any other run, and the resume context travels in the
same record under :data:`RESUME_CONTEXT_KEY`, so a paused run survives a
backend restart.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, cast

from .convergence import mean_score
from .failures import RunCancelled, describe_failure
from .models import OptimizeResult
from .outcomes import apply_outcome_fields
from .repeat import RoundEvidence

TIME_LIMIT_OPTION = "time_limit_s"
SPEND_LIMIT_OPTION = "spend_limit_usd"

PAUSED_REPORT_STATUS = "awaiting_approval"
STOPPED_REPORT_STATUS = "stopped"
CANCELLED_REPORT_STATUS = "cancelled"

RESUME_CONTEXT_KEY = "run_control"
"""Record-level RunStore key holding the resume context of a paused run."""


@dataclass(frozen=True, slots=True)
class RunControl:
    """Optional user-set limits; reaching one pauses at a Round boundary.

    ``time_limit_s`` remains a user approval control checked after a completed
    Round when another Round would start. It does not bound an in-flight
    Gateway operation; ``GatewayConfig.operation_timeout_s`` does that.
    """

    time_limit_s: float | None = None
    spend_limit_usd: float | None = None

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> RunControl:
        mapping = options or {}
        return cls(
            time_limit_s=_parse_limit(
                mapping.get(TIME_LIMIT_OPTION), TIME_LIMIT_OPTION
            ),
            spend_limit_usd=_parse_limit(
                mapping.get(SPEND_LIMIT_OPTION), SPEND_LIMIT_OPTION
            ),
        )

    @property
    def active(self) -> bool:
        return self.time_limit_s is not None or self.spend_limit_usd is not None

    def as_options(self) -> dict[str, float]:
        options: dict[str, float] = {}
        if self.time_limit_s is not None:
            options[TIME_LIMIT_OPTION] = self.time_limit_s
        if self.spend_limit_usd is not None:
            options[SPEND_LIMIT_OPTION] = self.spend_limit_usd
        return options


def _parse_limit(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(f"{name} must be a finite number at least 0")
    return amount


class BudgetPaused(RuntimeError):
    """A limit triggered at a Round boundary; the run awaits approval."""

    def __init__(
        self,
        *,
        reason: str,
        history: tuple[dict[str, Any], ...],
        spent_usd: float,
        elapsed_ms: int,
    ) -> None:
        super().__init__(f"run paused: {reason} limit reached")
        self.reason = reason
        self.history = history
        self.spent_usd = spent_usd
        self.elapsed_ms = elapsed_ms


@dataclass
class RoundTracker:
    """Completed-round evidence for cancel/pause/stop payloads."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    final_prompt: str | None = None
    original_kept: bool = True

    def record(self, request: Any, outcome: Any) -> None:
        evidence = RoundEvidence.from_outcome(
            round_number=request.round_number,
            outcome=outcome,
        )
        self.entries.append(evidence.to_dict())
        self.final_prompt = _best_tracked_prompt(self.entries, outcome.final_prompt)
        self.original_kept = (
            self.final_prompt == outcome.plan.prompt
            if hasattr(outcome, "plan")
            else outcome.original_kept
        )

    @classmethod
    def preload(
        cls,
        entries: list[dict[str, Any]],
        final_prompt: str | None,
        original_kept: bool,
    ) -> RoundTracker:
        copied_entries = [dict(entry) for entry in entries]
        best_prompt = _best_tracked_prompt(copied_entries, final_prompt)
        return cls(
            entries=copied_entries,
            final_prompt=best_prompt,
            original_kept=(
                best_prompt == final_prompt
                if best_prompt != final_prompt
                else original_kept
            ),
        )

    @property
    def history(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries]


def _best_tracked_prompt(
    entries: list[dict[str, Any]], fallback: str | None
) -> str | None:
    """Keep the best floor-passing selected prompt for pause/cancel payloads."""
    best_prompt = fallback
    best_score = -1.0
    for entry in entries:
        vector = entry.get("convergence")
        if not isinstance(vector, Mapping) or vector.get("passed") is not True:
            continue
        if vector.get("selected") is not True or not entry.get("selected_candidate_id"):
            continue
        scores = vector.get("scores")
        selection = entry.get("evidence")
        selection = selection if isinstance(selection, Mapping) else {}
        selection = selection.get("selection_evidence")
        selection = selection if isinstance(selection, Mapping) else {}
        candidate = selection.get("selected_candidate")
        if not isinstance(candidate, Mapping):
            continue
        prompt = candidate.get("text") or candidate.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            continue
        score = mean_score(scores) if isinstance(scores, Mapping) else -1.0
        if score >= best_score:
            best_prompt = prompt
            best_score = score
    return best_prompt


@dataclass
class RunControlState:
    """Per-run budget tracking shared by the round loop and progress events."""

    control: RunControl
    tracker: RoundTracker
    usage_total: Callable[[], float]
    started_perf: float
    cost_base: float = 0.0
    elapsed_base_ms: int = 0

    def spent_usd(self) -> float:
        return max(0.0, self.cost_base + self.usage_total())

    def elapsed_ms(self) -> int:
        return self.elapsed_base_ms + max(
            0, round((perf_counter() - self.started_perf) * 1000)
        )

    def note_completed(self, request: Any, outcome: Any) -> None:
        """Record a completed round, pausing when a limit is reached.

        A limit only pauses when the loop would otherwise spend more: the
        round must request continuation. There is no round cap (#169), so a
        limit stays able to pause the loop at any round boundary; a finished
        loop returns its final result instead of pausing.
        """
        self.tracker.record(request, outcome)
        if not self.control.active:
            return
        if not outcome.continue_rounds:
            return
        reason: str | None = None
        if (
            self.control.spend_limit_usd is not None
            and self.spent_usd() >= self.control.spend_limit_usd
        ):
            reason = "spend_limit"
        elif (
            self.control.time_limit_s is not None
            and self.elapsed_ms() >= self.control.time_limit_s * 1000
        ):
            reason = "time_limit"
        if reason is None:
            return
        raise BudgetPaused(
            reason=reason,
            history=tuple(self.tracker.history),
            spent_usd=self.spent_usd(),
            elapsed_ms=self.elapsed_ms(),
        )


def limit_text(reason: str, control: RunControl) -> str:
    if reason == "spend_limit" and control.spend_limit_usd is not None:
        return f"spend limit of ${control.spend_limit_usd:.2f}"
    if reason == "time_limit" and control.time_limit_s is not None:
        return f"time limit of {control.time_limit_s:g}s"
    return "limit"


def _rounds_text(count: int) -> str:
    return f"{count} completed round" + ("" if count == 1 else "s")


def build_paused_result(
    *,
    run_id: str,
    prompt: str,
    models: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    assumptions: Any,
    tracker: RoundTracker,
    control: RunControl,
    paused: BudgetPaused,
    cost: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    rounds = len(tracker.entries)
    final_prompt = tracker.final_prompt or prompt
    return {
        "status": "needs_input",
        "run_id": run_id,
        "final_prompt": final_prompt,
        "original_kept": tracker.original_kept,
        "report": {
            "status": PAUSED_REPORT_STATUS,
            "summary": (
                f"Paused after {_rounds_text(rounds)} at your "
                f"{limit_text(paused.reason, control)} "
                f"(${paused.spent_usd:.4f} spent). "
                "Continue to keep improving, or stop to keep what is done so far."
            ),
            "pause": {
                "reason": paused.reason,
                "limit": (
                    control.spend_limit_usd
                    if paused.reason == "spend_limit"
                    else control.time_limit_s
                ),
                "spent_usd": paused.spent_usd,
                "elapsed_ms": paused.elapsed_ms,
                "completed_rounds": rounds,
            },
            "diagnosis": dict(diagnosis),
            "assumptions": list(assumptions),
            "models": dict(models),
            "history": tracker.history,
        },
        "cost": cost,
        "timing": timing,
    }


def build_cancelled_result(
    *,
    run_id: str,
    prompt: str,
    tracker: RoundTracker,
    cost: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    rounds = len(tracker.entries)
    failure = describe_failure(RunCancelled(run_id))
    if rounds:
        kept = (
            "Your best prompt so far is kept below."
            if not tracker.original_kept
            else "Your original prompt was kept."
        )
        failure["hint"] = f"You cancelled this run after {_rounds_text(rounds)}. {kept}"
    return {
        "status": "failed",
        "run_id": run_id,
        "final_prompt": tracker.final_prompt or prompt,
        "original_kept": tracker.original_kept if rounds else True,
        "report": {
            "status": CANCELLED_REPORT_STATUS,
            "summary": failure["hint"],
            "error": failure["message"],
            "failure": failure,
            "diagnosis": {"confirmed_gaps": [], "problem_sentences": []},
            "assumptions": [],
            "history": tracker.history,
        },
        "cost": cost,
        "timing": timing,
    }


def build_stopped_result(
    *,
    paused_result: Mapping[str, Any],
    original_prompt: str,
    cost: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    raw_report = paused_result.get("report")
    report: Mapping[str, Any] = raw_report if isinstance(raw_report, Mapping) else {}
    history = list(report.get("history") or [])
    raw_pause = report.get("pause")
    pause: Mapping[str, Any] = raw_pause if isinstance(raw_pause, Mapping) else {}
    spent_value = pause.get("spent_usd", 0.0)
    spent = float(spent_value) if isinstance(spent_value, (int, float)) else 0.0
    result = dict(paused_result)
    result["status"] = "failed"
    stopped_report = {
        **dict(report),
        "status": STOPPED_REPORT_STATUS,
        "summary": (
            f"You stopped this run after {_rounds_text(len(history))} "
            f"(${spent:.4f} spent). What was completed so far is kept below."
        ),
        "failure": {
            "kind": "stopped",
            "headline": "Run stopped",
            "hint": (
                "You stopped this run before it converged. "
                "The completed rounds below are unchanged."
            ),
            "message": "stopped by the user",
        },
    }
    result["report"] = apply_outcome_fields(
        stopped_report,
        original_prompt=original_prompt,
        final_prompt=str(result.get("final_prompt") or original_prompt),
        control_state="stopped",
    )
    result["cost"] = dict(cost)
    result["timing"] = dict(timing)
    return result


def resume_context(
    *,
    options: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    assumptions: Any,
    seed: int,
    spent_usd: float,
    elapsed_ms: int,
) -> dict[str, Any]:
    return {
        "options": dict(options),
        "diagnosis": dict(diagnosis),
        "assumptions": list(assumptions),
        "seed": seed,
        "spent_usd": spent_usd,
        "elapsed_ms": elapsed_ms,
    }


def take_resume_context(result: dict[str, Any]) -> dict[str, Any] | None:
    """Remove the ephemeral resume context from a result before persisting it."""
    context = result.pop(RESUME_CONTEXT_KEY, None)
    return dict(context) if isinstance(context, Mapping) else None


def as_optimize_result(payload: Mapping[str, Any]) -> OptimizeResult:
    result: dict[str, Any] = {
        "status": payload["status"],
        "run_id": str(payload["run_id"]),
        "report": dict(payload.get("report") or {}),
        "cost": dict(payload.get("cost") or {}),
        "timing": dict(payload.get("timing") or {}),
        "final_prompt": str(payload.get("final_prompt") or ""),
        "original_kept": bool(payload.get("original_kept", True)),
    }
    if RESUME_CONTEXT_KEY in payload:
        # Ephemeral resume context: persisted to the record by the caller,
        # never shown to API clients as part of the result.
        result[RESUME_CONTEXT_KEY] = payload[RESUME_CONTEXT_KEY]
    return cast(OptimizeResult, result)
