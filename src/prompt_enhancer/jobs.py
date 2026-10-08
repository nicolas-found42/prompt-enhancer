"""Background runs with progress and cancellation for the HTTP surface.

An optimization can take several minutes, which is too long to hold one HTTP
request open: a dropped connection would lose the result.  ``RunJobs`` starts
each run on a single worker thread (the engine shares one gateway, so runs are
serialized), returns the run ID immediately, and lets clients poll, reattach
after a reload, or cancel at the next stage boundary.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .active_budget import ACTIVE_BUDGET_S, ActiveBudget, budget_scope
from .failures import RunCancelled
from .outcomes import apply_outcome_fields
from .run_control import RoundTracker, RunDeadlineReached, build_deadline_result
from .store import RunStore

ProgressCallback = Callable[[str, Mapping[str, Any]], None]
CancelCheck = Callable[[], bool]
OperationObserver = Callable[[Mapping[str, Any]], None]
JobWork = Callable[
    [ProgressCallback, CancelCheck, OperationObserver], Mapping[str, Any]
]
FailureBuilder = Callable[[BaseException], Mapping[str, Any]]


class JobNotFound(LookupError):
    """No job with this run ID is known to this server process."""


class JobBusy(RuntimeError):
    """A job for this run is already queued or running."""


@dataclass
class _Job:
    run_id: str
    kind: str
    prompt: str = ""
    state: str = "queued"
    stage: str | None = None
    round: dict[str, int] = field(default_factory=dict)
    stages_seen: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    monotonic_finished: float | None = None
    result: Mapping[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    monotonic_started: float = 0.0
    last_progress_monotonic: float | None = None
    last_progress_at: float | None = None
    operation: dict[str, Any] | None = None
    cost_total: float = 0.0
    options: dict[str, Any] = field(default_factory=dict)
    elapsed_offset_ms: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    timer: threading.Timer | None = field(default=None, repr=False)
    evidence: Callable[[], Mapping[str, Any]] | None = field(default=None, repr=False)
    on_finished: Callable[[], None] | None = field(default=None, repr=False)
    _monotonic: Callable[[], float] = field(default=time.monotonic, repr=False)

    def budget_running(self) -> bool:
        return self.state in {"queued", "running"} or (
            self.state == "done"
            and self.result is not None
            and (self.result.get("report") or {}).get("status") == "awaiting_approval"
        )

    def snapshot(self) -> dict[str, Any]:
        now = self._monotonic()
        end = self.monotonic_finished
        elapsed = self.elapsed_offset_ms + round(
            ((end if end is not None else now) - self.monotonic_started) * 1000
        )
        progress_age = (
            round((now - self.last_progress_monotonic) * 1000)
            if self.last_progress_monotonic is not None and end is None
            else (
                round(
                    ((end if end is not None else now) - self.last_progress_monotonic)
                    * 1000
                )
                if self.last_progress_monotonic is not None
                else None
            )
        )
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "prompt": self.prompt,
            "state": self.state,
            "stage": self.stage,
            "round": dict(self.round),
            "stages_seen": list(self.stages_seen),
            "elapsed_ms": max(0, elapsed),
            "last_progress_age_ms": progress_age,
            "cost_total": self.cost_total,
            "cancel_requested": self.cancel.is_set(),
            "cancellation_pending": self.cancel.is_set()
            and self.state in {"queued", "running"},
            "operation": dict(self.operation) if self.operation is not None else None,
            "result": self.result,
            "events": [dict(event) for event in self.events],
            "event_cursor": len(self.events),
            "remaining_active_ms": max(
                0, round(ACTIVE_BUDGET_S * 1000) - max(0, elapsed)
            ),
        }


class RunJobs:
    def __init__(
        self,
        *,
        keep: int = 50,
        store: RunStore | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prompt-run"
        )
        self._jobs: OrderedDict[str, _Job] = OrderedDict()
        self._lock = threading.Lock()
        self._persistence_lock = threading.RLock()
        self._keep = keep
        self._store = store
        self._monotonic = monotonic
        self._wall_time = wall_time
        if self._store is not None:
            self._recover()

    def _job_record(self, job: _Job) -> dict[str, Any]:
        return {
            "state": job.state,
            "kind": job.kind,
            "stage": job.stage,
            "round": dict(job.round),
            "stages_seen": list(job.stages_seen),
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "elapsed_ms": job.snapshot()["elapsed_ms"],
            "budget_active": job.budget_running(),
            "recorded_at": self._wall_time(),
            "last_progress_at": job.last_progress_at,
            "cancel_requested": job.cancel.is_set(),
            "operation": dict(job.operation) if job.operation else None,
            "cost_total": job.cost_total,
            "events": [dict(event) for event in job.events],
        }

    def _persist(
        self, job: _Job, *, recovery_result: Mapping[str, Any] | None = None
    ) -> None:
        if self._store is None:
            return

        def update(existing: dict[str, Any] | None) -> dict[str, Any]:
            record = existing or {
                "run_id": job.run_id,
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(job.started_at)
                ),
                "prompt": job.prompt,
                "options": {
                    key: value
                    for key, value in job.options.items()
                    if key != "configuration"
                },
                "result": {},
                "cost": {},
                "timing": {"started_at": job.started_at},
            }
            record["job"] = self._job_record(job)
            if "configuration" in job.options:
                record.setdefault("configuration", job.options["configuration"])
            saved_result = record.get("result")
            needs_result = (
                not isinstance(saved_result, Mapping)
                or not saved_result
                or saved_result.get("status") in {"queued", "running"}
            )
            if recovery_result is not None and needs_result:
                record["result"] = dict(recovery_result)
                timing = record.get("timing")
                record["timing"] = {
                    **(dict(timing) if isinstance(timing, Mapping) else {}),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            elif job.state == "done" and job.result is not None:
                # Optimizer usually saved its canonical result already. Keep any
                # richer record it wrote; generic jobs still get a durable result.
                if (
                    needs_result
                    or (job.result.get("report") or {}).get("control_state")
                    == "deadline_reached"
                ):
                    record["result"] = dict(job.result)
                    record["timing"] = dict(job.result.get("timing") or {})
                    if "jev_answers" in (job.result.get("report") or {}):
                        record["jev_answers"] = job.result["report"]["jev_answers"]
                    record["cost"] = dict(job.result.get("cost") or {})
            record["cost"] = record.get("cost") or {"total": job.cost_total}
            record["timing"] = record.get("timing") or {}
            return record

        # Ownership and the write are one critical section: an older job may
        # still finish a cancellation callback after a continuation replaced it.
        with self._persistence_lock:
            with self._lock:
                owns_record = self._jobs.get(job.run_id) is job
            if owns_record:
                self._store.update_run(job.run_id, update)

    def _recover(self) -> None:
        store = self._store
        if store is None:
            return
        all_runs = getattr(store, "all_runs", None)
        list_runs = getattr(store, "list_runs", None)
        if callable(all_runs):
            records = all_runs()
        elif callable(list_runs):
            try:
                records = list_runs(limit=200)
            except TypeError:
                records = list_runs()
        else:
            return
        for record in records:
            data = record.get("job")
            if not isinstance(data, Mapping):
                continue
            run_id = str(record.get("run_id") or "")
            if not run_id:
                continue
            now = self._monotonic()
            elapsed_ms = max(0, int(data.get("elapsed_ms") or 0))
            if data.get("budget_active") and isinstance(
                data.get("recorded_at"), (int, float)
            ):
                elapsed_ms += max(
                    0, round((self._wall_time() - data["recorded_at"]) * 1000)
                )
            if data.get("state") in {"done", "interrupted"}:
                self._jobs[run_id] = _Job(
                    run_id=run_id,
                    kind=str(data.get("kind") or "optimize"),
                    prompt=str(record.get("prompt") or ""),
                    state=str(data["state"]),
                    stage=data.get("stage"),
                    round=dict(data.get("round") or {}),
                    stages_seen=list(data.get("stages_seen") or []),
                    started_at=float(data.get("started_at") or time.time()),
                    finished_at=data.get("finished_at"),
                    monotonic_started=now,
                    monotonic_finished=None if data.get("budget_active") else now,
                    elapsed_offset_ms=elapsed_ms,
                    cost_total=float(data.get("cost_total") or 0),
                    result=dict(record.get("result") or {}),
                    events=[dict(event) for event in data.get("events", [])],
                    _monotonic=self._monotonic,
                )
                recovered = self._jobs[run_id]
                if recovered.budget_running():
                    self._start_timer(recovered)
                continue
            if data.get("state") not in {"queued", "running"}:
                continue
            saved_cost = record.get("cost")
            cost_total = data.get("cost_total")
            if isinstance(saved_cost, Mapping) and isinstance(
                saved_cost.get("total"), (int, float)
            ):
                recovered_cost = dict(saved_cost)
            else:
                recovered_cost = {"total": max(0.0, float(cost_total or 0.0))}
            prompt = str(record.get("prompt") or "")
            last_progress_at = data.get("last_progress_at")
            progress_age_ms = (
                max(0, round((time.time() - last_progress_at) * 1000))
                if isinstance(last_progress_at, (int, float))
                and not isinstance(last_progress_at, bool)
                else None
            )
            job = _Job(
                run_id=run_id,
                kind=str(data.get("kind") or "optimize"),
                prompt=prompt,
                state="interrupted",
                stage=data.get("stage"),
                round=dict(data.get("round") or {}),
                stages_seen=list(data.get("stages_seen") or []),
                started_at=float(data.get("started_at") or time.time()),
                finished_at=time.time(),
                monotonic_finished=self._monotonic(),
                result={
                    "status": "failed",
                    "run_id": run_id,
                    "original_prompt": prompt,
                    "final_prompt": prompt,
                    "original_kept": True,
                    "report": {
                        "status": "failed",
                        "summary": "The run stopped before finishing; your original prompt was kept unchanged.",
                        "outcome": "failed_operational",
                        "outcome_reason": "The server stopped before this run finished.",
                        "diagnosis": {
                            "confirmed_gaps": [],
                            "problem_sentences": [],
                        },
                        "assumptions": [],
                        "failure": {
                            "kind": "interrupted",
                            "headline": "The run stopped before finishing",
                            "hint": "Try again. The technical details below say what went wrong.",
                            "message": "The server stopped before this run finished.",
                        },
                    },
                    "cost": recovered_cost,
                    "timing": {"total_ms": elapsed_ms},
                },
                monotonic_started=now,
                elapsed_offset_ms=elapsed_ms,
                last_progress_monotonic=(
                    now - progress_age_ms / 1000
                    if progress_age_ms is not None
                    else None
                ),
                last_progress_at=last_progress_at,
                operation=dict(data["operation"])
                if isinstance(data.get("operation"), Mapping)
                else None,
                cost_total=max(0.0, float(cost_total or 0.0)),
                _monotonic=self._monotonic,
            )
            if data.get("cancel_requested"):
                job.cancel.set()
            saved_result = record.get("result")
            if (
                isinstance(saved_result, Mapping)
                and saved_result
                and saved_result.get("status") not in {"queued", "running"}
            ):
                job.result = dict(saved_result)
            self._jobs[run_id] = job
            self._persist(job, recovery_result=job.result)

    def submit(
        self,
        run_id: str,
        kind: str,
        work: JobWork,
        on_failure: FailureBuilder,
        *,
        prompt: str = "",
        options: Mapping[str, Any] | None = None,
        submitted_monotonic: float | None = None,
        evidence: Callable[[], Mapping[str, Any]] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        with self._persistence_lock:
            with self._lock:
                existing = self._jobs.get(run_id)
                if existing is not None and existing.state in {"queued", "running"}:
                    raise JobBusy(run_id)
                now = self._monotonic()
                job = _Job(
                    run_id,
                    kind,
                    prompt,
                    started_at=time.time(),
                    monotonic_started=now
                    if submitted_monotonic is None
                    else submitted_monotonic,
                    options=dict(options or {}),
                    evidence=evidence,
                    on_finished=on_finished,
                    _monotonic=self._monotonic,
                )
                saved = self._store.get_run(run_id) if self._store is not None else None
                if kind != "optimize":
                    job.cost_total = max(
                        0.0, float(((saved or {}).get("cost") or {}).get("total", 0))
                    )
                    job.elapsed_offset_ms = (
                        existing.snapshot()["elapsed_ms"]
                        if existing is not None
                        else max(
                            0,
                            int(((saved or {}).get("timing") or {}).get("total_ms", 0)),
                        )
                    )
                if existing is not None and existing.timer is not None:
                    existing.timer.cancel()
                job.events = [
                    dict(event)
                    for event in ((saved or {}).get("job") or {}).get("events", [])
                ]
                self._append_event(job, "queued", "Waiting for processing to begin.")
                self._jobs[run_id] = job
                self._jobs.move_to_end(run_id)
                while len(self._jobs) > self._keep:
                    oldest = next(iter(self._jobs))
                    if self._jobs[oldest].budget_running():
                        break
                    self._jobs.pop(oldest)
            self._persist(job)
        self._start_timer(job)
        self._executor.submit(self._run, job, work, on_failure)
        return job.snapshot()

    def _start_timer(self, job: _Job) -> None:
        job.timer = threading.Timer(
            max(0.0, (ACTIVE_BUDGET_S * 1000 - job.snapshot()["elapsed_ms"]) / 1000),
            self._expire,
            args=(job,),
        )
        job.timer.daemon = True
        job.timer.start()

    def _append_event(self, job: _Job, kind: str, summary: str, **facts: Any) -> None:
        job.events.append(
            {
                "cursor": len(job.events) + 1,
                "kind": kind,
                "summary": summary,
                "elapsed_ms": job.snapshot()["elapsed_ms"],
                "round": job.round.get("round"),
                **facts,
            }
        )

    def _expire(self, job: _Job) -> None:
        with self._persistence_lock:
            with self._lock:
                if self._jobs.get(job.run_id) is not job or not job.budget_running():
                    return
                saved = (
                    self._store.get_run(job.run_id) if self._store is not None else None
                )
                checkpoint = (saved or {}).get("checkpoint") or {}
                tracker = RoundTracker.preload(
                    list(checkpoint.get("history") or []),
                    checkpoint.get("final_prompt") or job.prompt,
                    bool(checkpoint.get("original_kept", True)),
                )
                live_evidence = dict(job.evidence()) if job.evidence is not None else {}
                live_cost = (
                    live_evidence.get("cost")
                    or checkpoint.get("cost")
                    or (saved or {}).get("cost")
                    or {}
                )
                job.cost_total = max(job.cost_total, float(live_cost.get("total") or 0))
                elapsed_ms = job.snapshot()["elapsed_ms"]
                job.cancel.set()
                job.finished_at = time.time()
                job.monotonic_finished = self._monotonic()
                job.result = build_deadline_result(
                    run_id=job.run_id,
                    prompt=job.prompt,
                    models=checkpoint.get("models") or {},
                    diagnosis=checkpoint.get("diagnosis") or {},
                    assumptions=checkpoint.get("assumptions") or [],
                    tracker=tracker,
                    deadline=RunDeadlineReached(
                        history=tuple(tracker.history),
                        spent_usd=job.cost_total,
                        elapsed_ms=elapsed_ms,
                        deadline_s=ACTIVE_BUDGET_S,
                    ),
                    cost={**dict(live_cost), "total": job.cost_total},
                    timing={"total_ms": elapsed_ms},
                )
                report = dict(job.result["report"])
                report.update(checkpoint.get("report_context") or {})
                report.update(live_evidence.get("report") or {})
                job.result["report"] = apply_outcome_fields(
                    report,
                    original_prompt=job.prompt,
                    final_prompt=job.result["final_prompt"],
                    control_state="deadline_reached",
                )
                job.state = "done"
                self._append_event(
                    job, "terminal", "The active processing deadline ended."
                )
            self._persist(job)
            self._release_evidence(job)

    def _release_evidence(self, job: _Job) -> None:
        if not job.budget_running() and self._jobs.get(job.run_id) is job:
            job.evidence = None
            if job.on_finished is not None:
                job.on_finished()
                job.on_finished = None

    def _run(self, job: _Job, work: JobWork, on_failure: FailureBuilder) -> None:
        from .publication import checkpoint_transaction, publication_allowed

        def progress(stage: str, round_info: Mapping[str, Any]) -> None:
            with self._persistence_lock:
                if job.state == "done":
                    raise RunCancelled(job.run_id)
                if job.cancel.is_set():
                    raise RunCancelled(job.run_id)
                if stage == "activity":
                    facts = {
                        key: round_info[key]
                        for key in (
                            "candidate_id",
                            "draft",
                            "diff",
                            "reasons",
                            "stage",
                            "checks",
                            "comparison",
                        )
                        if key in round_info
                    }
                    self._append_event(
                        job,
                        str(round_info.get("kind") or "waiting"),
                        str(round_info.get("summary") or "Processing your prompt."),
                        **facts,
                    )
                    self._persist(job)
                    return
                info = dict(round_info)
                info.pop("elapsed_ms", None)
                cost = info.pop("cost_total", None)
                job.stage = stage
                job.round = {
                    str(key): int(value)
                    for key, value in info.items()
                    if str(key) == "round"
                }
                if isinstance(cost, bool):
                    pass
                elif isinstance(cost, (int, float)):
                    job.cost_total = max(job.cost_total, float(cost))
                if stage not in job.stages_seen:
                    job.stages_seen.append(stage)
                self._append_event(
                    job,
                    "started",
                    stage.replace("_", " ").capitalize() + ".",
                    stage=stage,
                )
                job.last_progress_monotonic = self._monotonic()
                job.last_progress_at = time.time()
                self._persist(job)

        def cancel_check() -> bool:
            return job.cancel.is_set()

        def observe_operation(event: Mapping[str, Any]) -> None:
            with self._persistence_lock:
                if job.state == "done":
                    return
                job.operation = dict(event)
                kind = str(event.get("event") or "waiting")
                operation = (
                    str(event.get("operation") or "model service")
                    .replace("_", " ")
                    .replace(".", ": ")
                )
                summary = {
                    "start": f"Started {operation}.",
                    "end": f"Finished {operation}.",
                    "retry": "The model service is busy; waiting before another attempt.",
                    "error": f"Could not finish {operation}.",
                    "cancel_pending": "Cancellation requested.",
                }.get(kind, f"Model service: {kind.replace('_', ' ')}.")
                self._append_event(job, kind, summary, operation=operation)
                if event.get("event") == "start":
                    job.last_progress_monotonic = self._monotonic()
                    job.last_progress_at = time.time()
                self._persist(job)

        if job.state == "done":
            return
        if job.snapshot()["remaining_active_ms"] <= 0:
            self._expire(job)
            return
        job.state = "running"
        self._persist(job)

        def commit_checkpoint(transition: Callable[[], None]) -> None:
            with self._persistence_lock:
                if job.state == "done" or self._jobs.get(job.run_id) is not job:
                    raise RunCancelled(job.run_id)
                transition()

        checkpoint_token = checkpoint_transaction.set(commit_checkpoint)
        owner_token = publication_allowed.set(
            lambda: job.state != "done" and self._jobs.get(job.run_id) is job
        )
        try:
            if job.cancel.is_set():
                raise RunCancelled(job.run_id)
            with budget_scope(
                ActiveBudget(
                    self._monotonic, job.monotonic_started, job.elapsed_offset_ms
                )
            ):
                result = dict(work(progress, cancel_check, observe_operation))
            with self._persistence_lock:
                if job.state != "done":
                    if job.snapshot()["remaining_active_ms"] <= 0:
                        self._expire(job)
                    else:
                        job.result = result
        except Exception as exc:  # noqa: BLE001 - every failure must reach the client
            with self._persistence_lock:
                if job.state != "done":
                    if job.snapshot()["remaining_active_ms"] <= 0:
                        self._expire(job)
                    else:
                        job.result = dict(on_failure(exc))
        finally:
            publication_allowed.reset(owner_token)
            checkpoint_transaction.reset(checkpoint_token)
            with self._persistence_lock:
                with self._lock:
                    if job.state != "done":
                        job.finished_at = time.time()
                        job.state = "done"
                        if not job.budget_running():
                            job.monotonic_finished = self._monotonic()
                        self._append_event(
                            job,
                            "terminal",
                            "Waiting for approval; the active processing clock keeps running."
                            if job.budget_running()
                            else "Processing finished.",
                        )
                    if job.timer is not None and not job.budget_running():
                        job.timer.cancel()
                self._persist(job)
                self._release_evidence(job)

    def get(self, run_id: str, *, after_cursor: int = 0) -> dict[str, Any]:
        # A terminal result becomes observable only after its durable write.
        # Resume/edit callers may read the RunStore immediately after this poll.
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                raise JobNotFound(run_id)
            snapshot = job.snapshot()
        if snapshot["state"] != "done" and snapshot["remaining_active_ms"] > 0:
            snapshot["events"] = [
                event for event in snapshot["events"] if event["cursor"] > after_cursor
            ]
            return snapshot
        with self._persistence_lock:
            if job.budget_running() and job.snapshot()["remaining_active_ms"] <= 0:
                self._expire(job)
            snapshot = job.snapshot()
            snapshot["events"] = [
                event for event in snapshot["events"] if event["cursor"] > after_cursor
            ]
            return snapshot

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [
                job for job in self._jobs.values() if job.state in {"queued", "running"}
            ]
        return [job.snapshot() for job in jobs]

    def has_active_budget(self) -> bool:
        """Approval waits still own a watchdog after their worker has returned."""
        with self._lock:
            return any(job.budget_running() for job in self._jobs.values())

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            raise JobNotFound(run_id)
        if job.state in {"queued", "running"}:
            job.cancel.set()
            self._persist(job)
        return job.snapshot()

    def finish_approval(
        self, run_id: str, stop: Callable[[], Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Publish an explicit stop and release the approval-wait watchdog."""
        with self._persistence_lock:
            if run_id in self._jobs:
                self.get(run_id)
            result = stop()
            with self._lock:
                job = self._jobs.get(run_id)
                if job is None or job.state != "done" or not job.budget_running():
                    return result
                job.result = dict(result)
                job.monotonic_finished = self._monotonic()
                if job.timer is not None:
                    job.timer.cancel()
                self._append_event(job, "terminal", "You stopped processing.")
            self._persist(job)
            self._release_evidence(job)
            return result

    def close(self) -> None:
        """Stop local watchdogs when replacing an idle job manager."""
        for job in self._jobs.values():
            if job.timer is not None:
                job.timer.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def wait(self, run_id: str, timeout: float = 10.0) -> dict[str, Any]:
        """Block until a job finishes; used by tests."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = self.get(run_id)
            if snapshot["state"] == "done":
                return snapshot
            time.sleep(0.01)
        raise TimeoutError(run_id)
