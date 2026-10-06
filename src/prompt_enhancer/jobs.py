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

from .failures import RunCancelled
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
    _monotonic: Callable[[], float] = field(default=time.monotonic, repr=False)

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
        }


class RunJobs:
    def __init__(
        self,
        *,
        keep: int = 50,
        store: RunStore | None = None,
        monotonic: Callable[[], float] = time.monotonic,
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
            "last_progress_at": job.last_progress_at,
            "cancel_requested": job.cancel.is_set(),
            "operation": dict(job.operation) if job.operation else None,
            "cost_total": job.cost_total,
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
                "options": job.options,
                "result": {},
                "cost": {},
                "timing": {"started_at": job.started_at},
            }
            record["job"] = self._job_record(job)
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
                if needs_result:
                    record["result"] = dict(job.result)
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
            if not isinstance(data, Mapping) or data.get("state") not in {
                "queued",
                "running",
            }:
                continue
            run_id = str(record.get("run_id") or "")
            if not run_id:
                continue
            now = self._monotonic()
            elapsed_ms = max(0, int(data.get("elapsed_ms") or 0))
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
                    monotonic_started=now,
                    options=dict(options or {}),
                    _monotonic=self._monotonic,
                )
                self._jobs[run_id] = job
                self._jobs.move_to_end(run_id)
                while len(self._jobs) > self._keep:
                    oldest = next(iter(self._jobs))
                    if self._jobs[oldest].state in {"queued", "running"}:
                        break
                    self._jobs.pop(oldest)
            self._persist(job)
        self._executor.submit(self._run, job, work, on_failure)
        return job.snapshot()

    def _run(self, job: _Job, work: JobWork, on_failure: FailureBuilder) -> None:
        def progress(stage: str, round_info: Mapping[str, Any]) -> None:
            if job.cancel.is_set():
                raise RunCancelled(job.run_id)
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
                job.cost_total = max(0.0, float(cost))
            if stage not in job.stages_seen:
                job.stages_seen.append(stage)
            job.last_progress_monotonic = self._monotonic()
            job.last_progress_at = time.time()
            self._persist(job)

        def cancel_check() -> bool:
            return job.cancel.is_set()

        def observe_operation(event: Mapping[str, Any]) -> None:
            job.operation = dict(event)
            if event.get("event") == "start":
                job.last_progress_monotonic = self._monotonic()
                job.last_progress_at = time.time()
            self._persist(job)

        job.state = "running"
        job.started_at = time.time()
        job.monotonic_started = self._monotonic()
        self._persist(job)
        try:
            if job.cancel.is_set():
                raise RunCancelled(job.run_id)
            job.result = dict(work(progress, cancel_check, observe_operation))
        except Exception as exc:  # noqa: BLE001 - every failure must reach the client
            job.result = dict(on_failure(exc))
        finally:
            with self._persistence_lock:
                with self._lock:
                    job.finished_at = time.time()
                    job.monotonic_finished = self._monotonic()
                    job.state = "done"
                self._persist(job)

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            raise JobNotFound(run_id)
        return job.snapshot()

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [
                job for job in self._jobs.values() if job.state in {"queued", "running"}
            ]
        return [job.snapshot() for job in jobs]

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            raise JobNotFound(run_id)
        if job.state in {"queued", "running"}:
            job.cancel.set()
            self._persist(job)
        return job.snapshot()

    def wait(self, run_id: str, timeout: float = 10.0) -> dict[str, Any]:
        """Block until a job finishes; used by tests."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = self.get(run_id)
            if snapshot["state"] == "done":
                return snapshot
            time.sleep(0.01)
        raise TimeoutError(run_id)
