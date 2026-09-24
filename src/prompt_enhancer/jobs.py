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

JobWork = Callable[[Callable[[str, Mapping[str, Any]], None]], Mapping[str, Any]]
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
    result: Mapping[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        end = self.finished_at or time.time()
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "prompt": self.prompt,
            "state": self.state,
            "stage": self.stage,
            "round": dict(self.round),
            "stages_seen": list(self.stages_seen),
            "elapsed_ms": round((end - self.started_at) * 1000),
            "cancel_requested": self.cancel.is_set(),
            "result": self.result,
        }


class RunJobs:
    def __init__(self, *, keep: int = 50) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prompt-run")
        self._jobs: OrderedDict[str, _Job] = OrderedDict()
        self._lock = threading.Lock()
        self._keep = keep

    def submit(
        self, run_id: str, kind: str, work: JobWork, on_failure: FailureBuilder, *, prompt: str = ""
    ) -> dict[str, Any]:
        with self._lock:
            existing = self._jobs.get(run_id)
            if existing is not None and existing.state in {"queued", "running"}:
                raise JobBusy(run_id)
            job = _Job(run_id, kind, prompt)
            self._jobs[run_id] = job
            self._jobs.move_to_end(run_id)
            while len(self._jobs) > self._keep:
                oldest = next(iter(self._jobs))
                if self._jobs[oldest].state in {"queued", "running"}:
                    break
                self._jobs.pop(oldest)
        self._executor.submit(self._run, job, work, on_failure)
        return job.snapshot()

    def _run(self, job: _Job, work: JobWork, on_failure: FailureBuilder) -> None:
        def progress(stage: str, round_info: Mapping[str, Any]) -> None:
            if job.cancel.is_set():
                raise RunCancelled(job.run_id)
            job.stage = stage
            job.round = {str(key): int(value) for key, value in round_info.items()}
            if stage not in job.stages_seen:
                job.stages_seen.append(stage)

        job.state = "running"
        job.started_at = time.time()
        try:
            if job.cancel.is_set():
                raise RunCancelled(job.run_id)
            job.result = dict(work(progress))
        except Exception as exc:  # noqa: BLE001 - every failure must reach the client
            job.result = dict(on_failure(exc))
        finally:
            job.finished_at = time.time()
            job.state = "done"

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            raise JobNotFound(run_id)
        return job.snapshot()

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.state in {"queued", "running"}]
        return [job.snapshot() for job in jobs]

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            raise JobNotFound(run_id)
        if job.state != "done":
            job.cancel.set()
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
