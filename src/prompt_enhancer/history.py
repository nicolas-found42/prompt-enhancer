"""Browse persisted optimizer runs and store user feedback.

The bootstrap package owns SQLite persistence in :mod:`prompt_enhancer.store`.
This module adds the history view on top of that repository rather than
introducing a second database.  Records are normalised to a JSON-friendly
shape for the HTTP and web clients while preserving the original result and
report for engine consumers.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import Body

_VALID_FEEDBACK = {
    "accept": "accept",
    "accepted": "accept",
    "reject": "reject",
    "rejected": "reject",
}


class RunNotFound(LookupError):
    """Raised when a caller explicitly requires a missing run."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert engine/provider objects to deterministic JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _first(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _feedback(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("feedback must be 'accept' or 'reject'")
    return _VALID_FEEDBACK.get(str(value).strip().lower())


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }

def _normalise_record(record: Mapping[str, Any], existing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Flatten the canonical store record while retaining its original shape.

    Bootstrap stores ``result`` and ``report`` nested.  The history API also
    exposes those top-level fields because several UI panels need them
    independently, and later engine versions can add fields under ``metadata``.
    """
    source = dict(existing or {})
    source.update(_mapping(record))
    result = source.get("result")
    result_map = result if isinstance(result, Mapping) else {}
    report = source.get("report")
    if report is None and isinstance(result_map.get("report"), Mapping):
        report = result_map["report"]
    report_map = report if isinstance(report, Mapping) else {}

    def pick(*names: str, default: Any = None) -> Any:
        value = _first(source, *names, default=None)
        if value is None:
            value = _first(result_map, *names, default=None)
        if value is None:
            value = _first(report_map, *names, default=None)
        return _jsonable(default if value is None else value)

    original_prompt = pick("original_prompt", "originalPrompt", "prompt", "input_prompt", "inputPrompt", default="")
    final_prompt = pick("final_prompt", "finalPrompt", default=None)
    status = pick("status", default="completed" if final_prompt is not None else "needs_input")
    metadata = source.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    # Keep provider-specific and future fields searchable/visible without a
    # migration.  Known nested fields are omitted to avoid duplicate payloads.
    known = {
        "run_id", "runId", "id", "created_at", "createdAt", "updated_at", "updatedAt", "prompt",
        "input_prompt", "inputPrompt", "tier", "options", "result", "report", "metadata", "status",
        "final_prompt", "finalPrompt", "original_kept", "originalKept", "cost", "timing", "timings",
        "diagnosis", "tests", "candidates", "outputs", "grades", "questions", "answers", "models",
        "model_choices", "modelChoices", "jev_answers", "jevAnswers", "original", "original_results",
        "originalResults", "feedback", "decision", "feedback_at", "feedbackAt", "round", "round_index",
        "roundIndex", "max_rounds", "maxRounds", "escalation", "deep_offer", "deepOffer",
    }
    for key, value in source.items():
        if str(key) not in known and str(key) != "metadata":
            metadata.setdefault(str(key), _jsonable(value))
    # Repeat/escalation evidence can be supplied top-level or in result/report.
    for name in ("round", "round_index", "max_rounds", "escalation", "deep_offer"):
        aliases = {
            "round_index": ("round_index", "roundIndex"),
            "max_rounds": ("max_rounds", "maxRounds"),
            "deep_offer": ("deep_offer", "deepOffer"),
        }.get(name, (name,))
        value = _first(source, *aliases, default=None)
        if value is None:
            value = _first(result_map, *aliases, default=None)
        if value is None:
            value = _first(report_map, *aliases, default=None)
        if value is not None:
            metadata.setdefault(name, _jsonable(value))
    models = pick("models", "model_choices", "modelChoices", default={})
    if not models and isinstance(source.get("options"), Mapping):
        options = source["options"]
        models = options.get("models", options.get("model_choices", {}))
    feedback_value = _feedback(_first(source, "feedback", "decision", default=None))
    if feedback_value is None:
        feedback_value = _feedback(_first(result_map, "feedback", "decision", default=None))
    if feedback_value is None:
        feedback_value = _feedback(_first(report_map, "feedback", "decision", default=None))
    return {
        "run_id": str(_first(source, "run_id", "runId", "id", default="")),
        "id": str(_first(source, "run_id", "runId", "id", default="")),
        "created_at": _first(source, "created_at", "createdAt", default=None),
        "updated_at": _first(source, "updated_at", "updatedAt", default=None),
        "status": str(status),
        "prompt": str(original_prompt or ""),
        "original_prompt": str(original_prompt or ""),
        "final_prompt": final_prompt,
        "original_kept": pick("original_kept", "originalKept", default=None),
        "tier": pick("tier", default=None),
        "models": models,
        "diagnosis": pick("diagnosis", default={}),
        "tests": pick("tests", default=[]),
        "questions": pick("questions", default=[]),
        "answers": pick("answers", default={}),
        "candidates": pick("candidates", default=[]),
        "outputs": pick("outputs", "per_model", "perModel", default=[]),
        "grades": pick("grades", default=[]),
        "cost": pick("cost", "cost_breakdown", "costBreakdown", default={}),
        "timing": pick("timing", "timings", "duration", default={}),
        "timings": pick("timing", "timings", "duration", default={}),
        "report": _jsonable(report),
        "result": _jsonable(result),
        "original": pick("original", "original_results", "originalResults", default=None),
        "jev_answers": pick("jev_answers", "jevAnswers", "decisions", default=[]),
        "metadata": _jsonable(metadata),
        "feedback": feedback_value,
        "feedback_at": _first(source, "feedback_at", "feedbackAt", default=None),
    }


class RunHistory:
    """History repository adapter over the canonical ``RunStore``.

    The adapter intentionally calls only ``save_run``, ``get_run`` and
    ``list_runs``.  Any bootstrap-compatible store therefore remains the sole
    owner of SQLite durability, while history gains search, flattened detail,
    and feedback semantics.
    """

    def __init__(self, store: Any) -> None:
        for method in ("save_run", "get_run", "list_runs"):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"store must provide {method}()")
        self.store = store

    def _get(self, run_id: str) -> dict[str, Any] | None:
        value = self.store.get_run(str(run_id))
        if value is None:
            return None
        if isinstance(value, str):
            # A store returning an ID is not enough for a detail response.
            raise TypeError("store.get_run() must return a record or None")
        return _normalise_record(_mapping(value))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return complete evidence for a run, or ``None`` if unknown."""
        return self._get(run_id)

    def require_run(self, run_id: str) -> dict[str, Any]:
        detail = self.get_run(run_id)
        if detail is None:
            raise RunNotFound(f"run {run_id!r} was not found")
        return detail

    def save_run(self, record: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Persist a run and return its normalised detail."""
        payload = _mapping(record)
        run_id = _first(payload, "run_id", "runId", "id")
        if run_id is not None:
            run_id = str(run_id)
            existing = self._get(run_id)
            # The base store treats a save as a replacement.  Preserve durable
            # feedback and creation metadata when a later engine phase saves a
            # partial/needs-input record.
            if existing is not None:
                merged = dict(existing)
                merged.update(payload)
                if "feedback" not in payload and "decision" not in payload:
                    merged["feedback"] = existing.get("feedback")
                    merged["feedback_at"] = existing.get("feedback_at")
                if not _first(payload, "created_at", "createdAt"):
                    merged["created_at"] = existing.get("created_at")
                payload = merged
        saved = self.store.save_run(payload)
        if isinstance(saved, Mapping):
            run_id = _first(saved, "run_id", "runId", "id", default=run_id)
        elif saved is not None:
            run_id = saved
        if run_id is None:
            raise ValueError("store.save_run() did not return a run id")
        return self.require_run(str(run_id))

    def _store_list(self, *, limit: int, offset: int) -> list[Any]:
        # Ask for a generous page so local search can inspect all runs.  The
        # canonical store currently accepts limit and optional offset; fallback
        # calls keep this compatible with older bootstrap implementations.
        attempts = (
            lambda: self.store.list_runs(limit=limit, offset=offset),
            lambda: self.store.list_runs(limit=limit),
            lambda: self.store.list_runs(),
        )
        last_error: TypeError | None = None
        for attempt in attempts:
            try:
                value = attempt()
                return list(value or [])
            except TypeError as exc:
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError("store.list_runs() did not return a result")

    @staticmethod
    def _summary(detail: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: detail.get(key)
            for key in (
                "run_id", "id", "created_at", "updated_at", "status", "prompt", "original_prompt",
                "final_prompt", "original_kept", "tier", "feedback", "feedback_at", "cost", "timings",
            )
        } | {"models": detail.get("models", {})}

    @staticmethod
    def _metadata_value(metadata: Mapping[str, Any], path: str) -> Any:
        current: Any = metadata
        for part in str(path).split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    def list_runs(
        self,
        query: str | Mapping[str, Any] | None = None,
        *,
        search: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tier: str | None = None,
        status: str | None = None,
        feedback: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List newest-first runs, searching prompt text and metadata."""
        search_text = search if search is not None else (query if isinstance(query, str) else None)
        terms = dict(query) if isinstance(query, Mapping) else {}
        terms.update(metadata or {})
        wanted_feedback = _feedback(feedback) if feedback is not None else None
        # A local store is small; request enough rows to make filtering and
        # pagination stable.  Details are loaded only for filtering and then
        # reduced to summaries.
        raw = self._store_list(limit=100000, offset=0)
        details: list[dict[str, Any]] = []
        for item in raw:
            item_map = _mapping(item)
            run_id = _first(item_map, "run_id", "runId", "id")
            detail = self._get(str(run_id)) if run_id is not None else _normalise_record(item_map)
            if detail is not None:
                details.append(detail)
        details.sort(key=lambda item: (str(item.get("created_at") or ""), str(item.get("run_id") or "")), reverse=True)
        filtered: list[dict[str, Any]] = []
        for detail in details:
            if tier is not None and str(detail.get("tier") or "").casefold() != str(tier).casefold():
                continue
            if status is not None and str(detail.get("status") or "").casefold() != str(status).casefold():
                continue
            if wanted_feedback is not None and detail.get("feedback") != wanted_feedback:
                continue
            if any(self._metadata_value(detail.get("metadata", {}), str(key)) != value for key, value in terms.items()):
                continue
            if search_text:
                haystack = _dumps(detail).casefold()
                if str(search_text).casefold() not in haystack:
                    continue
            filtered.append(self._summary(detail))
        start = max(0, int(offset))
        count = max(0, int(limit))
        return filtered[start:] if count == 0 else filtered[start:start + count]

    def search_runs(self, query: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return self.list_runs(query, **kwargs)

    def record_feedback(self, run_id: str, decision: str) -> dict[str, Any]:
        """Durably store an accept/reject decision on a completed result."""
        value = _feedback(decision)
        if value is None:
            raise ValueError("feedback must be 'accept' or 'reject'")
        detail = self.require_run(str(run_id))
        if str(detail.get("status") or "").casefold() in {"needs_input", "needs-input", "paused", "running"}:
            raise ValueError("feedback is only available for a completed result")
        if detail.get("final_prompt") is None:
            raise ValueError("feedback is only available for a completed result")
        detail["feedback"] = value
        detail["feedback_at"] = _utc_now()
        self.store.save_run(detail)
        return self.require_run(str(run_id))

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()

    def __len__(self) -> int:
        return len(self._store_list(limit=100000, offset=0))


# A descriptive alias for callers that prefer repository terminology.
HistoryRepository = RunHistory


def register_history_routes(app: Any, store: Any) -> Any:
    """Attach history routes to a FastAPI app without owning its lifecycle."""
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError as exc:  # pragma: no cover - FastAPI is an API dependency
        raise RuntimeError("FastAPI is required to register history routes") from exc

    history = store if isinstance(store, RunHistory) else RunHistory(store)
    router = APIRouter(prefix="/api")

    @router.get("/runs")
    def list_runs(
        q: str | None = Query(default=None),
        search: str | None = Query(default=None),
        tier: str | None = Query(default=None),
        status: str | None = Query(default=None),
        feedback: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        runs = history.list_runs(q or search, tier=tier, status=status, feedback=feedback, limit=limit, offset=offset)
        return {"runs": runs, "count": len(runs), "total": len(runs)}

    @router.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        detail = history.get_run(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="run not found")
        return detail

    @router.post("/runs/{run_id}/feedback")
    def record_feedback(
        run_id: str, payload: Annotated[dict[str, Any], Body()]
    ) -> dict[str, Any]:
        decision = payload.get("decision", payload.get("feedback"))
        if decision is None:
            raise HTTPException(status_code=422, detail="decision is required")
        try:
            return history.record_feedback(run_id, str(decision))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except ValueError as exc:
            status = 409 if "completed" in str(exc) else 422
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    app.include_router(router)
    return app
