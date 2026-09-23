"""Thin HTTP surface for the local prompt optimizer."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings
from .optimizer import PromptOptimizer, RunNotFoundError
from .repeat import _reject_provider_secrets
from .store import RunStore


class OptimizeRequest(BaseModel):
    prompt: str
    tier: str = "standard"
    options: dict[str, Any] = Field(default_factory=dict)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    clarification_allowed: bool | None = None


class AnswersRequest(BaseModel):
    answers: dict[str, Any] = Field(default_factory=dict)


class AssumptionRequest(BaseModel):
    assumption: dict[str, Any]


def _public_catalog(settings: Settings) -> dict[str, Any]:
    models = {
        "judge": [
            {"id": settings.judge_model, "provider": "openrouter", "fixed": True}
        ],
        "writer": [{"id": settings.writer_model, "provider": "opencode-go"}],
        "strong_check": [
            {"id": settings.strong_check_model, "provider": "opencode-go"}
        ],
        "weak_panel": [
            {"id": model, "provider": "openrouter"} for model in settings.weak_models
        ],
    }
    return {"models": models, **models}


def create_app(
    optimizer: PromptOptimizer | None = None,
    store: RunStore | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create an app with injectable engine/store seams for local testing."""
    app_settings = settings or Settings.from_env()
    app_store = store or (
        optimizer.store
        if optimizer is not None
        else RunStore(app_settings.database_path)
    )
    app_optimizer = optimizer or PromptOptimizer(store=app_store, config=app_settings)
    app = FastAPI(title="Prompt Enhancer", version="0.1.0")
    app.state.optimizer = app_optimizer
    app.state.store = app_store
    app.state.settings = app_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/optimize")
    def optimize(request: OptimizeRequest) -> dict[str, Any]:
        options = dict(request.options)
        options["tier"] = request.tier
        if request.model_overrides:
            options["model_overrides"] = {
                **options.get("model_overrides", {}),
                **request.model_overrides,
            }
        if request.clarification_allowed is not None:
            options["clarification_allowed"] = request.clarification_allowed
        try:
            return dict(app_optimizer.optimize(request.prompt, options))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs(limit: int = 50, q: str | None = None) -> list[dict[str, Any]]:
        records = app_store.list_runs(limit=limit)
        if q:
            needle = q.casefold()
            records = [
                record
                for record in records
                if needle in str(record.get("prompt", "")).casefold()
            ]
        return records

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        record = app_store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return record

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str, request: AnswersRequest) -> dict[str, Any]:
        try:
            return dict(app_optimizer.resume(run_id, request.answers))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/api/optimize/resume/{run_id}")
    def optimize_resume(run_id: str, request: AnswersRequest) -> dict[str, Any]:
        try:
            return dict(app_optimizer.resume(run_id, request.answers))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/api/optimize/skip/{run_id}")
    def optimize_skip(run_id: str) -> dict[str, Any]:
        try:
            return dict(app_optimizer.resume(run_id, {"__skip__": True}))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/deep")
    def start_deep(run_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            _reject_provider_secrets(body or {})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        method = getattr(app_optimizer, "start_deep_pass", None)
        if body:
            raise HTTPException(
                status_code=422,
                detail="The deep endpoint does not accept request options",
            )
        if method is None:
            raise HTTPException(
                status_code=501, detail="deep escalation is not configured"
            )
        try:
            return dict(method(run_id))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/assumption")
    def update_assumption(run_id: str, request: AssumptionRequest) -> dict[str, Any]:
        try:
            return dict(app_optimizer.update_assumption(run_id, request.assumption))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return _public_catalog(app_settings)

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return app_settings.public_dict()

    @app.put("/api/settings")
    def put_settings(values: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "judge_model",
            "writer_model",
            "strong_check_model",
            "weak_models",
        }
        for key in allowed:
            if key in values:
                setattr(app_settings, key, values[key])
        return app_settings.public_dict()

    return app


app = create_app()
