"""Thin HTTP surface for the local prompt optimizer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .clarification import RunNotPausedError
from .config import Settings
from .history import RunNotFound
from .optimizer import PromptOptimizer, RunNotFoundError
from .repeat import _reject_provider_secrets
from .settings import ModelDefaults, SettingsStore
from .store import RunStore


class OptimizeRequest(BaseModel):
    prompt: str
    tier: str = "standard"
    options: dict[str, Any] = Field(default_factory=dict)
    model_overrides: dict[str, Any] = Field(default_factory=dict)
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
    app_settings = settings or getattr(optimizer, "config", None) or Settings.from_env()
    app_store = store or (
        optimizer.store
        if optimizer is not None
        else RunStore(app_settings.database_path)
    )
    settings_store = None
    if getattr(app_store, "path", ":memory:") != ":memory:":
        settings_store = SettingsStore(
            Path(app_store.path).with_suffix(".settings.json"),
            defaults=ModelDefaults(
                writer=app_settings.writer_model,
                strong=app_settings.strong_check_model,
                weak=app_settings.weak_models,
            ),
        )
        defaults = settings_store.load().defaults
        app_settings.writer_model = defaults.writer
        app_settings.strong_check_model = defaults.strong
        app_settings.weak_models = defaults.weak
    app_optimizer = optimizer or PromptOptimizer(store=app_store, config=app_settings)
    app = FastAPI(title="Prompt Enhancer", version="0.1.0")
    app.state.optimizer = app_optimizer
    app.state.store = app_store
    app.state.settings = app_settings
    app.state.settings_store = settings_store
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
    def list_runs(
        limit: int = 50,
        q: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        return app_optimizer.history.list_runs(q or search, limit=limit)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        record = app_optimizer.history.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return record

    @app.post("/api/runs/{run_id}/feedback")
    def record_feedback(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        decision = payload.get("decision", payload.get("feedback"))
        if decision is None:
            raise HTTPException(status_code=422, detail="decision is required")
        try:
            return app_optimizer.history.record_feedback(run_id, str(decision))
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except ValueError as exc:
            status = 409 if "completed" in str(exc) else 422
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    def continue_run(operation: Callable[[], Any]) -> dict[str, Any]:
        try:
            return dict(operation())
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except RunNotPausedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str, request: AnswersRequest) -> dict[str, Any]:
        return continue_run(lambda: app_optimizer.resume(run_id, request.answers))

    @app.post("/api/optimize/resume/{run_id}")
    def optimize_resume(run_id: str, request: AnswersRequest) -> dict[str, Any]:
        return continue_run(lambda: app_optimizer.resume(run_id, request.answers))

    @app.post("/api/optimize/skip/{run_id}")
    def optimize_skip(run_id: str) -> dict[str, Any]:
        return continue_run(lambda: app_optimizer.skip_clarification(run_id))

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
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/assumption")
    def update_assumption(run_id: str, request: AssumptionRequest) -> dict[str, Any]:
        try:
            return dict(app_optimizer.update_assumption(run_id, request.assumption))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        try:
            return app_optimizer.gateway.list_models(refresh=True).to_public_dict()
        except Exception as exc:
            if app_settings.openrouter_api_key or app_settings.opencode_go_key:
                raise HTTPException(status_code=503, detail="model catalog is unavailable") from exc
            return _public_catalog(app_settings)

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return app_settings.public_dict()

    @app.put("/api/settings")
    def put_settings(values: dict[str, Any]) -> dict[str, Any]:
        if "judge_model" in values and values["judge_model"] != app_settings.judge_model:
            raise HTTPException(status_code=422, detail="judge model is fixed")
        for key in ("writer_model", "strong_check_model"):
            if key in values:
                if not isinstance(values[key], str) or not values[key].strip():
                    raise HTTPException(status_code=422, detail=f"{key} must be a model ID")
                setattr(app_settings, key, values[key].strip())
        if "weak_models" in values:
            models = values["weak_models"]
            if not isinstance(models, list) or not models or any(not isinstance(model, str) or not model for model in models):
                raise HTTPException(status_code=422, detail="weak_models must be a non-empty model list")
            app_settings.weak_models = tuple(models)
        if settings_store is not None:
            settings_store.save(ModelDefaults(
                writer=app_settings.writer_model,
                strong=app_settings.strong_check_model,
                weak=app_settings.weak_models,
            ))
        return app_settings.public_dict()

    return app


app = create_app()
