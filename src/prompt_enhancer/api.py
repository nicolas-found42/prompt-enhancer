"""Thin HTTP surface for the local prompt optimizer."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .clarification import InvalidAnswerError, RunNotPausedError
from .config import Settings
from .history import RunNotFound
from .jobs import JobBusy, JobNotFound, RunJobs
from .models import new_run_id
from .optimizer import PromptOptimizer, RunNotFoundError
from .prompt_health import (
    PromptHealthService,
    PromptHealthStore,
    isolated_health_gateway,
)
from .repeat import _reject_provider_secrets
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


class PromptHealthRequest(BaseModel):
    prompt: str
    revision: int = Field(ge=0)
    session_id: str = Field(min_length=1, max_length=128)


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


def _optimize_options(request: OptimizeRequest) -> dict[str, Any]:
    options = dict(request.options)
    options["tier"] = request.tier
    if request.model_overrides:
        options["model_overrides"] = {
            **options.get("model_overrides", {}),
            **request.model_overrides,
        }
    if request.clarification_allowed is not None:
        options["clarification_allowed"] = request.clarification_allowed
    return options


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    return statistics.quantiles(ordered, n=100, method="inclusive")[
        round(fraction * 100) - 1
    ]


def run_estimates(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Typical time and cost per tier from completed local runs."""
    samples: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        if run.get("status") != "completed" or not run.get("tier"):
            continue
        timing = run.get("timings") or run.get("timing") or {}
        cost = run.get("cost") or {}
        total_ms = timing.get("total_ms") if isinstance(timing, Mapping) else None
        total_cost = cost.get("total") if isinstance(cost, Mapping) else None
        if (
            not isinstance(total_ms, (int, float))
            or not isinstance(total_cost, (int, float))
            or total_ms <= 0
        ):
            continue
        samples.setdefault(str(run["tier"]), []).append(
            (total_ms / 60000, float(total_cost))
        )
    estimates: dict[str, Any] = {}
    for tier, values in samples.items():
        minutes = [item[0] for item in values]
        costs = [item[1] for item in values]
        estimates[tier] = {
            "runs": len(values),
            "minutes": [_percentile(minutes, 0.5), _percentile(minutes, 0.9)],
            "cost": [_percentile(costs, 0.5), _percentile(costs, 0.9)],
        }
    return estimates


def create_app(
    optimizer: PromptOptimizer | None = None,
    store: RunStore | None = None,
    settings: Settings | None = None,
    health_gateway: Any | None = None,
) -> FastAPI:
    """Create an app with injectable engine/store seams for local testing."""
    app_settings = settings or getattr(optimizer, "config", None) or Settings.from_env()
    app_store = store or (
        optimizer.store
        if optimizer is not None
        else RunStore(app_settings.database_path)
    )
    app_optimizer = optimizer or PromptOptimizer(store=app_store, config=app_settings)
    app_settings = getattr(app_optimizer, "config", app_settings)
    app = FastAPI(title="Prompt Enhancer", version="0.1.0")
    jobs = RunJobs()
    app.state.jobs = jobs
    app.state.optimizer = app_optimizer
    app.state.store = app_store
    app.state.settings = app_settings
    optimizer_gateway = getattr(app_optimizer, "gateway", None)
    health_service = PromptHealthService(
        health_gateway
        or (
            isolated_health_gateway(optimizer_gateway)
            if optimizer_gateway is not None
            else None
        ),
        PromptHealthStore(getattr(app_store, "path", ":memory:")),
        decision_policy=getattr(app_optimizer, "decision_policy", None),
    )
    app.state.prompt_health = health_service
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

    @app.get("/api/prompt-health/settings")
    def prompt_health_settings() -> dict[str, Any]:
        return health_service.settings()

    @app.post("/api/prompt-health")
    def prompt_health(request: PromptHealthRequest) -> dict[str, Any]:
        return health_service.assess(
            request.prompt, request.revision, request.session_id
        )

    @app.post("/api/optimize")
    def optimize(request: OptimizeRequest) -> dict[str, Any]:
        try:
            return dict(
                app_optimizer.optimize(request.prompt, _optimize_options(request))
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def recorded_prompt(run_id: str) -> str:
        record = app_optimizer.store.get_run(run_id)
        return str((record or {}).get("prompt") or "")

    def submit(
        run_id: str, kind: str, work: Callable[[Any], Any], prompt: str | None = None
    ) -> dict[str, Any]:
        job_prompt = prompt if prompt is not None else recorded_prompt(run_id)

        def on_failure(exc: BaseException) -> dict[str, Any]:
            return dict(app_optimizer.failure_result(run_id, job_prompt, exc))

        try:
            return jobs.submit(run_id, kind, work, on_failure, prompt=job_prompt)
        except JobBusy as exc:
            raise HTTPException(
                status_code=409, detail="this run is already in progress"
            ) from exc

    def require_run(run_id: str) -> None:
        if app_optimizer.store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")

    @app.post("/api/jobs/optimize", status_code=202)
    def start_optimize(request: OptimizeRequest) -> dict[str, Any]:
        options = _optimize_options(request)
        try:
            app_optimizer.validate_request(request.prompt, options)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run_id = new_run_id()
        return submit(
            run_id,
            "optimize",
            lambda progress: app_optimizer.optimize(
                request.prompt, options, run_id=run_id, progress=progress
            ),
            prompt=request.prompt,
        )

    @app.post("/api/jobs/{run_id}/resume", status_code=202)
    def start_resume(run_id: str, request: AnswersRequest) -> dict[str, Any]:
        require_run(run_id)
        try:
            app_optimizer.validate_resume(run_id, request.answers)
        except InvalidAnswerError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_answer",
                    "question_id": exc.question_id,
                    "message": str(exc),
                },
            ) from exc
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except RunNotPausedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return submit(
            run_id,
            "resume",
            lambda progress: app_optimizer.resume(
                run_id, request.answers, progress=progress
            ),
        )

    @app.post("/api/jobs/{run_id}/skip", status_code=202)
    def start_skip(run_id: str) -> dict[str, Any]:
        require_run(run_id)
        return submit(
            run_id,
            "skip",
            lambda progress: app_optimizer.skip_clarification(
                run_id, progress=progress
            ),
        )

    @app.post("/api/jobs/{run_id}/deep", status_code=202)
    def start_deep_job(run_id: str) -> dict[str, Any]:
        require_run(run_id)
        return submit(
            run_id,
            "deep",
            lambda progress: app_optimizer.start_deep_pass(run_id, progress=progress),
        )

    @app.get("/api/jobs")
    def active_jobs() -> list[dict[str, Any]]:
        return jobs.active()

    @app.get("/api/jobs/{run_id}")
    def get_job(run_id: str, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        try:
            return jobs.get(run_id)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=404, detail="no run in progress with this ID"
            ) from exc

    @app.post("/api/jobs/{run_id}/cancel")
    def cancel_job(run_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(run_id)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=404, detail="no run in progress with this ID"
            ) from exc

    @app.get("/api/estimates")
    def estimates() -> dict[str, Any]:
        return run_estimates(app_optimizer.history.list_runs(None, limit=200))

    @app.get("/api/providers")
    def providers(probe: bool = False) -> dict[str, Any]:
        config = app_optimizer.config
        health = getattr(app_optimizer.gateway, "provider_health", None)
        models = (config.writer_model, config.strong_check_model) if probe else ()
        return {
            "providers": dict(health(probe_models=models))
            if health is not None
            else {},
            "fallback": {
                "writer": config.fallback_writer_model,
                "strong": config.fallback_strong_check_model,
            },
        }

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
            # The model catalog is not part of the Gateway interface; live and
            # scripted gateways offer it, and any failure falls back below.
            return (
                cast(Any, app_optimizer.gateway)
                .list_models(refresh=True)
                .to_public_dict()
            )
        except Exception as exc:
            if app_settings.openrouter_api_key or app_settings.opencode_go_key:
                raise HTTPException(
                    status_code=503, detail="model catalog is unavailable"
                ) from exc
            return _public_catalog(app_settings)

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return {
            **app_optimizer.get_model_settings(),
            "live_health": health_service.settings(),
        }

    @app.put("/api/settings")
    def put_settings(values: dict[str, Any]) -> dict[str, Any]:
        try:
            return app_optimizer.update_model_settings(values)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
