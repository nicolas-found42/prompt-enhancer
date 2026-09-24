"""HTTP adapter for the clarification part of the engine facade.

The bootstrap app can register these routes without knowing how paused runs are
persisted. Keeping the adapter in its own module also lets API contract tests
use a tiny fake optimizer while the engine tests exercise ClarificationService.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body


def register_clarification_routes(app: Any, optimizer: Any) -> None:
    """Add resume and explicit-skip routes to a FastAPI-like app.

    ``optimizer`` is intentionally duck typed. The base bootstrap can inject its
    ``PromptOptimizer`` facade, while a test can inject a deterministic fake.
    """

    from fastapi import HTTPException

    @app.post("/api/optimize/resume/{run_id}")
    def resume_run(
        run_id: str, payload: Annotated[dict[str, Any], Body()]
    ) -> dict[str, Any]:
        answers = payload.get("answers", payload) if isinstance(payload, dict) else {}
        try:
            result = optimizer.resume(run_id, answers)
        except (KeyError, LookupError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result

    @app.post("/api/optimize/skip/{run_id}")
    def skip_run(run_id: str) -> dict[str, Any]:
        skip = getattr(optimizer, "skip_clarification", None) or getattr(
            optimizer, "skip", None
        )
        if skip is None:
            raise HTTPException(
                status_code=501, detail="Clarification skip is unavailable"
            )
        try:
            result = skip(run_id)
        except (KeyError, LookupError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result


__all__ = ["register_clarification_routes"]
