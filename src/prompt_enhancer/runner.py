"""Deterministic parallel execution of prompts on a weak-model panel."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .catalog import DEFAULT_DEEP_WEAK_PANEL
from .strategies import TierBudget, budget_for_tier

DEFAULT_WEAK_PANEL: tuple[str, ...] = DEFAULT_DEEP_WEAK_PANEL


@dataclass(frozen=True)
class PanelRequest:
    """One model/sample execution request handed to a completion seam."""

    candidate_id: str
    prompt: str
    model: str
    sample: int
    seed: int
    temperature: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "prompt": self.prompt,
            "model": self.model,
            "sample": self.sample,
            "seed": self.seed,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class PanelResult:
    candidate_id: str
    model: str
    sample: int
    seed: int
    output: str
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model,
            "sample": self.sample,
            "seed": self.seed,
            "output": self.output,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class PanelRunResult:
    """All panel outputs, always ordered independently of thread completion."""

    results: tuple[PanelResult, ...]
    candidate_ids: tuple[str, ...]
    models: tuple[str, ...]
    samples: int
    run_seed: int

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def by_candidate(self, candidate_id: str) -> tuple[PanelResult, ...]:
        return tuple(
            result for result in self.results if result.candidate_id == candidate_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "models": list(self.models),
            "samples": self.samples,
            "run_seed": self.run_seed,
            "outputs": [result.to_dict() for result in self.results],
        }


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping):
            if name in value:
                return value[name]
        elif hasattr(value, name):
            return getattr(value, name)
    return default


def _candidate_prompt(candidate: Any) -> tuple[str, str]:
    if isinstance(candidate, str):
        return "original", candidate
    candidate_id = str(_field(candidate, "candidate_id", "id", default="original"))
    prompt = _field(candidate, "prompt", "text", default="")
    if prompt is None:
        prompt = ""
    return candidate_id, str(prompt)


def _model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    return str(_field(model, "model", "id", "name", default=model))


def _stable_seed(run_seed: int, candidate_id: str, model: str, sample: int) -> int:
    material = f"{run_seed}\0{candidate_id}\0{model}\0{sample}".encode()
    # SHA-256 avoids Python's process-randomized hash and makes replay portable.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFFFFFF


def _output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, Mapping):
        if isinstance(value.get("message"), Mapping):
            return _output_text(value["message"])
        choices = value.get("choices")
        if choices:
            return _output_text(choices[0])
        for key in ("output", "text", "content", "response", "completion"):
            if key in value and value[key] is not None:
                return _output_text(value[key])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _output_text(value[0]) if value else ""
    return str(value)


def _invoke_execute(execute: Any, request: PanelRequest, run_id: str | None) -> Any:
    """Call a callable or a model gateway without making a live API default."""

    if hasattr(execute, "chat"):
        return execute.chat(
            request.model,
            [{"role": "user", "content": request.prompt}],
            role="weak",
            run_id=run_id,
            seed=request.seed,
            temperature=request.temperature,
        )
    if not callable(execute):
        raise TypeError("execute must be callable or expose complete/chat")

    # The public callback receives PanelRequest.  The small compatibility path
    # for model,prompt,seed keeps simple test doubles pleasant to write.
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):
        return execute(request)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if len(positional) >= 3 or has_varargs:
        try:
            return execute(request.model, request.prompt, request.seed, request.sample)
        except TypeError:
            return execute(request.model, request.prompt, request.seed)
    return execute(request)


def _execute_one(
    execute: Any,
    candidate_id: str,
    prompt: str,
    model: str,
    sample: int,
    run_seed: int,
    run_id: str | None,
) -> PanelResult:
    seed = _stable_seed(run_seed, candidate_id, model, sample)
    request = PanelRequest(candidate_id, prompt, model, sample, seed)
    output = _output_text(_invoke_execute(execute, request, run_id))
    return PanelResult(candidate_id, model, sample, seed, output, prompt)


def run_candidate_panel(
    candidate: Any,
    weak_models: Sequence[Any],
    execute: Any,
    *,
    samples: int = 1,
    run_seed: int = 0,
    max_workers: int | None = None,
    run_id: str | None = None,
) -> PanelRunResult:
    """Run one candidate on every model and sample, concurrently."""

    if samples < 1:
        raise ValueError("samples must be at least 1")
    candidate_id, prompt = _candidate_prompt(candidate)
    models = tuple(_model_name(model) for model in weak_models)
    if not models:
        raise ValueError("weak_models must contain at least one model")
    requests = [
        (candidate_id, prompt, model, sample, run_seed, run_id)
        for model in models
        for sample in range(samples)
    ]
    workers = len(requests) if max_workers is None else min(len(requests), max_workers)
    workers = max(workers, 1)
    if workers == 1:
        completed = [_execute_one(execute, *request) for request in requests]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            completed = list(
                executor.map(lambda request: _execute_one(execute, *request), requests)
            )
    # executor.map preserves input order, and requests are deliberately ordered.
    return PanelRunResult(tuple(completed), (candidate_id,), models, samples, run_seed)


def run_candidates(
    candidates: Sequence[Any],
    weak_models: Sequence[Any] | None,
    execute: Any,
    *,
    original: Any = None,
    samples: int | None = None,
    budget: TierBudget | str | None = None,
    run_seed: int = 0,
    max_workers: int | None = None,
    run_id: str | None = None,
) -> PanelRunResult:
    """Run the original and all candidates on a tier-bounded weak panel.

    The original is included automatically when ``original`` is provided.  A
    tier budget truncates the configured model panel and supplies its sample
    count; an explicit ``samples`` value can still be used by a test or an
    override.
    """

    selected_budget = budget_for_tier(budget) if budget is not None else None
    if selected_budget is not None:
        models = list(weak_models or DEFAULT_WEAK_PANEL)[: selected_budget.models]
        selected_samples = selected_budget.samples if samples is None else samples
    else:
        models = list(weak_models or DEFAULT_WEAK_PANEL)
        selected_samples = 1 if samples is None else samples
    if not models:
        raise ValueError("weak_models must contain at least one model")

    items: list[tuple[Any, str]] = []
    if original is not None:
        items.append((original, "original"))
    items.extend((candidate, "") for candidate in candidates)
    if not items:
        raise ValueError("at least one candidate or original is required")

    all_results: list[PanelResult] = []
    candidate_ids: list[str] = []
    for candidate, forced_id in items:
        candidate_id, prompt = _candidate_prompt(candidate)
        if forced_id:
            candidate_id = forced_id
        candidate_ids.append(candidate_id)
        result = run_candidate_panel(
            {"id": candidate_id, "prompt": prompt},
            models,
            execute,
            samples=selected_samples,
            run_seed=run_seed,
            max_workers=max_workers,
            run_id=run_id,
        )
        all_results.extend(result.results)
    return PanelRunResult(
        tuple(all_results),
        tuple(candidate_ids),
        tuple(_model_name(model) for model in models),
        selected_samples,
        run_seed,
    )


# A discoverable alias for callers that refer to the operation as a panel run.
run_panel = run_candidates


__all__ = [
    "DEFAULT_WEAK_PANEL",
    "PanelRequest",
    "PanelResult",
    "PanelRunResult",
    "run_candidate_panel",
    "run_candidates",
    "run_panel",
]
