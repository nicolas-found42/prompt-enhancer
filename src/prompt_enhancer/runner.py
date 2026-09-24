"""Deterministic parallel execution of prompts on a weak-model panel."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .catalog import DEFAULT_DEEP_WEAK_PANEL
from .gateway import Gateway
from .models import Tier
from .strategies import CandidateDraft

DEFAULT_WEAK_PANEL: tuple[str, ...] = DEFAULT_DEEP_WEAK_PANEL


# Weak models sample at this temperature; replay keys include it.
WEAK_TEMPERATURE = 0.7


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


def _execute_one(
    gateway: Gateway,
    candidate_id: str,
    prompt: str,
    model: str,
    sample: int,
    run_seed: int,
    run_id: str | None,
) -> PanelResult:
    seed = _stable_seed(run_seed, candidate_id, model, sample)
    output = _output_text(
        gateway.chat(
            model,
            [{"role": "user", "content": prompt}],
            role="weak",
            run_id=run_id,
            seed=seed,
            temperature=WEAK_TEMPERATURE,
        )
    )
    return PanelResult(candidate_id, model, sample, seed, output, prompt)


def _run_panel(
    candidate_id: str,
    prompt: str,
    models: tuple[str, ...],
    gateway: Gateway,
    *,
    samples: int,
    run_seed: int,
    max_workers: int | None,
    run_id: str | None,
) -> list[PanelResult]:
    """Run one prompt on every model and sample, concurrently, in a stable order."""
    requests = [
        (candidate_id, prompt, model, sample, run_seed, run_id)
        for model in models
        for sample in range(samples)
    ]
    workers = len(requests) if max_workers is None else min(len(requests), max_workers)
    if max(workers, 1) == 1:
        return [_execute_one(gateway, *request) for request in requests]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # executor.map preserves input order, and requests are deliberately ordered.
        return list(
            executor.map(lambda request: _execute_one(gateway, *request), requests)
        )


def run_candidates(
    candidates: Sequence[CandidateDraft],
    weak_models: Sequence[str] | None,
    gateway: Gateway,
    *,
    original: str | None = None,
    samples: int | None = None,
    budget: Tier | str | None = None,
    run_seed: int = 0,
    max_workers: int | None = None,
    run_id: str | None = None,
) -> PanelRunResult:
    """Run the original and all candidates on a tier-bounded weak panel.

    The original runs first, as ``original``, when it is provided. A tier
    budget truncates the configured model panel and supplies its sample count;
    an explicit ``samples`` value overrides the count.
    """

    selected_budget = Tier.parse(budget).budget if budget is not None else None
    if selected_budget is not None:
        models = tuple(weak_models or DEFAULT_WEAK_PANEL)[: selected_budget.models]
        selected_samples = selected_budget.samples if samples is None else samples
    else:
        models = tuple(weak_models or DEFAULT_WEAK_PANEL)
        selected_samples = 1 if samples is None else samples
    if not models:
        raise ValueError("weak_models must contain at least one model")
    if selected_samples < 1:
        raise ValueError("samples must be at least 1")

    items = ([("original", original)] if original is not None else []) + [
        (candidate.candidate_id, candidate.text) for candidate in candidates
    ]
    if not items:
        raise ValueError("at least one candidate or original is required")

    results: list[PanelResult] = []
    for candidate_id, prompt in items:
        results.extend(
            _run_panel(
                candidate_id,
                prompt,
                models,
                gateway,
                samples=selected_samples,
                run_seed=run_seed,
                max_workers=max_workers,
                run_id=run_id,
            )
        )
    return PanelRunResult(
        tuple(results),
        tuple(candidate_id for candidate_id, _ in items),
        models,
        selected_samples,
        run_seed,
    )


__all__ = [
    "DEFAULT_WEAK_PANEL",
    "PanelResult",
    "PanelRunResult",
    "run_candidates",
]
