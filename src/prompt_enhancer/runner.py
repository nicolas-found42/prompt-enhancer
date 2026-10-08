"""Deterministic parallel execution of prompts on a weak-model panel."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any, cast

from .config import DEFAULT_FIXED_WEAK_PANEL, Settings
from .gateway import MAX_CONCURRENT_TRANSPORT_REQUESTS, Gateway, ProviderError
from .strategies import CandidateDraft
from .tuning_profile import ALLOWED_PROVIDERS, PROFILE_ID, WEAK_MODEL, provider_policy

DEFAULT_WEAK_PANEL: tuple[str, ...] = DEFAULT_FIXED_WEAK_PANEL


# Weak models sample at this temperature; replay keys include it.
WEAK_TEMPERATURE = 0.7


class ComparisonFailure(ProviderError):
    """A failed comparison with every completed and failed sample accounted for."""

    def __init__(self, cause: ProviderError, comparison: Mapping[str, Any]) -> None:
        super().__init__(
            cause.provider,
            cause.model,
            cause.status,
            "matched comparison did not complete",
            role=cause.role,
            kind=cause.kind,
            response_details=cause.response_details,
        )
        self.comparison = dict(comparison)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "comparison": self.comparison}


class _PanelAttemptFailed(Exception):
    def __init__(self, cause: Exception, results: list[PanelResult]) -> None:
        self.cause = cause
        self.results = results
        super().__init__(str(cause))


@dataclass(frozen=True)
class PanelResult:
    candidate_id: str
    model: str
    sample: int
    seed: int
    output: str
    prompt: str = ""
    response_details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **(
                {"response_details": dict(self.response_details)}
                if self.response_details
                else {}
            ),
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
    comparison: Mapping[str, Any] = field(default_factory=dict)

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
            **({"comparison": dict(self.comparison)} if self.comparison else {}),
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


def _output_text(value: Any, *, strict: bool = False) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, Mapping):
        if isinstance(value.get("message"), Mapping):
            return _output_text(value["message"], strict=strict)
        choices = value.get("choices")
        if choices:
            return _output_text(choices[0], strict=strict)
        for key in ("output", "text", "content", "response", "completion"):
            if key in value and value[key] is not None:
                return _output_text(value[key], strict=strict)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _output_text(value[0], strict=strict) if value else ""
    return "" if strict else str(value)


def _execute_one(
    gateway: Gateway,
    candidate_id: str,
    prompt: str,
    model: str,
    sample: int,
    run_seed: int,
    run_id: str | None,
    max_output_tokens: int | None,
    validate_response: bool,
    comparison_provider: str | None = None,
) -> PanelResult:
    seed = _stable_seed(
        run_seed,
        "matched_comparison" if comparison_provider else candidate_id,
        model,
        sample,
    )
    response = gateway.chat(
        model,
        [{"role": "user", "content": prompt}],
        role="weak",
        run_id=run_id,
        seed=seed,
        temperature=WEAK_TEMPERATURE,
        **(
            {
                "provider": provider_policy(comparison_provider),
                "route_provider": "openrouter",
            }
            if comparison_provider
            else {}
        ),
        **({"max_tokens": max_output_tokens} if max_output_tokens is not None else {}),
    )
    output = _output_text(response, strict=validate_response)
    details: dict[str, Any] = {}
    if validate_response:
        details = _response_details(response, output)
        if comparison_provider:
            raw = response if isinstance(response, Mapping) else {}
            served = raw.get("provider")
            served_name = served.casefold() if isinstance(served, str) else None
            served_model = raw.get("model")
            generation_id = raw.get("id")
            details.update(
                {
                    "requested_provider": comparison_provider,
                    "requested_policy": provider_policy(comparison_provider),
                    "requested_model": model,
                    "served_provider": served_name
                    if served_name in ALLOWED_PROVIDERS
                    else None,
                    "served_model": model if served_model == model else None,
                    "generation_id": generation_id
                    if isinstance(generation_id, str)
                    and re.fullmatch(r"gen-[A-Za-z0-9-]{1,150}", generation_id)
                    else None,
                    "sampling": {
                        "requested": {"seed": seed, "temperature": WEAK_TEMPERATURE},
                        "supported": None,
                        "effective": None,
                    },
                }
            )
            if (served_name and served_name != comparison_provider) or (
                isinstance(served_model, str) and served_model and served_model != model
            ):
                raise ProviderError(
                    "openrouter",
                    model,
                    None,
                    "served identity did not match the pinned comparison",
                    role="weak",
                    kind="provider_identity_mismatch",
                    response_details=details,
                )
        incomplete = details.get("status") in {
            "incomplete",
            "failed",
            "cancelled",
        } or details.get("finish_reason") in {
            "length",
            "max_tokens",
            "content_filter",
            "tool_calls",
            "tool_use",
        }
        if incomplete or not output.strip():
            if comparison_provider:
                details["partial_output"] = output
            # Provider metadata is allowlisted; never retain body diagnostics,
            # request headers, reasoning text or arbitrary error messages.
            provider = next(
                (
                    str(call["provider"])
                    for call in reversed(getattr(gateway, "calls", ()))
                    if isinstance(call, Mapping)
                    and call.get("model") == model
                    and call.get("role") == "weak"
                    and call.get("provider") in {"go", "openrouter"}
                ),
                "model",
            )
            raise ProviderError(
                provider,
                model,
                None,
                "panel answer did not complete",
                role="weak",
                kind="incomplete_response" if incomplete else "empty_response",
                response_details=details,
            )
    return PanelResult(candidate_id, model, sample, seed, output, prompt, details)


def _response_details(response: Any, output: str) -> dict[str, Any]:
    details: dict[str, Any] = {"visible_chars": len(output)}
    if not isinstance(response, Mapping):
        return details
    status = response.get("status")
    if status in {"completed", "incomplete", "failed", "cancelled"}:
        details["status"] = status
    stop_reason = response.get("stop_reason")
    if stop_reason in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}:
        details["finish_reason"] = stop_reason
    incomplete = response.get("incomplete_details")
    if isinstance(incomplete, Mapping) and incomplete.get("reason") in {
        "max_output_tokens",
        "content_filter",
    }:
        details["incomplete_reason"] = incomplete["reason"]
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish = choices[0].get("finish_reason")
        if finish in {"stop", "length", "max_tokens", "content_filter", "tool_calls"}:
            details["finish_reason"] = finish
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        for source in ("output_tokens", "completion_tokens"):
            count = usage.get(source)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                details["output_tokens"] = count
        for source in ("output_tokens_details", "completion_tokens_details"):
            tokens = usage.get(source)
            if isinstance(tokens, Mapping):
                count = tokens.get("reasoning_tokens")
                if (
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                ):
                    details["reasoning_tokens"] = count
    return details


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
    max_output_tokens: int | None,
    validate_response: bool,
    comparison_provider: str | None = None,
    request_records: list[dict[str, Any]] | None = None,
) -> list[PanelResult]:
    """Run one prompt on every model and sample, concurrently, in a stable order."""
    requests = [
        (
            candidate_id,
            prompt,
            model,
            sample,
            run_seed,
            run_id,
            max_output_tokens,
            validate_response,
            comparison_provider,
        )
        for model in models
        for sample in range(samples)
    ]
    requested_workers = len(requests) if max_workers is None else max_workers
    workers = min(len(requests), requested_workers, MAX_CONCURRENT_TRANSPORT_REQUESTS)
    if max(workers, 1) == 1 and request_records is None:
        return [_execute_one(gateway, *request) for request in requests]
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        # Each model call needs its own context copy: Context instances cannot
        # be entered by multiple threads, and executor tasks do not inherit the
        # caller's Gateway cancellation/observer context automatically.
        futures = [
            executor.submit(context.run, _execute_one, gateway, *request)
            for context, request in zip(
                (copy_context() for _ in requests), requests, strict=True
            )
        ]
        # Reading futures in request order keeps results stable regardless of
        # which model finishes first.
        if request_records is None:
            return [cast(PanelResult, future.result()) for future in futures]
        results: list[PanelResult] = []
        first_error: Exception | None = None
        for request, future in zip(requests, futures, strict=True):
            record: dict[str, Any] = {
                "candidate_id": request[0],
                "model": request[2],
                "sample": request[3],
                "seed": _stable_seed(
                    run_seed, "matched_comparison", request[2], request[3]
                ),
                "requested_provider": comparison_provider,
            }
            try:
                result = cast(PanelResult, future.result())
                results.append(result)
                record.update(
                    status="completed", response_details=dict(result.response_details)
                )
            except Exception as exc:
                first_error = first_error or exc
                record.update(
                    status="failed",
                    error=exc.to_dict()
                    if isinstance(exc, ProviderError)
                    else {"kind": type(exc).__name__},
                )
            request_records.append(record)
        if first_error is not None:
            raise _PanelAttemptFailed(first_error, results) from first_error
        return results


def run_candidates(
    candidates: Sequence[CandidateDraft],
    weak_models: Sequence[str] | None,
    gateway: Gateway,
    *,
    original: str | None = None,
    samples: int | None = None,
    settings: Settings | None = None,
    run_seed: int = 0,
    max_workers: int | None = None,
    run_id: str | None = None,
    validate_response: bool = True,
    on_comparison: Callable[[Mapping[str, Any]], None] | None = None,
) -> PanelRunResult:
    """Run the original and candidates on the settings-backed weak panel.

    The original runs first, as ``original``, when it is provided. The panel
    uses the first configured number of distinct models and configured sample
    count; an explicit ``samples`` value overrides the count.
    """

    workload = settings or Settings()
    model_count = workload.weak_model_count
    configured_models = tuple(weak_models or workload.weak_models or DEFAULT_WEAK_PANEL)
    if (
        not isinstance(model_count, int)
        or isinstance(model_count, bool)
        or model_count < 1
        or len(configured_models) < model_count
    ):
        raise ValueError(f"weak_models must contain at least {model_count} models")
    models = configured_models[:model_count]
    if any(not isinstance(model, str) or not model for model in models):
        raise ValueError("weak_models must contain non-empty model IDs")
    if len(set(models)) != model_count:
        raise ValueError(f"weak_models must contain {model_count} distinct models")
    selected_samples = workload.weak_samples if samples is None else samples
    output_limit = workload.weak_max_output_tokens if validate_response else None
    if output_limit is not None and (
        not isinstance(output_limit, int)
        or isinstance(output_limit, bool)
        or output_limit < 1
    ):
        raise ValueError("weak_max_output_tokens must be a positive integer")
    if (
        not isinstance(selected_samples, int)
        or isinstance(selected_samples, bool)
        or selected_samples < 1
    ):
        raise ValueError("samples must be a positive integer")
    if workload.evaluation_profile and (
        workload.evaluation_profile != PROFILE_ID
        or models != (WEAK_MODEL,)
        or selected_samples != 3
        or not validate_response
    ):
        raise ValueError("the tuning profile requires three validated Llama samples")

    items = ([("original", original)] if original is not None else []) + [
        (candidate.candidate_id, candidate.text) for candidate in candidates
    ]
    if not items:
        raise ValueError("at least one candidate or original is required")

    attempts: list[dict[str, Any]] = []
    providers: tuple[str | None, ...] = (
        (
            workload.evaluation_provider or ALLOWED_PROVIDERS[0],
            *(
                item
                for item in ALLOWED_PROVIDERS
                if item != (workload.evaluation_provider or ALLOWED_PROVIDERS[0])
            ),
        )
        if workload.evaluation_profile
        else (None,)
    )
    if workload.evaluation_profile:
        provider_policy(str(providers[0]))
    for provider in providers:
        results: list[PanelResult] = []
        request_records: list[dict[str, Any]] = []
        try:
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
                        max_output_tokens=output_limit,
                        validate_response=validate_response,
                        comparison_provider=provider,
                        request_records=request_records if provider else None,
                    )
                )
        except _PanelAttemptFailed as panel_error:
            results.extend(panel_error.results)
            exc = panel_error.cause
            attempts.append(
                {
                    "requested_provider": provider,
                    "status": "incomplete",
                    "outputs": [item.to_dict() for item in results],
                    "requests": request_records,
                    "error": exc.to_dict()
                    if isinstance(exc, ProviderError)
                    else {"kind": type(exc).__name__},
                }
            )
            can_fallback = (
                isinstance(exc, ProviderError)
                and provider != providers[-1]
                and (
                    exc.kind in {"context_length", "timeout", "network"}
                    or exc.status in {502, 503, 504}
                    or (
                        exc.status == 429
                        and exc.response_details.get("quota_scope") == "shared_pool"
                    )
                )
            )
            if on_comparison is not None:
                on_comparison(
                    {
                        "profile": workload.evaluation_profile,
                        "attempts": list(attempts),
                        "fallback_to": providers[providers.index(provider) + 1]
                        if can_fallback
                        else None,
                    }
                )
            if can_fallback:
                continue
            if isinstance(exc, ProviderError):
                raise ComparisonFailure(
                    exc, {"profile": workload.evaluation_profile, "attempts": attempts}
                ) from exc
            raise exc from panel_error
        if provider:
            attempts.append(
                {
                    "requested_provider": provider,
                    "status": "completed",
                    "outputs": [item.to_dict() for item in results],
                    "requests": request_records,
                }
            )
            if on_comparison is not None:
                on_comparison(
                    {"profile": workload.evaluation_profile, "attempts": list(attempts)}
                )
        break
    return PanelRunResult(
        tuple(results),
        tuple(candidate_id for candidate_id, _ in items),
        models,
        selected_samples,
        run_seed,
        {
            "profile": workload.evaluation_profile,
            "requested_provider": provider,
            "policy": provider_policy(provider),
            "attempts": attempts,
            "matched_requested_samples": True,
            # Equal requests are observable; providers do not attest effective
            # sampling parameters in these responses. Never turn that into True.
            "matched_effective_settings": None,
        }
        if provider
        else {},
    )


__all__ = [
    "DEFAULT_WEAK_PANEL",
    "PanelResult",
    "PanelRunResult",
    "run_candidates",
]
