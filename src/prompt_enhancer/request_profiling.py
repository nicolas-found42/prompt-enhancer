"""Opt-in request timing sidecars; raw Gateway answers remain unchanged."""

from __future__ import annotations

import json
import math
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any


def _number(value: Any) -> int | float | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value >= 0 else None
    if isinstance(value, float):
        return value if math.isfinite(value) and value >= 0 else None
    return None


def _identity(value: Any) -> str | None:
    return (
        value
        if isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", value)
        else None
    )


def _provider(value: Any) -> str | None:
    if isinstance(value, str) and value.lower() in {"novita", "groq"}:
        return value.lower()
    return None


class RequestProfile:
    """One physical adapter attempt, closed once even if its worker finishes late."""

    def __init__(
        self,
        record: dict[str, Any],
        *,
        started_at: float,
        clock: Callable[[], float],
        publish: Callable[[dict[str, Any]], None],
    ) -> None:
        self.record = record
        self.started_at = started_at
        self.clock = clock
        self.publish = publish
        self.dispatched_at: float | None = None
        self.closed = False
        self.lock = threading.Lock()

    def dispatched(self) -> None:
        with self.lock:
            if not self.closed:
                self.dispatched_at = self.clock()

    def finish(
        self,
        *,
        ended_at: float,
        status: str,
        http_status: int | None = None,
        response: Any = None,
        error_kind: str | None = None,
    ) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            raw = response if isinstance(response, Mapping) else {}
            usage = raw.get("usage")
            usage = usage if isinstance(usage, Mapping) else {}
            details = usage.get("completion_tokens_details")
            details = details if isinstance(details, Mapping) else {}
            generation = raw.get("id")
            record = {
                **self.record,
                "status": status,
                "http_status": http_status,
                "error_kind": error_kind,
                "started_monotonic_s": self.started_at,
                "dispatched_monotonic_s": self.dispatched_at,
                "dispatched": self.dispatched_at is not None,
                "finished_monotonic_s": ended_at,
                "adapter_attempt_ms": max(0.0, (ended_at - self.started_at) * 1000),
                "queue_ms": max(0.0, (self.dispatched_at - self.started_at) * 1000)
                if self.dispatched_at is not None
                else None,
                "transport_ms": max(0.0, (ended_at - self.dispatched_at) * 1000)
                if self.dispatched_at is not None
                else None,
                "served_model": _identity(raw.get("model")),
                "served_provider": _provider(raw.get("provider")),
                "generation_id": generation
                if isinstance(generation, str)
                and re.fullmatch(r"gen-[A-Za-z0-9-]{1,100}", generation)
                else None,
                "input_tokens": _number(
                    usage.get("prompt_tokens", usage.get("input_tokens"))
                ),
                "output_tokens": _number(
                    usage.get("completion_tokens", usage.get("output_tokens"))
                ),
                "reasoning_tokens": _number(details.get("reasoning_tokens")),
                "reported_cost": _number(usage.get("cost")),
                "headers_ms": None,
                "first_byte_ms": None,
                "ttft_ms": None,
                "visible_generation_interval_ms": None,
                "output_tokens_per_second": None,
                "visible_token_timing_source": "unavailable_nonstreaming_transport",
            }
        self.publish(record)


class RequestProfiles:
    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.generation = 0

    def clear(self) -> None:
        with self.lock:
            self.generation += 1
            self.records.clear()

    def _publish(self, record: dict[str, Any], generation: int) -> None:
        with self.lock:
            if generation == self.generation:
                self.records.append(deepcopy(record))

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return deepcopy(self.records)

    def begin(
        self,
        payload: Mapping[str, Any],
        *,
        model: str,
        gateway_provider: str,
        role: str,
        run_id: str | None,
        operation: str,
        attempt: int,
    ) -> RequestProfile:
        with self.lock:
            generation = self.generation
        policy = payload.get("provider")
        only = policy.get("only") if isinstance(policy, Mapping) else None
        requested_provider = (
            _provider(only[0]) if isinstance(only, list) and len(only) == 1 else None
        )
        sampling = {
            key: _number(payload.get(key)) for key in ("seed", "temperature", "top_p")
        }
        reasoning = payload.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, Mapping) else {}
        effort = reasoning.get("effort")
        effort = (
            effort
            if isinstance(effort, str)
            and effort in {"none", "minimal", "low", "medium", "high", "xhigh"}
            else None
        )
        return RequestProfile(
            {
                "attempt_id": uuid.uuid4().hex,
                "run_id": run_id,
                "role": role,
                "operation": operation,
                "attempt": attempt,
                "gateway_provider": gateway_provider,
                "requested_model": model,
                "requested_provider": requested_provider,
                "request_bytes": len(
                    json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                ),
                "request_size_source": "canonical_json_utf8_without_headers",
                "max_output_tokens": _number(
                    payload.get("max_tokens", payload.get("max_output_tokens"))
                ),
                "sampling": {
                    "requested": sampling,
                    "supported": None,
                    "effective": None,
                },
                "reasoning": {
                    "requested_effort": effort,
                    "requested_allowance_tokens": _number(reasoning.get("max_tokens")),
                    "supported": None,
                    "effective": None,
                },
            },
            started_at=self.clock(),
            clock=self.clock,
            publish=lambda record: self._publish(record, generation),
        )
