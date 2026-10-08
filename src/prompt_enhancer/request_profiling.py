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

from .streaming import (
    INCOMPLETE_FINISH_REASONS,
    is_raw_stream,
    stream_metadata,
    visible_content,
)


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
        self.phases: dict[str, float] = {}
        self.visible_frames = 0
        self.chunk_received_at: float | None = None

    def dispatched(self) -> None:
        with self.lock:
            if not self.closed:
                self.dispatched_at = self.clock()

    def observe_stream(self, kind: str) -> None:
        with self.lock:
            if self.closed or kind not in {
                "headers",
                "first_byte",
                "visible_content",
                "chunk_received",
            }:
                return
            if kind == "chunk_received":
                self.chunk_received_at = self.clock()
                return
            now = (
                self.chunk_received_at
                if kind in {"first_byte", "visible_content"}
                and self.chunk_received_at is not None
                else self.clock()
            )
            self.phases.setdefault(kind, now)
            if kind == "visible_content":
                self.phases["last_visible_content"] = now
                self.visible_frames += 1

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
            stream_complete = (
                bool(raw.get("complete") and not raw.get("errors"))
                if is_raw_stream(raw)
                else None
            )
            stream_errors = raw.get("errors") if is_raw_stream(raw) else None
            response_bytes = (
                _number(raw.get("received_bytes")) if is_raw_stream(raw) else None
            )
            events = raw.get("events")
            visible_chars = (
                sum(len(visible_content(event) or "") for event in events)
                if is_raw_stream(raw) and isinstance(events, list)
                else None
            )
            if is_raw_stream(raw):
                raw = stream_metadata(raw)
                terminal = raw.get("choices", [])
                if (
                    terminal
                    and terminal[0].get("finish_reason") in INCOMPLETE_FINISH_REASONS
                ):
                    stream_complete = False
            terminal_choices = raw.get("choices", []) if is_raw_stream(response) else []
            finish_reason = (
                terminal_choices[0].get("finish_reason") if terminal_choices else None
            )
            usage = raw.get("usage")
            usage = usage if isinstance(usage, Mapping) else {}
            details = usage.get("completion_tokens_details")
            details = details if isinstance(details, Mapping) else {}
            generation = raw.get("id")

            def phase_ms(kind: str) -> float | None:
                at = self.phases.get(kind)
                return (
                    max(0.0, (at - self.dispatched_at) * 1000)
                    if at is not None and self.dispatched_at is not None
                    else None
                )

            first_visible = self.phases.get("visible_content")
            last_visible = self.phases.get("last_visible_content")
            interval_ms = (
                max(0.0, (last_visible - first_visible) * 1000)
                if first_visible is not None and last_visible is not None
                else None
            )
            output_tokens = _number(
                usage.get("completion_tokens", usage.get("output_tokens"))
            )
            reasoning_tokens = _number(details.get("reasoning_tokens"))
            output_rate = None
            if (
                interval_ms
                and output_tokens is not None
                and reasoning_tokens is not None
                and 0 <= reasoning_tokens <= output_tokens
                and stream_complete is True
                and not stream_errors
            ):
                try:
                    output_rate = _number(
                        (output_tokens - reasoning_tokens) / (interval_ms / 1000)
                    )
                except OverflowError:
                    pass
            record = {
                **self.record,
                "status": status,
                "stream_complete": stream_complete,
                "stream_terminal_reason": finish_reason
                if finish_reason in INCOMPLETE_FINISH_REASONS | {"stop"}
                else None,
                "stream_error_count": len(stream_errors)
                if isinstance(stream_errors, list)
                else None,
                "response_bytes": response_bytes,
                "response_size_source": "received_sse_bytes"
                if response_bytes is not None
                else None,
                "visible_output_chars": visible_chars,
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
                "identity_conflicts": raw.get("identity_conflicts")
                if is_raw_stream(response)
                else [],
                "generation_id": generation
                if isinstance(generation, str)
                and re.fullmatch(r"gen-[A-Za-z0-9-]{1,100}", generation)
                else None,
                "input_tokens": _number(
                    usage.get("prompt_tokens", usage.get("input_tokens"))
                ),
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "reported_cost": _number(usage.get("cost")),
                "headers_ms": phase_ms("headers"),
                "first_byte_ms": phase_ms("first_byte"),
                "ttft_ms": phase_ms("visible_content"),
                "visible_generation_interval_ms": interval_ms,
                "output_tokens_per_second": output_rate,
                "output_tokens_per_second_definition": "Reported completion tokens minus reported reasoning tokens, divided by the first-to-last visible-content frame interval; unavailable without both counts and a positive interval.",
                "visible_content_frames": self.visible_frames,
                "first_visible_monotonic_s": self.phases.get("visible_content"),
                "last_visible_monotonic_s": self.phases.get("last_visible_content"),
                "stream_timing_resolution": "socket_read_return",
                "visible_token_timing_source": "sse_visible_content_frame_arrival"
                if self.visible_frames
                else "unavailable_no_visible_content_frame"
                if self.phases
                else "unavailable_nonstreaming_transport",
                "stream_phase_clock": "adapter_monotonic_since_dispatch",
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
                "streaming_requested": payload.get("stream") is True,
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
