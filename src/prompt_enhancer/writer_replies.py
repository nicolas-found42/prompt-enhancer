"""Bounded, caller-owned recovery of required writer replies."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from .gateway import Gateway, ProviderError, completion_text, writer_messages

WRITER_REPLY_RECOVERY_MIN_VERSION = 14
T = TypeVar("T")


def read_writer_reply(
    gateway: Gateway,
    *,
    model: str,
    instructions: str,
    state: Mapping[str, Any],
    read: Callable[[Any], T],
    operation: str,
    instruction_version: int,
    attempts: list[dict[str, Any]],
    run_id: str | None = None,
    round_number: int | None = None,
) -> T:
    """Read once for historical protocols; new protocols allow one retry.

    Only caller reading/shape errors are retried. Gateway failures retain their
    existing handling. The retry marker makes two different answers recordable
    in the existing request-keyed recording format. Usage stays in the Gateway.
    """
    if instruction_version < WRITER_REPLY_RECOVERY_MIN_VERSION:
        return read(
            gateway.chat(
                model,
                writer_messages(instructions, state),
                role="writer",
                run_id=run_id,
            )
        )

    for attempt in (1, 2):
        request_state = dict(state)
        if attempt == 2:
            request_state["writer_reply_retry"] = {
                "operation": operation,
                "attempt": attempt,
            }
        entry: dict[str, Any] = {
            "operation": operation,
            "round": round_number,
            "attempt": attempt,
            "model": model,
        }
        try:
            response = gateway.chat(
                model,
                writer_messages(instructions, request_state),
                role="writer",
                run_id=run_id,
            )
        except ProviderError:
            attempts.append({**entry, "outcome": "provider_error"})
            raise
        try:
            reason = "empty_completion"
            text = completion_text(response)
            if not text.strip():
                raise ValueError("writer reply is empty")
            reason = "invalid_reply_shape"
            value = read(response)
        except (ValueError, TypeError) as exc:
            if isinstance(exc, json.JSONDecodeError):
                reason = "invalid_json"
            attempts.append({**entry, "outcome": "invalid_response", "reason": reason})
            if attempt == 2:
                raise ProviderError(
                    "writer",
                    model,
                    None,
                    f"invalid {operation} response after two writer replies",
                    role="writer",
                    kind="invalid_response",
                ) from exc
        else:
            attempts.append({**entry, "outcome": "success"})
            return value
    raise AssertionError("writer reply attempts exhausted without a result")
