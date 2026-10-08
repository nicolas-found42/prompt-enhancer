"""Source-preserving accounting for bounded diagnosis observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

MAX_SOURCE_WINDOWS = 16


def identity(request: Mapping[str, Any]) -> str:
    return json.dumps(request, ensure_ascii=False, sort_keys=True)


class DiagnosisRecovery:
    """Retain original question meaning without promoting partial-window answers."""

    def __init__(
        self, prompt: str, on_change: Callable[[], None] | None = None
    ) -> None:
        self.prompt = prompt
        self.questions: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self.on_change = on_change

    def register(
        self, request: Mapping[str, Any], dispatched: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        canonical = identity(request)
        key = self.aliases.get(canonical, canonical)
        if key not in self.questions:
            self.questions[key] = {
                "id": hashlib.sha256(canonical.encode()).hexdigest(),
                "key": str(request.get("key", "")),
                "original_request": dict(request),
                "source_span": {
                    "start": 0,
                    "end": len(self.prompt),
                    "unit": "unicode_codepoints",
                },
                "required": False,
                "status": "unused",
                "raw_answer": None,
                "answered_by": None,
                "attempts": [],
                "windows": [],
            }
        self.aliases[canonical] = key
        if dispatched is not None:
            self.aliases[identity(dispatched)] = key
        return self.questions[key]

    def required(self, request: Mapping[str, Any]) -> None:
        self.register(request)["required"] = True

    def held(self, request: Mapping[str, Any], reason: str) -> None:
        record = self.register(request)
        if not record["windows"]:
            record.update(status="held", reason=reason)

    def error(
        self, requests: Sequence[Mapping[str, Any]], error: Mapping[str, Any]
    ) -> None:
        for request in requests:
            record = self.register(request)
            record["attempts"].append(
                {"status": "provider_error", "error": dict(error)}
            )
            record.update(status="provider_error", reason="provider_error")

    def observed(
        self,
        request: Mapping[str, Any],
        raw: Any,
        *,
        valid: bool,
        answered_by: str | None,
    ) -> None:
        record = self.register(request)
        record.update(raw_answer=raw, answered_by=answered_by)
        if record["windows"]:
            return
        previous_error = record["status"] == "provider_error"
        if valid:
            record.update(status="completed", reason=None)
        elif not previous_error and raw is not None:
            record.update(
                status="invalid_answer", reason="malformed_or_unusable_answer"
            )
        elif not previous_error and record["status"] != "held":
            record.update(status="held", reason="missing_answer")

    def windowed(
        self,
        request: Mapping[str, Any],
        *,
        fits: Callable[[Sequence[Mapping[str, Any]]], bool],
        dispatch: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        record = self.register(request)
        if record["windows"]:
            return
        state = request.get("state")
        source = state.get("prompt") if isinstance(state, Mapping) else None
        if not isinstance(source, str) or source != self.prompt:
            self.held(request, "unsupported_source_scope")
            return
        record.update(
            status="source_windows_only", reason="whole_source_decision_unresolved"
        )
        start = 0
        while start < len(source):
            index = len(record["windows"])

            def window_request(
                end: int, *, window_start: int = start, window_index: int = index
            ) -> dict[str, Any]:
                return {
                    **request,
                    "key": f"{request.get('key', '')}:source_window:{window_index}",
                    "state": {
                        "prompt": source[window_start:end],
                        "source_window": {
                            "start": window_start,
                            "end": end,
                            "unit": "unicode_codepoints",
                            "scope": "partial_original_prompt",
                        },
                    },
                }

            low, high = start, len(source)
            while low < high:
                middle = (low + high + 1) // 2
                if fits([window_request(middle)]):
                    low = middle
                else:
                    high = middle - 1
            end = low
            held_reason = None
            if end == start:
                end = len(source)
                held_reason = "question_metadata_exceeds_request_limit"
            elif index >= MAX_SOURCE_WINDOWS - 1 and end < len(source):
                end = len(source)
                held_reason = "source_window_count_cap"
            item = {
                "source": source[start:end],
                "source_span": {
                    "start": start,
                    "end": end,
                    "unit": "unicode_codepoints",
                },
                "scope": "partial_original_prompt",
                "status": "held",
                "raw_answer": None,
                "answered_by": None,
            }
            if held_reason is not None:
                item["reason"] = held_reason
            else:
                item.update(dispatch(window_request(end)))
            record["windows"].append(item)
            if self.on_change is not None:
                self.on_change()
            start = end

    def to_dict(self) -> dict[str, Any]:
        return {
            "accounting_protocol": "diagnosis-source-accounting-v1",
            "questions": list(self.questions.values()),
            "question_count": len(self.questions),
        }
