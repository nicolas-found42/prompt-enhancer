"""Raw chat SSE framing; answer interpretation remains with Gateway callers."""

from __future__ import annotations

import base64
import codecs
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

STREAM_PROTOCOL = "raw-chat-sse-1"
INCOMPLETE_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "content_filter", "tool_calls", "tool_use", "error"}
)


def is_raw_stream(answer: Any) -> bool:
    return isinstance(answer, Mapping) and answer.get("protocol") == STREAM_PROTOCOL


def _choice_zero(event: Any) -> Mapping[str, Any] | None:
    if not isinstance(event, Mapping):
        return None
    choices = event.get("choices")
    if not isinstance(choices, list):
        return None
    return next(
        (c for c in choices if isinstance(c, Mapping) and c.get("index", 0) == 0),
        None,
    )


def visible_content(event: Any) -> str | None:
    choice = _choice_zero(event)
    delta = choice.get("delta") if choice is not None else None
    content = delta.get("content") if isinstance(delta, Mapping) else None
    return content if isinstance(content, str) and content else None


def stream_metadata(answer: Mapping[str, Any]) -> dict[str, Any]:
    """Read accounting/identity fields without changing the retained answer."""
    result: dict[str, Any] = {}
    conflicts: list[str] = []
    events = answer.get("events", [])
    if isinstance(events, list):
        for event in events:
            if isinstance(event, Mapping):
                for key in ("id", "model", "provider", "usage"):
                    if key in event:
                        value = event[key]
                        if key == "usage":
                            result[key] = value
                        elif key not in conflicts and value is not None:
                            previous = result.get(key)
                            comparable = (
                                value.casefold()
                                if key == "provider" and isinstance(value, str)
                                else value
                            )
                            old_comparable = (
                                previous.casefold()
                                if key == "provider" and isinstance(previous, str)
                                else previous
                            )
                            if previous is not None and old_comparable != comparable:
                                conflicts.append(key)
                                result.pop(key, None)
                            else:
                                result[key] = value
                choice = _choice_zero(event)
                if (
                    choice is not None
                    and isinstance(choice.get("finish_reason"), str)
                    and choice.get("finish_reason")
                ):
                    previous = result.get("choices", [])
                    # A later accounting frame cannot erase known truncation.
                    if (
                        not previous
                        or previous[0].get("finish_reason")
                        not in INCOMPLETE_FINISH_REASONS
                    ):
                        result["choices"] = [choice]
    result["identity_conflicts"] = conflicts
    return result


def stream_completion_text(
    answer: Mapping[str, Any], *, allow_incomplete_terminal: bool = False
) -> str:
    """Caller-side reading, rejecting partial, malformed or failed streams."""
    if not answer.get("complete") or answer.get("errors"):
        raise ValueError("model stream is incomplete or malformed")
    events = answer.get("events")
    if not isinstance(events, list):
        raise ValueError("model stream has no events")
    if any(isinstance(event, Mapping) and "error" in event for event in events):
        raise ValueError("model stream reports a provider error")
    terminal = stream_metadata(answer).get("choices", [])
    if (
        not allow_incomplete_terminal
        and terminal
        and terminal[0].get("finish_reason") in INCOMPLETE_FINISH_REASONS
    ):
        raise ValueError("model stream did not finish a usable completion")
    content = [visible_content(event) for event in events]
    if not any(content):
        raise ValueError("model stream returned no visible completion")
    return "".join(item for item in content if item is not None)


class RawChatStream:
    """Incrementally frame SSE, preserving every received byte separately."""

    def __init__(self, observe: Callable[[str], None] | None = None) -> None:
        self.observe = observe
        self.decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
        self.buffer = ""
        self.data: list[str] = []
        self.events: list[Any] = []
        self.errors: list[dict[str, Any]] = []
        self.complete = False
        self.raw_chunks: list[bytes] = []
        self.received_bytes = 0

    def feed(self, chunk: bytes) -> None:
        self.raw_chunks.append(chunk)
        self.received_bytes += len(chunk)
        self.buffer += self.decoder.decode(chunk)
        self._lines()

    def _lines(self, *, final: bool = False) -> None:
        while (newline := re.search(r"\r\n|\r|\n", self.buffer)) is not None:
            # A CR at the current chunk boundary may be half of CRLF.
            if (
                not final
                and newline.group() == "\r"
                and newline.end() == len(self.buffer)
            ):
                break
            line, self.buffer = (
                self.buffer[: newline.start()],
                self.buffer[newline.end() :],
            )
            if not line:
                self._dispatch()
            elif line.startswith("data:"):
                value = line[5:]
                self.data.append(value[1:] if value.startswith(" ") else value)
            elif line == "data":
                self.data.append("")

    def _dispatch(self) -> None:
        if not self.data:
            return
        data = "\n".join(self.data)
        self.data.clear()
        if self.complete:
            self.errors.append({"kind": "event_after_done", "event": len(self.events)})
            return
        if data == "[DONE]":
            self.complete = True
            return
        try:

            def invalid_constant(value: str) -> Any:
                raise ValueError("invalid JSON constant")

            event = json.loads(data, parse_constant=invalid_constant)
        except (ValueError, RecursionError):
            self.errors.append({"kind": "invalid_json", "event": len(self.events)})
            return
        self.events.append(event)
        if not isinstance(event, Mapping):
            self.errors.append(
                {"kind": "invalid_chat_event", "event": len(self.events) - 1}
            )
        else:
            choices = event.get("choices")
            if "choices" in event:
                malformed = not isinstance(choices, list)
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, Mapping):
                            malformed = True
                            continue
                        index = choice.get("index")
                        if "index" in choice and (
                            not isinstance(index, int)
                            or isinstance(index, bool)
                            or index < 0
                        ):
                            malformed = True
                        reason = choice.get("finish_reason")
                        if reason is not None and not isinstance(reason, str):
                            malformed = True
                        delta = choice.get("delta")
                        if delta is not None and not isinstance(delta, Mapping):
                            malformed = True
                        elif isinstance(delta, Mapping):
                            content = delta.get("content")
                            if content is not None and not isinstance(content, str):
                                malformed = True
                if malformed:
                    self.errors.append(
                        {"kind": "invalid_chat_delta", "event": len(self.events) - 1}
                    )
            failed = "error" in event or (
                isinstance(choices, list)
                and any(
                    isinstance(choice, Mapping)
                    and choice.get("finish_reason") == "error"
                    for choice in choices
                )
            )
            if failed:
                self.errors.append(
                    {"kind": "provider_error", "event": len(self.events) - 1}
                )
        if visible_content(event) is not None and self.observe is not None:
            self.observe("visible_content")

    def finish(self) -> dict[str, Any]:
        self.buffer += self.decoder.decode(b"", final=True)
        self._lines(final=True)
        if self.buffer.startswith("data") or self.data:
            self.errors.append(
                {"kind": "unterminated_event", "event": len(self.events)}
            )
        body = b"".join(self.raw_chunks)
        try:
            body.decode("utf-8")
        except UnicodeDecodeError:
            self.errors.append({"kind": "invalid_utf8"})
        return {
            "protocol": STREAM_PROTOCOL,
            "raw_body": body.decode("utf-8", errors="replace"),
            "raw_body_base64": base64.b64encode(body).decode("ascii"),
            "events": self.events,
            "errors": self.errors,
            "complete": self.complete,
            "received_bytes": self.received_bytes,
        }
