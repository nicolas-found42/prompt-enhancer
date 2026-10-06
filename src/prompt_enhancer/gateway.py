"""The Gateway: the single way the engine reaches any model.

:class:`Gateway` is the interface. :class:`HttpGateway` routes live traffic to
OpenCode Go and OpenRouter; :class:`ScriptedGateway` answers tests;
:class:`ReplayGateway` replays recordings made by ``RecordingGateway``. API
keys are read from server-side configuration and are never placed in a
response or exception.
"""

from __future__ import annotations

import copy
import hashlib
import json
import json as json_module
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from .catalog import (
    DEFAULT_GO_STRONG,
    DEFAULT_GO_WRITER,
    JEV_MODEL,
    CatalogSnapshot,
    ModelInfo,
    StaticModelCatalog,
)
from .jev import batch_decision_payload, decision_payload
from .usage import UsageLedger

MAX_CONCURRENT_TRANSPORT_REQUESTS = 8

LEGACY_JEV_ALIAS = "typesafe/jev-1.13"

DEFAULT_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_USER_AGENT = "prompt-enhancer/0.1"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
ACCESS_DENIED_STATUS = {401, 402, 403}
PROVIDER_STATUS_TTL = 600.0


class Gateway(Protocol):
    """The single way the engine reaches any model.

    Answers are returned raw, exactly as recordings store them; callers read
    them with ``completion_text`` and ``jev.parse_decision``. Every adapter
    raises ``ProviderError`` when a model cannot be reached or its reply cannot
    be used.
    """

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        role: str,
        run_id: str | None = None,
        **params: Any,
    ) -> Any: ...

    def decide(
        self,
        payload: Mapping[str, Any],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> Any: ...

    def decide_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> list[Any]: ...

    def new_run(self, run_id: str | None = None) -> str: ...

    def usage_report(self) -> dict[str, Any]: ...

    @property
    def decision_log(self) -> list[dict[str, Any]]: ...

    @property
    def jev_model(self) -> str: ...


def writer_messages(instructions: str, state: Any = None) -> list[dict[str, str]]:
    """Messages for a writer call: instructions as system, untrusted state as JSON."""
    if state is None:
        return [{"role": "user", "content": instructions}]
    return [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": json.dumps(state, ensure_ascii=False, sort_keys=True),
        },
    ]


def completion_text(value: Any) -> str:
    """The text of a raw chat answer, in any provider's reply shape."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if isinstance(value.get("message"), Mapping):
            return completion_text(value["message"])
        for key in ("content", "text", "output", "completion", "answer"):
            if isinstance(value.get(key), str):
                return value[key]
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            return completion_text(choices[0])
    raise ValueError("model gateway returned no text completion")


class GatewayTransport(Protocol):
    """Synchronous transport boundary.

    Optional request-scoped cancellation is supported by adapters that set
    ``supports_request_id = True``, accept ``request_id`` in ``request``, and
    provide ``abort_request(request_id)``. Other adapters remain source
    compatible; their call worker is kept daemonized and bounded by the Gateway
    worker limit if they ignore the supplied timeout.
    """

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any | None = None,
        timeout: float | None = None,
    ) -> Any: ...


@dataclass(slots=True)
class GatewayConfig:
    go_api_key: str | None = None
    openrouter_api_key: str | None = None
    go_base_url: str = DEFAULT_GO_BASE_URL
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    go_models_url: str | None = None
    openrouter_models_url: str | None = None
    decisions_path: str = "/alpha/decisions"
    jev_model: str = JEV_MODEL
    timeout: float = 120.0
    # A deadline for one complete Gateway operation, including retries and
    # retry delays. RunControl.time_limit_s remains a separate Round-boundary
    # pause control.
    operation_timeout_s: float = 180.0
    max_retries: int = 2
    backoff: float = 0.0
    user_agent: str = DEFAULT_USER_AGENT
    referer: str = "https://localhost/prompt-enhancer"
    title: str = "Prompt Enhancer"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GatewayConfig:
        env = os.environ if environ is None else environ
        go_base = env.get("OPENCODE_GO_BASE_URL", DEFAULT_GO_BASE_URL).rstrip("/")
        openrouter_base = env.get(
            "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
        ).rstrip("/")
        return cls(
            go_api_key=env.get("OPENCODE_GO_KEY") or env.get("OPENCODE_GO_API_KEY"),
            openrouter_api_key=env.get("OPENROUTER_API_KEY"),
            go_base_url=go_base,
            openrouter_base_url=openrouter_base,
            go_models_url=env.get("OPENCODE_GO_MODELS_URL", f"{go_base}/models"),
            openrouter_models_url=env.get(
                "OPENROUTER_MODELS_URL", f"{openrouter_base}/models"
            ),
            timeout=float(env.get("PROMPT_ENHANCER_TIMEOUT", "120")),
            operation_timeout_s=float(
                env.get("PROMPT_ENHANCER_OPERATION_TIMEOUT", "180")
            ),
            max_retries=int(env.get("PROMPT_ENHANCER_MAX_RETRIES", "2")),
            jev_model=env.get("PROMPT_ENHANCER_JEV_MODEL", JEV_MODEL),
            backoff=float(env.get("PROMPT_ENHANCER_RETRY_BACKOFF", "0")),
            user_agent=env.get("PROMPT_ENHANCER_USER_AGENT", DEFAULT_USER_AGENT),
        )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    provider: str
    model: str
    url: str
    headers: Mapping[str, str]


class ProviderError(RuntimeError):
    """A sanitized provider failure.

    The response body and request payload are intentionally not retained.  A
    run store can retain the original prompt, while API error responses cannot
    accidentally disclose credentials or provider diagnostics.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        status: int | None,
        message: str = "provider request failed",
        *,
        role: str | None = None,
        kind: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status = status
        self.role = role
        # ``http`` has a status; ``network`` never reached a response; and
        # ``invalid_response`` means a reply arrived but could not be used.
        self.kind = kind or ("http" if status is not None else "network")
        detail = {
            "http": f"HTTP {status}",
            "network": "no response",
            "invalid_response": "invalid response",
        }.get(self.kind, self.kind)
        super().__init__(f"{provider} request for {model} failed ({detail}): {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "http_status": self.status,
            "role": self.role,
            "kind": self.kind,
            "message": str(self),
        }


class _GatewayDeadlineExceeded(TimeoutError):
    """A Gateway retry or operation could not fit within its deadline."""


class HttpTransport:
    """Minimal JSON HTTP transport; no provider-specific behavior."""

    supports_request_id = True

    def __init__(self, opener: Callable[..., Any] | None = None) -> None:
        self._opener = opener or urllib.request.urlopen
        self._active_responses: dict[str, Any] = {}
        self._active_lock = threading.Lock()

    def abort_request(self, request_id: str) -> None:
        """Close a response body that is currently being read, when available."""
        with self._active_lock:
            response = self._active_responses.get(request_id)
        if response is None:
            return
        # Closing the underlying socket interrupts a blocking read on the
        # worker thread. Some test/custom response wrappers expose only close.
        raw = getattr(getattr(response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        close_socket = getattr(sock, "close", None)
        if callable(close_socket):
            shutdown = getattr(sock, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # The worker may already have closed its socket.
            close_socket()
            return
        close = getattr(response, "close", None)
        if callable(close):
            close()

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any | None = None,
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        # Preserve insertion order in Choice criteria: their declared option
        # order is an experimental input. Replay keys use the canonical,
        # sorted json_module_dumps helper below instead.
        data = (
            None
            if json is None
            else json_module.dumps(
                json, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        request = urllib.request.Request(url, data=data, method=method)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self._opener(request, timeout=timeout) as response:
                if request_id is not None:
                    with self._active_lock:
                        self._active_responses[request_id] = response
                try:
                    body = self._read_body(response, timeout)
                finally:
                    if request_id is not None:
                        with self._active_lock:
                            self._active_responses.pop(request_id, None)
                status = getattr(
                    response, "status", getattr(response, "status_code", 200)
                )
                parsed = _decode_body(body)
                return {
                    "status_code": status,
                    "json": parsed,
                    "headers": dict(response.headers.items()),
                }
        except urllib.error.HTTPError as exc:
            if request_id is not None:
                with self._active_lock:
                    self._active_responses[request_id] = exc
            try:
                body = self._read_body(exc, timeout) if hasattr(exc, "read") else b""
            finally:
                if request_id is not None:
                    with self._active_lock:
                        self._active_responses.pop(request_id, None)
            return {
                "status_code": exc.code,
                "json": _decode_body(body),
                "headers": dict(exc.headers.items()),
            }
        # URLError, timeout, and connection errors intentionally propagate to
        # HttpGateway, which retries and wraps them as ProviderError.

    @staticmethod
    def _read_body(response: Any, timeout: float | None) -> bytes:
        read_chunk = getattr(response, "read1", None)
        if not callable(read_chunk) or timeout is None:
            return response.read()
        deadline = time.monotonic() + max(0.001, timeout)
        chunks: list[bytes] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("HTTP response exceeded its total deadline")
            raw = getattr(getattr(response, "fp", None), "raw", None)
            sock = getattr(raw, "_sock", None)
            settimeout = getattr(sock, "settimeout", None)
            if callable(settimeout):
                settimeout(remaining)
            chunk = read_chunk(64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


def json_module_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _decode_body(body: Any) -> Any:
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if not body:
        return {}
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {"text": str(body)}


def _response_status(response: Any) -> int | None:
    if isinstance(response, Mapping):
        value = response.get("status_code", response.get("status"))
    else:
        value = getattr(response, "status_code", getattr(response, "status", None))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(str(value))
            seconds = (deadline - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _decision_answers(response: Any, keys: Sequence[str]) -> list[Any]:
    answers = response.get("answers") if isinstance(response, Mapping) else None
    if not isinstance(answers, Mapping) or any(key not in answers for key in keys):
        raise ProviderError(
            "openrouter",
            JEV_MODEL,
            None,
            "decision answers are missing",
            role="judge",
            kind="invalid_response",
        )
    return [answers[key] for key in keys]


def _response_json(response: Any) -> Any:
    if isinstance(response, Mapping):
        value = response.get("json", response.get("data", response))
    else:
        value = getattr(response, "json", response)
    return value() if callable(value) else value


class HttpGateway:
    """Route chat and Jev requests while tracking usage by role."""

    def __init__(
        self,
        transport: GatewayTransport | None = None,
        *,
        config: GatewayConfig | None = None,
        catalog: Any | None = None,
        usage: UsageLedger | None = None,
        go_models: Iterable[str | ModelInfo] | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or GatewayConfig()
        self.transport = transport or HttpTransport()
        self.catalog = catalog
        self.usage = usage or UsageLedger()
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._session = uuid.uuid4().hex
        self._sessions: dict[str, str] = {}
        self._go_model_ids: set[str] = {
            DEFAULT_GO_WRITER,
            "deepseek-v4.1-flash",
            DEFAULT_GO_STRONG,
            "mimo-v2.6-flash",
            "qwen3.8-flash",
            "muse-spark-1.3-contributor",
        } | {item if isinstance(item, str) else item.id for item in (go_models or ())}
        self.calls: list[dict[str, Any]] = []
        self.transport_attempts_by_role: dict[str, int] = {}
        self.decision_log: list[dict[str, Any]] = []
        self._provider_status: dict[str, dict[str, Any]] = {}
        # A timed-out generic transport has no portable interruption API. Keep
        # abandoned calls bounded even when a custom adapter cannot abort them.
        self._transport_workers = threading.BoundedSemaphore(
            MAX_CONCURRENT_TRANSPORT_REQUESTS
        )
        self._operation_context: ContextVar[dict[str, Any] | None] = ContextVar(
            f"gateway-operation-context-{id(self)}", default=None
        )

    @contextmanager
    def operation_context(
        self,
        *,
        cancel_check: Callable[[], bool] | None = None,
        observer: Callable[[Mapping[str, Any]], None] | None = None,
        substage: str | None = None,
        run_id: str | None = None,
    ):
        """Attach per-run cancellation and sanitized operation evidence.

        ContextVars keep concurrent callers isolated and do not add control
        fields to provider payloads or recordings.
        """
        current = dict(self._operation_context.get() or {})
        if cancel_check is not None:
            current["cancel_check"] = cancel_check
        if observer is not None:
            current["observer"] = observer
        if substage is not None:
            current["substage"] = substage
        if run_id is not None:
            current["run_id"] = run_id
        token = self._operation_context.set(current)
        started = self._monotonic()
        if substage is not None:
            self._emit_operation(
                {
                    "event": "start",
                    "operation": substage,
                    "run_id": current.get("run_id"),
                }
            )
        try:
            yield
        except BaseException as exc:
            if substage is not None:
                self._emit_operation(
                    {
                        "event": "error",
                        "operation": substage,
                        "run_id": current.get("run_id"),
                        "elapsed_ms": max(
                            0, round((self._monotonic() - started) * 1000)
                        ),
                        "error_kind": type(exc).__name__,
                    }
                )
            raise
        else:
            if substage is not None:
                self._emit_operation(
                    {
                        "event": "end",
                        "operation": substage,
                        "run_id": current.get("run_id"),
                        "elapsed_ms": max(
                            0, round((self._monotonic() - started) * 1000)
                        ),
                    }
                )
        finally:
            self._operation_context.reset(token)

    def _emit_operation(self, event: Mapping[str, Any]) -> None:
        observer = (self._operation_context.get() or {}).get("observer")
        if callable(observer):
            # The event is deliberately allow-listed at construction sites;
            # never include request payloads, headers, response bodies or keys.
            observer(dict(event))

    def _check_cancelled(self, run_id: str | None) -> None:
        check = (self._operation_context.get() or {}).get("cancel_check")
        if callable(check) and check():
            from .failures import RunCancelled

            self._emit_operation({"event": "cancel_pending", "run_id": run_id})
            raise RunCancelled(run_id or "")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, **kwargs: Any
    ) -> HttpGateway:
        return cls(
            transport=kwargs.pop("transport", None),
            config=GatewayConfig.from_env(environ),
            **kwargs,
        )

    @property
    def session_id(self) -> str:
        return self._session

    @property
    def jev_model(self) -> str:
        return self.config.jev_model

    def new_run(self, run_id: str | None = None) -> str:
        """Set a stable Go session value and return it."""
        self.usage = UsageLedger()
        self.decision_log = []
        self.transport_attempts_by_role = {}
        if run_id:
            session = str(run_id)
        else:
            session = uuid.uuid4().hex
        self._sessions[session] = session
        self._session = session
        return session

    def session_for(self, run_id: str | None = None) -> str:
        if run_id is None:
            return self._session
        key = str(run_id)
        return self._sessions.setdefault(key, key)

    def list_models(self, *, refresh: bool = False) -> CatalogSnapshot:
        if self.catalog is None:
            raise RuntimeError("model catalog is not configured")
        if hasattr(self.catalog, "fetch"):
            snapshot = self.catalog.fetch(force=refresh)
        else:
            snapshot = self.catalog
        if isinstance(snapshot, CatalogSnapshot):
            self._go_model_ids.update(snapshot.go_ids)
            return snapshot
        if isinstance(snapshot, Mapping):
            # Be liberal for tiny local catalog implementations.
            result = snapshot.get("snapshot", snapshot)
            if isinstance(result, CatalogSnapshot):
                self._go_model_ids.update(result.go_ids)
                return result
        raise TypeError("catalog.fetch() must return CatalogSnapshot")

    def route_model(self, model: str, *, run_id: str | None = None) -> RouteDecision:
        provider = "openrouter"
        if model != JEV_MODEL:
            try:
                snapshot = self.list_models()
                provider = "go" if model in snapshot.go_ids else "openrouter"
            except (RuntimeError, TypeError):
                provider = "go" if model in self._go_model_ids else "openrouter"
        base = (
            self.config.go_base_url
            if provider == "go"
            else self.config.openrouter_base_url
        )
        path = "/chat/completions"
        if provider == "go":
            if model.startswith(("qwen3.", "minimax-")):
                path = "/messages"
            elif model.startswith(("grok-", "gpt-", "muse-spark-")):
                path = "/responses"
        headers: dict[str, str] = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
        }
        key = (
            self.config.go_api_key
            if provider == "go"
            else self.config.openrouter_api_key
        )
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if provider == "go":
            headers["x-opencode-session"] = self.session_for(run_id)
            if path == "/messages":
                headers["anthropic-version"] = "2023-06-01"
                if key:
                    headers.pop("Authorization", None)
                    headers["x-api-key"] = key
        else:
            headers["HTTP-Referer"] = self.config.referer
            headers["X-Title"] = self.config.title
        return RouteDecision(provider, model, f"{base.rstrip('/')}{path}", headers)

    def _attempt(self, decision: RouteDecision, payload: Mapping[str, Any]) -> Any:
        return self.transport.request(
            decision.url,
            method="POST",
            headers=decision.headers,
            json=dict(payload),
            timeout=self.config.timeout,
        )

    def _request(
        self,
        decision: RouteDecision | None,
        payload: Mapping[str, Any] | Callable[[RouteDecision], Mapping[str, Any]],
        *,
        role: str,
        operation: str,
        run_id: str | None,
        route_factory: Callable[[], RouteDecision] | None = None,
    ) -> Any:
        started = self._monotonic()
        deadline = started + max(0.001, float(self.config.operation_timeout_s))
        self._emit_operation(
            {
                "event": "start",
                "operation": self._named_operation(operation),
                "role": role,
                "run_id": run_id,
                "timeout_s": self.config.operation_timeout_s,
            }
        )
        model_info: ModelInfo | None = None
        if decision is None and route_factory is None:
            raise ValueError("a Gateway route or route factory is required")

        def resolve_with_pricing() -> RouteDecision:
            nonlocal model_info
            if decision is None:
                if route_factory is None:
                    raise ValueError("a Gateway route or route factory is required")
                resolved = route_factory()
            else:
                resolved = decision
            # Default Jev routing skips catalog I/O, but its first pricing
            # lookup may still fetch the live catalog. Keep both in the same
            # bounded worker; accounting must not trigger an unbounded fetch.
            model_info = self._model_info(resolved.model)
            return resolved

        try:
            self._check_cancelled(run_id)
            decision = self._bounded_catalog_route(
                resolve_with_pricing,
                model="unknown",
                role=role,
                run_id=run_id,
                deadline=deadline,
            )
        except Exception as exc:
            from .failures import RunCancelled

            error_kind = (
                "cancelled"
                if isinstance(exc, RunCancelled)
                else "deadline_exceeded"
                if isinstance(exc, _GatewayDeadlineExceeded)
                else type(exc).__name__
            )
            self._emit_operation(
                {
                    "event": "error",
                    "operation": self._named_operation(operation),
                    "role": role,
                    "run_id": run_id,
                    "elapsed_ms": max(0, round((self._monotonic() - started) * 1000)),
                    "error_kind": error_kind,
                }
            )
            if isinstance(exc, RunCancelled):
                raise
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(
                "catalog",
                "model routing",
                None,
                "model routing failed or exceeded its deadline",
                role=role,
                kind="timeout"
                if isinstance(exc, _GatewayDeadlineExceeded)
                else "catalog",
            ) from exc
        request_payload = (
            cast(Mapping[str, Any], payload(decision)) if callable(payload) else payload
        )
        attempts = max(0, self.config.max_retries) + 1
        last_status: int | None = None
        for attempt in range(attempts):
            try:
                self._check_cancelled(run_id)
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise _GatewayDeadlineExceeded(
                        "gateway operation deadline exceeded"
                    )
                self.transport_attempts_by_role[role] = (
                    self.transport_attempts_by_role.get(role, 0) + 1
                )
                response = self._bounded_transport_request(
                    decision,
                    request_payload,
                    timeout=min(max(0.001, self.config.timeout), remaining),
                    deadline=deadline,
                    run_id=run_id,
                    role=role,
                )
                status = _response_status(response)
                last_status = status
                if status is not None and status >= 400:
                    self._check_cancelled(run_id)
                    if deadline - self._monotonic() <= 0:
                        raise _GatewayDeadlineExceeded(
                            "gateway operation deadline exceeded"
                        )
                    if status in RETRYABLE_STATUS and attempt + 1 < attempts:
                        headers = (
                            response.get("headers", {})
                            if isinstance(response, Mapping)
                            else getattr(response, "headers", {})
                        )
                        retry_after = (
                            next(
                                (
                                    value
                                    for key, value in headers.items()
                                    if key.lower() == "retry-after"
                                ),
                                None,
                            )
                            if isinstance(headers, Mapping)
                            else None
                        )
                        delay = _retry_after_seconds(retry_after)
                        if delay is None:
                            delay = self.config.backoff * (2**attempt)
                        self._emit_operation(
                            {
                                "event": "retry",
                                "operation": self._named_operation(operation),
                                "role": role,
                                "run_id": run_id,
                                "attempt": attempt + 1,
                                "delay_s": min(
                                    delay, max(0.0, deadline - self._monotonic())
                                ),
                                "status": status,
                            }
                        )
                        self._wait_retry(delay, deadline, run_id)
                        continue
                    if status in ACCESS_DENIED_STATUS:
                        self._note_provider(
                            decision.provider, "unavailable", status, decision.model
                        )
                    raise ProviderError(
                        decision.provider, decision.model, status, role=role
                    )
                self._note_provider(decision.provider, "ok", status, decision.model)
                decoded = _response_json(response)
                usage = decoded if isinstance(decoded, Mapping) else {}
                self.usage.record(
                    role=role,
                    provider=decision.provider,
                    model=decision.model,
                    response=usage,
                    input_cost_per_token=model_info.input_cost_per_token
                    if model_info
                    else None,
                    output_cost_per_token=model_info.output_cost_per_token
                    if model_info
                    else None,
                    cap=model_info.monthly_cap if model_info else None,
                    cost=(
                        float(usage["usage"]["cost"])
                        if isinstance(usage.get("usage"), Mapping)
                        and usage["usage"].get("cost") is not None
                        else None
                    ),
                )
                # A completed response is billable even when cancellation wins
                # the race to its caller. Never expose the answer after cancel.
                self._check_cancelled(run_id)
                if deadline - self._monotonic() <= 0:
                    raise _GatewayDeadlineExceeded(
                        "gateway operation deadline exceeded"
                    )
                self._emit_operation(
                    {
                        "event": "end",
                        "operation": self._named_operation(operation),
                        "role": role,
                        "run_id": run_id,
                        "elapsed_ms": max(
                            0, round((self._monotonic() - started) * 1000)
                        ),
                        "attempts": attempt + 1,
                        "status": status,
                    }
                )
                return decoded
            except ProviderError as exc:
                self._emit_operation(
                    {
                        "event": "error",
                        "operation": self._named_operation(operation),
                        "role": role,
                        "run_id": run_id,
                        "elapsed_ms": max(
                            0, round((self._monotonic() - started) * 1000)
                        ),
                        "attempts": attempt + 1,
                        "status": exc.status,
                        "error_kind": exc.kind,
                    }
                )
                raise
            except Exception as exc:
                from .failures import RunCancelled

                if isinstance(exc, RunCancelled):
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": self._named_operation(operation),
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "attempts": attempt + 1,
                            "error_kind": "cancelled",
                        }
                    )
                    raise
                remaining = deadline - self._monotonic()
                if remaining <= 0 or isinstance(exc, _GatewayDeadlineExceeded):
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": self._named_operation(operation),
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "attempts": attempt + 1,
                            "error_kind": "deadline_exceeded",
                        }
                    )
                    raise ProviderError(
                        decision.provider,
                        decision.model,
                        last_status,
                        "operation deadline exceeded",
                        role=role,
                        kind="timeout",
                    ) from exc
                if attempt + 1 >= attempts:
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": self._named_operation(operation),
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "attempts": attempt + 1,
                            "error_kind": type(exc).__name__,
                        }
                    )
                    raise ProviderError(
                        decision.provider, decision.model, last_status, role=role
                    ) from exc
                try:
                    self._wait_retry(
                        self.config.backoff * (2**attempt), deadline, run_id
                    )
                except _GatewayDeadlineExceeded as deadline_error:
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": self._named_operation(operation),
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "attempts": attempt + 1,
                            "error_kind": "deadline_exceeded",
                        }
                    )
                    raise ProviderError(
                        decision.provider,
                        decision.model,
                        last_status,
                        "operation deadline exceeded",
                        role=role,
                        kind="timeout",
                    ) from deadline_error
        raise ProviderError(decision.provider, decision.model, last_status, role=role)

    def _named_operation(self, operation: str) -> str:
        substage = (self._operation_context.get() or {}).get("substage")
        return f"{substage}.{operation}" if isinstance(substage, str) else operation

    def _acquire_worker(self, deadline: float, run_id: str | None) -> bool:
        """Wait for bounded capacity without extending the operation deadline."""
        while True:
            self._check_cancelled(run_id)
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return False
            if self._transport_workers.acquire(timeout=min(0.05, remaining)):
                try:
                    self._check_cancelled(run_id)
                    if deadline - self._monotonic() <= 0:
                        self._transport_workers.release()
                        return False
                except BaseException:
                    self._transport_workers.release()
                    raise
                return True

    def _bounded_catalog_route(
        self,
        route_factory: Callable[[], RouteDecision],
        *,
        model: str,
        role: str,
        run_id: str | None,
        deadline: float,
    ) -> RouteDecision:
        """Resolve a possibly live model catalog within the operation budget."""
        self._emit_operation(
            {
                "event": "start",
                "operation": "route_model",
                "role": role,
                "run_id": run_id,
                "timeout_s": max(0.0, deadline - self._monotonic()),
            }
        )
        finished = threading.Event()
        result: list[Any] = []
        started = self._monotonic()
        try:
            acquired = self._acquire_worker(deadline, run_id)
        except BaseException as exc:
            from .failures import RunCancelled

            self._emit_operation(
                {
                    "event": "error",
                    "operation": "route_model",
                    "role": role,
                    "run_id": run_id,
                    "elapsed_ms": max(0, round((self._monotonic() - started) * 1000)),
                    "error_kind": "cancelled"
                    if isinstance(exc, RunCancelled)
                    else type(exc).__name__,
                }
            )
            raise
        if not acquired:
            self._emit_operation(
                {
                    "event": "error",
                    "operation": "route_model",
                    "role": role,
                    "run_id": run_id,
                    "elapsed_ms": 0,
                    "error_kind": "transport_busy",
                }
            )
            raise ProviderError(
                "catalog",
                model,
                None,
                "too many provider requests are still shutting down",
                role=role,
                kind="transport_busy",
            )

        def resolve() -> None:
            try:
                result.append(route_factory())
            except BaseException as exc:
                result.append(exc)
            finally:
                self._transport_workers.release()
                finished.set()

        worker = threading.Thread(target=resolve, name="gateway-route", daemon=True)
        try:
            worker.start()
        except BaseException as exc:
            self._transport_workers.release()
            self._emit_operation(
                {
                    "event": "error",
                    "operation": "route_model",
                    "role": role,
                    "run_id": run_id,
                    "elapsed_ms": max(0, round((self._monotonic() - started) * 1000)),
                    "error_kind": type(exc).__name__,
                }
            )
            raise
        cancel_reported = False
        while True:
            if finished.is_set():
                try:
                    self._check_cancelled(run_id)
                except Exception as exc:
                    from .failures import RunCancelled

                    if not isinstance(exc, RunCancelled):
                        raise
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": "route_model",
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "error_kind": "cancelled",
                        }
                    )
                    raise
                if deadline - self._monotonic() <= 0:
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": "route_model",
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "error_kind": "deadline_exceeded",
                        }
                    )
                    raise _GatewayDeadlineExceeded(
                        "Gateway catalog routing exceeded its deadline"
                    )
                value = result[0]
                if isinstance(value, BaseException):
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": "route_model",
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "error_kind": type(value).__name__,
                        }
                    )
                    raise value
                self._emit_operation(
                    {
                        "event": "end",
                        "operation": "route_model",
                        "role": role,
                        "run_id": run_id,
                        "elapsed_ms": max(
                            0, round((self._monotonic() - started) * 1000)
                        ),
                    }
                )
                return value
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                try:
                    self._check_cancelled(run_id)
                except Exception as exc:
                    from .failures import RunCancelled

                    if not isinstance(exc, RunCancelled):
                        raise
                    self._emit_operation(
                        {
                            "event": "error",
                            "operation": "route_model",
                            "role": role,
                            "run_id": run_id,
                            "elapsed_ms": max(
                                0, round((self._monotonic() - started) * 1000)
                            ),
                            "error_kind": "cancelled",
                        }
                    )
                    raise
                self._emit_operation(
                    {
                        "event": "error",
                        "operation": "route_model",
                        "role": role,
                        "run_id": run_id,
                        "elapsed_ms": max(
                            0, round((self._monotonic() - started) * 1000)
                        ),
                        "error_kind": "deadline_exceeded",
                    }
                )
                raise _GatewayDeadlineExceeded(
                    "Gateway catalog routing exceeded its deadline"
                )
            if not cancel_reported:
                check = (self._operation_context.get() or {}).get("cancel_check")
                if callable(check) and check():
                    self._emit_operation(
                        {
                            "event": "cancel_pending",
                            "operation": "route_model",
                            "run_id": run_id,
                        }
                    )
                    cancel_reported = True
            finished.wait(min(0.05, remaining))

    def _bounded_transport_request(
        self,
        decision: RouteDecision,
        payload: Mapping[str, Any],
        *,
        timeout: float,
        deadline: float,
        run_id: str | None,
        role: str,
    ) -> Any:
        """Return by the logical deadline even if a transport ignores timeout.

        The transport runs in a daemon thread because a generic synchronous
        transport has no portable abort method. Cancellation stays pending
        until that request returns or reaches its deadline; the worker cannot
        keep the process alive after the caller has recorded a terminal result.
        """
        finished = threading.Event()
        result: list[Any] = []
        request_id = uuid.uuid4().hex
        payload_snapshot = copy.deepcopy(dict(payload))
        headers_snapshot = dict(decision.headers)
        if not self._acquire_worker(deadline, run_id):
            raise ProviderError(
                decision.provider,
                decision.model,
                None,
                "too many provider requests are still shutting down",
                role=role,
                kind="transport_busy",
            )

        def request() -> None:
            try:
                result.append(
                    self.transport.request(
                        decision.url,
                        method="POST",
                        headers=headers_snapshot,
                        json=payload_snapshot,
                        timeout=timeout,
                        **(
                            {"request_id": request_id}
                            if getattr(self.transport, "supports_request_id", False)
                            else {}
                        ),
                    )
                )
            except BaseException as exc:  # propagate in the caller thread
                result.append(exc)
            finally:
                self._transport_workers.release()
                finished.set()

        worker = threading.Thread(target=request, name="gateway-request", daemon=True)
        try:
            worker.start()
        except BaseException:
            self._transport_workers.release()
            raise
        cancel_reported = False
        abort_requested = False
        while True:
            if finished.is_set():
                # Closing an active request can make its worker report a
                # transport error. Once the run requested cancellation, that
                # error is a consequence of cancellation and must not enter
                # Gateway retry or provider-error handling.
                value = result[0]
                if isinstance(value, BaseException):
                    self._check_cancelled(run_id)
                    raise value
                return value
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                abort = getattr(self.transport, "abort_request", None)
                if (
                    callable(abort)
                    and getattr(self.transport, "supports_request_id", False)
                    and not abort_requested
                ):
                    try:
                        abort(request_id)
                        abort_requested = True
                    except Exception:  # noqa: BLE001 - timeout outcome takes precedence
                        abort_requested = True
                self._check_cancelled(run_id)
                raise _GatewayDeadlineExceeded("gateway operation deadline exceeded")
            if not cancel_reported:
                check = (self._operation_context.get() or {}).get("cancel_check")
                if callable(check) and check():
                    self._emit_operation({"event": "cancel_pending", "run_id": run_id})
                    cancel_reported = True
                    abort = getattr(self.transport, "abort_request", None)
                    if callable(abort) and getattr(
                        self.transport, "supports_request_id", False
                    ):
                        self._emit_operation(
                            {
                                "event": "cancel_abort_requested",
                                "role": role,
                                "run_id": run_id,
                            }
                        )
                        try:
                            abort(request_id)
                        except Exception:  # noqa: BLE001 - cancellation takes precedence
                            pass
                        abort_requested = True
            finished.wait(min(0.05, remaining))

    def _wait_retry(self, delay: float, deadline: float, run_id: str | None) -> None:
        remaining = deadline - self._monotonic()
        if remaining <= 0 or delay >= remaining:
            raise _GatewayDeadlineExceeded("gateway operation deadline exceeded")
        if not callable((self._operation_context.get() or {}).get("cancel_check")):
            self._sleep(max(0.0, delay))
            return
        end = self._monotonic() + max(0.0, delay)
        while True:
            self._check_cancelled(run_id)
            left = end - self._monotonic()
            if left <= 0:
                return
            self._sleep(min(0.1, left))

    def _note_provider(
        self, provider: str, status: str, http_status: int | None, model: str
    ) -> None:
        self._provider_status[provider] = {
            "status": status,
            "http_status": http_status,
            "model": model,
            "checked_at": time.time(),
        }

    def _probe(self, model: str) -> None:
        # Sent straight through the transport so the probe never lands in the
        # usage ledger of a run that may be in progress.
        decision = self.route_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 16,
        }
        if decision.url.endswith("/responses"):
            payload = {
                "model": model,
                "input": payload["messages"],
                "max_output_tokens": 16,
            }
        try:
            status = _response_status(self._attempt(decision, payload))
        except (OSError, ValueError):
            self._note_provider(decision.provider, "unknown", None, model)
            return
        if status is not None and status in ACCESS_DENIED_STATUS:
            self._note_provider(decision.provider, "unavailable", status, model)
        elif status is not None and status >= 400:
            self._note_provider(decision.provider, "unknown", status, model)
        else:
            self._note_provider(decision.provider, "ok", status, model)

    def provider_health(
        self, *, probe_models: Iterable[str] = ()
    ) -> dict[str, dict[str, Any]]:
        """Return the last known access state of each provider.

        A provider that has not been used recently is probed with a one-line
        chat request to the first of ``probe_models`` routed to it, so the UI
        can warn before a run is spent on a provider that refuses every call.
        """
        now = time.time()
        for model in probe_models:
            provider = self.route_model(model).provider
            known = self._provider_status.get(provider)
            if (
                known is not None
                and now - float(known["checked_at"]) < PROVIDER_STATUS_TTL
            ):
                continue
            self._probe(model)
        return {
            provider: {
                key: value for key, value in state.items() if key != "checked_at"
            }
            for provider, state in self._provider_status.items()
        }

    def _backoff(self, attempt: int) -> None:
        if self.config.backoff:
            self._sleep(self.config.backoff * (2**attempt))

    def _model_info(self, model: str) -> ModelInfo | None:
        if self.catalog is None:
            return None
        try:
            return self.list_models().get(model)
        except (RuntimeError, TypeError):
            return None

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]] | str,
        *,
        role: str = "writer",
        run_id: str | None = None,
        **params: Any,
    ) -> Any:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        routed_decisions: list[RouteDecision] = []

        def build_payload(decision: RouteDecision) -> Mapping[str, Any]:
            routed_decisions.append(decision)
            payload: dict[str, Any] = {
                "model": model,
                "messages": list(messages),
                **params,
            }
            if decision.url.endswith("/messages"):
                system = "\n".join(
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "system"
                )
                payload = {
                    "model": model,
                    "messages": [
                        dict(message)
                        for message in messages
                        if message.get("role") != "system"
                    ],
                    "max_tokens": params.get("max_tokens", 1024),
                    **({"system": system} if system else {}),
                    **(
                        {"temperature": params["temperature"]}
                        if "temperature" in params
                        else {}
                    ),
                }
            elif decision.url.endswith("/responses"):
                payload = {
                    "model": model,
                    "input": list(messages),
                    "max_output_tokens": params.get(
                        "max_tokens",
                        4096 if model.startswith("muse-spark-") else 1024,
                    ),
                }
            self.calls.append(
                {
                    "operation": "chat",
                    "role": role,
                    "model": model,
                    "provider": decision.provider,
                    "run_id": run_id,
                }
            )
            return payload

        response = self._request(
            None,
            build_payload,
            role=role,
            operation="chat",
            run_id=run_id,
            route_factory=lambda: self.route_model(model, run_id=run_id),
        )
        decision = routed_decisions[0]
        if decision.url.endswith("/messages") and isinstance(response, Mapping):
            content = response.get("content", [])
            text = (
                "".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, Mapping) and item.get("type") == "text"
                )
                if isinstance(content, list)
                else ""
            )
            return {**response, "choices": [{"message": {"content": text}}]}
        if decision.url.endswith("/responses") and isinstance(response, Mapping):
            text = response.get("output_text")
            if not isinstance(text, str):
                output = response.get("output", [])
                text = (
                    "".join(
                        str(part.get("text", ""))
                        for item in output
                        if isinstance(item, Mapping)
                        for part in item.get("content", [])
                        if isinstance(part, Mapping)
                        and part.get("type") == "output_text"
                    )
                    if isinstance(output, list)
                    else ""
                )
            return {**response, "choices": [{"message": {"content": text}}]}
        return response

    def decide(
        self,
        payload: Mapping[str, Any],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> Any:
        request = dict(payload)
        key, envelope = decision_payload(request, model=self.config.jev_model)
        self.calls.append(
            {
                "operation": "decide",
                "role": role,
                "model": self.config.jev_model,
                "provider": "openrouter",
                "run_id": run_id,
            }
        )
        response = self._request(
            None,
            envelope,
            role=role,
            operation="decide",
            run_id=run_id,
            route_factory=lambda: self._decisions_route(run_id),
        )
        try:
            answer = _decision_answers(response, [key])[0]
        except ProviderError:
            self._record_received_decision_answers([request], [key], response)
            raise
        model = response.get("model") if isinstance(response, Mapping) else None
        if not isinstance(model, str) or not model:
            self._record_received_decision_answers([request], [key], response)
        self.decision_log.append(self._decision_entry(request, answer, response))
        return answer

    def decide_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> list[Any]:
        if not requests:
            return []
        keys, envelope = batch_decision_payload(requests, model=self.config.jev_model)
        self.calls.append(
            {
                "operation": "decide",
                "role": role,
                "model": self.config.jev_model,
                "provider": "openrouter",
                "run_id": run_id,
            }
        )
        response = self._request(
            None,
            envelope,
            role=role,
            operation="decide_batch",
            run_id=run_id,
            route_factory=lambda: self._decisions_route(run_id),
        )
        try:
            answers = _decision_answers(response, keys)
        except ProviderError:
            self._record_received_decision_answers(requests, keys, response)
            raise
        model = response.get("model") if isinstance(response, Mapping) else None
        if not isinstance(model, str) or not model:
            self._record_received_decision_answers(requests, keys, response)
        self.decision_log.extend(
            self._decision_entry(request, answer, response)
            for request, answer in zip(requests, answers, strict=True)
        )
        return answers

    def _record_received_decision_answers(
        self,
        requests: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
        response: Any,
    ) -> None:
        """Log actual received answers when a response fails validation.

        This diagnostic path deliberately does not turn a partial or
        unidentified response into a successful return value. If the provider
        omitted its model snapshot, ``None`` records that identity as unknown;
        the caller still receives the original strict-validation error.
        """
        raw_answers = response.get("answers") if isinstance(response, Mapping) else None
        if not isinstance(raw_answers, Mapping):
            return
        model = response.get("model") if isinstance(response, Mapping) else None
        answered_by = model if isinstance(model, str) and model else None
        usage = response.get("usage") if isinstance(response, Mapping) else None
        usage_record = dict(usage) if isinstance(usage, Mapping) else {}
        for request, key in zip(requests, keys, strict=True):
            if key in raw_answers:
                self.decision_log.append(
                    {
                        "question": dict(request),
                        "answer": raw_answers[key],
                        "answered_by": answered_by,
                        "usage": usage_record,
                    }
                )

    def _decisions_route(self, run_id: str | None) -> RouteDecision:
        route = self.route_model(self.config.jev_model, run_id=run_id)
        base = self.config.openrouter_base_url.rstrip("/").removesuffix("/v1")
        return RouteDecision(
            route.provider,
            route.model,
            f"{base}{self.config.decisions_path}",
            route.headers,
        )

    def _decision_entry(
        self, request: Mapping[str, Any], answer: Any, response: Any
    ) -> dict[str, Any]:
        model = response.get("model") if isinstance(response, Mapping) else None
        if not isinstance(model, str) or not model:
            raise ProviderError(
                "openrouter",
                self.config.jev_model,
                None,
                "decision response is missing model snapshot",
                role="judge",
                kind="invalid_response",
            )
        return {
            "question": dict(request),
            "answer": answer,
            "answered_by": model,
            "usage": response.get("usage")
            if isinstance(response.get("usage"), Mapping)
            else {},
        }

    def usage_report(self) -> dict[str, Any]:
        return self.usage.to_dict()


def _call_or_value(value: Any, *args: Any, **kwargs: Any) -> Any:
    return value(*args, **kwargs) if callable(value) else value


class ScriptedGateway:
    """Deterministic gateway for engine tests and local demos."""

    def __init__(
        self,
        responses: Sequence[Any] = (),
        *,
        jev_model: str = JEV_MODEL,
        chat: Callable[..., Any] | None = None,
        decision: Callable[..., Any] | None = None,
        catalog: Any | None = None,
        usage: UsageLedger | None = None,
    ) -> None:
        """Answer from ``chat``/``decision`` handlers, else pop ``responses`` in order."""
        self.responses = list(responses)
        self.jev_model = jev_model
        self.chat_handler = chat
        self.decision_handler = decision
        self.catalog = catalog or StaticModelCatalog((), ())
        self.usage = usage or UsageLedger()
        self.calls: list[dict[str, Any]] = []
        self.decision_log: list[dict[str, Any]] = []
        self._run_id: str | None = None

    def new_run(self, run_id: str | None = None) -> str:
        self.usage = UsageLedger()
        self.decision_log = []
        self._run_id = str(run_id) if run_id is not None else "scripted"
        return self._run_id

    def _next(
        self, handler: Callable[..., Any] | None, *args: Any, **kwargs: Any
    ) -> Any:
        if handler is not None:
            return handler(*args, **kwargs)
        if not self.responses:
            raise ProviderError(
                "scripted", "response", None, "no scripted response remains"
            )
        return self.responses.pop(0)

    def chat(
        self,
        model: str,
        messages: Any,
        *,
        role: str = "writer",
        run_id: str | None = None,
        **params: Any,
    ) -> Any:
        self.calls.append(
            {"operation": "chat", "role": role, "model": model, "run_id": run_id}
        )
        return self._next(
            self.chat_handler, model, messages, role=role, run_id=run_id, **params
        )

    def decide(
        self,
        payload: Mapping[str, Any],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "operation": "decide",
                "role": role,
                "model": self.jev_model,
                "run_id": run_id,
            }
        )
        answer = self._next(
            self.decision_handler, dict(payload), role=role, run_id=run_id
        )
        self.decision_log.append(
            {
                "question": dict(payload),
                "answer": answer,
                "answered_by": self.jev_model,
                "usage": {},
            }
        )
        return answer

    def decide_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> list[Any]:
        return [self.decide(request, role=role, run_id=run_id) for request in requests]

    def list_models(self, *, refresh: bool = False) -> CatalogSnapshot:
        return (
            self.catalog.fetch(force=refresh)
            if hasattr(self.catalog, "fetch")
            else self.catalog
        )

    def provider_health(
        self, *, probe_models: Iterable[str] = ()
    ) -> dict[str, dict[str, Any]]:
        del probe_models
        return {}

    usage_report = HttpGateway.usage_report


class ReplayGateway(ScriptedGateway):
    """Replay recorded answers, found by a hash of the exact request.

    A request that was not recorded raises ``ProviderError``; there is no
    fallback, so a recording can never answer a different request.
    """

    def __init__(
        self,
        recordings: Mapping[str, Any],
        *,
        decision_provenance: Mapping[str, Any] | None = None,
        expected_snapshot: str | None = None,
        allow_snapshot_mismatch: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.recordings = dict(recordings)
        self.decision_provenance = dict(decision_provenance or {})
        self.diagnosis_request_byte_limit: int | None = None
        self.retry_reservation_multiplier: int | None = None
        self.criterion_reading: dict[str, Any] | None = None
        if not allow_snapshot_mismatch:
            snapshots = {
                item.get("answered_by")
                for item in self.decision_provenance.values()
                if isinstance(item, Mapping)
            }
            expected = expected_snapshot or self.jev_model
            mismatches = snapshots - {expected}
            if mismatches:
                raise ValueError(
                    f"recorded Jev snapshot {sorted(mismatches)!r} differs from configured pin {expected!r}; use --allow-snapshot-mismatch to override"
                )
        self.replayed_keys: list[str] = []

    @staticmethod
    def request_key(operation: str, model: str, payload: Any, role: str) -> str:
        raw = json_module_dumps(
            {"operation": operation, "model": model, "payload": payload, "role": role}
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _lookup(self, operation: str, model: str, payload: Any, role: str) -> Any:
        key = self.request_key(operation, model, payload, role)
        if (
            key not in self.recordings
            and operation == "decide"
            and not self.decision_provenance
        ):
            legacy_key = self.request_key(operation, LEGACY_JEV_ALIAS, payload, role)
            if legacy_key in self.recordings:
                key = legacy_key
        self.replayed_keys.append(key)
        if key not in self.recordings:
            raise ProviderError("replay", model, None, "no recorded response")
        return self.recordings[key]

    def chat(
        self,
        model: str,
        messages: Any,
        *,
        role: str = "writer",
        run_id: str | None = None,
        **params: Any,
    ) -> Any:
        payload = {
            "model": model,
            "messages": list(messages) if not isinstance(messages, str) else messages,
            **params,
        }
        self.calls.append(
            {"operation": "chat", "role": role, "model": model, "run_id": run_id}
        )
        return self._lookup("chat", model, payload, role)

    def decide(
        self,
        payload: Mapping[str, Any],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> Any:
        request = dict(payload)
        if "state" not in request and "prompt" in request:
            request["state"] = request.pop("prompt")
        self.calls.append(
            {
                "operation": "decide",
                "role": role,
                "model": self.jev_model,
                "run_id": run_id,
            }
        )
        key = self.request_key("decide", self.jev_model, request, role)
        answer = self._lookup("decide", self.jev_model, request, role)
        provenance = self.decision_provenance.get(key, {})
        if self.decision_provenance and not provenance:
            raise ProviderError(
                "replay",
                self.jev_model,
                None,
                "recorded Jev decision has no snapshot provenance",
            )
        self.decision_log.append(
            {
                "question": request,
                "answer": answer,
                "answered_by": provenance.get("answered_by", "unknown"),
                "usage": provenance.get("usage", {}),
            }
        )
        return answer


if TYPE_CHECKING:
    # Every adapter must satisfy the Gateway interface exactly; ty enforces it.
    _ADAPTERS: tuple[type[Gateway], ...] = (HttpGateway, ScriptedGateway, ReplayGateway)


__all__ = [
    "DEFAULT_GO_BASE_URL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "Gateway",
    "GatewayConfig",
    "GatewayTransport",
    "HttpGateway",
    "HttpTransport",
    "ProviderError",
    "ReplayGateway",
    "RouteDecision",
    "ScriptedGateway",
    "writer_messages",
]
