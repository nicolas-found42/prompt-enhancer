"""Provider-routing model gateway with deterministic local seams.

All outbound traffic goes through a small transport protocol.  Production can
use :class:`HttpTransport`; tests and local mode use a recording transport,
:class:`ScriptedGateway`, or :class:`ReplayGateway`.  API keys are read from
server-side configuration and are never placed in a response or exception.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

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

DEFAULT_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_USER_AGENT = "prompt-enhancer/0.1"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class GatewayTransport(Protocol):
    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any | None = None,
        timeout: float | None = None,
    ) -> Any:
        ...


@dataclass(slots=True)
class GatewayConfig:
    go_api_key: str | None = None
    openrouter_api_key: str | None = None
    go_base_url: str = DEFAULT_GO_BASE_URL
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    go_models_url: str | None = None
    openrouter_models_url: str | None = None
    decisions_path: str = "/alpha/decisions"
    timeout: float = 120.0
    max_retries: int = 2
    backoff: float = 0.0
    user_agent: str = DEFAULT_USER_AGENT
    referer: str = "https://localhost/prompt-enhancer"
    title: str = "Prompt Enhancer"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GatewayConfig:
        env = os.environ if environ is None else environ
        go_base = env.get("OPENCODE_GO_BASE_URL", DEFAULT_GO_BASE_URL).rstrip("/")
        openrouter_base = env.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
        return cls(
            go_api_key=env.get("OPENCODE_GO_KEY") or env.get("OPENCODE_GO_API_KEY"),
            openrouter_api_key=env.get("OPENROUTER_API_KEY"),
            go_base_url=go_base,
            openrouter_base_url=openrouter_base,
            go_models_url=env.get("OPENCODE_GO_MODELS_URL", f"{go_base}/models"),
            openrouter_models_url=env.get("OPENROUTER_MODELS_URL", f"{openrouter_base}/models"),
            timeout=float(env.get("PROMPT_ENHANCER_TIMEOUT", "120")),
            max_retries=int(env.get("PROMPT_ENHANCER_MAX_RETRIES", "2")),
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

    def __init__(self, provider: str, model: str, status: int | None, message: str = "provider request failed") -> None:
        self.provider = provider
        self.model = model
        self.status = status
        super().__init__(f"{provider} request for {model} failed ({status or 'network error'}): {message}")


class HttpTransport:
    """Minimal JSON HTTP transport; no provider-specific behavior."""

    def __init__(self, opener: Callable[..., Any] | None = None) -> None:
        self._opener = opener or urllib.request.urlopen

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = None if json is None else json_module_dumps(json).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self._opener(request, timeout=timeout) as response:
                body = response.read()
                status = getattr(response, "status", getattr(response, "status_code", 200))
                parsed = _decode_body(body)
                return {"status_code": status, "json": parsed, "headers": dict(response.headers.items())}
        except urllib.error.HTTPError as exc:
            body = exc.read() if hasattr(exc, "read") else b""
            return {"status_code": exc.code, "json": _decode_body(body), "headers": dict(exc.headers.items())}
        # URLError, timeout, and connection errors intentionally propagate to
        # ModelGateway, which retries and wraps them as ProviderError.


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


def _response_json(response: Any) -> Any:
    if isinstance(response, Mapping):
        value = response.get("json", response.get("data", response))
    else:
        value = getattr(response, "json", response)
    return value() if callable(value) else value


class ModelGateway:
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
    ) -> None:
        self.config = config or GatewayConfig()
        self.transport = transport or HttpTransport()
        self.catalog = catalog
        self.usage = usage or UsageLedger()
        self._sleep = sleep or time.sleep
        self._session = uuid.uuid4().hex
        self._sessions: dict[str, str] = {}
        self._go_model_ids: set[str] = {
            DEFAULT_GO_WRITER,
            "deepseek-v4.1-flash",
            DEFAULT_GO_STRONG,
            "mimo-v2.6-flash",
            "qwen3.8-flash",
            "muse-spark-1.3-contributor",
        } | {
            item if isinstance(item, str) else item.id for item in (go_models or ())
        }
        self.calls: list[dict[str, Any]] = []
        self.decision_log: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **kwargs: Any) -> ModelGateway:
        return cls(transport=kwargs.pop("transport", None), config=GatewayConfig.from_env(environ), **kwargs)

    @property
    def session_id(self) -> str:
        return self._session

    def new_run(self, run_id: str | None = None) -> str:
        """Set a stable Go session value and return it."""
        self.usage = UsageLedger()
        self.decision_log = []
        if run_id:
            session = str(run_id)
        else:
            session = uuid.uuid4().hex
        self._sessions[session] = session
        self._session = session
        return session

    set_run = new_run

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

    catalog_snapshot = list_models

    def route_model(self, model: str, *, run_id: str | None = None) -> RouteDecision:
        provider = "openrouter"
        if model != JEV_MODEL:
            try:
                snapshot = self.list_models()
                provider = "go" if model in snapshot.go_ids else "openrouter"
            except (RuntimeError, TypeError):
                provider = "go" if model in self._go_model_ids else "openrouter"
        base = self.config.go_base_url if provider == "go" else self.config.openrouter_base_url
        path = "/chat/completions"
        if provider == "go":
            if model.startswith(("qwen3.", "minimax-")):
                path = "/messages"
            elif model.startswith(("grok-", "gpt-", "muse-spark-")):
                path = "/responses"
        headers: dict[str, str] = {"User-Agent": self.config.user_agent, "Accept": "application/json"}
        key = self.config.go_api_key if provider == "go" else self.config.openrouter_api_key
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

    route = route_model

    def _attempt(self, decision: RouteDecision, payload: Mapping[str, Any]) -> Any:
        return self.transport.request(
            decision.url,
            method="POST",
            headers=decision.headers,
            json=dict(payload),
            timeout=self.config.timeout,
        )

    def _request(self, decision: RouteDecision, payload: Mapping[str, Any], *, role: str) -> Any:
        attempts = max(0, self.config.max_retries) + 1
        last_status: int | None = None
        for attempt in range(attempts):
            try:
                response = self._attempt(decision, payload)
                status = _response_status(response)
                last_status = status
                if status is not None and status >= 400:
                    if status in RETRYABLE_STATUS and attempt + 1 < attempts:
                        self._backoff(attempt)
                        continue
                    raise ProviderError(decision.provider, decision.model, status)
                decoded = _response_json(response)
                usage = decoded if isinstance(decoded, Mapping) else {}
                model_info = self._model_info(decision.model)
                self.usage.record(
                    role=role,
                    provider=decision.provider,
                    model=decision.model,
                    response=usage,
                    input_cost_per_token=model_info.input_cost_per_token if model_info else None,
                    output_cost_per_token=model_info.output_cost_per_token if model_info else None,
                    cap=model_info.monthly_cap if model_info else None,
                    cost=(
                        float(usage["usage"]["cost"])
                        if isinstance(usage.get("usage"), Mapping)
                        and usage["usage"].get("cost") is not None
                        else None
                    ),
                )
                return decoded
            except ProviderError:
                raise
            except Exception as exc:
                if attempt + 1 >= attempts:
                    raise ProviderError(decision.provider, decision.model, last_status) from exc
                self._backoff(attempt)
        raise ProviderError(decision.provider, decision.model, last_status)

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
        decision = self.route_model(model, run_id=run_id)
        payload: dict[str, Any] = {"model": model, "messages": list(messages), **params}
        if decision.url.endswith("/messages"):
            system = "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "system")
            payload = {
                "model": model,
                "messages": [dict(message) for message in messages if message.get("role") != "system"],
                "max_tokens": params.get("max_tokens", 1024),
                **({"system": system} if system else {}),
                **({"temperature": params["temperature"]} if "temperature" in params else {}),
            }
        elif decision.url.endswith("/responses"):
            payload = {
                "model": model,
                "input": list(messages),
                "max_output_tokens": params.get("max_tokens", 4096 if model.startswith("muse-spark-") else 1024),
            }
        self.calls.append({"operation": "chat", "role": role, "model": model, "provider": decision.provider, "run_id": run_id})
        response = self._request(decision, payload, role=role)
        if decision.url.endswith("/messages") and isinstance(response, Mapping):
            content = response.get("content", [])
            text = "".join(str(item.get("text", "")) for item in content if isinstance(item, Mapping) and item.get("type") == "text") if isinstance(content, list) else ""
            return {**response, "choices": [{"message": {"content": text}}]}
        if decision.url.endswith("/responses") and isinstance(response, Mapping):
            text = response.get("output_text")
            if not isinstance(text, str):
                output = response.get("output", [])
                text = "".join(
                    str(part.get("text", ""))
                    for item in output if isinstance(item, Mapping)
                    for part in item.get("content", []) if isinstance(part, Mapping) and part.get("type") == "output_text"
                ) if isinstance(output, list) else ""
            return {**response, "choices": [{"message": {"content": text}}]}
        return response

    def complete(self, model: str | Mapping[str, Any], messages: Any = None, **kwargs: Any) -> Any:
        if isinstance(model, Mapping) and messages is None:
            request = dict(model)
            model_id = str(request.get("model", ""))
            state = request.get("state")
            instructions = str(request.get("instructions", ""))
            if state is not None:
                messages = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(state, ensure_ascii=False, sort_keys=True)},
                ]
            else:
                messages = [{"role": "user", "content": instructions}]
            kwargs.setdefault("role", str(request.get("role", "writer")))
            return self.chat(model_id, messages, **kwargs)
        return self.chat(str(model), messages, **kwargs)
    def decide(self, payload: Mapping[str, Any], *, role: str = "judge", run_id: str | None = None) -> Any:
        request = dict(payload)
        key, envelope = decision_payload(request, model=JEV_MODEL)
        decision = self._decisions_route(run_id)
        self.calls.append({"operation": "decide", "role": role, "model": JEV_MODEL, "provider": "openrouter", "run_id": run_id})
        response = self._request(
            decision,
            envelope,
            role=role,
        )
        answer = response["answers"][key]
        self.decision_log.append({"question": request, "answer": answer})
        return answer

    decision = decide
    jev = decide

    def jev_batch(self, requests: Sequence[Mapping[str, Any]], *, role: str = "judge", run_id: str | None = None) -> list[Any]:
        if not requests:
            return []
        keys, envelope = batch_decision_payload(requests, model=JEV_MODEL)
        self.calls.append({"operation": "decide", "role": role, "model": JEV_MODEL, "provider": "openrouter", "run_id": run_id})
        response = self._request(
            self._decisions_route(run_id),
            envelope,
            role=role,
        )
        answers = [response["answers"][key] for key in keys]
        self.decision_log.extend(
            {"question": request, "answer": answer}
            for request, answer in zip(requests, answers, strict=True)
        )
        return answers

    def _decisions_route(self, run_id: str | None) -> RouteDecision:
        route = self.route_model(JEV_MODEL, run_id=run_id)
        base = self.config.openrouter_base_url.rstrip("/").removesuffix("/v1")
        return RouteDecision(route.provider, route.model, f"{base}{self.config.decisions_path}", route.headers)

    def usage_report(self) -> dict[str, Any]:
        return self.usage.to_dict()


def _call_or_value(value: Any, *args: Any, **kwargs: Any) -> Any:
    return value(*args, **kwargs) if callable(value) else value


class ScriptedGateway:
    """Deterministic gateway for engine tests and local demos."""

    def __init__(
        self,
        responses: Any = None,
        *,
        chat: Callable[..., Any] | None = None,
        decision: Callable[..., Any] | None = None,
        catalog: Any | None = None,
        usage: UsageLedger | None = None,
    ) -> None:
        self.responses = list(responses) if isinstance(responses, (list, tuple)) else responses
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

    set_run = new_run

    def _next(self, handler: Callable[..., Any] | None, *args: Any, **kwargs: Any) -> Any:
        if handler is not None:
            return handler(*args, **kwargs)
        if callable(self.responses):
            return self.responses(*args, **kwargs)
        if isinstance(self.responses, (list, tuple)):
            if not self.responses:
                raise ProviderError("scripted", "response", None, "no scripted response remains")
            return self.responses.pop(0)
        if isinstance(self.responses, Mapping):
            for key, value in self.responses.items():
                if key in kwargs or key in args:
                    return value
            if len(self.responses) == 1:
                return next(iter(self.responses.values()))
        raise ProviderError("scripted", "response", None, "no scripted response")

    def chat(self, model: str, messages: Any, *, role: str = "writer", run_id: str | None = None, **params: Any) -> Any:
        self.calls.append({"operation": "chat", "role": role, "model": model, "run_id": run_id})
        return self._next(self.chat_handler, model, messages, role=role, run_id=run_id, **params)

    def complete(self, model: str | Mapping[str, Any], messages: Any = None, **kwargs: Any) -> Any:
        if isinstance(model, Mapping) and messages is None:
            request = dict(model)
            model_id = str(request.get("model", ""))
            state = request.get("state")
            instructions = str(request.get("instructions", ""))
            messages = (
                [{"role": "system", "content": instructions}, {"role": "user", "content": json.dumps(state, ensure_ascii=False, sort_keys=True)}]
                if state is not None
                else [{"role": "user", "content": instructions}]
            )
            kwargs.setdefault("role", str(request.get("role", "writer")))
            return self.chat(model_id, messages, **kwargs)
        return self.chat(str(model), messages, **kwargs)

    def decide(self, payload: Mapping[str, Any], *, role: str = "judge", run_id: str | None = None) -> Any:
        self.calls.append({"operation": "decide", "role": role, "model": JEV_MODEL, "run_id": run_id})
        answer = self._next(self.decision_handler, dict(payload), role=role, run_id=run_id)
        self.decision_log.append({"question": dict(payload), "answer": answer})
        return answer

    decision = decide
    jev = decide

    def jev_batch(self, requests: Sequence[Mapping[str, Any]], *, role: str = "judge", run_id: str | None = None) -> list[Any]:
        return [self.jev(request, role=role, run_id=run_id) for request in requests]
    def list_models(self, *, refresh: bool = False) -> CatalogSnapshot:
        return self.catalog.fetch(force=refresh) if hasattr(self.catalog, "fetch") else self.catalog

    usage_report = ModelGateway.usage_report


class ReplayGateway(ScriptedGateway):
    """Replay recorded responses by a deterministic request key.

    ``responses`` may map tuples, call-shaped keys, or canonical request hashes
    to JSON responses.  The matcher first checks explicit keys and then falls
    back to a stable SHA-256 key, making recorded suites deterministic without
    network access.
    """

    def __init__(self, responses: Mapping[Any, Any] | Iterable[Any], *, strict: bool = False, **kwargs: Any) -> None:
        super().__init__(responses, **kwargs)
        self.strict = strict
        self.replayed_keys: list[str] = []

    @staticmethod
    def request_key(operation: str, model: str, payload: Any, role: str) -> str:
        raw = json_module_dumps({"operation": operation, "model": model, "payload": payload, "role": role})
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _lookup(self, responses: Mapping[Any, Any], operation: str, model: str, payload: Any, role: str) -> Any:
        key = self.request_key(operation, model, payload, role)
        self.replayed_keys.append(key)
        if key in responses:
            return responses[key]
        if self.strict:
            raise ProviderError("replay", model, None, "no recorded response")
        for key in ((operation, model, role), (model, role), (operation, model), model):
            if key in responses:
                return responses[key]
        alias = f"{'decision' if operation == 'decide' else 'complete'}:{model if operation == 'decide' else role}"
        if alias in responses:
            value = responses[alias]
            if operation == "decide" and isinstance(value, Mapping):
                kind = str(payload.get("type", "noul")) if isinstance(payload, Mapping) else "noul"
                value = value.get(kind, value)
            return value
        if not self.strict and len(responses) == 1:
            return next(iter(responses.values()))
        raise ProviderError("replay", model, None, "no recorded response")

    def chat(self, model: str, messages: Any, *, role: str = "writer", run_id: str | None = None, **params: Any) -> Any:
        payload = {"model": model, "messages": list(messages) if not isinstance(messages, str) else messages, **params}
        self.calls.append({"operation": "chat", "role": role, "model": model, "run_id": run_id})
        if self.chat_handler is not None or callable(self.responses):
            return self._next(self.chat_handler, model, messages, role=role, run_id=run_id, **params)
        return self._lookup(
            cast(Mapping[Any, Any], self.responses), "chat", model, payload, role
        )

    def decide(self, payload: Mapping[str, Any], *, role: str = "judge", run_id: str | None = None) -> Any:
        request = dict(payload)
        if "state" not in request and "prompt" in request:
            request["state"] = request.pop("prompt")
        self.calls.append({"operation": "decide", "role": role, "model": JEV_MODEL, "run_id": run_id})
        answer = (
            self._next(self.decision_handler, request, role=role, run_id=run_id)
            if self.decision_handler is not None or callable(self.responses)
            else self._lookup(cast(Mapping[Any, Any], self.responses), "decide", JEV_MODEL, request, role)
        )
        self.decision_log.append({"question": request, "answer": answer})
        return answer

    decision = decide
    jev = decide


__all__ = [
    "DEFAULT_GO_BASE_URL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "GatewayConfig",
    "GatewayTransport",
    "HttpTransport",
    "ModelGateway",
    "ProviderError",
    "ReplayGateway",
    "RouteDecision",
    "ScriptedGateway",
]
