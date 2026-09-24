"""Live provider model catalogs and safe public model metadata.

The catalog is deliberately separate from the engine.  A transport can be
injected in tests (or by a local application) and the public representation
never contains provider credentials.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

JEV_MODEL = "typesafe/jev-1.13-20260917"
DEFAULT_GO_WRITER = "space-bunny-free"
DEFAULT_GO_STRONG = "glm-5.3-flash"
DEFAULT_WEAK_PANEL = (
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-nemo",
    "meta-llama/llama-3.2-3b-instruct",
)
DEFAULT_DEEP_WEAK_PANEL = DEFAULT_WEAK_PANEL + (
    "mimo-v2.6-flash",
    "muse-spark-1.3-contributor",
)

# The catalog flags models whose provider policies retain or train on prompts.
# Muse is included in Deep only because the user explicitly selected it.
_EXCLUDED_DEFAULT_FRAGMENTS = ("muse", "spark", "gpt-5.6-luna")


class CatalogTransport(Protocol):
    """Minimal transport seam used by :class:`LiveModelCatalog`."""

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


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Public model metadata.

    Rates are per token and are the rates reported by the provider.  ``None``
    means the provider did not expose a rate; callers must not guess it.
    """

    id: str
    provider: str
    name: str | None = None
    context_window: int | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    monthly_cap: float | None = None
    supports_chat: bool = True
    safe_for_default: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def model_id(self) -> str:
        return self.id

    @property
    def display_name(self) -> str:
        return self.name or self.id

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize without headers, URLs containing keys, or secrets."""
        result: dict[str, Any] = {
            "id": self.id,
            "model_id": self.id,
            "provider": self.provider,
            "name": self.display_name,
            "context_window": self.context_window,
            "input_cost_per_token": self.input_cost_per_token,
            "output_cost_per_token": self.output_cost_per_token,
            "monthly_cap": self.monthly_cap,
            "supports_chat": self.supports_chat,
            "safe_for_default": self.safe_for_default,
        }
        # Metadata is provider-owned and may contain arbitrary fields.  Only
        # copy scalar, non-secret-looking metadata; never expose auth data.
        if self.metadata:
            for key, value in self.metadata.items():
                lowered = key.lower()
                if any(word in lowered for word in ("key", "token", "secret", "authorization")):
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    result[key] = value
        return result


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """Immutable public view of both provider catalogs."""

    go: tuple[ModelInfo, ...] = ()
    openrouter: tuple[ModelInfo, ...] = ()
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def judge(self) -> ModelInfo:
        # Jev is intentionally fixed, even if a provider catalog is stale or
        # has not yet indexed the judge.
        return ModelInfo(
            id=JEV_MODEL,
            provider="openrouter",
            name="Jev judge (fixed)",
            supports_chat=False,
            safe_for_default=False,
        )

    @property
    def models(self) -> tuple[ModelInfo, ...]:
        return self.go + self.openrouter

    @property
    def go_ids(self) -> frozenset[str]:
        return frozenset(model.id for model in self.go)

    def get(self, model_id: str) -> ModelInfo | None:
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def supports(self, model_id: str) -> bool:
        return model_id == JEV_MODEL or self.get(model_id) is not None

    def default_safe(self) -> tuple[ModelInfo, ...]:
        return tuple(model for model in self.models if model.safe_for_default)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "judge": self.judge.to_public_dict(),
            "providers": {
                "go": [model.to_public_dict() for model in self.go],
                "openrouter": [model.to_public_dict() for model in self.openrouter],
            },
            "fetched_at": self.fetched_at.isoformat(),
        }


class CatalogError(RuntimeError):
    """Raised when a provider catalog cannot be fetched or decoded."""


def _as_list(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("data", "models", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
        # Some OpenCode deployments return a bare mapping keyed by model id.
        if payload and all(isinstance(value, Mapping) for value in payload.values()):
            return [{"id": key, **dict(value)} for key, value in payload.items()]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    return []


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _decode_json(response: Any) -> Any:
    if isinstance(response, Mapping) and "json" in response:
        value = response["json"]
        return value() if callable(value) else value
    if isinstance(response, Mapping) and "data" in response and not any(
        key in response for key in ("id", "models")
    ):
        data = response["data"]
        return data() if callable(data) else data
    json_method = getattr(response, "json", None)
    if callable(json_method):
        return json_method()
    return response


def _status(response: Any) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(response, name, None)
        if value is None and isinstance(response, Mapping):
            value = response.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _model_from_item(item: Mapping[str, Any], provider: str) -> ModelInfo | None:
    model_id = item.get("id") or item.get("model") or item.get("slug")
    if not model_id or not isinstance(model_id, str):
        return None
    pricing = item.get("pricing")
    if not isinstance(pricing, Mapping):
        pricing = {}
    input_rate = _number(
        pricing.get("prompt")
        if pricing.get("prompt") is not None
        else pricing.get("input")
        if pricing.get("input") is not None
        else item.get("input_cost_per_token")
    )
    output_rate = _number(
        pricing.get("completion")
        if pricing.get("completion") is not None
        else pricing.get("output")
        if pricing.get("output") is not None
        else item.get("output_cost_per_token")
    )
    lowered = model_id.lower()
    safe = not any(fragment in lowered for fragment in _EXCLUDED_DEFAULT_FRAGMENTS)
    return ModelInfo(
        id=model_id,
        provider=provider,
        name=item.get("name") if isinstance(item.get("name"), str) else None,
        context_window=_int(item.get("context_length") or item.get("context_window")),
        input_cost_per_token=input_rate,
        output_cost_per_token=output_rate,
        monthly_cap=_number(item.get("monthly_cap") or item.get("cap")),
        supports_chat=bool(item.get("supports_chat", True)),
        safe_for_default=safe,
        metadata={
            key: value
            for key, value in item.items()
            if key not in {"id", "model", "slug", "name", "pricing", "context_length", "context_window"}
        },
    )


def parse_catalog(payload: Any, provider: str) -> tuple[ModelInfo, ...]:
    """Decode common OpenRouter/OpenCode model-list response shapes."""
    decoded = _decode_json(payload)
    models: list[ModelInfo] = []
    seen: set[str] = set()
    for item in _as_list(decoded):
        model = _model_from_item(item, provider)
        if model is not None and model.id not in seen:
            models.append(model)
            seen.add(model.id)
    if not models:
        raise CatalogError(f"{provider} catalog response contained no models")
    return tuple(models)


class LiveModelCatalog:
    """Fetch Go and OpenRouter catalogs through a replaceable transport."""

    def __init__(
        self,
        transport: CatalogTransport,
        *,
        go_url: str,
        openrouter_url: str,
        go_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.transport = transport
        self.go_url = go_url
        self.openrouter_url = openrouter_url
        # Keys are intentionally held only by the server-side catalog object.
        self._go_api_key = go_api_key
        self._openrouter_api_key = openrouter_api_key
        self.timeout = timeout
        self._snapshot: CatalogSnapshot | None = None

    @property
    def snapshot(self) -> CatalogSnapshot | None:
        return self._snapshot

    def _fetch_one(self, url: str, provider: str, key: str | None) -> tuple[ModelInfo, ...]:
        headers = {"Accept": "application/json", "User-Agent": "prompt-enhancer/0.1"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            response = self.transport.request(
                url, method="GET", headers=headers, timeout=self.timeout
            )
        except Exception as exc:
            raise CatalogError(f"unable to fetch {provider} catalog") from exc
        status = _status(response)
        if status is not None and status >= 400:
            raise CatalogError(f"{provider} catalog request failed with HTTP {status}")
        return parse_catalog(response, provider)

    def fetch(self, *, force: bool = False) -> CatalogSnapshot:
        if self._snapshot is not None and not force:
            return self._snapshot
        # Fetch both even if one provider is unavailable: callers can decide
        # whether to use a partial catalog, but an error is surfaced clearly.
        go = self._fetch_one(self.go_url, "go", self._go_api_key)
        openrouter = self._fetch_one(
            self.openrouter_url, "openrouter", self._openrouter_api_key
        )
        self._snapshot = CatalogSnapshot(go=go, openrouter=openrouter)
        return self._snapshot

    list_models = fetch

    def route_provider(self, model_id: str) -> str:
        if model_id == JEV_MODEL:
            return "openrouter"
        snapshot = self.snapshot or self.fetch()
        return "go" if model_id in snapshot.go_ids else "openrouter"


class StaticModelCatalog:
    """Small deterministic catalog useful for local mode and public tests."""

    def __init__(self, go: Iterable[ModelInfo | str], openrouter: Iterable[ModelInfo | str] = ()) -> None:
        self._snapshot = CatalogSnapshot(
            go=tuple(self._coerce(item, "go") for item in go),
            openrouter=tuple(self._coerce(item, "openrouter") for item in openrouter),
        )

    @staticmethod
    def _coerce(item: ModelInfo | str, provider: str) -> ModelInfo:
        return item if isinstance(item, ModelInfo) else ModelInfo(id=item, provider=provider)

    @property
    def snapshot(self) -> CatalogSnapshot:
        return self._snapshot

    def fetch(self, *, force: bool = False) -> CatalogSnapshot:
        return self._snapshot

    list_models = fetch

    def route_provider(self, model_id: str) -> str:
        if model_id == JEV_MODEL:
            return "openrouter"
        return "go" if model_id in self._snapshot.go_ids else "openrouter"


__all__ = [
    "DEFAULT_DEEP_WEAK_PANEL",
    "DEFAULT_GO_STRONG",
    "DEFAULT_GO_WRITER",
    "DEFAULT_WEAK_PANEL",
    "JEV_MODEL",
    "CatalogError",
    "CatalogSnapshot",
    "CatalogTransport",
    "LiveModelCatalog",
    "ModelInfo",
    "StaticModelCatalog",
    "parse_catalog",
]
