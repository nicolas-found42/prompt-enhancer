"""User-selected model defaults and per-run overrides.

The settings store is a tiny JSON repository suitable for a local single-user
app.  It intentionally stores no API keys.  A server can replace the storage
implementation later without changing the public settings surface.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .catalog import (
    DEFAULT_GO_STRONG,
    DEFAULT_GO_WRITER,
    DEFAULT_WEAK_PANEL,
    JEV_MODEL,
)

ROLES = ("writer", "strong", "weak")


@dataclass(frozen=True, slots=True)
class ModelDefaults:
    writer: str = DEFAULT_GO_WRITER
    strong: str = DEFAULT_GO_STRONG
    weak: tuple[str, ...] = DEFAULT_WEAK_PANEL

    def __post_init__(self) -> None:
        if not self.writer or not isinstance(self.writer, str):
            raise ValueError("writer must be a non-empty model id")
        if not self.strong or not isinstance(self.strong, str):
            raise ValueError("strong must be a non-empty model id")
        if not self.weak or any(not isinstance(model, str) or not model for model in self.weak):
            raise ValueError("weak must contain non-empty model ids")

    def to_dict(self) -> dict[str, Any]:
        return {"writer": self.writer, "strong": self.strong, "weak": list(self.weak)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> ModelDefaults:
        payload = payload or {}
        writer = payload.get("writer", DEFAULT_GO_WRITER)
        strong = payload.get("strong", DEFAULT_GO_STRONG)
        weak = payload.get("weak", payload.get("weak_panel", DEFAULT_WEAK_PANEL))
        if isinstance(weak, str):
            weak = (weak,)
        return cls(writer=writer, strong=strong, weak=tuple(weak))

    def merged(self, overrides: Mapping[str, Any] | None) -> ModelDefaults:
        """Return a per-run selection without mutating saved defaults."""
        if not overrides:
            return self
        data = self.to_dict()
        for role in ROLES:
            if role not in overrides:
                continue
            value = overrides[role]
            if role == "weak" and not isinstance(value, str):
                value = tuple(value)
            data[role] = value
        return ModelDefaults.from_dict(data)

    resolve = merged


@dataclass(frozen=True, slots=True)
class Settings:
    defaults: ModelDefaults = ModelDefaults()

    def to_public_dict(self) -> dict[str, Any]:
        return {"models": self.defaults.to_dict(), "judge": JEV_MODEL}

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> Settings:
        return Settings(defaults=self.defaults.merged(overrides))

    resolve = with_overrides


class SettingsStore:
    """Atomic local JSON settings persistence with an injectable initial path."""

    def __init__(self, path: str | os.PathLike[str] = "settings.json", defaults: ModelDefaults | None = None) -> None:
        self.path = Path(path)
        self._defaults = defaults or ModelDefaults()
        self._lock = RLock()
        self._settings = self._load()

    def _load(self) -> Settings:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return Settings(self._defaults)
        if not isinstance(payload, Mapping):
            return Settings(self._defaults)
        # Accept both {models: {...}} and the bare model map for easy migration.
        model_payload = payload.get("models", payload)
        return Settings(ModelDefaults.from_dict(model_payload))

    def load(self) -> Settings:
        with self._lock:
            return self._settings

    get = load

    def save(self, settings: Settings | ModelDefaults | Mapping[str, Any]) -> Settings:
        if isinstance(settings, Settings):
            normalized = settings
        elif isinstance(settings, ModelDefaults):
            normalized = Settings(settings)
        else:
            normalized = Settings(ModelDefaults.from_dict(settings))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".settings-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(normalized.to_public_dict(), stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            self._settings = normalized
            return normalized

    update = save

    def update_defaults(self, **roles: Any) -> Settings:
        with self._lock:
            return self.save(self._settings.defaults.merged(roles))

    def resolve(self, overrides: Mapping[str, Any] | None = None) -> Settings:
        return self.load().with_overrides(overrides)

    def public(self) -> dict[str, Any]:
        return self.load().to_public_dict()


__all__ = ["ROLES", "ModelDefaults", "Settings", "SettingsStore"]
