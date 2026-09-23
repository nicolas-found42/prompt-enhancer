"""Typed parsing for Jev decision responses.

The gateway deliberately returns provider payloads unchanged.  This module is the
single place where provider-shaped responses become small, immutable values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias


class JevResponseError(ValueError):
    """Raised when a Jev response cannot be represented as a known decision."""


@dataclass(frozen=True, slots=True)
class NoulDecision:
    probability: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    value: str
    probability: float


@dataclass(frozen=True, slots=True)
class ChoiceDecision:
    selected: str
    options: tuple[ChoiceOption, ...]
    confidence: float

    @property
    def probabilities(self) -> dict[str, float]:
        return {option.value: option.probability for option in self.options}


@dataclass(frozen=True, slots=True)
class ScoreLevel:
    level: int | float | str
    probability: float


@dataclass(frozen=True, slots=True)
class ScoreDecision:
    level: int | float | str
    levels: tuple[ScoreLevel, ...]
    confidence: float

    @property
    def probabilities(self) -> dict[int | float | str, float]:
        return {option.level: option.probability for option in self.levels}


JevDecision: TypeAlias = NoulDecision | ChoiceDecision | ScoreDecision


def _payload(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    dump = getattr(payload, "model_dump", None)
    if callable(dump):
        value = dump()
        if isinstance(value, Mapping):
            return value
    return vars(payload) if hasattr(payload, "__dict__") else {}


def _unwrap(payload: Any) -> tuple[Mapping[str, Any], str | None]:
    body = _payload(payload)
    kind_value = body.get("type") or body.get("kind")
    kind = str(kind_value).lower() if kind_value is not None else None

    for name in ("noul", "choice", "score", "decision", "answer", "result"):
        nested = body.get(name)
        if isinstance(nested, Mapping):
            nested_body = _payload(nested)
            return nested_body, str(nested_body.get("type") or kind or name).lower()

    if isinstance(body.get("data"), Mapping):
        return _unwrap(body["data"])
    return body, kind


def _probability(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise JevResponseError(f"{field} must be a probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise JevResponseError(f"{field} must be between 0 and 1")
    return result


def _confidence(body: Mapping[str, Any]) -> float:
    value = body.get("confidence", body.get("certainty", 0.0))
    return _probability(value, field="confidence")


def _probability_items(raw: Any) -> list[tuple[str, float]]:
    if isinstance(raw, Mapping):
        items = [(str(key), value) for key, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        items = []
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                key = item.get("value", item.get("level", item.get("choice", index)))
                value = item.get("probability", item.get("probability_true"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                key, value = item
            else:
                key, value = index, item
            items.append((str(key), value))
    else:
        raise JevResponseError("decision probabilities are required")

    parsed = [(key, _probability(value, field=f"probability for {key}")) for key, value in items]
    total = sum(value for _, value in parsed)
    if not parsed or total <= 0.0:
        raise JevResponseError("decision probabilities must not be empty")
    return [(key, value / total) for key, value in parsed]


def _selected(
    body: Mapping[str, Any], candidates: Sequence[object], *, field: str
) -> str:
    for name in (field, "answer", "value", "label"):
        value = body.get(name)
        if value is not None:
            return str(value)
    raise JevResponseError(f"{field} is required")


def parse_decision(payload: Any) -> JevDecision:
    """Parse a Jev ``noul``, ``choice``, or ``score`` response.

    Provider envelopes are accepted, but malformed or unknown decisions fail
    closed.  ``parse_decision`` is idempotent for values returned by this
    function.
    """

    if isinstance(payload, (NoulDecision, ChoiceDecision, ScoreDecision)):
        return payload

    body, kind = _unwrap(payload)
    if kind is None:
        if "probability_true" in body or "probability" in body:
            kind = "noul"
        elif "selected" in body or "choice" in body or "options" in body:
            kind = "choice"
        elif "level" in body or "levels" in body or "score" in body:
            kind = "score"

    if kind == "noul" or ("probability_true" in body and "options" not in body):
        raw_probability = body.get("probability_true", body.get("probability"))
        return NoulDecision(
            probability=_probability(raw_probability, field="probability_true"),
            confidence=_confidence(body),
        )

    if kind == "choice":
        raw_options = body.get("options", body.get("probabilities"))
        options = tuple(
            ChoiceOption(value=value, probability=probability)
            for value, probability in _probability_items(raw_options)
        )
        selected = _selected(body, [option.value for option in options], field="choice")
        if selected not in {option.value for option in options}:
            raise JevResponseError("choice must name one of the returned options")
        return ChoiceDecision(
            selected=selected,
            options=options,
            confidence=_confidence(body),
        )

    if kind == "score":
        raw_levels = body.get("levels", body.get("probabilities"))
        levels = tuple(
            ScoreLevel(level=value, probability=probability)
            for value, probability in _probability_items(raw_levels)
        )
        level_value: int | float | str
        selected = _selected(body, [option.level for option in levels], field="score")
        try:
            level_value = int(selected)
        except ValueError:
            try:
                level_value = float(selected)
            except ValueError:
                level_value = selected
        if level_value not in {option.level for option in levels}:
            raise JevResponseError("score must name one of the returned levels")
        return ScoreDecision(level=level_value, levels=levels, confidence=_confidence(body))

    raise JevResponseError(f"unsupported Jev decision kind: {kind or 'unknown'}")
