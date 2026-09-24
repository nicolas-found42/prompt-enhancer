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
    score: float
    level: int
    levels: tuple[ScoreLevel, ...]
    legend: dict[str, str]
    confidence: float

    @property
    def probabilities(self) -> dict[str, float]:
        return {str(option.level): option.probability for option in self.levels}


JevDecision: TypeAlias = NoulDecision | ChoiceDecision | ScoreDecision


def decision_question(request: Mapping[str, Any]) -> dict[str, Any]:
    """Build one Decisions API question from a provider-neutral request."""
    kind = str(request.get("type") or "noul")
    instruction_key = next(
        (name for name in ("query", "question", "instructions") if name in request),
        None,
    )
    instructions = request[instruction_key] if instruction_key is not None else None
    if (
        not isinstance(instructions, (str, dict, list))
        or not instructions
        or isinstance(instructions, str)
        and not instructions.strip()
        or not _json_compatible(instructions)
    ):
        raise ValueError("Jev question instructions are required")
    question: dict[str, Any] = {"type": kind, "instructions": instructions}
    criteria = request.get("criteria")
    if criteria is None and kind == "choice":
        options = request.get("options", ())
        criteria = (
            dict(options)
            if isinstance(options, Mapping)
            else {str(option): str(option) for option in options}
        )
    elif criteria is None and kind == "score":
        criteria = list(request.get("levels", ()))
    if criteria is not None:
        if not _json_compatible(criteria):
            raise ValueError("Jev question criteria must be JSON-compatible")
        question["criteria"] = criteria
    return question


def _json_compatible(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_compatible(item)
            for key, item in value.items()
        )
    return False


def decision_payload(
    request: Mapping[str, Any], *, model: str
) -> tuple[str, dict[str, Any]]:
    """Keep user-controlled text in state when building a single Jev call."""
    key = str(request.get("key") or "decision")
    state = request.get("state", request.get("prompt"))
    return key, {
        "model": model,
        "state": state,
        "questions": {key: decision_question(request)},
    }


def batch_decision_payload(
    requests: Sequence[Mapping[str, Any]], *, model: str
) -> tuple[list[str], dict[str, Any]]:
    """Build a batch envelope, addressing each question's corresponding state."""
    keys = [
        str(request.get("key") or f"decision_{index}")
        for index, request in enumerate(requests)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Jev batch question keys must be unique")
    states = [request.get("state", request.get("prompt")) for request in requests]
    shared_state = all(state == states[0] for state in states)
    state = (
        states[0] if shared_state else {"items": dict(zip(keys, states, strict=True))}
    )
    questions = {}
    for key, request in zip(keys, requests, strict=True):
        question = decision_question(request)
        if not shared_state:
            item = f"state.items[{key!r}]"
            instructions = question["instructions"]
            if isinstance(instructions, str):
                question["instructions"] = f"For {item}: {instructions}"
            elif isinstance(instructions, dict):
                if "item" in instructions and instructions["item"] != item:
                    raise ValueError(
                        "Jev batch instruction item field conflicts with state"
                    )
                question["instructions"] = {**instructions, "item": item}
            else:
                question["instructions"] = {"item": item, "question": instructions}
        questions[key] = question
    return keys, {"model": model, "state": state, "questions": questions}


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

    parsed = [
        (key, _probability(value, field=f"probability for {key}"))
        for key, value in items
    ]
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
        raw_probability = body.get(
            "noul", body.get("probability_true", body.get("probability"))
        )
        probability = _probability(raw_probability, field="noul")
        return NoulDecision(
            probability=probability,
            confidence=_confidence(body)
            if "confidence" in body or "certainty" in body
            else max(probability, 1 - probability),
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
        raw_score = body.get("score")
        if raw_score is None:
            raise JevResponseError("score is required")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise JevResponseError("score must be numeric") from exc
        if not math.isfinite(score):
            raise JevResponseError("score must be finite")
        highest_probability = max(option.probability for option in levels)
        try:
            level = min(
                int(option.level)
                for option in levels
                if option.probability == highest_probability
            )
        except ValueError as exc:
            raise JevResponseError("score levels must have numeric indexes") from exc
        raw_legend = body.get("legend", {})
        if not isinstance(raw_legend, Mapping):
            raise JevResponseError("score legend must be a mapping")
        return ScoreDecision(
            score=score,
            level=level,
            levels=levels,
            legend={str(key): str(value) for key, value in raw_legend.items()},
            confidence=_confidence(body),
        )

    raise JevResponseError(f"unsupported Jev decision kind: {kind or 'unknown'}")
