"""Retain typed Noul evidence without provider explanations or diagnostics."""

from collections.abc import Mapping
from typing import Any

_FIELDS = {
    "type",
    "kind",
    "probability_true",
    "probability",
    "confidence",
    "certainty",
    "noul",
}
_ENVELOPES = {"noul", "choice", "score", "decision", "answer", "result", "data"}


def typed_evidence(raw: Any, *, _depth: int = 0) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"malformed_type": type(raw).__name__}
    if _depth >= 12:
        return {"evidence_gap": "Provider envelope exceeds the retained depth bound."}
    result = {}
    for key, value in raw.items():
        if key in _ENVELOPES and isinstance(value, Mapping):
            result[key] = typed_evidence(value, _depth=_depth + 1)
        elif key in _FIELDS and (
            value is None or isinstance(value, (str, int, float, bool))
        ):
            result[key] = value
    return result
