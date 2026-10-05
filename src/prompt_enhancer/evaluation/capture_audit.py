"""Validate exact per-request identity before recording or auditing a capture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..gateway import ReplayGateway


def validate_capture(
    records: Sequence[Mapping[str, Any]],
    expected_keys: Sequence[str],
    responses: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject missing, reused, reordered, or incorrectly associated evidence."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("capture records must be an array")
    if not isinstance(responses, Mapping):
        raise ValueError("capture responses must be an object")
    if (
        not isinstance(expected_keys, Sequence)
        or isinstance(expected_keys, (str, bytes))
        or not all(isinstance(key, str) for key in expected_keys)
    ):
        raise ValueError("expected request keys must be a string array")
    if len(records) != len(expected_keys):
        raise ValueError("capture record count does not match expected request count")
    identifiers: set[str] = set()
    for record, expected in zip(records, expected_keys, strict=True):
        if (
            not isinstance(record, Mapping)
            or "payload" not in record
            or "answer" not in record
        ):
            raise ValueError("capture record must include its payload and answer")
        correlation = record.get("correlation_id")
        if (
            not isinstance(correlation, str)
            or not correlation
            or correlation in identifiers
        ):
            raise ValueError("capture correlation identifier is missing or reused")
        identifiers.add(correlation)
        for field in ("operation", "model", "role"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"capture {field} is missing")
        key = ReplayGateway.request_key(
            record["operation"], record["model"], record.get("payload"), record["role"]
        )
        if key != expected or record.get("request_key") != key:
            raise ValueError(
                "capture request identity or payload hash does not reconcile"
            )
        if key not in responses or record.get("answer") != responses[key]:
            raise ValueError("capture answer is associated with the wrong request")
    if set(expected_keys) != set(responses):
        raise ValueError("capture response keys do not reconcile with requests")
    return {
        "status": "complete",
        "request_count": len(records),
        "response_count": len(responses),
    }
