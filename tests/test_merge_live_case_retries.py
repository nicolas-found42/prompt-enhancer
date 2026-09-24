"""Targeted live reruns replace only their own case metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_live_case_retries import merge


def test_merge_replaces_retry_case_and_keeps_unaffected_case() -> None:
    base = {
        "responses": {"same": {"noul": 0.2}, "old": {"text": "old"}},
        "case_costs": {"a": {"total": 1.0}, "b": {"total": 2.0}},
        "case_latency_ms": {"a": 10, "b": 20},
        "rubric_thresholds": {"context": 0.87},
    }
    retry = {
        "responses": {"same": {"noul": 0.3}, "new": {"text": "new"}},
        "case_costs": {"b": {"total": 3.0}},
        "case_latency_ms": {"b": 30},
        "rubric_thresholds": {"context": 0.87},
    }

    result = merge(base, retry, retry_case_ids={"b"})

    assert result["case_costs"] == {"a": {"total": 1.0}, "b": {"total": 3.0}}
    assert result["case_latency_ms"] == {"a": 10, "b": 30}
    assert result["responses"]["same"] == {"noul": 0.3}
    assert result["responses"]["old"] == {"text": "old"}


def test_merge_requires_matching_rubric() -> None:
    base = {"responses": {}, "case_costs": {"a": {}}, "case_latency_ms": {"a": 1}, "rubric_thresholds": {"context": 0.87}}
    retry = {"responses": {}, "case_costs": {"a": {}}, "case_latency_ms": {"a": 1}, "rubric_thresholds": {"context": 0.8}}

    with pytest.raises(ValueError, match="rubric thresholds"):
        merge(base, retry, retry_case_ids={"a"})


def test_merge_requires_matching_faithfulness_threshold() -> None:
    base = {"responses": {}, "case_costs": {"a": {}}, "case_latency_ms": {"a": 1}, "rubric_thresholds": None}
    retry = {**base, "faithfulness_threshold": 0.8}

    with pytest.raises(ValueError, match="faithfulness thresholds"):
        merge(base, retry, retry_case_ids={"a"})
