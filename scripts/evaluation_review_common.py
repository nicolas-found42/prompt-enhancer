"""Shared screening and binary calibration measures for evaluation scripts."""

from __future__ import annotations

import re
from collections.abc import Iterable

SOAR_PLACEHOLDER = re.compile(
    r"\[[A-Z _]{3,}\]|<[^>]{3,}>|\b(?:placeholder|omitted|removed|redacted|code snippet)\b",
    re.IGNORECASE,
)


def has_soar_placeholder(prompt: str) -> bool:
    return SOAR_PLACEHOLDER.search(prompt) is not None


def binary_metrics(rows: Iterable[tuple[float, bool]], threshold: float) -> dict[str, float | int]:
    observations = list(rows)
    tp = sum(probability >= threshold and expected for probability, expected in observations)
    fp = sum(probability >= threshold and not expected for probability, expected in observations)
    fn = sum(probability < threshold and expected for probability, expected in observations)
    tn = sum(probability < threshold and not expected for probability, expected in observations)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f0_5": f05}
