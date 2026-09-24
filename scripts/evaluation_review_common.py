"""Shared screening, splitting, calibration, and output helpers for evaluation scripts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOAR_PLACEHOLDER = re.compile(
    r"\[[A-Z _]{3,}\]|<[^>]{3,}>|\b(?:placeholder|omitted|removed|redacted|code snippet)\b",
    re.IGNORECASE,
)

# Credential-like text that must never be sent to a provider or saved in review output.
SECRET = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|"
    r"(?:api[_ -]?key|password|access[_ -]?token|bearer|secret)\s*[:=]\s*\S{8,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
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


def in_holdout_group(group: str) -> bool:
    """Assign about one fifth of source groups to holdout, stably by SHA-256."""
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0


def save_json(path: Path, value: Any) -> None:
    """Write indented JSON atomically so an interrupted run keeps the prior file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
