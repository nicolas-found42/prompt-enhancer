"""Keep/reject labels linked to winning score vectors and floor calibration.

Calibration requires at least twelve completed labeled runs, with at least
three keeps and three rejections. Per dimension, the proposed floor is the
midpoint between the mean kept and rejected scores. This deliberately simple
rule uses both labels and is auditable; the result is clamped to 75% of the
shipped floor and 1.0. Twelve is an operational minimum, not a statistical
certainty claim. Below the sample minimum, floors are left untouched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .score_vector import SCORE_DIMENSIONS

MINIMUM_LABELED_SAMPLE = 12
MINIMUM_PER_LABEL = 3
CONSERVATIVE_FLOOR_FACTOR = 0.75


@dataclass(frozen=True, slots=True)
class FloorCalibration:
    """An auditable proposed floor update computed from labeled run vectors."""

    shipped_floors: Mapping[str, float]
    floors_before: Mapping[str, float]
    adjusted_floors: Mapping[str, float]
    movement: Mapping[str, float]
    sample_size: int
    keep_count: int
    reject_count: int
    changed: bool
    applied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shipped_floors": dict(self.shipped_floors),
            "floors_before": dict(self.floors_before),
            "adjusted_floors": dict(self.adjusted_floors),
            "movement": dict(self.movement),
            "sample_size": self.sample_size,
            "keep_count": self.keep_count,
            "reject_count": self.reject_count,
            "changed": self.changed,
            "applied": self.applied,
            "reason": self.reason,
            "minimum_labeled_sample": MINIMUM_LABELED_SAMPLE,
            "minimum_per_label": MINIMUM_PER_LABEL,
            "conservative_floor_factor": CONSERVATIVE_FLOOR_FACTOR,
        }


def calibrate_score_floors(
    records: Sequence[Mapping[str, Any]],
    current_floors: Mapping[str, float],
    *,
    conservative_floors: Mapping[str, float] | None = None,
) -> FloorCalibration:
    """Compute floor values from records carrying ``feedback_labels`` evidence."""
    floors = {name: float(current_floors[name]) for name in SCORE_DIMENSIONS}
    shipped = {
        name: float((conservative_floors or current_floors)[name])
        for name in SCORE_DIMENSIONS
    }
    labeled: list[tuple[str, dict[str, float]]] = []
    for record in records:
        labels = record.get("feedback_labels")
        if not isinstance(labels, Mapping):
            continue
        decision = labels.get("decision")
        vector = labels.get("score_vector")
        if (
            decision not in {"accept", "reject"}
            or labels.get("status") != "linked"
            or not labels.get("candidate_id")
            or not isinstance(vector, Mapping)
        ):
            continue
        scores: dict[str, float] = {}
        for dimension in SCORE_DIMENSIONS:
            try:
                score = float(vector[dimension])
            except (KeyError, TypeError, ValueError):
                break
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                break
            scores[dimension] = score
        if len(scores) == len(SCORE_DIMENSIONS):
            labeled.append((str(decision), scores))

    keep_count = sum(label == "accept" for label, _ in labeled)
    reject_count = sum(label == "reject" for label, _ in labeled)
    adjusted = dict(floors)
    ready = (
        len(labeled) >= MINIMUM_LABELED_SAMPLE
        and keep_count >= MINIMUM_PER_LABEL
        and reject_count >= MINIMUM_PER_LABEL
    )
    if ready:
        for dimension in SCORE_DIMENSIONS:
            kept = [scores[dimension] for label, scores in labeled if label == "accept"]
            rejected = [
                scores[dimension] for label, scores in labeled if label == "reject"
            ]
            midpoint = (sum(kept) / len(kept) + sum(rejected) / len(rejected)) / 2
            lower_bound = shipped[dimension] * CONSERVATIVE_FLOOR_FACTOR
            adjusted[dimension] = min(1.0, max(lower_bound, midpoint))

    movement = {
        dimension: round(adjusted[dimension] - floors[dimension], 6)
        for dimension in SCORE_DIMENSIONS
    }
    changed = any(movement.values())
    return FloorCalibration(
        shipped_floors=shipped,
        floors_before=floors,
        adjusted_floors=adjusted,
        movement=movement,
        sample_size=len(labeled),
        keep_count=keep_count,
        reject_count=reject_count,
        changed=changed,
        applied=ready and changed,
        reason=(
            "calibrated from labeled keep/reject vectors"
            if ready
            else f"requires {MINIMUM_LABELED_SAMPLE} labels including at least "
            f"{MINIMUM_PER_LABEL} keeps and {MINIMUM_PER_LABEL} rejections"
        ),
    )
