"""Quality score vector: six judged dimensions with max-gate floors.

Every candidate is scored on fidelity, style fit, clarity, specificity,
coherence, and safety. Each non-fidelity dimension is judged by its own
narrow noul question (see :mod:`prompt_enhancer.jev_questions`); the
fidelity dimension reuses the existing fidelity checks as its hard gate
(``1.0`` when they pass, ``0.0`` otherwise). Code normalizes every
dimension to the 0-1 scale and applies per-dimension floors read from the
engine :class:`~prompt_enhancer.config.Settings`: any dimension below its
floor rejects the candidate outright — a breach is never averaged away by
strong siblings.

The vector shape (``dimensions``/``scores``/``floors``/``breaches``/
``passed``) is pinned in the report-snapshot fixtures; keep it stable for
the convergence loop (#169) and floor recalibration (#170).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import jev_questions
from .fidelity import FidelityResult
from .gateway import Gateway, ProviderError
from .jev import JevResponseError, NoulDecision, parse_decision

#: Six score-vector dimensions in pinned report order.
SCORE_DIMENSIONS = (
    "fidelity",
    "style_fit",
    "clarity",
    "specificity",
    "coherence",
    "safety",
)

#: Dimensions judged by their own noul question (fidelity is the hard gate).
JUDGED_DIMENSIONS = tuple(
    dimension for dimension in SCORE_DIMENSIONS if dimension != "fidelity"
)


def score_request_key(dimension: str) -> str:
    """Gateway request key for one judged dimension (``score:<dimension>``)."""
    return f"score:{dimension}"


@dataclass(frozen=True)
class ScoreVector:
    """One candidate's normalized scores against the configured floors."""

    scores: Mapping[str, float]
    floors: Mapping[str, float]
    breaches: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Mapping[str, float],
        *,
        fidelity_passed: bool,
        floors: Mapping[str, float],
        evidence: Mapping[str, Any] | None = None,
    ) -> ScoreVector:
        """Normalize raw probabilities to 0-1 and name every floor breach."""
        normalized_floors = {
            dimension: float(floors[dimension]) for dimension in SCORE_DIMENSIONS
        }
        scores: dict[str, float] = {
            "fidelity": 1.0 if fidelity_passed else 0.0,
        }
        for dimension in JUDGED_DIMENSIONS:
            try:
                raw = float(probabilities.get(dimension, 0.0))
            except (TypeError, ValueError):
                raw = 0.0
            scores[dimension] = min(1.0, max(0.0, raw))
        breaches = tuple(
            dimension
            for dimension in SCORE_DIMENSIONS
            if scores[dimension] < normalized_floors[dimension]
        )
        return cls(
            scores=scores,
            floors=normalized_floors,
            breaches=breaches,
            evidence=dict(evidence or {}),
        )

    @property
    def passed(self) -> bool:
        """Max-gate verdict: no dimension may sit below its floor."""
        return not self.breaches

    @property
    def breach_reasons(self) -> tuple[str, ...]:
        """Human-readable rejection reasons naming each breaching dimension."""
        return tuple(
            f"score floor breached: {dimension} "
            f"({self.scores[dimension]:.2f} < floor {self.floors[dimension]:.2f})"
            for dimension in self.breaches
        )

    @property
    def judged_breach_reasons(self) -> tuple[str, ...]:
        """Breach reasons for judged dimensions only.

        The fidelity dimension's hard gate is the fidelity check itself, so
        its breach would duplicate ``candidate failed fidelity checks`` (a
        fidelity breach is equivalent to a failed gate: the dimension scores
        1.0 exactly when the gate passes). The full breach list stays in
        :attr:`breaches` and ``to_dict()``.
        """
        return tuple(
            f"score floor breached: {dimension} "
            f"({self.scores[dimension]:.2f} < floor {self.floors[dimension]:.2f})"
            for dimension in self.breaches
            if dimension != "fidelity"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": list(SCORE_DIMENSIONS),
            "scores": dict(self.scores),
            "floors": dict(self.floors),
            "breaches": list(self.breaches),
            "passed": self.passed,
            "evidence": {
                key: dict(value) if isinstance(value, Mapping) else value
                for key, value in self.evidence.items()
            },
        }


def score_candidate(
    gateway: Gateway,
    original_prompt: str,
    candidate_prompt: str,
    *,
    fidelity: FidelityResult,
    applied_style: str,
    style_bundle: Sequence[str],
    floors: Mapping[str, float],
    judge_model: str,
    run_id: str,
) -> ScoreVector:
    """Judge the five non-fidelity dimensions in one batch; fail closed.

    A fidelity failure already rejects the candidate; its dimension still
    records ``0.0`` so the vector names why. Unparseable answers or incomplete
    judgment evidence score ``0.0`` for affected dimensions — a candidate is
    never passed on missing evidence. A :class:`ProviderError` propagates as
    an operational failure so the uncapped retry loop stops instead of
    retrying an outage.
    """
    state = {
        "original_prompt": original_prompt,
        "candidate_prompt": candidate_prompt,
        "applied_style": applied_style,
        "style_bundle": list(style_bundle),
    }
    requests: list[dict[str, Any]] = []
    for dimension in JUDGED_DIMENSIONS:
        if dimension == "style_fit":
            query = jev_questions.style_fit_question(applied_style, style_bundle)
        else:
            query = jev_questions.SCORE_QUESTIONS[dimension]
        requests.append(
            {
                "model": judge_model,
                "key": score_request_key(dimension),
                "type": "noul",
                "query": query,
                "state": state,
            }
        )
    try:
        raw_answers = gateway.decide_batch(requests, role="judge", run_id=run_id)
        if not isinstance(raw_answers, list) or len(raw_answers) != len(requests):
            raise JevResponseError("incomplete score-vector response")
    except ProviderError:
        # Provider outages are operational failures, not a low-quality vector.
        # Propagation lets the optimizer stop the uncapped retry loop cleanly.
        raise
    except JevResponseError as exc:
        return ScoreVector.from_probabilities(
            {},
            fidelity_passed=fidelity.passed,
            floors=floors,
            evidence={"error": type(exc).__name__},
        )
    probabilities: dict[str, float] = {}
    evidence: dict[str, Any] = {"fidelity": {"passed": fidelity.passed}}
    for dimension, answer in zip(JUDGED_DIMENSIONS, raw_answers, strict=True):
        try:
            decision = parse_decision(answer)
            if not isinstance(decision, NoulDecision):
                raise JevResponseError("wrong Jev decision type")
            probability, confidence = decision.probability, decision.confidence
        except JevResponseError:
            probability, confidence = 0.0, 0.0
        probabilities[dimension] = probability
        evidence[dimension] = {"probability": probability, "confidence": confidence}
    return ScoreVector.from_probabilities(
        probabilities,
        fidelity_passed=fidelity.passed,
        floors=floors,
        evidence=evidence,
    )


__all__ = [
    "JUDGED_DIMENSIONS",
    "SCORE_DIMENSIONS",
    "ScoreVector",
    "score_candidate",
    "score_request_key",
]
