"""Convergence loop: retry until the quality vector meets its floors and stalls.

The Perfect Prompt Loop (#169) stops a healthy run when the best candidate's
score vector has met *every* dimension floor and the marginal gain over the
previous round's best has fallen below the documented epsilon. Until then the
loop keeps retrying, handing each round's losing candidates to the next
round's strategy search through the existing prior-failures seam.

The loop adds no attempt cap of its own: run control (cancel, budget pause),
a permanent provider failure, and the Route stage's impossible outcome are
the stop surfaces. This module reads the per-round vector that
:mod:`prompt_enhancer.score_vector` records — never a one-shot end-of-run
computation — and decides from the last two rounds' evidence alone, so it is
a pure function that scripted vectors can test directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Report/history outcome for a run that met its floors and stopped gaining.
CONVERGED_STATUS = "converged"

#: What the loop decision reads while the loop must keep trying.
CONTINUE_STATUS = "continue"


@dataclass(frozen=True, slots=True)
class ConvergenceDecision:
    """The loop decision after one round, with the vector evidence behind it."""

    status: str
    scores: Mapping[str, float]
    floors: Mapping[str, float]
    breaches: tuple[str, ...]
    gain: float | None
    epsilon: float
    passed: bool = False

    @property
    def converged(self) -> bool:
        return self.status == CONVERGED_STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "converged": self.converged,
            "scores": dict(self.scores),
            "floors": dict(self.floors),
            "breaches": list(self.breaches),
            "passed": self.passed,
            "gain": self.gain,
            "epsilon": self.epsilon,
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def mean_score(scores: Mapping[str, Any]) -> float:
    """The mean of a score vector, used for the marginal-gain comparison."""

    values = [
        float(value)
        for value in scores.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(values) / len(values) if values else 0.0


def best_candidate_vector(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """The best available candidate's score vector from a round report.

    The winning candidate is preferred; when the round kept the original (no
    candidate qualified) the highest-ranked candidate's vector is the round's
    best available measurement, which is what the loop compares across rounds.
    """

    selection = _mapping(_mapping(report).get("selection_evidence"))
    ordered: list[Mapping[str, Any]] = []
    winner = selection.get("selected_candidate")
    has_winner = isinstance(winner, Mapping)
    if has_winner:
        ordered.append(winner)
    elif selection.get("original_kept") is True:
        original = _mapping(selection.get("original"))
        vector = _mapping(_mapping(original.get("metadata")).get("score_vector"))
        if vector.get("scores"):
            return {
                "scores": dict(_mapping(vector.get("scores"))),
                "floors": dict(_mapping(vector.get("floors"))),
                "breaches": list(vector.get("breaches") or ()),
                "passed": bool(vector.get("passed")),
                "selected": False,
                "source": vector.get("source", "original_baseline"),
                "candidate_id": original.get("candidate_id", "original"),
            }
    ranking = selection.get("ranking")
    if isinstance(ranking, list):
        ordered.extend(item for item in ranking if isinstance(item, Mapping))
    for candidate in ordered:
        vector = _mapping(candidate.get("metadata")).get("score_vector")
        if isinstance(vector, Mapping) and vector.get("scores"):
            return {
                "scores": dict(_mapping(vector.get("scores"))),
                "floors": dict(_mapping(vector.get("floors"))),
                "breaches": list(vector.get("breaches") or ()),
                # A rejected or merely top-ranked candidate cannot establish
                # that the prompt returned to the user met every floor.
                "passed": bool(vector.get("passed")) and has_winner,
                "selected": has_winner,
                "source": vector.get("source", "selected_candidate"),
                "candidate_id": candidate.get("candidate_id"),
            }
    return None


def convergence_decision(
    vector: Mapping[str, Any] | None,
    *,
    previous_vector: Mapping[str, Any] | None,
    epsilon: float,
) -> ConvergenceDecision:
    """Decide whether the run has converged on this round's best vector.

    Converged means every dimension met its floor *and* the marginal gain over
    the previous round's best vector fell to or below epsilon. A first round
    whose vector already meets its floors has no further gain to find and
    converges immediately; a below-floor vector, or a gain still above
    epsilon, keeps the loop retrying.
    """

    scores = dict(_mapping(_mapping(vector).get("scores")))
    floors = dict(_mapping(_mapping(vector).get("floors")))
    breaches = tuple(str(item) for item in _mapping(vector).get("breaches") or ())
    passed = bool(vector) and bool(_mapping(vector).get("passed"))
    if isinstance(vector, Mapping) and "selected" in vector:
        passed = passed and bool(vector.get("selected"))
    gain: float | None = None
    if vector and previous_vector is not None:
        previous_scores = _mapping(previous_vector).get("scores")
        if isinstance(previous_scores, Mapping) and scores:
            gain = mean_score(scores) - mean_score(previous_scores)
    # The vectors are means of decimal probabilities. Allow a tiny arithmetic
    # tolerance so a nominal gain exactly on the configured boundary converges.
    converged = passed and (gain is None or gain <= epsilon + 1e-12)
    return ConvergenceDecision(
        status=CONVERGED_STATUS if converged else CONTINUE_STATUS,
        scores=scores,
        floors=floors,
        breaches=breaches,
        gain=gain,
        epsilon=epsilon,
        passed=passed,
    )


def convergence_summary(
    scores: Mapping[str, float],
    floors: Mapping[str, float],
    *,
    gain: float | None = None,
) -> str:
    """Summarize measured convergence without claiming an unmeasured plateau."""

    evidence = (
        "every quality dimension met its floor on the first round; no earlier round existed for a gain comparison"
        if gain is None
        else "every quality dimension met its floor and the measured gain was within the configured epsilon"
    )
    if scores and floors:
        dimensions = ", ".join(
            f"{name} {scores.get(name, 0.0):.2f} (floor {floors.get(name, 0.0):.2f})"
            for name in sorted(scores)
        )
        return f"The prompt converged: {evidence}. Final vector — {dimensions}."
    return f"The prompt converged: {evidence}."


__all__ = [
    "CONTINUE_STATUS",
    "CONVERGED_STATUS",
    "ConvergenceDecision",
    "best_candidate_vector",
    "convergence_decision",
    "convergence_summary",
    "mean_score",
]
