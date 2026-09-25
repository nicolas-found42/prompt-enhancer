"""Grade panel outputs into the metrics used for robust candidate ranking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from . import jev_questions
from .evaluation.order_bias import LEGACY_GRADING_POLICY, OrderBiasPolicy
from .gateway import Gateway
from .jev import (
    ChoiceDecision,
    JevResponseError,
    NoulDecision,
    ScoreDecision,
    parse_decision,
)
from .runner import PanelResult


@dataclass(frozen=True)
class GradeReport:
    """Per-model pass rates and the aggregate metrics required by the spec."""

    candidate_id: str
    per_model: Mapping[str, float]
    per_model_samples: Mapping[str, tuple[float, ...]]
    worst: float
    mean: float
    spread: float
    sample_scores: tuple[float, ...] = ()
    graded_outputs: int = 0
    threshold: float = 0.5

    @property
    def worst_model_pass_rate(self) -> float:
        return self.worst

    @property
    def mean_pass_rate(self) -> float:
        return self.mean

    @property
    def sample_spread(self) -> float:
        return self.spread

    @property
    def per_model_pass_rates(self) -> Mapping[str, float]:
        return self.per_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "per_model": dict(self.per_model),
            "per_model_pass_rates": dict(self.per_model),
            "per_model_samples": {
                model: list(scores) for model, scores in self.per_model_samples.items()
            },
            "worst": self.worst,
            "worst_model_pass_rate": self.worst,
            "mean": self.mean,
            "mean_pass_rate": self.mean,
            "spread": self.spread,
            "sample_spread": self.spread,
            "sample_scores": list(self.sample_scores),
            "graded_outputs": self.graded_outputs,
            "threshold": self.threshold,
        }


def _noul_probability(answer: Any) -> float | None:
    """A Jev yes/no probability, or None for an unusable answer."""
    try:
        decision = parse_decision(answer)
    except JevResponseError:
        return None
    return decision.probability if isinstance(decision, NoulDecision) else None


def metrics_from_scores(
    per_model_scores: Mapping[str, Sequence[float]],
    *,
    threshold: float = 0.5,
) -> tuple[
    dict[str, float],
    dict[str, tuple[float, ...]],
    float,
    float,
    float,
    tuple[float, ...],
]:
    """Calculate report metrics from raw model/sample scores.

    Per-model values are pass rates (scores at or above ``threshold``).  The
    aggregate mean is the mean of all output scores, while spread is the range
    of per-sample means.  This keeps both model robustness and sample
    consistency visible to the selector.
    """

    rates: dict[str, float] = {}
    samples: dict[str, tuple[float, ...]] = {}
    for model in sorted(per_model_scores):
        values = tuple(float(value) for value in per_model_scores[model])
        samples[model] = values
        rates[model] = (
            fmean(1.0 if value >= threshold else 0.0 for value in values)
            if values
            else 0.0
        )
    mean = fmean(rates.values()) if rates else 0.0
    worst = min(rates.values(), default=0.0)
    sample_means = [
        fmean(values[index] for values in samples.values() if index < len(values))
        for index in range(max((len(values) for values in samples.values()), default=0))
    ]
    sample_means = [value for value in sample_means if value is not None]
    spread = max(sample_means) - min(sample_means) if len(sample_means) > 1 else 0.0
    return rates, samples, worst, mean, spread, tuple(sample_means)


def grade_candidate(
    candidate_id: str,
    panel: Sequence[PanelResult],
    score: Callable[[PanelResult], float],
    *,
    threshold: float = 0.5,
) -> GradeReport:
    """Grade one candidate's panel outputs into robust aggregate metrics.

    ``score`` rates one output between 0 and 1. No gateway calls happen here;
    ``grade_panel_with_jev`` supplies scores from Jev.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    runs = [run for run in panel if run.candidate_id == candidate_id]
    per_model_scores: dict[str, list[float]] = {}
    for run in runs:
        per_model_scores.setdefault(run.model, []).append(float(score(run)))
    rates, model_samples, worst, mean, spread, sample_scores = metrics_from_scores(
        per_model_scores, threshold=threshold
    )
    return GradeReport(
        candidate_id=candidate_id,
        per_model=rates,
        per_model_samples=model_samples,
        worst=worst,
        mean=mean,
        spread=spread,
        sample_scores=sample_scores,
        graded_outputs=len(runs),
        threshold=threshold,
    )


def grade_panel_with_jev(
    panel: Sequence[PanelResult],
    tests: Sequence[Mapping[str, Any]],
    gateway: Gateway,
    *,
    judge_model: str,
    run_id: str,
    grading_policy: OrderBiasPolicy | None = None,
) -> tuple[dict[str, GradeReport], list[dict[str, Any]]]:
    """Grade panel outputs using a compatible persisted order-bias policy.

    Noul remains a single direct ask. Choice and Score use one order only when
    a compatible artifact recommends ``single``; ``mean_pair`` averages the
    two semantically aligned asks. Missing or incompatible artifacts preserve
    the historical conservative ``legacy_min_pair`` behavior.
    """
    requests: list[dict[str, Any]] = []
    response_indices: dict[tuple[int, int], list[int]] = {}
    policy_evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for output_index, run in enumerate(panel):
        for test_index, test in enumerate(tests):
            state = {"prompt": run.prompt, "output": run.output, "test": dict(test)}
            kind = str(test.get("kind", "noul"))
            options = tuple(
                str(item)
                for item in (
                    test.get("options") if kind == "choice" else test.get("levels")
                )
                or ()
            )
            descriptions = test.get("option_descriptions")
            if not isinstance(descriptions, Mapping):
                descriptions = {}
            if kind == "noul":
                resolution = {
                    "policy": "noul_direct",
                    "reason": "outside_order_bias_experiment",
                    "snapshot": gateway.jev_model,
                }
            elif grading_policy is None:
                resolution = {
                    "policy": LEGACY_GRADING_POLICY,
                    "reason": "no_order_bias_artifact",
                    "snapshot": gateway.jev_model,
                }
            else:
                resolution = grading_policy.resolve(test, snapshot=gateway.jev_model)
            resolution = {
                **resolution,
                "primitive": kind,
                "test_id": str(test.get("id", f"test-{test_index + 1:03d}")),
                "question": str(test.get("question", "")),
            }
            selected_policy = str(resolution.get("policy", LEGACY_GRADING_POLICY))
            policy_evidence[(output_index, test_index)] = resolution
            request_indexes: list[int] = []
            orders = (
                (False,)
                if kind == "noul" or selected_policy == "single"
                else (False, True)
            )
            for second in orders:
                request_indexes.append(len(requests))
                requests.append(
                    {
                        "key": f"grade_{output_index}_{test_index}_{'second' if second else 'first'}",
                        "model": judge_model,
                        "type": kind,
                        "state": state,
                        "question": jev_questions.GRADING_NOUL_QUESTION
                        if kind == "noul"
                        else jev_questions.GRADING_OTHER_QUESTION,
                        **(
                            {
                                "criteria": {
                                    option: descriptions.get(option)
                                    for option in (
                                        reversed(options) if second else options
                                    )
                                }
                            }
                            if kind == "choice"
                            else {}
                        ),
                        **(
                            {"criteria": list(reversed(options) if second else options)}
                            if kind == "score"
                            else {}
                        ),
                    }
                )
            response_indices[(output_index, test_index)] = request_indexes
    responses: list[Any] = []
    for offset in range(0, len(requests), 40):
        responses.extend(
            gateway.decide_batch(
                requests[offset : offset + 40], role="judge", run_id=run_id
            )
        )
    if len(responses) != len(requests):
        raise ValueError("Jev returned an incomplete grading batch")
    evidence = [
        {
            "question": request,
            "answer": response,
            "grading_policy": policy_evidence[
                next(
                    pair
                    for pair, indexes in response_indices.items()
                    if request_index in indexes
                )
            ],
        }
        for request_index, (request, response) in enumerate(
            zip(requests, responses, strict=True)
        )
    ]
    scores: dict[tuple[str, str, int, int], float] = {}
    for output_index, run in enumerate(panel):
        test_scores = []
        for test_index in range(len(tests)):
            indexes = response_indices[(output_index, test_index)]
            test = tests[test_index]
            kind = str(test.get("kind", "noul"))
            expected = str(test.get("expected", "yes"))
            if kind == "noul":
                direct = _noul_probability(responses[indexes[0]])
                test_scores.append(
                    0.0
                    if direct is None
                    else 1.0 - direct
                    if expected.casefold() in {"no", "false"}
                    else direct
                )
            else:
                try:
                    first = parse_decision(responses[indexes[0]])
                    second = (
                        parse_decision(responses[indexes[1]])
                        if len(indexes) > 1
                        else None
                    )
                    if kind == "choice" and isinstance(first, ChoiceDecision):
                        masses = [first.probabilities.get(expected, 0.0)]
                        if isinstance(second, ChoiceDecision):
                            masses.append(second.probabilities.get(expected, 0.0))
                        selected_policy = str(
                            policy_evidence[(output_index, test_index)]["policy"]
                        )
                        test_scores.append(
                            _combine_order_masses(masses, selected_policy)
                        )
                    elif kind == "score" and isinstance(first, ScoreDecision):
                        levels = tuple(str(level) for level in test.get("levels", ()))
                        first_mass = _score_expected_mass(
                            first, levels, expected, reverse=False
                        )
                        masses = [first_mass]
                        if isinstance(second, ScoreDecision):
                            masses.append(
                                _score_expected_mass(
                                    second, levels, expected, reverse=True
                                )
                            )
                        selected_policy = str(
                            policy_evidence[(output_index, test_index)]["policy"]
                        )
                        test_scores.append(
                            _combine_order_masses(masses, selected_policy)
                        )
                    else:
                        test_scores.append(0.0)
                except ValueError:
                    test_scores.append(0.0)
        scores[(run.candidate_id, run.model, run.sample, run.seed)] = min(
            test_scores, default=0.0
        )
    grades = {
        candidate_id: grade_candidate(
            candidate_id,
            panel,
            lambda run: scores[(run.candidate_id, run.model, run.sample, run.seed)],
        )
        for candidate_id in dict.fromkeys(run.candidate_id for run in panel)
    }
    return grades, evidence


def _score_expected_mass(
    decision: ScoreDecision,
    levels: Sequence[str],
    expected: str,
    *,
    reverse: bool,
) -> float:
    if expected not in levels:
        return 0.0
    probabilities = decision.probabilities
    semantic_mass: dict[str, float] = {}
    for index in range(len(levels)):
        semantic_index = len(levels) - 1 - index if reverse else index
        semantic_mass[levels[semantic_index]] = probabilities.get(str(index), 0.0)
    expected_index = levels.index(expected)
    return sum(semantic_mass.get(level, 0.0) for level in levels[expected_index:])


def _combine_order_masses(masses: Sequence[float], policy: str) -> float:
    if not masses:
        return 0.0
    if policy == "single" or len(masses) == 1:
        return masses[0]
    if policy == "mean_pair":
        return fmean(masses)
    return min(masses)


__all__ = [
    "GradeReport",
    "grade_candidate",
    "grade_panel_with_jev",
    "metrics_from_scores",
]
