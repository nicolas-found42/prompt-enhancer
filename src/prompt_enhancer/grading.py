"""Grade panel outputs into the metrics used for robust candidate ranking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .gateway import Gateway
from .jev import (
    ChoiceDecision,
    JevResponseError,
    NoulDecision,
    ScoreDecision,
    parse_decision,
)


@dataclass(frozen=True)
class GradeRequest:
    """A single candidate/model/sample item for the judge seam."""

    candidate_id: str
    model: str
    sample: int
    seed: int
    output: str
    tests: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model,
            "sample": self.sample,
            "seed": self.seed,
            "output": self.output,
            "tests": list(self.tests),
        }


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


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping):
            if name in value:
                return value[name]
        elif hasattr(value, name):
            return getattr(value, name)
    return default


def _runs(runs: Any) -> list[Any]:
    if hasattr(runs, "results"):
        return list(runs.results)
    if isinstance(runs, Mapping):
        return list(runs.get("outputs", runs.get("results", ())))
    return list(runs)


def _candidate_id(candidate: Any, runs: Sequence[Any]) -> str:
    value = _field(candidate, "candidate_id", "id")
    if value is not None:
        return str(value)
    if runs:
        value = _field(runs[0], "candidate_id", "id")
        if value is not None:
            return str(value)
    return "candidate"


def _run_fields(run: Any) -> tuple[str, str, int, int, str]:
    candidate_id = str(_field(run, "candidate_id", "id", default="candidate"))
    model = str(_field(run, "model", "model_id", default="unknown"))
    sample = int(_field(run, "sample", "sample_index", default=0))
    seed = int(_field(run, "seed", default=0))
    output = str(_field(run, "output", "text", "completion", default=""))
    return candidate_id, model, sample, seed, output


def _noul_probability(answer: Any) -> float:
    """A Jev yes/no answer as a probability; unusable answers score zero."""
    try:
        decision = parse_decision(answer)
    except JevResponseError:
        return 0.0
    return decision.probability if isinstance(decision, NoulDecision) else 0.0


def metrics_from_scores(
    per_model_scores: Mapping[str, Sequence[float]],
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, float], dict[str, tuple[float, ...]], float, float, float, tuple[float, ...]]:
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
        rates[model] = fmean(1.0 if value >= threshold else 0.0 for value in values) if values else 0.0
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
    candidate: Any,
    panel_runs: Any,
    judge: Callable[[GradeRequest], float],
    *,
    tests: Sequence[Any] = (),
    threshold: float = 0.5,
) -> GradeReport:
    """Grade every panel output and return robust aggregate metrics.

    ``judge`` scores one :class:`GradeRequest` between 0 and 1.  No gateway
    calls happen here; ``grade_panel_with_jev`` supplies scores from Jev.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    runs = _runs(panel_runs)
    runs = [
        run
        for run in runs
        if _field(run, "candidate_id", "id", default=_candidate_id(candidate, runs))
        == _candidate_id(candidate, runs)
    ]
    candidate_id = _candidate_id(candidate, runs)
    per_model_scores: dict[str, list[float]] = {}
    for run in runs:
        _, model, sample, seed, output = _run_fields(run)
        request = GradeRequest(candidate_id, model, sample, seed, output, tuple(tests))
        score = float(judge(request))
        per_model_scores.setdefault(model, []).append(score)
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
    panel_runs: Any,
    tests: Sequence[Mapping[str, Any]],
    gateway: Gateway,
    *,
    judge_model: str,
    run_id: str,
) -> tuple[dict[str, GradeReport], list[dict[str, Any]]]:
    """Batch every output/test decision and its reversed consistency check."""
    panel = _runs(panel_runs)
    requests: list[dict[str, Any]] = []
    for output_index, run in enumerate(panel):
        for test_index, test in enumerate(tests):
            state = {"prompt": _field(run, "prompt", default=""), "output": _field(run, "output", default=""), "test": dict(test)}
            kind = str(test.get("kind", "noul"))
            options = tuple(str(item) for item in (test.get("options") if kind == "choice" else test.get("levels")) or ())
            for second in (False, True):
                question = "Is the answer to the success criterion in state.test yes?"
                if kind == "noul" and second:
                    question = "Is the answer to the success criterion in state.test no?"
                requests.append({
                    "key": f"grade_{output_index}_{test_index}_{'second' if second else 'first'}",
                    "model": judge_model,
                    "type": kind,
                    "state": state,
                    "question": question if kind == "noul" else "Answer the success criterion in state.test using the provided criteria.",
                    **({"criteria": {option: option for option in (reversed(options) if second else options)}} if kind == "choice" else {}),
                    **({"criteria": list(reversed(options) if second else options)} if kind == "score" else {}),
                })
    responses: list[Any] = []
    for offset in range(0, len(requests), 40):
        responses.extend(gateway.decide_batch(requests[offset : offset + 40], role="judge", run_id=run_id))
    if len(responses) != len(requests):
        raise ValueError("Jev returned an incomplete grading batch")
    evidence = [
        {"question": request, "answer": response}
        for request, response in zip(requests, responses, strict=True)
    ]
    scores: dict[tuple[str, str, int, int], float] = {}
    for output_index, run in enumerate(panel):
        test_scores = []
        for test_index in range(len(tests)):
            pair_index = 2 * (output_index * len(tests) + test_index)
            test = tests[test_index]
            kind = str(test.get("kind", "noul"))
            expected = str(test.get("expected", "yes"))
            if kind == "noul":
                direct = _noul_probability(responses[pair_index])
                reverse = _noul_probability(responses[pair_index + 1])
                test_scores.append(min(reverse, 1.0 - direct) if expected.casefold() in {"no", "false"} else min(direct, 1.0 - reverse))
            else:
                try:
                    first = parse_decision(responses[pair_index])
                    second = parse_decision(responses[pair_index + 1])
                    if kind == "choice" and isinstance(first, ChoiceDecision) and isinstance(second, ChoiceDecision) or kind == "score" and isinstance(first, ScoreDecision) and isinstance(second, ScoreDecision):
                        test_scores.append(min(first.probabilities.get(expected, 0.0), second.probabilities.get(expected, 0.0)))
                    else:
                        test_scores.append(0.0)
                except ValueError:
                    test_scores.append(0.0)
        scores[(str(_field(run, "candidate_id")), str(_field(run, "model")), int(_field(run, "sample")), int(_field(run, "seed")))] = min(test_scores, default=0.0)
    grades = {
        candidate_id: grade_candidate(
            {"candidate_id": candidate_id},
            panel_runs,
            lambda request: scores[(request.candidate_id, request.model, request.sample, request.seed)],
            tests=tests,
        )
        for candidate_id in dict.fromkeys(str(_field(run, "candidate_id")) for run in panel)
    }
    return grades, evidence


__all__ = [
    "GradeReport",
    "GradeRequest",
    "grade_candidate",
    "grade_panel_with_jev",
    "metrics_from_scores",
]
