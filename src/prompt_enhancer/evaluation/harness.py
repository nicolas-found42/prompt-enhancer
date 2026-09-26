"""A reproducible maintainer harness around the public optimizer seam."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from ..diagnosis import (
    DEFAULT_RUBRIC,
    HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
    HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
    SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
    TASK_TAXONOMY_PROTOCOL_VERSION,
    GapImpact,
)
from ..rewrite import WRITER_INSTRUCTION_VERSIONS
from .datasets import (
    Dataset,
    canonical_json,
    normalize_gap_type,
    replay_digest,
)

IMPROVEMENT_EPSILON = 1e-12
REPORT_SCHEMA_VERSION = 1
# Recordings without engine metadata predate it: they used the version 1 writer
# instruction and the 0.90 success-test faithfulness gate.
HISTORICAL_WRITER_INSTRUCTION_VERSION = 1
HISTORICAL_FAITHFULNESS_THRESHOLD = 0.9


class EvaluationError(RuntimeError):
    """Raised when the evaluation cannot be run safely."""


class Engine(Protocol):
    """The only engine surface the harness is allowed to call."""

    def optimize(self, prompt: str, options: dict[str, Any] | None = None) -> object:
        """Optimize one prompt."""

    def resume(self, run_id: str, answers: dict[str, Any]) -> object:
        """Continue a paused run with externally supplied answers."""


EngineFactory = Callable[[Path | None], Engine]


@dataclass(frozen=True, slots=True)
class HarnessOptions:
    """Configuration sent to the engine and included in the run identity."""

    tier: str = "standard"
    seed: int = 0
    clarification_allowed: bool = False
    model_overrides: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    fail_fast: bool = False

    def __post_init__(self) -> None:
        tier = self.tier.strip().lower()
        if tier not in {"fast", "standard", "deep"}:
            raise EvaluationError("tier must be fast, standard, or deep")
        if not isinstance(self.seed, int):
            raise EvaluationError("seed must be an integer")
        for name in ("model_overrides", "settings", "extra"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise EvaluationError(f"{name} must be an object")
            object.__setattr__(self, name, dict(value))

    def optimize_options(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "seed": self.seed,
            "clarification_allowed": self.clarification_allowed,
            "model_overrides": dict(self.model_overrides),
            "settings": dict(self.settings),
            **dict(self.extra),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "seed": self.seed,
            "clarification_allowed": self.clarification_allowed,
            "model_overrides": dict(self.model_overrides),
            "settings": dict(self.settings),
            "extra": dict(self.extra),
            "fail_fast": self.fail_fast,
        }


@dataclass(frozen=True, slots=True)
class GapMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProblemSentenceMetrics:
    status: str
    metric: str
    labeled_cases: int
    excluded_failed_cases: int
    true_positives: int | None
    false_positives: int | None
    false_negatives: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    false_flags: int | None
    reason: str | None = None
    per_kind: Mapping[str, GapMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metric": self.metric,
            "labeled_cases": self.labeled_cases,
            "excluded_failed_cases": self.excluded_failed_cases,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_flags": self.false_flags,
            "reason": self.reason,
            "per_kind": {
                name: metric.to_dict() for name, metric in sorted(self.per_kind.items())
            },
        }


@dataclass(frozen=True, slots=True)
class TaskClassificationMetrics:
    status: str
    metric: str
    total_cases: int
    labeled_cases: int
    excluded_failed_cases: int
    leaf_correct: int | None
    leaf_accuracy: float | None
    parent_correct: int | None
    parent_cases: int
    parent_accuracy: float | None
    parent_coverage: float | None
    fallback_count: int | None
    fallback_frequency: float | None
    mean_provider_requests: float | None
    mean_provider_request_delta_vs_legacy: float | None
    mean_classification_latency_ms: float | None
    latency_comparison_status: str
    latency_comparison_reason: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiagnosisSummary:
    labeled_cases: int
    predicted_gap_occurrences: int
    expected_gap_occurrences: int
    micro: GapMetrics
    per_gap: Mapping[str, GapMetrics]
    problem_sentences: ProblemSentenceMetrics
    task_classification: TaskClassificationMetrics
    excluded_failed_cases: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "labeled_cases": self.labeled_cases,
            "predicted_gap_occurrences": self.predicted_gap_occurrences,
            "expected_gap_occurrences": self.expected_gap_occurrences,
            "excluded_failed_cases": self.excluded_failed_cases,
            "micro": self.micro.to_dict(),
            "per_gap": {
                name: metric.to_dict() for name, metric in sorted(self.per_gap.items())
            },
            "problem_sentences": self.problem_sentences.to_dict(),
            "task_classification": self.task_classification.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ImprovementSummary:
    comparable_cases: int
    unavailable_cases: int
    improved: int
    unchanged: int
    regressed: int
    improvement_rate: float
    no_change_rate: float
    regression_rate: float
    mean_score_delta: float
    mean_improvement: float
    mean_regression_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RestructuringSummary:
    status: str
    total_cases: int
    selected_for_generation: int
    built_candidates: int
    wins: int
    selection_rate: float | None
    win_rate: float | None
    strong_checked: int
    strong_regressions: int
    strong_rejection_rate: float | None
    evidence_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CostSummary:
    currency: str
    total: float
    mean_per_case: float
    by_role: Mapping[str, float]
    unavailable_cases: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "total": self.total,
            "mean_per_case": self.mean_per_case,
            "by_role": dict(sorted(self.by_role.items())),
            "unavailable_cases": self.unavailable_cases,
        }


@dataclass(frozen=True, slots=True)
class LatencySummary:
    measurement: str
    unit: str
    total: float | None
    mean_per_case: float | None
    p50: float | None
    p95: float | None
    unavailable_cases: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    source: str
    status: str
    expected_gaps: tuple[str, ...]
    predicted_gaps: tuple[str, ...]
    original_kept: bool | None
    final_prompt: str | None
    original_score: float | None
    winner_score: float | None
    score_delta: float | None
    outcome: str
    cost: float | None
    cost_by_role: Mapping[str, float]
    latency_ms: float | None
    error: str | None = None
    labels_present: bool = False
    expected_problem_sentences: tuple[tuple[str, str], ...] = ()
    problem_sentence_labels_present: bool = False
    predicted_problem_sentences: tuple[tuple[str, str], ...] = ()
    expected_task_type: str | None = None
    task_type_labels_present: bool = False
    predicted_task_type: str | None = None
    task_type_fallback_reason: str | None = None
    task_taxonomy_provider_requests: int | None = None
    task_taxonomy_provider_request_delta_vs_legacy: int | None = None
    task_taxonomy_latency_ms: float | None = None
    diagnosis_request_evidence: Mapping[str, Any] = field(default_factory=dict)
    lossless_restructuring: Mapping[str, Any] = field(default_factory=dict)
    lossless_strong_check: Mapping[str, Any] = field(default_factory=dict)
    failure_attribution: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": self.source,
            "status": self.status,
            "expected_gaps": list(self.expected_gaps),
            "predicted_gaps": list(self.predicted_gaps),
            "original_kept": self.original_kept,
            "final_prompt": self.final_prompt,
            "original_score": self.original_score,
            "winner_score": self.winner_score,
            "score_delta": self.score_delta,
            "outcome": self.outcome,
            "cost": self.cost,
            "cost_by_role": dict(sorted(self.cost_by_role.items())),
            "latency_ms": self.latency_ms,
            "error": self.error,
            "labels_present": self.labels_present,
            "problem_sentence_labels_present": self.problem_sentence_labels_present,
            "expected_problem_sentences": [
                {"kind": kind, "sentence_id": sentence_id}
                for kind, sentence_id in self.expected_problem_sentences
            ],
            "predicted_problem_sentences": [
                {"kind": kind, "sentence_id": sentence_id}
                for kind, sentence_id in self.predicted_problem_sentences
            ],
            "expected_task_type": self.expected_task_type,
            "task_type_labels_present": self.task_type_labels_present,
            "predicted_task_type": self.predicted_task_type,
            "task_type_fallback_reason": self.task_type_fallback_reason,
            "task_taxonomy_provider_requests": self.task_taxonomy_provider_requests,
            "task_taxonomy_provider_request_delta_vs_legacy": self.task_taxonomy_provider_request_delta_vs_legacy,
            "task_taxonomy_latency_ms": self.task_taxonomy_latency_ms,
            "diagnosis_request_evidence": dict(self.diagnosis_request_evidence),
            "lossless_restructuring": dict(self.lossless_restructuring),
            "lossless_strong_check": dict(self.lossless_strong_check),
            "failure_attribution": dict(self.failure_attribution),
        }


@dataclass(frozen=True, slots=True)
class HarnessReport:
    schema_version: int
    run_identity: Mapping[str, Any]
    dataset: Mapping[str, Any]
    options: Mapping[str, Any]
    replay: Mapping[str, Any] | None
    cases: tuple[CaseEvaluation, ...]
    diagnosis: DiagnosisSummary
    improvement: ImprovementSummary
    restructuring: RestructuringSummary
    cost: CostSummary
    latency_ms: LatencySummary
    failure_attribution: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_identity": dict(self.run_identity),
            "dataset": dict(self.dataset),
            "options": dict(self.options),
            "replay": dict(self.replay) if self.replay is not None else None,
            "cases": [case.to_dict() for case in self.cases],
            "diagnosis": self.diagnosis.to_dict(),
            "improvement": self.improvement.to_dict(),
            "restructuring": self.restructuring.to_dict(),
            "cost": self.cost.to_dict(),
            "latency_ms": self.latency_ms.to_dict(),
            "failure_attribution": dict(self.failure_attribution),
        }

    def to_json(self, *, pretty: bool = False) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )


@dataclass(frozen=True, slots=True)
class _ReplayBundle:
    path: Path
    digest: str
    gateway_recordings: Mapping[str, Any] | str | Path
    jev_model: str | None = None
    decision_provenance: Mapping[str, Any] = field(default_factory=dict)
    case_latency_ms: Mapping[str, float] = field(default_factory=dict)
    case_costs: Mapping[str, tuple[float, Mapping[str, float]]] = field(
        default_factory=dict
    )
    rubric_thresholds: Mapping[str, float] | None = None
    writer_instruction_version: int = HISTORICAL_WRITER_INSTRUCTION_VERSION
    faithfulness_threshold: float = HISTORICAL_FAITHFULNESS_THRESHOLD
    checklist_keys: tuple[str, ...] | None = None
    checklist_impacts: Mapping[str, str] | None = None
    sentence_diagnosis_version: int = HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION
    task_taxonomy_version: int = HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION
    speculative_diagnosis: bool = False
    observe_sequential_diagnosis: bool = False
    diagnosis_request_byte_limit: int | None = None
    pricing_models: tuple[Mapping[str, Any], ...] = ()
    decision_policy_artifacts: tuple[Mapping[str, Any], ...] = ()
    decision_policy_version: str | None = None
    cascade_settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _CaseObservation:
    case_id: str
    source: str
    status: str
    expected_gaps: tuple[str, ...]
    expected_task_type: str | None = None
    task_type_labels_present: bool = False
    predicted_task_type: str | None = None
    task_type_fallback_reason: str | None = None
    task_taxonomy_provider_requests: int | None = None
    task_taxonomy_provider_request_delta_vs_legacy: int | None = None
    task_taxonomy_latency_ms: float | None = None
    diagnosis_request_evidence: Mapping[str, Any] = field(default_factory=dict)
    lossless_restructuring: Mapping[str, Any] = field(default_factory=dict)
    lossless_strong_check: Mapping[str, Any] = field(default_factory=dict)
    failure_attribution: Mapping[str, Any] = field(default_factory=dict)
    labels_present: bool = False
    predicted_gaps: tuple[str, ...] = ()
    expected_problem_sentences: tuple[tuple[str, str], ...] = ()
    problem_sentence_labels_present: bool = False
    predicted_problem_sentences: tuple[tuple[str, str], ...] = ()
    original_kept: bool | None = None
    final_prompt: str | None = None
    original_score: float | None = None
    winner_score: float | None = None
    cost: float | None = None
    cost_by_role: Mapping[str, float] = field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None

    def finish(self) -> CaseEvaluation:
        delta = (
            self.winner_score - self.original_score
            if self.original_score is not None and self.winner_score is not None
            else None
        )
        if delta is None:
            outcome = "unavailable"
        elif delta > IMPROVEMENT_EPSILON:
            outcome = "improved"
        elif delta < -IMPROVEMENT_EPSILON:
            outcome = "regressed"
        else:
            outcome = "unchanged"
        return CaseEvaluation(
            case_id=self.case_id,
            source=self.source,
            status=self.status,
            expected_gaps=self.expected_gaps,
            labels_present=self.labels_present,
            predicted_gaps=self.predicted_gaps,
            original_kept=self.original_kept,
            final_prompt=self.final_prompt,
            original_score=self.original_score,
            winner_score=self.winner_score,
            score_delta=delta,
            outcome=outcome,
            cost=self.cost,
            cost_by_role=dict(self.cost_by_role),
            latency_ms=self.latency_ms,
            error=self.error,
            expected_problem_sentences=self.expected_problem_sentences,
            problem_sentence_labels_present=self.problem_sentence_labels_present,
            predicted_problem_sentences=self.predicted_problem_sentences,
            expected_task_type=self.expected_task_type,
            task_type_labels_present=self.task_type_labels_present,
            predicted_task_type=self.predicted_task_type,
            task_type_fallback_reason=self.task_type_fallback_reason,
            task_taxonomy_provider_requests=self.task_taxonomy_provider_requests,
            task_taxonomy_provider_request_delta_vs_legacy=self.task_taxonomy_provider_request_delta_vs_legacy,
            task_taxonomy_latency_ms=self.task_taxonomy_latency_ms,
            diagnosis_request_evidence=dict(self.diagnosis_request_evidence),
            lossless_restructuring=dict(self.lossless_restructuring),
            lossless_strong_check=dict(self.lossless_strong_check),
            failure_attribution=dict(self.failure_attribution),
        )


class EvaluationHarness:
    """Run datasets through the same ``engine.optimize`` seam as the product.

    Pass a ready engine for live or fully injected tests.  Pass an engine
    factory to let the harness create a fresh engine for each run.  A replay
    path is given to that factory, which must guarantee that missing responses
    fail rather than fall back to a provider.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        engine_factory: EngineFactory | None = None,
        allow_snapshot_mismatch: bool = False,
    ) -> None:
        if engine is not None and engine_factory is not None:
            raise EvaluationError("provide either engine or engine_factory, not both")
        if engine is not None and not callable(getattr(engine, "optimize", None)):
            raise EvaluationError("engine must provide optimize(prompt, options)")
        self._engine = engine
        self._engine_factory = engine_factory
        self._allow_snapshot_mismatch = allow_snapshot_mismatch

    def run(
        self,
        dataset: Dataset,
        *,
        options: HarnessOptions | None = None,
        replay_path: str | Path | None = None,
    ) -> HarnessReport:
        options = options or HarnessOptions()
        replay = _load_replay(replay_path) if replay_path is not None else None
        engine = self._engine_for(replay)
        observations = tuple(
            self._evaluate_case(engine, case, options, replay) for case in dataset.cases
        )
        return _build_report(dataset, options, replay, observations)

    def _engine_for(self, replay: _ReplayBundle | None) -> Engine:
        if self._engine is not None:
            if replay is not None:
                raise EvaluationError(
                    "a replay path requires an engine_factory that binds ReplayGateway"
                )
            return self._engine
        factory = self._engine_factory
        if factory is not None:
            engine = factory(replay.path if replay is not None else None)
        elif replay is not None:
            engine = default_engine_factory(
                replay.path, allow_snapshot_mismatch=self._allow_snapshot_mismatch
            )
        else:
            raise EvaluationError(
                "a live engine must be injected explicitly; replay or inject a gateway"
            )
        if not callable(getattr(engine, "optimize", None)):
            raise EvaluationError("engine factory must return an object with optimize")
        return engine

    def _evaluate_case(
        self,
        engine: Engine,
        case: Any,
        options: HarnessOptions,
        replay: _ReplayBundle | None,
    ) -> _CaseObservation:
        observation = _CaseObservation(
            case_id=case.id,
            source=case.source,
            status="error",
            expected_gaps=case.expected_gaps,
            expected_task_type=case.expected_task_type,
            task_type_labels_present=case.task_type_labels_present,
            labels_present=case.labels_present,
            expected_problem_sentences=case.expected_problem_sentences,
            problem_sentence_labels_present=case.problem_sentence_labels_present,
        )
        started = perf_counter()
        try:
            result = _as_mapping(
                engine.optimize(case.prompt, options.optimize_options())
            )
            requested_clarification = _status(result) == "needs_input"
            resumed = False
            answers = case.metadata.get("clarification_answers")
            if requested_clarification and answers is not None:
                provenance = case.metadata.get("clarification_answer_provenance")
                if not isinstance(provenance, str) or provenance not in {
                    "human",
                    "source",
                }:
                    raise EvaluationError(
                        "clarification answers require human or source provenance"
                    )
                if (
                    not isinstance(answers, Mapping)
                    or not answers
                    or any(not isinstance(key, str) for key in answers)
                ):
                    raise EvaluationError(
                        "clarification answers must be a non-empty object"
                    )
                run_id = result.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    raise EvaluationError("paused engine result requires a run_id")
                result = _as_mapping(engine.resume(run_id, dict(answers)))
                resumed = True
            report = _as_mapping(result.get("report", {}))
            observation.failure_attribution = _attribution_case_evaluation(
                report, case.metadata.get("attribution_labels")
            )
            restructuring = report.get("lossless_restructuring")
            if isinstance(restructuring, Mapping):
                observation.lossless_restructuring = dict(restructuring)
            strong = _as_mapping(report.get("strong_check", {}))
            strong_candidates = strong.get("candidates")
            if isinstance(strong_candidates, list):
                for candidate in strong_candidates:
                    if (
                        isinstance(candidate, Mapping)
                        and candidate.get("strategy") == "restructure_lossless"
                    ):
                        observation.lossless_strong_check = dict(candidate)
                        break
            observation.predicted_gaps = _predicted_gaps(result, report)
            observation.predicted_problem_sentences = _predicted_problem_sentences(
                report
            )
            diagnosis = _as_mapping(report.get("diagnosis", {}))
            request_evidence = diagnosis.get("request_evidence")
            if isinstance(request_evidence, Mapping):
                observation.diagnosis_request_evidence = dict(request_evidence)
            observation.predicted_task_type = _optional_string(
                diagnosis.get("task_type")
            )
            observation.task_type_fallback_reason = _optional_string(
                diagnosis.get("task_type_fallback_reason")
            )
            taxonomy_evidence = _as_mapping(diagnosis.get("taxonomy_evidence", {}))
            request_count = taxonomy_evidence.get("provider_requests")
            if isinstance(request_count, int) and not isinstance(request_count, bool):
                observation.task_taxonomy_provider_requests = request_count
            request_delta = taxonomy_evidence.get("provider_request_delta_vs_legacy")
            if isinstance(request_delta, int) and not isinstance(request_delta, bool):
                observation.task_taxonomy_provider_request_delta_vs_legacy = (
                    request_delta
                )
            observation.task_taxonomy_latency_ms = _optional_number(
                taxonomy_evidence.get("classification_latency_ms")
            )
            if requested_clarification:
                observation.predicted_gaps = tuple(
                    sorted({*observation.predicted_gaps, "clarification_need"})
                )
            evaluation_gaps = case.metadata.get("evaluation_gaps")
            if isinstance(evaluation_gaps, list):
                scope = {normalize_gap_type(item) for item in evaluation_gaps}
                observation.expected_gaps = tuple(
                    gap for gap in observation.expected_gaps if gap in scope
                )
                observation.predicted_gaps = tuple(
                    gap for gap in observation.predicted_gaps if gap in scope
                )
            observation.original_kept = _optional_bool(result.get("original_kept"))
            observation.final_prompt = _optional_string(result.get("final_prompt"))
            observation.original_score, observation.winner_score = _scores(
                result, report, observation.original_kept
            )
            observation.cost, observation.cost_by_role = _cost(result)
            if replay is not None:
                if case.id in replay.case_costs:
                    observation.cost, observation.cost_by_role = replay.case_costs[
                        case.id
                    ]
                observation.latency_ms = replay.case_latency_ms.get(case.id)
            else:
                observation.latency_ms = (
                    None if resumed else _latency_ms(result, report)
                )
                if observation.latency_ms is None:
                    observation.latency_ms = (perf_counter() - started) * 1000
            observation.status = _status(result)
            if observation.status == "failed":
                observation.error = (
                    _optional_string(report.get("error")) or "engine returned failed"
                )
        except Exception as exc:
            if options.fail_fast:
                raise
            observation.error = f"{type(exc).__name__}: {exc}"
            if replay is not None:
                observation.latency_ms = replay.case_latency_ms.get(case.id)
        return observation


def default_engine_factory(
    replay_path: Path | None = None, *, allow_snapshot_mismatch: bool = False
) -> Engine:
    """Build the product optimizer with a strict replay gateway when requested."""

    from dataclasses import replace

    from ..catalog import ModelInfo, StaticModelCatalog
    from ..config import Settings
    from ..diagnosis import (
        DEFAULT_RUBRIC,
        HISTORICAL_CHECKLIST_EXCLUSIONS,
        HISTORICAL_CHECKLIST_IMPACTS,
        checklist_keys,
        restrict_checklist,
        with_impacts,
    )
    from ..optimizer import PromptOptimizer
    from .calibration import DecisionPolicy

    if replay_path is None:
        return PromptOptimizer()
    bundle = _load_replay(replay_path)
    from ..gateway import ReplayGateway

    recordings = cast(Mapping[Any, Any], bundle.gateway_recordings)
    rubric = (
        replace(DEFAULT_RUBRIC, gap_thresholds=bundle.rubric_thresholds)
        if bundle.rubric_thresholds is not None
        else DEFAULT_RUBRIC
    )
    recorded_keys = bundle.checklist_keys
    if recorded_keys is None:
        recorded_keys = tuple(
            key
            for key in checklist_keys(rubric)
            if key not in HISTORICAL_CHECKLIST_EXCLUSIONS
        )
    rubric = restrict_checklist(rubric, recorded_keys)
    rubric = with_impacts(
        rubric,
        bundle.checklist_impacts
        if bundle.checklist_impacts is not None
        else HISTORICAL_CHECKLIST_IMPACTS,
    )
    configured_pin = Settings.from_env().judge_model
    recorded_pin = bundle.jev_model or configured_pin
    try:
        priced_models = [
            ModelInfo(
                id=str(item["id"]),
                provider=str(item["provider"]),
                input_cost_per_token=item.get("input_cost_per_token"),
                output_cost_per_token=item.get("output_cost_per_token"),
            )
            for item in bundle.pricing_models
        ]
        catalog = StaticModelCatalog(
            [model for model in priced_models if model.provider == "go"],
            [model for model in priced_models if model.provider == "openrouter"],
        )
        replay_gateway = ReplayGateway(
            recordings,
            decision_provenance=bundle.decision_provenance,
            jev_model=recorded_pin,
            expected_snapshot=configured_pin,
            allow_snapshot_mismatch=allow_snapshot_mismatch,
            catalog=catalog,
        )
        replay_gateway.diagnosis_request_byte_limit = (
            bundle.diagnosis_request_byte_limit
        )
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    replay_settings = replace(
        Settings.from_env(),
        judge_model=recorded_pin,
        **bundle.cascade_settings,
    )
    return PromptOptimizer(
        gateway=replay_gateway,
        config=replay_settings,
        diagnosis_rubric=rubric,
        writer_instruction_version=bundle.writer_instruction_version,
        faithfulness_threshold=bundle.faithfulness_threshold,
        sentence_diagnosis_version=bundle.sentence_diagnosis_version,
        task_taxonomy_version=bundle.task_taxonomy_version,
        speculative_diagnosis=bundle.speculative_diagnosis,
        observe_sequential_diagnosis=bundle.observe_sequential_diagnosis,
        decision_policy=DecisionPolicy(
            artifacts=bundle.decision_policy_artifacts,
            policy_version=bundle.decision_policy_version or "issue-50-v1",
        )
        if bundle.decision_policy_artifacts
        else None,
    )


def _load_replay(path: str | Path) -> _ReplayBundle:
    replay_path = Path(path)
    digest = replay_digest(replay_path)
    try:
        raw = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"could not load replay {replay_path}: {exc}") from exc

    if isinstance(raw, Mapping) and "responses" in raw:
        recordings: Mapping[str, Any] | str | Path = raw["responses"]
        provenance = raw.get("decision_provenance", {})
        recorded_jev_model = raw.get("jev_model")
        if not isinstance(recordings, (Mapping, list)):
            raise EvaluationError("replay responses must be an object or list")
        raw_latencies = raw.get("case_latency_ms", {})
        raw_costs = raw.get("case_costs", {})
        raw_thresholds = raw.get("rubric_thresholds")
        writer_version = raw.get(
            "writer_instruction_version", HISTORICAL_WRITER_INSTRUCTION_VERSION
        )
        faithfulness = raw.get(
            "faithfulness_threshold", HISTORICAL_FAITHFULNESS_THRESHOLD
        )
        raw_checklist = raw.get("checklist_keys")
        raw_impacts = raw.get("checklist_impacts")
        sentence_diagnosis_version = raw.get(
            "sentence_diagnosis_version",
            HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
        )
        task_taxonomy_version = raw.get(
            "task_taxonomy_version",
            HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
        )
        speculative_diagnosis = raw.get("speculative_diagnosis", False)
        observe_sequential_diagnosis = raw.get("observe_sequential_diagnosis", False)
        diagnosis_request_byte_limit = raw.get("diagnosis_request_byte_limit")
        raw_pricing_models = raw.get("pricing_models", [])
        raw_policy_artifacts = raw.get("decision_policy_artifacts", [])
        raw_policy_version = raw.get("decision_policy_version")
        raw_cascade_settings = raw.get("cascade_settings", {})
    else:
        recordings = replay_path
        provenance = {}
        recorded_jev_model = None
        raw_latencies = {}
        raw_costs = {}
        raw_thresholds = None
        writer_version = HISTORICAL_WRITER_INSTRUCTION_VERSION
        faithfulness = HISTORICAL_FAITHFULNESS_THRESHOLD
        raw_checklist = None
        raw_impacts = None
        sentence_diagnosis_version = HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION
        task_taxonomy_version = HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION
        speculative_diagnosis = False
        observe_sequential_diagnosis = False
        diagnosis_request_byte_limit = None
        raw_pricing_models = []
        raw_policy_artifacts = []
        raw_policy_version = None
        raw_cascade_settings = {}
    if not isinstance(raw_cascade_settings, Mapping) or set(raw_cascade_settings) - {
        "grading_cascade_pair_cap",
        "grading_cascade_dollar_cap",
        "grading_confirmation_reservation_usd",
        "attribution_pair_cap",
        "attribution_dollar_cap",
    }:
        raise EvaluationError("replay cascade_settings contains unknown fields")
    pair_cap = raw_cascade_settings.get("grading_cascade_pair_cap")
    dollar_cap = raw_cascade_settings.get("grading_cascade_dollar_cap")
    reservation = raw_cascade_settings.get("grading_confirmation_reservation_usd")
    attribution_pair_cap = raw_cascade_settings.get("attribution_pair_cap")
    attribution_dollar_cap = raw_cascade_settings.get("attribution_dollar_cap")
    if pair_cap is not None and (
        isinstance(pair_cap, bool) or not isinstance(pair_cap, int) or pair_cap < 0
    ):
        raise EvaluationError("replay cascade pair cap must be non-negative")
    if attribution_pair_cap is not None and (
        isinstance(attribution_pair_cap, bool)
        or not isinstance(attribution_pair_cap, int)
        or attribution_pair_cap < 0
    ):
        raise EvaluationError("replay attribution pair cap must be non-negative")
    if any(
        value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        )
        for value in (dollar_cap, reservation, attribution_dollar_cap)
    ):
        raise EvaluationError("replay cascade dollar settings must be non-negative")
    if not isinstance(raw_pricing_models, list) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("id"), str)
        or not isinstance(item.get("provider"), str)
        or item.get("provider") not in {"go", "openrouter"}
        or any(
            rate is not None
            and (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(rate)
                or rate < 0
            )
            for rate in (
                item.get("input_cost_per_token"),
                item.get("output_cost_per_token"),
            )
        )
        for item in raw_pricing_models
    ):
        raise EvaluationError("replay pricing_models must contain valid model rates")
    if not isinstance(raw_policy_artifacts, list) or any(
        not isinstance(item, Mapping) for item in raw_policy_artifacts
    ):
        raise EvaluationError("replay decision_policy_artifacts must be objects")
    if raw_policy_artifacts and (
        not isinstance(raw_policy_version, str) or not raw_policy_version
    ):
        raise EvaluationError("replay decision_policy_version is required")
    if raw_impacts is not None and (
        not isinstance(raw_impacts, Mapping)
        or any(
            not isinstance(key, str)
            or value not in {impact.value for impact in GapImpact}
            for key, value in raw_impacts.items()
        )
    ):
        raise EvaluationError(
            "replay checklist_impacts must map question ids to known impacts"
        )
    if not isinstance(provenance, Mapping) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("answered_by"), str)
        or not item["answered_by"]
        for item in provenance.values()
    ):
        raise EvaluationError(
            "replay decision_provenance requires an answering snapshot for each decision"
        )
    if recorded_jev_model is not None and (
        not isinstance(recorded_jev_model, str) or not recorded_jev_model
    ):
        raise EvaluationError("replay jev_model must be a model ID")
    if raw_checklist is not None and (
        not isinstance(raw_checklist, list)
        or any(not isinstance(key, str) or not key for key in raw_checklist)
    ):
        raise EvaluationError("replay checklist_keys must be a list of question ids")
    if isinstance(
        sentence_diagnosis_version, bool
    ) or sentence_diagnosis_version not in {
        HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
        SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
    }:
        raise EvaluationError("replay sentence_diagnosis_version is not supported")
    if isinstance(task_taxonomy_version, bool) or task_taxonomy_version not in {
        HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
        TASK_TAXONOMY_PROTOCOL_VERSION,
    }:
        raise EvaluationError("replay task_taxonomy_version is not supported")
    if not isinstance(speculative_diagnosis, bool):
        raise EvaluationError("replay speculative_diagnosis must be a boolean")
    if not isinstance(observe_sequential_diagnosis, bool):
        raise EvaluationError("replay observe_sequential_diagnosis must be a boolean")
    if diagnosis_request_byte_limit is not None and (
        isinstance(diagnosis_request_byte_limit, bool)
        or not isinstance(diagnosis_request_byte_limit, int)
        or diagnosis_request_byte_limit < 0
    ):
        raise EvaluationError("replay diagnosis_request_byte_limit must be nonnegative")
    if (
        isinstance(writer_version, bool)
        or writer_version not in WRITER_INSTRUCTION_VERSIONS
    ):
        raise EvaluationError(
            "replay writer_instruction_version is not a known version"
        )
    if (
        isinstance(faithfulness, bool)
        or not isinstance(faithfulness, (int, float))
        or not 0 <= faithfulness <= 1
    ):
        raise EvaluationError("replay faithfulness_threshold must be a probability")
    if not isinstance(raw_latencies, Mapping):
        raise EvaluationError("replay case_latency_ms must be an object")
    latencies: dict[str, float] = {}
    for case_id, value in raw_latencies.items():
        if not isinstance(case_id, str) or not case_id:
            raise EvaluationError("replay latency case ids must be non-empty strings")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EvaluationError(f"replay latency for {case_id!r} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise EvaluationError(
                f"replay latency for {case_id!r} must be finite and non-negative"
            )
        latencies[case_id] = float(value)
    if not isinstance(raw_costs, Mapping):
        raise EvaluationError("replay case_costs must be an object")
    costs: dict[str, tuple[float, Mapping[str, float]]] = {}
    for case_id, value in raw_costs.items():
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(value, Mapping)
        ):
            raise EvaluationError(
                "replay case costs require non-empty case ids and objects"
            )
        total, roles = _cost({"cost": value})
        if total is None or total < 0 or any(amount < 0 for amount in roles.values()):
            raise EvaluationError(
                f"replay cost for {case_id!r} must be finite and non-negative"
            )
        costs[case_id] = (total, roles)
    thresholds: dict[str, float] | None = None
    if raw_thresholds is not None:
        if not isinstance(raw_thresholds, Mapping):
            raise EvaluationError("replay rubric_thresholds must be an object")
        thresholds = {}
        for question_id, value in raw_thresholds.items():
            if (
                not isinstance(question_id, str)
                or not question_id
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise EvaluationError(
                    "replay rubric thresholds require question ids and probabilities"
                )
            thresholds[question_id] = float(value)
    return _ReplayBundle(
        path=replay_path,
        digest=digest,
        gateway_recordings=recordings,
        jev_model=recorded_jev_model,
        decision_provenance=provenance,
        case_latency_ms=latencies,
        case_costs=costs,
        rubric_thresholds=thresholds,
        writer_instruction_version=writer_version,
        faithfulness_threshold=float(faithfulness),
        checklist_keys=tuple(raw_checklist) if raw_checklist is not None else None,
        checklist_impacts=dict(raw_impacts) if raw_impacts is not None else None,
        sentence_diagnosis_version=sentence_diagnosis_version,
        task_taxonomy_version=task_taxonomy_version,
        speculative_diagnosis=speculative_diagnosis,
        observe_sequential_diagnosis=observe_sequential_diagnosis,
        diagnosis_request_byte_limit=diagnosis_request_byte_limit,
        pricing_models=tuple(raw_pricing_models),
        decision_policy_artifacts=tuple(raw_policy_artifacts),
        decision_policy_version=raw_policy_version,
        cascade_settings=dict(raw_cascade_settings),
    )


def _build_report(
    dataset: Dataset,
    options: HarnessOptions,
    replay: _ReplayBundle | None,
    observations: Sequence[_CaseObservation],
) -> HarnessReport:
    cases = tuple(observation.finish() for observation in observations)
    diagnosis = _diagnosis_summary(cases)
    improvement = _improvement_summary(cases)
    restructuring = _restructuring_summary(cases)
    cost = _cost_summary(cases)
    latency = _latency_summary(cases, replayed=replay is not None)
    dataset_input = dataset.to_dict()
    options_input = options.to_dict()
    replay_input = (
        {"digest": replay.digest, "format": "strict"} if replay is not None else None
    )
    identity_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": dataset_input,
        "options": options_input,
        "replay": replay_input,
    }
    run_digest = hashlib.sha256(canonical_json(identity_payload).encode()).hexdigest()
    run_identity = {
        "run_digest": run_digest,
        "dataset_digest": dataset.digest,
        "deterministic_replay": replay is not None,
    }
    return HarnessReport(
        schema_version=REPORT_SCHEMA_VERSION,
        run_identity=run_identity,
        dataset=dataset_input,
        options=options_input,
        replay=replay_input,
        cases=cases,
        diagnosis=diagnosis,
        improvement=improvement,
        restructuring=restructuring,
        cost=cost,
        latency_ms=latency,
        failure_attribution=_attribution_harness_summary(cases),
    )


def _attribution_case_evaluation(
    report: Mapping[str, Any], raw_labels: object
) -> dict[str, Any]:
    rounds = report.get("history")
    entries = (
        [
            _as_mapping(_as_mapping(item).get("evidence", {})).get(
                "failure_attribution"
            )
            for item in rounds
        ]
        if isinstance(rounds, list)
        else []
    )
    entries = [item for item in entries if isinstance(item, Mapping)]
    if not entries and isinstance(report.get("failure_attribution"), Mapping):
        entries = [report["failure_attribution"]]
    pairs = [
        pair
        for entry in entries
        for pair in entry.get("pairs", [])
        if isinstance(pair, Mapping)
    ]
    labels = (
        [
            item
            for item in raw_labels
            if isinstance(item, Mapping)
            and item.get("provenance") in {"human", "source"}
        ]
        if isinstance(raw_labels, list)
        else []
    )
    label_keys = ("prompt_digest", "model", "sample", "test_id")
    matching = [
        (pair, label)
        for pair in pairs
        if pair.get("status") == "supported"
        for label in labels
        if all(pair.get(key) == label.get(key) for key in label_keys)
    ]
    correct = sum(
        pair.get("sentence_id") == label.get("sentence_id")
        and pair.get("kind") == label.get("kind")
        for pair, label in matching
    )
    return {
        "status": "available" if entries else "unavailable",
        "attributed_count": sum(pair.get("status") == "supported" for pair in pairs),
        "unresolved_count": sum(pair.get("status") == "unresolved" for pair in pairs),
        "skipped_count": sum(pair.get("status") == "skipped" for pair in pairs),
        "correctness": {
            "status": "available" if matching else "unavailable",
            "labeled_predictions": len(matching),
            "correct_predictions": correct,
            "accuracy": correct / len(matching) if matching else None,
            "provenance": sorted({str(label["provenance"]) for _, label in matching}),
        },
    }


def _attribution_harness_summary(cases: Sequence[CaseEvaluation]) -> dict[str, Any]:
    observations = [
        case.failure_attribution
        for case in cases
        if case.failure_attribution.get("status") == "available"
    ]
    correct = sum(
        int(_as_mapping(item.get("correctness", {})).get("correct_predictions", 0))
        for item in observations
    )
    labeled = sum(
        int(_as_mapping(item.get("correctness", {})).get("labeled_predictions", 0))
        for item in observations
    )
    return {
        "status": "available" if observations else "unavailable",
        "attributed_count": sum(
            int(item.get("attributed_count", 0)) for item in observations
        ),
        "unresolved_count": sum(
            int(item.get("unresolved_count", 0)) for item in observations
        ),
        "skipped_count": sum(
            int(item.get("skipped_count", 0)) for item in observations
        ),
        "correctness": {
            "status": "available" if labeled else "unavailable",
            "labeled_predictions": labeled,
            "correct_predictions": correct,
            "accuracy": correct / labeled if labeled else None,
            "provenance": sorted(
                {
                    provenance
                    for item in observations
                    for provenance in _as_mapping(item.get("correctness", {})).get(
                        "provenance", []
                    )
                }
            ),
        },
    }


def _restructuring_summary(cases: Sequence[CaseEvaluation]) -> RestructuringSummary:
    usable = [case for case in cases if case.status not in {"failed", "error"}]
    selected = sum(bool(case.lossless_restructuring) for case in usable)
    built = sum(
        case.lossless_restructuring.get("outcome") == "candidate_built"
        for case in usable
    )
    wins = sum(
        case.lossless_restructuring.get("selection_outcome") == "selected"
        for case in usable
    )
    checked = sum(bool(case.lossless_strong_check) for case in usable)
    regressions = sum(
        case.lossless_strong_check.get("regression") is True for case in usable
    )
    return RestructuringSummary(
        status="available" if selected else "unavailable",
        total_cases=len(usable),
        selected_for_generation=selected,
        built_candidates=built,
        wins=wins,
        selection_rate=selected / len(usable) if usable else None,
        win_rate=wins / built if built else None,
        strong_checked=checked,
        strong_regressions=regressions,
        strong_rejection_rate=regressions / checked if checked else None,
        evidence_kind="observed_run_outcomes" if selected else "unavailable",
    )


def _diagnosis_summary(cases: Sequence[CaseEvaluation]) -> DiagnosisSummary:
    counts: dict[str, Counter[str]] = {}
    labeled_cases = 0
    predicted_occurrences = 0
    expected_occurrences = 0
    excluded_failed_cases = 0
    for case in cases:
        if not case.labels_present:
            continue
        if case.status in {"failed", "error"}:
            excluded_failed_cases += 1
            continue
        labeled_cases += 1
        expected = set(case.expected_gaps)
        predicted = set(case.predicted_gaps)
        predicted_occurrences += len(predicted)
        expected_occurrences += len(expected)
        for gap_type in expected | predicted:
            counter = counts.setdefault(gap_type, Counter())
            if gap_type in expected and gap_type in predicted:
                counter["tp"] += 1
            elif gap_type in predicted:
                counter["fp"] += 1
            else:
                counter["fn"] += 1
    per_gap = {
        gap_type: _gap_metrics(counter["tp"], counter["fp"], counter["fn"])
        for gap_type, counter in counts.items()
    }
    micro = _gap_metrics(
        sum(metric.true_positives for metric in per_gap.values()),
        sum(metric.false_positives for metric in per_gap.values()),
        sum(metric.false_negatives for metric in per_gap.values()),
    )
    return DiagnosisSummary(
        labeled_cases=labeled_cases,
        predicted_gap_occurrences=predicted_occurrences,
        expected_gap_occurrences=expected_occurrences,
        excluded_failed_cases=excluded_failed_cases,
        micro=micro,
        per_gap=per_gap,
        problem_sentences=_problem_sentence_metrics(cases),
        task_classification=_task_classification_metrics(cases),
    )


def _task_classification_metrics(
    cases: Sequence[CaseEvaluation],
) -> TaskClassificationMetrics:
    task_labeled = [case for case in cases if case.task_type_labels_present]
    usable = [case for case in task_labeled if case.status not in {"failed", "error"}]
    excluded_failed = len(task_labeled) - len(usable)
    if not usable:
        reason = (
            "No evaluation cases include expected_task_type labels."
            if not task_labeled
            else "All task type labeled cases failed or errored."
        )
        return TaskClassificationMetrics(
            status="unavailable",
            metric="exact_task_type_and_parent_fallback",
            total_cases=len(cases),
            labeled_cases=0,
            excluded_failed_cases=excluded_failed,
            leaf_correct=None,
            leaf_accuracy=None,
            parent_correct=None,
            parent_cases=0,
            parent_accuracy=None,
            parent_coverage=None,
            fallback_count=None,
            fallback_frequency=None,
            mean_provider_requests=None,
            mean_provider_request_delta_vs_legacy=None,
            mean_classification_latency_ms=None,
            latency_comparison_status="unavailable",
            latency_comparison_reason="No matched historical classification-only timings are recorded.",
            reason=reason,
        )

    branch_keys = {branch.key for branch in DEFAULT_RUBRIC.task_branches}
    parent_by_task = {
        task_key: branch.key
        for branch in DEFAULT_RUBRIC.task_branches
        for task_key in branch.children
    }
    leaf_correct = sum(
        case.predicted_task_type == case.expected_task_type for case in usable
    )
    parent_cases = 0
    parent_correct = 0
    fallback_count = 0
    for case in usable:
        is_fallback = case.task_type_fallback_reason is not None
        fallback_count += int(is_fallback)
        if not is_fallback or case.predicted_task_type not in branch_keys:
            continue
        parent_cases += 1
        expected_parent = parent_by_task.get(case.expected_task_type or "")
        if case.expected_task_type in branch_keys:
            expected_parent = case.expected_task_type
        parent_correct += int(case.predicted_task_type == expected_parent)

    provider_requests = [
        case.task_taxonomy_provider_requests
        for case in usable
        if case.task_taxonomy_provider_requests is not None
    ]
    request_deltas = [
        case.task_taxonomy_provider_request_delta_vs_legacy
        for case in usable
        if case.task_taxonomy_provider_request_delta_vs_legacy is not None
    ]
    classification_latencies = [
        case.task_taxonomy_latency_ms
        for case in usable
        if case.task_taxonomy_latency_ms is not None
    ]
    return TaskClassificationMetrics(
        status="available",
        metric="exact_task_type_and_parent_fallback",
        total_cases=len(cases),
        labeled_cases=len(usable),
        excluded_failed_cases=excluded_failed,
        leaf_correct=leaf_correct,
        leaf_accuracy=leaf_correct / len(usable),
        parent_correct=parent_correct,
        parent_cases=parent_cases,
        parent_accuracy=(parent_correct / parent_cases if parent_cases else None),
        parent_coverage=parent_cases / len(usable),
        fallback_count=fallback_count,
        fallback_frequency=fallback_count / len(usable),
        mean_provider_requests=(
            math.fsum(provider_requests) / len(provider_requests)
            if provider_requests
            else None
        ),
        mean_provider_request_delta_vs_legacy=(
            math.fsum(request_deltas) / len(request_deltas) if request_deltas else None
        ),
        mean_classification_latency_ms=(
            math.fsum(classification_latencies) / len(classification_latencies)
            if classification_latencies
            else None
        ),
        latency_comparison_status="unavailable",
        latency_comparison_reason="Matched historical classification-only timings are not recorded.",
    )


def _problem_sentence_metrics(
    cases: Sequence[CaseEvaluation],
) -> ProblemSentenceMetrics:
    labeled_cases = 0
    excluded_failed_cases = 0
    counts: dict[str, Counter[str]] = {}
    for case in cases:
        if not case.problem_sentence_labels_present:
            continue
        if case.status in {"failed", "error"}:
            excluded_failed_cases += 1
            continue
        labeled_cases += 1
        expected = set(case.expected_problem_sentences)
        predicted = set(case.predicted_problem_sentences)
        for kind in {item[0] for item in expected | predicted}:
            expected_kind = {item for item in expected if item[0] == kind}
            predicted_kind = {item for item in predicted if item[0] == kind}
            counter = counts.setdefault(kind, Counter())
            counter["tp"] += len(expected_kind & predicted_kind)
            counter["fp"] += len(predicted_kind - expected_kind)
            counter["fn"] += len(expected_kind - predicted_kind)
    if labeled_cases == 0:
        return ProblemSentenceMetrics(
            status="unavailable",
            metric="problem_sentence_kind_and_id",
            labeled_cases=0,
            excluded_failed_cases=excluded_failed_cases,
            true_positives=None,
            false_positives=None,
            false_negatives=None,
            precision=None,
            recall=None,
            f1=None,
            false_flags=None,
            reason="No evaluation cases include expected_problem_sentences labels.",
        )
    per_kind = {
        kind: _gap_metrics(counter["tp"], counter["fp"], counter["fn"])
        for kind, counter in counts.items()
    }
    true_positives = sum(metric.true_positives for metric in per_kind.values())
    false_positives = sum(metric.false_positives for metric in per_kind.values())
    false_negatives = sum(metric.false_negatives for metric in per_kind.values())
    aggregate = _gap_metrics(true_positives, false_positives, false_negatives)
    return ProblemSentenceMetrics(
        status="available",
        metric="problem_sentence_kind_and_id",
        labeled_cases=labeled_cases,
        excluded_failed_cases=excluded_failed_cases,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=aggregate.precision,
        recall=aggregate.recall,
        f1=aggregate.f1,
        false_flags=false_positives,
        per_kind=per_kind,
    )


def _gap_metrics(tp: int, fp: int, fn: int) -> GapMetrics:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return GapMetrics(tp, fp, fn, precision, recall, f1)


def _improvement_summary(cases: Sequence[CaseEvaluation]) -> ImprovementSummary:
    deltas = [case.score_delta for case in cases if case.score_delta is not None]
    improved = sum(delta > IMPROVEMENT_EPSILON for delta in deltas)
    regressed = sum(delta < -IMPROVEMENT_EPSILON for delta in deltas)
    unchanged = len(deltas) - improved - regressed
    count = len(deltas)
    positive = [delta for delta in deltas if delta > IMPROVEMENT_EPSILON]
    negative = [-delta for delta in deltas if delta < -IMPROVEMENT_EPSILON]
    return ImprovementSummary(
        comparable_cases=count,
        unavailable_cases=len(cases) - count,
        improved=improved,
        unchanged=unchanged,
        regressed=regressed,
        improvement_rate=improved / count if count else 0.0,
        no_change_rate=unchanged / count if count else 0.0,
        regression_rate=regressed / count if count else 0.0,
        mean_score_delta=math.fsum(deltas) / count if count else 0.0,
        mean_improvement=math.fsum(positive) / improved if improved else 0.0,
        mean_regression_magnitude=math.fsum(negative) / regressed if regressed else 0.0,
    )


def _cost_summary(cases: Sequence[CaseEvaluation]) -> CostSummary:
    values = [case.cost for case in cases if case.cost is not None]
    role_totals: dict[str, float] = {}
    for case in cases:
        for role, value in case.cost_by_role.items():
            role_totals[role] = role_totals.get(role, 0.0) + value
    return CostSummary(
        currency="USD",
        total=math.fsum(values),
        mean_per_case=math.fsum(values) / len(cases) if cases else 0.0,
        by_role=role_totals,
        unavailable_cases=len(cases) - len(values),
    )


def _latency_summary(
    cases: Sequence[CaseEvaluation], *, replayed: bool
) -> LatencySummary:
    values = sorted(case.latency_ms for case in cases if case.latency_ms is not None)
    if replayed:
        measurement = "recorded_case_latency" if values else "unavailable_in_replay"
    else:
        measurement = "engine_or_wall_clock"
    total = math.fsum(values) if values else None
    return LatencySummary(
        measurement=measurement,
        unit="milliseconds",
        total=total,
        mean_per_case=total / len(values) if total is not None else None,
        p50=_nearest_rank(values, 0.50),
        p95=_nearest_rank(values, 0.95),
        unavailable_cases=len(cases) - len(values),
    )


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        converted = asdict(cast(Any, value))
        return converted
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        converted = model_dump()
        if isinstance(converted, Mapping):
            return converted
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return attributes
    raise EvaluationError(
        f"engine.optimize returned unsupported {type(value).__name__}; expected mapping"
    )


def _predicted_gaps(
    result: Mapping[str, Any], report: Mapping[str, Any]
) -> tuple[str, ...]:
    diagnosis = report.get("diagnosis", result.get("diagnosis", {}))
    if isinstance(diagnosis, Mapping):
        value = diagnosis.get(
            "gaps",
            diagnosis.get(
                "confirmed_gaps",
                diagnosis.get("missing_pieces", result.get("gaps", [])),
            ),
        )
    else:
        value = diagnosis
    if value is None:
        value = result.get("confirmed_gaps", [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    gaps: set[str] = set()
    for item in value:
        if isinstance(item, str) and item.strip():
            gaps.add(normalize_gap_type(item))
        elif isinstance(item, Mapping):
            gap_type = _mapping_gap_type(item)
            if gap_type:
                gaps.add(gap_type)
    return tuple(sorted(gaps))


def _predicted_problem_sentences(
    report: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    diagnosis = report.get("diagnosis", {})
    if not isinstance(diagnosis, Mapping):
        return ()
    problems = diagnosis.get("problem_sentences", ())
    if not isinstance(problems, Sequence) or isinstance(problems, (str, bytes)):
        return ()
    pairs: set[tuple[str, str]] = set()
    for item in problems:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("kind")
        sentence_id = item.get("sentence_id")
        sentence = item.get("sentence")
        if sentence_id is None and isinstance(sentence, Mapping):
            sentence_id = sentence.get("id")
        if (
            isinstance(kind, str)
            and kind.strip()
            and isinstance(sentence_id, str)
            and sentence_id.strip()
        ):
            pairs.add((kind.strip().lower(), sentence_id.strip()))
    return tuple(sorted(pairs))


def _mapping_gap_type(value: Mapping[str, Any]) -> str | None:
    for key in ("gap_type", "type", "kind", "name", "key", "id"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return normalize_gap_type(candidate)
    for key in ("checklist_item", "gap"):
        nested = value.get(key)
        if isinstance(nested, str) and nested.strip():
            return normalize_gap_type(nested)
        if isinstance(nested, Mapping):
            nested_type = _mapping_gap_type(nested)
            if nested_type:
                return nested_type
    return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _scores(
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    original_kept: bool | None,
) -> tuple[float | None, float | None]:
    ranking: Mapping[str, Any] = {}
    for candidate in (
        report.get("selection_evidence"),
        report.get("ranking"),
        report.get("selection"),
    ):
        if isinstance(candidate, Mapping):
            ranking = candidate
            break
    original = _score(ranking.get("original_score", result.get("original_score")))
    winner = _score(
        ranking.get(
            "winner_score",
            ranking.get("selected_score", result.get("winner_score")),
        )
    )
    if original_kept is True and original is not None and winner is None:
        winner = original
    return original, winner


def _score(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, Mapping):
        for key in (
            "worst_model_pass_rate",
            "worst_model_score",
            "mean_pass_rate",
            "score",
        ):
            candidate = value.get(key)
            if (
                isinstance(candidate, (int, float))
                and not isinstance(candidate, bool)
                and math.isfinite(float(candidate))
            ):
                return float(candidate)
        metrics = value.get("metrics")
        if isinstance(metrics, Mapping):
            return _score(metrics)
    return None


def _cost(result: Mapping[str, Any]) -> tuple[float | None, Mapping[str, float]]:
    value = result.get("cost", result.get("cost_usd"))
    if isinstance(value, bool):
        return None, {}
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value), {}
    if not isinstance(value, Mapping):
        return None, {}
    total = None
    for key in ("total", "total_usd", "amount", "amount_usd"):
        candidate = value.get(key)
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and math.isfinite(float(candidate))
        ):
            total = float(candidate)
            break
    roles: dict[str, float] = {}
    raw_roles = value.get("cost_by_role", value.get("by_role", value.get("roles", {})))
    if isinstance(raw_roles, Mapping):
        for role, amount in raw_roles.items():
            if (
                isinstance(role, str)
                and isinstance(amount, (int, float))
                and not isinstance(amount, bool)
                and math.isfinite(float(amount))
            ):
                roles[role] = float(amount)
    return total, roles


def _latency_ms(result: Mapping[str, Any], report: Mapping[str, Any]) -> float | None:
    candidates = (result.get("timing"), result.get("latency_ms"), report.get("timing"))
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
            return float(candidate)
        if isinstance(candidate, Mapping):
            for key in ("total_ms", "latency_ms", "duration_ms", "elapsed_ms"):
                amount = candidate.get(key)
                if (
                    isinstance(amount, (int, float))
                    and not isinstance(amount, bool)
                    and math.isfinite(float(amount))
                ):
                    return float(amount)
    return None


def _status(result: Mapping[str, Any]) -> str:
    explicit = result.get("status")
    if isinstance(explicit, str) and explicit:
        return explicit
    if result.get("questions") is not None or result.get("needs_input") is True:
        return "needs_input"
    if "final_prompt" in result:
        return "completed"
    return "unknown"


def compare_reports(before: HarnessReport, after: HarnessReport) -> dict[str, Any]:
    """Compare two reports without assuming they share dataset case order."""

    if isinstance(before, Mapping):
        before = _report_from_dict(before)
    if isinstance(after, Mapping):
        after = _report_from_dict(after)
    before_cases = {case.case_id: case for case in before.cases}
    after_cases = {case.case_id: case for case in after.cases}
    common = sorted(before_cases.keys() & after_cases.keys())

    def delta(path: Sequence[str]) -> float:
        return float(_nested(after.to_dict(), path) - _nested(before.to_dict(), path))

    diagnosis_deltas: dict[str, Any] = {}
    for gap_type in sorted(
        before.diagnosis.per_gap.keys() | after.diagnosis.per_gap.keys()
    ):
        diagnosis_deltas[gap_type] = {
            metric: delta(("diagnosis", "per_gap", gap_type, metric))
            for metric in ("precision", "recall", "f1")
        }
    return {
        "before_run_digest": before.run_identity["run_digest"],
        "after_run_digest": after.run_identity["run_digest"],
        "common_cases": len(common),
        "only_before_cases": sorted(before_cases.keys() - after_cases.keys()),
        "only_after_cases": sorted(after_cases.keys() - before_cases.keys()),
        "diagnosis": diagnosis_deltas,
        "task_classification": {
            field_name: _optional_numeric_delta(
                getattr(before.diagnosis.task_classification, field_name),
                getattr(after.diagnosis.task_classification, field_name),
            )
            for field_name in (
                "leaf_accuracy",
                "parent_accuracy",
                "parent_coverage",
                "fallback_frequency",
                "mean_provider_requests",
                "mean_provider_request_delta_vs_legacy",
                "mean_classification_latency_ms",
            )
        },
        "improvement": {
            "improvement_rate": delta(("improvement", "improvement_rate")),
            "no_change_rate": delta(("improvement", "no_change_rate")),
            "regression_rate": delta(("improvement", "regression_rate")),
            "mean_score_delta": delta(("improvement", "mean_score_delta")),
        },
        "cost": {
            "total": delta(("cost", "total")),
            "mean_per_case": delta(("cost", "mean_per_case")),
        },
        "latency_ms": {
            "mean_per_case": _optional_numeric_delta(
                before.latency_ms.mean_per_case, after.latency_ms.mean_per_case
            ),
            "p95": _optional_numeric_delta(before.latency_ms.p95, after.latency_ms.p95),
        },
    }


def _optional_numeric_delta(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else after - before


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> float:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping):
            raise EvaluationError(f"report path {'.'.join(path)} is unavailable")
        current = current[key]
    if not isinstance(current, (int, float)):
        raise EvaluationError(f"report path {'.'.join(path)} is not numeric")
    return float(current)


def _report_from_dict(value: Mapping[str, Any]) -> HarnessReport:
    """Coerce a JSON report for compare_reports callers.

    Full reconstruction is intentionally avoided: comparison only needs the
    stable aggregate and case fields and should remain tolerant of reports
    produced by a newer harness schema.
    """

    cases = tuple(
        CaseEvaluation(
            case_id=str(item["case_id"]),
            source=str(item.get("source", "real")),
            status=str(item.get("status", "unknown")),
            expected_gaps=tuple(item.get("expected_gaps", [])),
            labels_present=bool(
                item.get("labels_present", bool(item.get("expected_gaps")))
            ),
            predicted_gaps=tuple(item.get("predicted_gaps", [])),
            original_kept=item.get("original_kept"),
            final_prompt=item.get("final_prompt"),
            original_score=item.get("original_score"),
            winner_score=item.get("winner_score"),
            score_delta=item.get("score_delta"),
            outcome=str(item.get("outcome", "unavailable")),
            cost=item.get("cost"),
            cost_by_role=item.get("cost_by_role", {}),
            latency_ms=item.get("latency_ms"),
            error=item.get("error"),
            expected_problem_sentences=tuple(
                (str(label["kind"]), str(label["sentence_id"]))
                for label in item.get("expected_problem_sentences", [])
                if isinstance(label, Mapping)
                and label.get("kind") is not None
                and label.get("sentence_id") is not None
            ),
            problem_sentence_labels_present=bool(
                item.get("problem_sentence_labels_present", False)
            ),
            predicted_problem_sentences=tuple(
                (str(label["kind"]), str(label["sentence_id"]))
                for label in item.get("predicted_problem_sentences", [])
                if isinstance(label, Mapping)
                and label.get("kind") is not None
                and label.get("sentence_id") is not None
            ),
            expected_task_type=(
                str(item["expected_task_type"])
                if item.get("expected_task_type") is not None
                else None
            ),
            task_type_labels_present=bool(item.get("task_type_labels_present", False)),
            predicted_task_type=(
                str(item["predicted_task_type"])
                if item.get("predicted_task_type") is not None
                else None
            ),
            task_type_fallback_reason=(
                str(item["task_type_fallback_reason"])
                if item.get("task_type_fallback_reason") is not None
                else None
            ),
            task_taxonomy_provider_requests=item.get("task_taxonomy_provider_requests"),
            task_taxonomy_provider_request_delta_vs_legacy=item.get(
                "task_taxonomy_provider_request_delta_vs_legacy"
            ),
            task_taxonomy_latency_ms=item.get("task_taxonomy_latency_ms"),
            diagnosis_request_evidence=item.get("diagnosis_request_evidence", {}),
            lossless_restructuring=item.get("lossless_restructuring", {}),
            lossless_strong_check=item.get("lossless_strong_check", {}),
            failure_attribution=item.get("failure_attribution", {}),
        )
        for item in value.get("cases", [])
        if isinstance(item, Mapping)
    )
    gap_metrics = {
        name: _gap_metrics(
            int(metric.get("true_positives", 0)),
            int(metric.get("false_positives", 0)),
            int(metric.get("false_negatives", 0)),
        )
        for name, metric in value.get("diagnosis", {}).get("per_gap", {}).items()
    }
    micro_data = value.get("diagnosis", {}).get("micro", {})
    micro = _gap_metrics(
        int(micro_data.get("true_positives", 0)),
        int(micro_data.get("false_positives", 0)),
        int(micro_data.get("false_negatives", 0)),
    )
    diagnosis_data = value.get("diagnosis", {})
    improvement_data = value.get("improvement", {})
    cost_data = value.get("cost", {})
    latency_data = value.get("latency_ms", {})
    return HarnessReport(
        schema_version=int(value.get("schema_version", 1)),
        run_identity=value.get("run_identity", {}),
        dataset=value.get("dataset", {}),
        options=value.get("options", {}),
        replay=value.get("replay"),
        cases=cases,
        diagnosis=DiagnosisSummary(
            labeled_cases=int(diagnosis_data.get("labeled_cases", 0)),
            predicted_gap_occurrences=int(
                diagnosis_data.get("predicted_gap_occurrences", 0)
            ),
            expected_gap_occurrences=int(
                diagnosis_data.get("expected_gap_occurrences", 0)
            ),
            excluded_failed_cases=int(diagnosis_data.get("excluded_failed_cases", 0)),
            micro=micro,
            per_gap=gap_metrics,
            problem_sentences=_problem_sentence_metrics(cases),
            task_classification=_task_classification_metrics(cases),
        ),
        improvement=ImprovementSummary(
            comparable_cases=int(improvement_data.get("comparable_cases", 0)),
            unavailable_cases=int(improvement_data.get("unavailable_cases", 0)),
            improved=int(improvement_data.get("improved", 0)),
            unchanged=int(improvement_data.get("unchanged", 0)),
            regressed=int(improvement_data.get("regressed", 0)),
            improvement_rate=float(improvement_data.get("improvement_rate", 0.0)),
            no_change_rate=float(improvement_data.get("no_change_rate", 0.0)),
            regression_rate=float(improvement_data.get("regression_rate", 0.0)),
            mean_score_delta=float(improvement_data.get("mean_score_delta", 0.0)),
            mean_improvement=float(improvement_data.get("mean_improvement", 0.0)),
            mean_regression_magnitude=float(
                improvement_data.get("mean_regression_magnitude", 0.0)
            ),
        ),
        restructuring=_restructuring_summary(cases),
        cost=CostSummary(
            currency=str(cost_data.get("currency", "USD")),
            total=float(cost_data.get("total", 0.0)),
            mean_per_case=float(cost_data.get("mean_per_case", 0.0)),
            by_role=cost_data.get("by_role", {}),
            unavailable_cases=int(cost_data.get("unavailable_cases", 0)),
        ),
        latency_ms=LatencySummary(
            measurement=str(latency_data.get("measurement", "unknown")),
            unit=str(latency_data.get("unit", "milliseconds")),
            total=latency_data.get("total"),
            mean_per_case=latency_data.get("mean_per_case"),
            p50=latency_data.get("p50"),
            p95=latency_data.get("p95"),
            unavailable_cases=int(latency_data.get("unavailable_cases", 0)),
        ),
        failure_attribution=value.get("failure_attribution", {}),
    )


__all__ = [
    "CaseEvaluation",
    "CostSummary",
    "DiagnosisSummary",
    "Engine",
    "EngineFactory",
    "EvaluationError",
    "EvaluationHarness",
    "GapMetrics",
    "HarnessOptions",
    "HarnessReport",
    "ImprovementSummary",
    "RestructuringSummary",
    "LatencySummary",
    "compare_reports",
    "default_engine_factory",
]
