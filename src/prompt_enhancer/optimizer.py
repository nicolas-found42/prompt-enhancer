"""Public local optimization engine.

The facade is deliberately dependency-injected with a Gateway: production uses
HttpGateway; tests and local demos use ScriptedGateway or ReplayGateway.
The public result always includes the original prompt, evidence, cost, timing,
and a durable run identifier.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from . import jev_questions
from .candidate_evaluation import round_judgment_provenance, summarize_capabilities
from .catalog import LiveModelCatalog
from .clarification import (
    ClarificationService,
    InMemoryClarificationRepository,
    RunNotPausedError,
    SQLiteClarificationRepository,
    UnknownRunError,
)
from .clarifier import Clarifier
from .config import Settings
from .criterion_reading import (
    CRITERION_READING_MIN_VERSION,
    recording_metadata,
)
from .diagnosis import (
    DEFAULT_RUBRIC,
    HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
    HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
    PREVIOUS_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
    SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
    TASK_TAXONOMY_PROTOCOL_VERSION,
    ConfirmedGap,
    Diagnoser,
    DiagnosisReport,
    DiagnosisRubric,
    GapImpact,
    split_sentences,
)
from .failures import RunCancelled, describe_failure
from .feedback_labels import calibrate_score_floors as calibrate_feedback_floors
from .gateway import (
    Gateway,
    GatewayConfig,
    HttpGateway,
    HttpTransport,
    ScriptedGateway,
)
from .history import RunHistory
from .improve import failure_impossible
from .jev import NoulDecision, parse_decision
from .models import (
    CostBreakdown,
    OptimizeResult,
    new_run_id,
    utc_now,
)
from .outcomes import apply_outcome_fields
from .repeat import (
    RepeatCoordinator,
    RoundEvidence,
    RoundRequest,
    RoundRunner,
    _history_from_run,
)
from .rewrite import (
    CURRENT_WRITER_INSTRUCTION_VERSION,
    WRITER_INSTRUCTION_VERSIONS,
)
from .rounds import RoundOutcome, RoundPlan, prompt_diff, run_round
from .rubric_revisions import SQLiteRubricStore
from .run_control import (
    PAUSED_REPORT_STATUS,
    RESUME_CONTEXT_KEY,
    BudgetPaused,
    RoundTracker,
    RunControl,
    RunControlState,
    as_optimize_result,
    build_cancelled_result,
    build_paused_result,
    build_stopped_result,
    resume_context,
    take_resume_context,
)
from .settings import ModelDefaults, SettingsStore
from .store import RunStore
from .strategies import RouteResult, run_route
from .styles import parse_improvement_style
from .success_tests import (
    DEFAULT_FAITHFULNESS_THRESHOLD,
    SuccessTestScreenCache,
)
from .understand import UnderstandResult, run_understand
from .writer_replies import WRITER_REPLY_RECOVERY_MIN_VERSION

RUN_OPTION_KEYS = frozenset(
    {
        "clarification_allowed",
        "improvement_style",
        "model_overrides",
        "prior_round_failures",
        "seed",
        "spend_limit_usd",
        "time_limit_s",
    }
)
MODEL_OVERRIDE_KEYS = frozenset(
    {
        "judge",
        "judge_model",
        "strong",
        "strong_check_model",
        "weak",
        "weak_models",
        "writer",
        "writer_model",
    }
)

if TYPE_CHECKING:
    from .evaluation.calibration import CalibrationArtifact, DecisionPolicy
    from .evaluation.order_bias import OrderBiasPolicy
    from .evaluation.recording import RecordingGateway


class RunNotFoundError(KeyError):
    """Raised when a caller resumes or edits an unknown run."""


ProgressCallback = Callable[[str, Mapping[str, Any]], None]
"""Called with a stage name at each stage boundary; may raise RunCancelled."""

STAGES = (
    "diagnosing",
    "clarifying",
    "writing_tests",
    "choosing_strategy",
    "writing_candidates",
    "running_weak_models",
    "grading",
    "checking_fidelity",
    "strong_check",
    "evaluating_candidates",
    "accepting_candidates",
)


@dataclass(frozen=True, slots=True)
class _RunContext:
    prompt: str
    run_id: str
    diagnosis: Mapping[str, Any]
    assumptions: Any
    settings: Settings
    seed: int
    improvement_style: str = "auto"
    understand: UnderstandResult | None = None
    route: RouteResult | None = None


class PromptOptimizer:
    """Single public engine entry point for optimization and run lifecycle."""

    def __init__(
        self,
        gateway: Gateway | None = None,
        store: RunStore | None = None,
        config: Settings | None = None,
        rubric_store: SQLiteRubricStore | None = None,
        diagnosis_rubric: DiagnosisRubric = DEFAULT_RUBRIC,
        writer_instruction_version: int = CURRENT_WRITER_INSTRUCTION_VERSION,
        faithfulness_threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
        decision_policy: DecisionPolicy | Mapping[str, Any] | str | Path | None = None,
        calibration: CalibrationArtifact | Mapping[str, Any] | str | Path | None = None,
        grading_policy: OrderBiasPolicy | Mapping[str, Any] | str | Path | None = None,
        grading_policy_from_env: bool = True,
        sentence_diagnosis_version: int = SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
        task_taxonomy_version: int = TASK_TAXONOMY_PROTOCOL_VERSION,
        speculative_diagnosis: bool = True,
        observe_sequential_diagnosis: bool = False,
    ) -> None:
        if writer_instruction_version not in WRITER_INSTRUCTION_VERSIONS:
            raise ValueError("unknown candidate writer instruction version")
        if not 0 <= faithfulness_threshold <= 1:
            raise ValueError("faithfulness threshold must be a probability")
        if sentence_diagnosis_version not in {
            HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
            PREVIOUS_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
            SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
        }:
            raise ValueError("unsupported sentence diagnosis protocol version")
        if task_taxonomy_version not in {
            HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
            TASK_TAXONOMY_PROTOCOL_VERSION,
        }:
            raise ValueError("unsupported task taxonomy protocol version")
        self.store = store or RunStore()
        self.config = config or Settings.from_env()
        self.diagnosis_rubric = diagnosis_rubric
        self.writer_instruction_version = writer_instruction_version
        self.faithfulness_threshold = faithfulness_threshold
        self.success_test_screen_cache = SuccessTestScreenCache()
        self.sentence_diagnosis_version = sentence_diagnosis_version
        self.task_taxonomy_version = task_taxonomy_version
        self.speculative_diagnosis = speculative_diagnosis
        self.observe_sequential_diagnosis = observe_sequential_diagnosis
        from .evaluation.calibration import (
            DEFAULT_POLICY_VERSION,
            CalibrationArtifact,
            CalibrationError,
            DecisionPolicy,
        )

        selected_policy = (
            decision_policy if decision_policy is not None else calibration
        )
        if selected_policy is None or isinstance(selected_policy, DecisionPolicy):
            self.decision_policy = selected_policy
        else:
            artifact = (
                CalibrationArtifact.load(selected_policy)
                if isinstance(selected_policy, (str, Path))
                else CalibrationArtifact.from_dict(selected_policy)
                if isinstance(selected_policy, Mapping)
                else selected_policy
            )
            verdict_policy = artifact.metadata.get("verdict_policy")
            policy_version = (
                verdict_policy.get("policy_version", DEFAULT_POLICY_VERSION)
                if isinstance(verdict_policy, Mapping)
                else DEFAULT_POLICY_VERSION
            )
            if not isinstance(policy_version, str) or not policy_version.strip():
                raise CalibrationError("artifact policy_version must be non-empty")
            self.decision_policy = DecisionPolicy.from_artifact(
                artifact, policy_version=policy_version
            )
        self.settings_store = (
            SettingsStore(
                Path(self.store.path).with_suffix(".settings.json"),
                defaults=ModelDefaults(
                    writer=self.config.writer_model,
                    strong=self.config.strong_check_model,
                    weak=self.config.weak_models,
                ),
            )
            if self.store.path != ":memory:"
            else None
        )
        if self.settings_store is not None:
            saved_settings = self.settings_store.load()
            defaults = saved_settings.defaults
            self.config.writer_model = defaults.writer
            self.config.strong_check_model = defaults.strong
            self.config.weak_models = defaults.weak
            for dimension, floor in (saved_settings.score_floors or {}).items():
                if dimension in self.config.score_floors:
                    try:
                        numeric_floor = float(floor)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= numeric_floor <= 1.0:
                        setattr(self.config, f"score_floor_{dimension}", numeric_floor)
        self.gateway: Gateway = (
            gateway if gateway is not None else self._default_gateway()
        )
        self._run_log_start = len(self.gateway.decision_log)
        from .evaluation.order_bias import OrderBiasPolicy

        if grading_policy is None and grading_policy_from_env:
            configured_grading_policy = os.getenv("PROMPT_ENHANCER_ORDER_BIAS_POLICY")
            if configured_grading_policy and configured_grading_policy.strip():
                grading_policy = Path(configured_grading_policy.strip())
        self.grading_policy = (
            grading_policy
            if grading_policy is None or isinstance(grading_policy, OrderBiasPolicy)
            else OrderBiasPolicy(grading_policy)
        )
        self._configure_recording_gateway()
        self.history = RunHistory(self.store)
        self.rubric_store = rubric_store or (
            SQLiteRubricStore(self.store.path)
            if self.store.path != ":memory:"
            else None
        )
        self.repeat = RepeatCoordinator()
        self._progress: ProgressCallback | None = None
        self._round: dict[str, int] = {}
        self._scope_perf: float | None = None
        self._scope_cost_base: float = 0.0
        self._scope_elapsed_base_ms: int = 0
        self._clarification = ClarificationService(
            self._clarification_repository(),
            continuation=self._continue_clarification,
        )

    def _configure_recording_gateway(self) -> None:
        from .diagnosis import checklist_impacts, checklist_keys
        from .evaluation.recording import RecordingGateway
        from .grading_cascade import _retry_multiplier

        recording = self.gateway
        if not isinstance(recording, RecordingGateway):
            return
        recording.rubric_thresholds = dict(self.diagnosis_rubric.gap_thresholds)
        recording.writer_instruction_version = self.writer_instruction_version
        recording.faithfulness_threshold = self.faithfulness_threshold
        recording.sentence_diagnosis_version = self.sentence_diagnosis_version
        recording.task_taxonomy_version = self.task_taxonomy_version
        recording.speculative_diagnosis = self.speculative_diagnosis
        recording.observe_sequential_diagnosis = self.observe_sequential_diagnosis
        recording.diagnosis_retry_reservation_multiplier = _retry_multiplier(recording)
        recording.checklist_keys = list(checklist_keys(self.diagnosis_rubric))
        recording.checklist_impacts = checklist_impacts(self.diagnosis_rubric)
        if self.decision_policy is not None:
            recording.decision_policy_artifacts = list(
                self.decision_policy.artifact_dicts
            )
            recording.decision_policy_version = self.decision_policy.policy_version
        if self.grading_policy is not None:
            recording.grading_policy_artifact = dict(self.grading_policy.artifact)
        if self.writer_instruction_version >= CRITERION_READING_MIN_VERSION:
            recording.criterion_reading = recording_metadata()
        if self.writer_instruction_version >= 7:
            recording.cascade_settings = {
                "retry_reservation_multiplier": _retry_multiplier(recording),
                "grading_cascade_pair_cap": self.config.grading_cascade_pair_cap,
                "grading_cascade_dollar_cap": self.config.grading_cascade_dollar_cap,
                "grading_confirmation_reservation_usd": self.config.grading_confirmation_reservation_usd,
                **(
                    {
                        "attribution_pair_cap": self.config.attribution_pair_cap,
                        "attribution_dollar_cap": self.config.attribution_dollar_cap,
                    }
                    if self.writer_instruction_version >= 8
                    else {}
                ),
            }

    def attach_recording_gateway(self, path: Path) -> RecordingGateway:
        from .evaluation.recording import RecordingGateway

        if isinstance(self.gateway, RecordingGateway):
            raise ValueError("a recording gateway is already attached")
        recording = RecordingGateway(self.gateway, path)
        self.gateway = recording
        self._configure_recording_gateway()
        return recording

    def get_model_settings(self) -> dict[str, Any]:
        return self.config.public_dict()

    def recalibrate_score_floors(self) -> dict[str, Any]:
        """Explicitly recompute, persist, and activate floors from run feedback."""
        from .config import Settings as DefaultSettings

        result = calibrate_feedback_floors(
            self.history.all_runs(),
            self.config.score_floors,
            conservative_floors=DefaultSettings().score_floors,
        )
        if result.applied:
            for dimension, floor in result.adjusted_floors.items():
                setattr(self.config, f"score_floor_{dimension}", floor)
        if self.settings_store is not None:
            self.settings_store.update_score_floors(
                result.adjusted_floors, result.to_dict()
            )
        return {**result.to_dict(), "active_floors": self.config.score_floors}

    def update_model_settings(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if "judge_model" in values and values["judge_model"] != self.config.judge_model:
            raise ValueError("judge model is fixed")
        writer = values.get("writer_model", self.config.writer_model)
        strong = values.get("strong_check_model", self.config.strong_check_model)
        weak = values.get("weak_models", self.config.weak_models)
        if not isinstance(writer, str) or not writer.strip():
            raise ValueError("writer_model must be a model ID")
        if not isinstance(strong, str) or not strong.strip():
            raise ValueError("strong_check_model must be a model ID")
        if (
            not isinstance(weak, (list, tuple))
            or len(weak) < self.config.weak_model_count
            or any(not isinstance(item, str) or not item.strip() for item in weak)
            or len(set(weak)) != len(weak)
        ):
            raise ValueError(
                f"weak_models must contain at least {self.config.weak_model_count} distinct model IDs"
            )
        selected = ModelDefaults(
            writer=writer.strip(), strong=strong.strip(), weak=tuple(weak)
        )
        if self.settings_store is not None:
            self.settings_store.save(selected)
        self.config.writer_model = selected.writer
        self.config.strong_check_model = selected.strong
        self.config.weak_models = selected.weak
        return self.get_model_settings()

    def _default_gateway(self) -> Gateway:
        if not self.config.openrouter_api_key or not self.config.opencode_go_key:
            return ScriptedGateway(jev_model=self.config.judge_model)
        gateway_config = GatewayConfig.from_env()
        gateway_config.openrouter_api_key = self.config.openrouter_api_key
        gateway_config.go_api_key = self.config.opencode_go_key
        gateway_config.jev_model = self.config.judge_model
        transport = HttpTransport()
        catalog = LiveModelCatalog(
            transport,
            go_url=gateway_config.go_models_url
            or f"{gateway_config.go_base_url}/models",
            openrouter_url=gateway_config.openrouter_models_url
            or f"{gateway_config.openrouter_base_url}/models",
            go_api_key=gateway_config.go_api_key,
            openrouter_api_key=gateway_config.openrouter_api_key,
        )
        return HttpGateway(transport, config=gateway_config, catalog=catalog)

    def _clarification_repository(self) -> Any:
        if self.store.path == ":memory:":
            return InMemoryClarificationRepository()
        return SQLiteClarificationRepository(self.store.path)

    def _run_settings(self, options: Mapping[str, Any]) -> Settings:
        overrides = options.get("model_overrides") or {}
        if not isinstance(overrides, Mapping):
            raise TypeError("model_overrides must be a mapping")
        if "judge" in overrides or "judge_model" in overrides:
            raise ValueError("judge model is fixed to Jev")
        writer = overrides.get(
            "writer", overrides.get("writer_model", self.config.writer_model)
        )
        strong = overrides.get(
            "strong",
            overrides.get("strong_check_model", self.config.strong_check_model),
        )
        selected_weak = overrides.get("weak", overrides.get("weak_models"))
        if selected_weak is None:
            selected_weak = self.config.weak_models
        if (
            not isinstance(writer, str)
            or not writer
            or not isinstance(strong, str)
            or not strong
        ):
            raise ValueError("writer and strong model overrides must be model IDs")
        if isinstance(selected_weak, str):
            selected_weak = (selected_weak,)
        if (
            not isinstance(selected_weak, (list, tuple))
            or not selected_weak
            or any(not isinstance(item, str) or not item for item in selected_weak)
        ):
            raise ValueError("weak model overrides must be a non-empty list")
        count = self.config.weak_model_count
        if len(selected_weak) < count or len(set(selected_weak[:count])) != count:
            raise ValueError(f"weak panel requires {count} distinct models")
        return replace(
            self.config,
            writer_model=writer,
            strong_check_model=strong,
            weak_models=tuple(selected_weak[:count]),
        )

    def validate_request(
        self, prompt: str, options: Mapping[str, Any] | None = None
    ) -> None:
        """Raise ValueError for a request that optimize() would refuse."""
        self._prepare(prompt, options)

    def _prepare(
        self, prompt: str, options: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], Settings, int]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        supplied_options = _safe_options(options or {})
        unknown_options = set(supplied_options).difference(RUN_OPTION_KEYS)
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise ValueError(f"unknown run option(s): {names}")
        overrides = supplied_options.get("model_overrides")
        if isinstance(overrides, Mapping):
            unknown_overrides = set(overrides).difference(MODEL_OVERRIDE_KEYS)
            if unknown_overrides:
                names = ", ".join(sorted(unknown_overrides))
                raise ValueError(f"unknown model override(s): {names}")
        style = parse_improvement_style(
            supplied_options.get("improvement_style", "auto")
        )
        supplied_options["improvement_style"] = style
        # Optional run-control limits ride along in the options; reject an
        # invalid limit here so callers fail fast with a ValueError.
        RunControl.from_options(supplied_options)
        run_settings = self._run_settings(supplied_options)
        run_seed = _run_seed(prompt, supplied_options.get("seed"))
        return supplied_options, run_settings, run_seed

    @contextmanager
    def _progress_scope(
        self,
        progress: ProgressCallback | None,
        *,
        started_perf: float | None = None,
        cost_base: float = 0.0,
        elapsed_base_ms: int = 0,
    ) -> Iterator[None]:
        previous = (
            self._progress,
            self._round,
            self._scope_perf,
            self._scope_cost_base,
            self._scope_elapsed_base_ms,
        )
        self._progress, self._round = progress, {}
        self._scope_perf = started_perf
        self._scope_cost_base = cost_base
        self._scope_elapsed_base_ms = elapsed_base_ms
        try:
            yield
        finally:
            (
                self._progress,
                self._round,
                self._scope_perf,
                self._scope_cost_base,
                self._scope_elapsed_base_ms,
            ) = previous

    def _stage(self, name: str) -> None:
        if self._progress is not None:
            payload: dict[str, Any] = dict(self._round)
            if self._scope_perf is not None:
                payload["elapsed_ms"] = self._scope_elapsed_base_ms + max(
                    0, round((perf_counter() - self._scope_perf) * 1000)
                )
                payload["cost_total"] = max(
                    0.0, self._scope_cost_base + self._current_usage_total()
                )
            self._progress(name, payload)

    def _current_usage_total(self) -> float:
        try:
            report = self.gateway.usage_report() or {}
            total = report.get("total", 0.0)
            return float(total) if isinstance(total, (int, float)) else 0.0
        except Exception:  # noqa: BLE001 - progress must never fail a run
            return 0.0

    def optimize(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> OptimizeResult:
        supplied_options, run_settings, run_seed = self._prepare(prompt, options)
        run_id = run_id or new_run_id()
        started_perf = perf_counter()
        started_at = utc_now()
        self.gateway.new_run(run_id)
        self._run_log_start = len(self.gateway.decision_log)

        try:
            with self._progress_scope(progress, started_perf=started_perf):
                result = self._optimize_started(
                    prompt,
                    supplied_options,
                    run_settings,
                    run_seed,
                    run_id,
                    started_at,
                    started_perf,
                )
        except Exception as exc:  # noqa: BLE001 - a failed run is still saved and explained
            result = self.failure_result(run_id, prompt, exc, started_at=started_at)
            result["report"]["models"] = run_settings.model_roles()
        result["timing"]["total_ms"] = max(
            0, round((perf_counter() - started_perf) * 1000)
        )
        result["timing"]["started_at"] = started_at
        result["timing"]["finished_at"] = utc_now()
        self._save_result(prompt, supplied_options, result, started_at)
        return result

    def failure_result(
        self,
        run_id: str,
        prompt: str,
        exc: BaseException,
        *,
        started_at: str | None = None,
    ) -> OptimizeResult:
        """Build the public result for a run that stopped before finishing."""
        failure = describe_failure(exc)
        cancelled = failure["kind"] == "cancelled"
        payload = OptimizeResult(
            status="failed",
            run_id=run_id,
            final_prompt=prompt,
            original_kept=True,
            report=apply_outcome_fields(
                {
                    "status": "cancelled" if cancelled else "failed",
                    "summary": failure["hint"]
                    if cancelled
                    else "The run stopped before finishing; your original prompt was saved unchanged.",
                    "error": failure["message"],
                    "failure": failure,
                    "diagnosis": {"confirmed_gaps": [], "problem_sentences": []},
                    "assumptions": [],
                },
                original_prompt=prompt,
                final_prompt=prompt,
                control_state="cancelled" if cancelled else None,
            ),
            cost=self._usage_cost(),
            timing={
                "total_ms": 0,
                "started_at": started_at or utc_now(),
                "finished_at": utc_now(),
            },
        )
        self._attach_current_run_jev(payload, original_prompt=prompt)
        return payload

    def _attach_current_run_jev(
        self,
        payload: OptimizeResult,
        previous: Sequence[Mapping[str, Any]] = (),
        *,
        original_prompt: str | None = None,
    ) -> None:
        """Retain all answered raw judgments when a run stops mid-pipeline."""
        records = [dict(item) for item in previous] + [
            dict(item)
            for item in round_judgment_provenance(
                self.gateway, self._run_log_start, {}, None
            )
        ]
        report = dict(payload.get("report", {}))
        report["judgment_provenance"] = records
        report["capabilities_fired"] = summarize_capabilities(records)
        report = apply_outcome_fields(
            report,
            original_prompt=(
                original_prompt
                if original_prompt is not None
                else str(
                    payload.get("original_prompt")
                    or payload.get("prompt")
                    or payload.get("final_prompt")
                    or ""
                )
            ),
            final_prompt=str(payload.get("final_prompt") or ""),
        )
        payload["report"] = report

    def _optimize_started(
        self,
        prompt: str,
        options: Mapping[str, Any],
        run_settings: Settings,
        run_seed: int,
        run_id: str,
        started_at: str,
        started_perf: float,
    ) -> OptimizeResult:
        self._stage("diagnosing")
        diagnosis_log_start = len(self.gateway.decision_log)
        diagnosis = self._diagnose(prompt)
        diagnosis_payload = (
            diagnosis.as_dict()
            if diagnosis is not None
            else {"confirmed_gaps": [], "problem_sentences": []}
        )
        diagnosis_payload["judgment_provenance"] = [
            dict(item)
            for item in round_judgment_provenance(
                self.gateway, diagnosis_log_start, {}, None
            )
        ]
        request_evidence = diagnosis_payload.get("request_evidence", {})
        if (
            isinstance(request_evidence, Mapping)
            and request_evidence.get("complete") is False
        ):
            diagnosis_records = diagnosis_payload["judgment_provenance"]
            return OptimizeResult(
                status="completed",
                run_id=run_id,
                final_prompt=prompt,
                original_kept=True,
                report={
                    "status": "completed",
                    "summary": "Diagnosis evidence was incomplete; your original prompt was kept.",
                    "diagnosis": diagnosis_payload,
                    "assumptions": [],
                    "history": [],
                    "models": run_settings.model_roles(),
                    "judgment_provenance": diagnosis_records,
                    "capabilities_fired": summarize_capabilities(diagnosis_records),
                },
                cost=self._usage_cost(),
                timing={
                    "total_ms": max(0, round((perf_counter() - started_perf) * 1000)),
                    "started_at": started_at,
                    "finished_at": utc_now(),
                },
            )
        gaps = tuple(diagnosis.confirmed_gaps) if diagnosis is not None else ()
        self._stage("clarifying")
        plan = Clarifier(
            self.gateway,
            writer_model=run_settings.writer_model,
            judge_model=run_settings.judge_model,
        ).plan(
            prompt,
            gaps,
            allow_clarification=options.get("clarification_allowed", True),
            run_id=run_id,
        )
        if plan.questions:
            state = self._clarification.start(
                run_id,
                prompt,
                plan,
                metadata={
                    "options": dict(options),
                    "diagnosis": diagnosis_payload,
                },
            )
            result = self._needs_input_result(run_id, state, started_at, started_perf)
            result["report"]["models"] = run_settings.model_roles()
            result["report"]["diagnosis"] = diagnosis_payload
            return result
        return self._run_rounds(
            _RunContext(
                prompt,
                run_id,
                diagnosis_payload,
                [item.as_dict() for item in plan.assumptions],
                run_settings,
                run_seed,
                str(options.get("improvement_style", "auto")),
            ),
            prior_failures=options.get("prior_round_failures", ()),
            control=RunControl.from_options(options),
            options=options,
            started_perf=started_perf,
            started_at=started_at,
        )

    def _round_executor(
        self,
        context: _RunContext,
        understand: UnderstandResult,
        route: RouteResult,
        writer_attempts: list[dict[str, Any]],
    ) -> RoundRunner:
        def execute(request: RoundRequest) -> RoundOutcome:
            self._round = {"round": request.round_number}
            plan = RoundPlan(
                prompt=context.prompt,
                working_prompt=_prompt_with_assumptions(
                    context.prompt, context.assumptions
                ),
                run_id=context.run_id,
                seed=context.seed,
                diagnosis=context.diagnosis,
                assumptions=context.assumptions,
                settings=context.settings,
                faithfulness_threshold=self.faithfulness_threshold,
                writer_instruction_version=self.writer_instruction_version,
                writer_attempts=writer_attempts,
                prior_failures=tuple(request.prior_failures),
                prior_vector=request.prior_vector,
                round_number=request.round_number,
                grading_policy=self.grading_policy,
                screen_cache=self.success_test_screen_cache,
                decision_policy=self.decision_policy,
                applied_style=route.applied_style,
                hard_constraints=tuple(understand.hard_constraints),
                exact_output=understand.exact_output,
                route_strategies=tuple(route.strategies),
            )
            return run_round(self.gateway, plan, on_stage=self._stage)

        return execute

    def _understand_and_route(
        self, context: _RunContext
    ) -> tuple[UnderstandResult, RouteResult]:
        """Run the Understand and Route stages for a round context."""
        provenance_start = len(self.gateway.decision_log)
        understand = run_understand(
            self.gateway,
            context.prompt,
            requested_style=context.improvement_style,
            diagnosis=context.diagnosis,
            judge_model=context.settings.judge_model,
            run_id=context.run_id,
        )
        route = run_route(
            self.gateway,
            context.prompt,
            applied_style=understand.applied_style,
            hard_constraints=understand.hard_constraints,
            exact_output=understand.exact_output,
            judge_model=context.settings.judge_model,
            run_id=context.run_id,
        )
        stage_judgments = round_judgment_provenance(
            self.gateway, provenance_start, {}, None
        )
        understand = replace(
            understand,
            provenance={
                **understand.provenance,
                "judgment_provenance": [dict(item) for item in stage_judgments],
            },
        )
        return understand, route

    @staticmethod
    def _attach_understand_route(
        payload: OptimizeResult,
        understand: UnderstandResult,
        route: RouteResult,
        *,
        original_prompt: str,
    ) -> None:
        """Record the style stages on a completed run payload."""
        report = dict(payload.get("report", {}))
        report["improvement_style"] = understand.requested_style
        report["applied_style"] = understand.applied_style
        report["understand"] = understand.to_dict()
        report["route"] = route.to_dict()
        understand_judgments = understand.provenance.get("judgment_provenance", ())
        diagnosis_report = report.get("diagnosis", {})
        diagnosis_judgments = (
            diagnosis_report.get("judgment_provenance", ())
            if isinstance(diagnosis_report, Mapping)
            else ()
        )
        records = [
            dict(item) for item in diagnosis_judgments if isinstance(item, Mapping)
        ] + [dict(item) for item in understand_judgments if isinstance(item, Mapping)]
        history = report.get("history", ())
        if isinstance(history, list):
            for round_item in history:
                if not isinstance(round_item, Mapping):
                    continue
                evidence = round_item.get("evidence", {})
                round_records = (
                    evidence.get("judgment_provenance", ())
                    if isinstance(evidence, Mapping)
                    else ()
                )
                if isinstance(round_records, list):
                    records.extend(
                        dict(item)
                        for item in round_records
                        if isinstance(item, Mapping)
                    )
        report["judgment_provenance"] = records
        report["capabilities_fired"] = summarize_capabilities(records)
        report = apply_outcome_fields(
            report,
            original_prompt=original_prompt,
            final_prompt=str(payload.get("final_prompt") or original_prompt),
        )
        payload["report"] = report

    def _run_rounds(
        self,
        context: _RunContext,
        *,
        prior_failures: Any = (),
        prior_history: tuple[RoundEvidence, ...] = (),
        tracker: RoundTracker | None = None,
        control: RunControl | None = None,
        options: Mapping[str, Any] | None = None,
        started_perf: float | None = None,
        started_at: str | None = None,
        cost_base: float = 0.0,
        elapsed_base_ms: int = 0,
    ) -> OptimizeResult:
        if context.understand is not None and context.route is not None:
            understand, route = context.understand, context.route
        else:
            understand, route = self._understand_and_route(context)
            context = replace(context, understand=understand, route=route)
        if route.impossible_reason is not None:
            return self._impossible_result(context, understand, route)
        active = tracker if tracker is not None else RoundTracker()
        state = RunControlState(
            control=control or RunControl(),
            tracker=active,
            usage_total=self._current_usage_total,
            started_perf=started_perf if started_perf is not None else perf_counter(),
            cost_base=cost_base,
            elapsed_base_ms=elapsed_base_ms,
        )
        writer_attempts: list[dict[str, Any]] = [
            dict(attempt)
            for entry in prior_history
            for attempt in entry.evidence.get("writer_attempts", ())
        ]

        def with_writer_attempts(result: OptimizeResult) -> OptimizeResult:
            if self.writer_instruction_version >= WRITER_REPLY_RECOVERY_MIN_VERSION:
                result["report"]["writer_attempts"] = [
                    dict(item) for item in writer_attempts
                ]
            return result

        base_execute = self._round_executor(context, understand, route, writer_attempts)

        def execute_round(request: RoundRequest) -> RoundOutcome:
            outcome = base_execute(request)
            state.note_completed(request, outcome)
            return outcome

        try:
            repeated = self.repeat.run(
                run_id=context.run_id,
                prompt=context.prompt,
                execute_round=execute_round,
                initial_failures=prior_failures,
                prior_history=prior_history,
            )
        except BudgetPaused as paused:
            return with_writer_attempts(
                self._paused_result(context, state, paused, options, started_at)
            )
        except RunCancelled:
            cancelled = as_optimize_result(
                build_cancelled_result(
                    run_id=context.run_id,
                    prompt=context.prompt,
                    tracker=active,
                    cost=self._usage_cost(),
                    timing={
                        "total_ms": state.elapsed_ms(),
                        "started_at": started_at or utc_now(),
                        "finished_at": utc_now(),
                    },
                )
            )
            cancelled_report = dict(cancelled.get("report", {}))
            cancelled_report["improvement_style"] = context.improvement_style
            cancelled_report["applied_style"] = (
                understand.applied_style if understand is not None else None
            )
            cancelled_report = apply_outcome_fields(
                cancelled_report,
                original_prompt=context.prompt,
                final_prompt=str(cancelled.get("final_prompt") or context.prompt),
                control_state="cancelled",
            )
            cancelled["report"] = cancelled_report
            self._attach_current_run_jev(cancelled, original_prompt=context.prompt)
            return with_writer_attempts(cancelled)
        except Exception as exc:  # noqa: BLE001 - keep resolved run context on failure
            failed = self.failure_result(
                context.run_id,
                context.prompt,
                exc,
                started_at=started_at,
            )
            failure_report = dict(failed.get("report", {}))
            failure_report["improvement_style"] = context.improvement_style
            if understand is not None:
                failure_report["applied_style"] = understand.applied_style
                failure_report["understand"] = understand.to_dict()
            if route is not None:
                failure_report["route"] = route.to_dict()
            failure_report["history"] = [dict(item) for item in active.entries]
            failed["report"] = failure_report
            self._attach_current_run_jev(
                failed,
                original_prompt=context.prompt,
            )
            return with_writer_attempts(failed)
        payload = cast(OptimizeResult, repeated.as_payload())
        self._attach_understand_route(
            payload, understand, route, original_prompt=context.prompt
        )
        return with_writer_attempts(payload)

    def _impossible_result(
        self,
        context: _RunContext,
        understand: UnderstandResult,
        route: RouteResult,
    ) -> OptimizeResult:
        """The honest stop when no style bundle survives the hard gates."""
        failure = failure_impossible(
            understand.applied_style, understand.hard_constraints
        )
        diagnosis_report = context.diagnosis
        diagnosis_records = (
            diagnosis_report.get("judgment_provenance", ())
            if isinstance(diagnosis_report, Mapping)
            else ()
        )
        records = [
            dict(item)
            for item in (
                *diagnosis_records,
                *understand.provenance.get("judgment_provenance", ()),
            )
            if isinstance(item, Mapping)
        ]
        return OptimizeResult(
            status="failed",
            run_id=context.run_id,
            final_prompt=context.prompt,
            original_kept=True,
            report=apply_outcome_fields(
                {
                    "status": "impossible",
                    "summary": failure["message"],
                    "failure": failure,
                    "improvement_style": understand.requested_style,
                    "applied_style": understand.applied_style,
                    "understand": understand.to_dict(),
                    "route": route.to_dict(),
                    "diagnosis": dict(context.diagnosis),
                    "assumptions": list(context.assumptions),
                    "selection_evidence": {
                        "selected_candidate_id": None,
                        "rejection_reasons": {},
                    },
                    "history": [],
                    "models": context.settings.model_roles(),
                    "judgment_provenance": records,
                    "capabilities_fired": summarize_capabilities(records),
                },
                original_prompt=context.prompt,
                final_prompt=context.prompt,
            ),
            cost=self._usage_cost(),
            timing={
                "total_ms": 0,
                "started_at": utc_now(),
                "finished_at": utc_now(),
            },
        )

    def _paused_result(
        self,
        context: _RunContext,
        state: RunControlState,
        paused: BudgetPaused,
        options: Mapping[str, Any] | None,
        started_at: str | None,
    ) -> OptimizeResult:
        payload = build_paused_result(
            run_id=context.run_id,
            prompt=context.prompt,
            models=context.settings.model_roles(),
            diagnosis=context.diagnosis,
            assumptions=context.assumptions,
            tracker=state.tracker,
            control=state.control,
            paused=paused,
            cost=self._usage_cost(),
            timing={
                "total_ms": state.elapsed_ms(),
                "started_at": started_at or utc_now(),
                "finished_at": utc_now(),
            },
        )
        payload[RESUME_CONTEXT_KEY] = resume_context(
            options=dict(options or {}),
            diagnosis=context.diagnosis,
            assumptions=context.assumptions,
            seed=context.seed,
            spent_usd=paused.spent_usd,
            elapsed_ms=paused.elapsed_ms,
        )
        saved_context = payload[RESUME_CONTEXT_KEY]
        if context.understand is not None:
            saved_context["understand"] = context.understand.to_dict()
        if context.route is not None:
            saved_context["route"] = context.route.to_dict()
        saved_context["improvement_style"] = context.improvement_style
        paused_report = dict(payload.get("report", {}))
        paused_report["improvement_style"] = context.improvement_style
        if context.understand is not None:
            paused_report["applied_style"] = context.understand.applied_style
            paused_report["understand"] = context.understand.to_dict()
        if context.route is not None:
            paused_report["route"] = context.route.to_dict()
        paused_report = apply_outcome_fields(
            paused_report,
            original_prompt=context.prompt,
            final_prompt=str(payload.get("final_prompt") or context.prompt),
            control_state="awaiting_approval",
        )
        payload["report"] = paused_report
        self._attach_current_run_jev(
            cast(OptimizeResult, payload), original_prompt=context.prompt
        )
        return as_optimize_result(payload)

    def _diagnose(self, prompt: str) -> DiagnosisReport | None:
        rubric = None
        if self.rubric_store is not None:
            try:
                rubric = self.rubric_store.active_rubric()
            except RuntimeError:
                pass
        suppressed = (
            {item.question_id for item in rubric.questions}
            | set(rubric.disabled_default_question_ids)
            if rubric is not None
            else set()
        )
        diagnosis_rubric = replace(
            self.diagnosis_rubric,
            task_types=tuple(
                replace(
                    task,
                    checklist=tuple(
                        item for item in task.checklist if item.key not in suppressed
                    ),
                )
                for task in self.diagnosis_rubric.task_types
            ),
            task_branches=tuple(
                replace(
                    branch,
                    checklist=(
                        tuple(
                            item
                            for item in branch.checklist
                            if item.key not in suppressed
                        )
                        if branch.checklist is not None
                        else None
                    ),
                )
                for branch in self.diagnosis_rubric.task_branches
            ),
        )
        active_rubric_version = (
            rubric.version_id if rubric is not None else "default-v1"
        )
        questions = (
            [
                {
                    "key": f"rubric:{item.question_id}",
                    "model": self.config.judge_model,
                    "type": item.response_type,
                    "query": item.text,
                    "state": {"prompt": prompt},
                }
                for item in rubric.questions
            ]
            if rubric is not None
            else []
        )
        diagnoser = Diagnoser(
            self.gateway,
            rubric=diagnosis_rubric,
            decision_policy=self.decision_policy,
            rubric_version=active_rubric_version,
            sentence_protocol_version=self.sentence_diagnosis_version,
            task_taxonomy_version=self.task_taxonomy_version,
            speculative_fanout=self.speculative_diagnosis,
            record_request_evidence=(
                self.speculative_diagnosis or self.observe_sequential_diagnosis
            ),
            additional_requests=questions,
        )
        report = diagnoser.diagnose(prompt)
        calibration_evidence = dict(report.calibration or {})
        if rubric is None:
            return report
        if not questions:
            return replace(
                report,
                rubric_version=rubric.version_id,
                calibration=calibration_evidence or None,
            )
        if (report.request_evidence or {}).get("complete") is False:
            return report
        if self.speculative_diagnosis or self.observe_sequential_diagnosis:
            observations = diagnoser.observe_additional(questions)
            responses = [item.raw_answer for item in observations]
            entries = [{"answered_by": item.answered_by} for item in observations]
            report = replace(
                report, request_evidence=diagnoser.request_evidence(prompt)
            )
            if (report.request_evidence or {}).get("complete") is False:
                return report
        else:
            log = getattr(self.gateway, "decision_log", [])
            before = len(log)
            responses = list(self.gateway.decide_batch(questions))
            entries = list(log)[before:]
        gaps = list(report.confirmed_gaps)
        default_impacts = {
            item.key: item.impact
            for task in DEFAULT_RUBRIC.task_types
            for item in task.checklist
        }
        for index, (item, response) in enumerate(
            zip(rubric.questions, responses, strict=True)
        ):
            decision = parse_decision(response)
            if not isinstance(decision, NoulDecision):
                continue
            entry = entries[index] if index < len(entries) else {}
            snapshot = entry.get("answered_by") if isinstance(entry, Mapping) else None
            if (
                item.calibration_snapshot is not None
                and item.calibration_snapshot != snapshot
            ):
                calibration_evidence[f"rubric:{item.question_id}"] = {
                    "disposition": "abstain",
                    "reason": "rubric_calibration_snapshot_mismatch",
                }
                continue
            from .evaluation.calibration import (
                DEFAULT_POLICY_VERSION,
                runtime_question_identity,
            )

            identity = runtime_question_identity(
                f"rubric:{item.question_id}",
                questions[index],
                family="rubric",
                rubric_version=rubric.version_id,
                snapshot=snapshot if isinstance(snapshot, str) else None,
                policy_version=(
                    self.decision_policy.policy_version
                    if self.decision_policy is not None
                    else DEFAULT_POLICY_VERSION
                ),
            )
            identity = replace(
                identity,
                event_mapping={
                    "polarity": "positive" if item.missing_when == "yes" else "negative"
                },
            )
            policy_decision = None
            if self.decision_policy is not None:
                policy_decision = self.decision_policy.apply(
                    question_id=f"rubric:{item.question_id}",
                    identity=identity,
                    decision=decision,
                    raw_answer=response,
                    snapshot=snapshot if isinstance(snapshot, str) else None,
                )
                calibration_evidence[f"rubric:{item.question_id}"] = {
                    "disposition": policy_decision.disposition,
                    "verdict": policy_decision.verdict,
                    "reason": policy_decision.reason,
                    "threshold": policy_decision.threshold,
                    "predicate": dict(policy_decision.predicate),
                    "event_probability": policy_decision.evidence.get(
                        "event_probability"
                    ),
                    "fit": policy_decision.evidence.get("fit"),
                }
            missing = (
                decision.probability
                if item.missing_when == "yes"
                else 1.0 - decision.probability
            )
            if policy_decision is not None and not policy_decision.is_legacy:
                if not policy_decision.may_gate or policy_decision.threshold is None:
                    continue
                threshold = policy_decision.threshold
            else:
                threshold = item.threshold
            gate_probability = (
                policy_decision.evidence.get("event_probability", missing)
                if policy_decision is not None and policy_decision.may_gate
                else missing
            )
            legacy_confident = (
                decision.confidence >= diagnosis_rubric.confidence_threshold
            )
            if gate_probability >= threshold and (
                policy_decision is not None
                and not policy_decision.is_legacy
                or legacy_confident
            ):
                gaps.append(
                    ConfirmedGap(
                        item.question_id,
                        item.text,
                        default_impacts.get(item.question_id, GapImpact.MEDIUM),
                        gate_probability,
                        decision.confidence,
                        threshold,
                    )
                )
        return replace(
            report,
            confirmed_gaps=tuple(gaps),
            rubric_version=rubric.version_id,
            calibration=calibration_evidence or None,
        )

    def _assumption_meaning_check(
        self, original_prompt: str, updated_prompt: str, run_id: str
    ) -> dict[str, Any]:
        try:
            response = self.gateway.decide(
                {
                    "model": self.config.judge_model,
                    "key": "assumption_meaning",
                    "state": {
                        "original_prompt": original_prompt,
                        "updated_prompt": updated_prompt,
                    },
                    "question": jev_questions.ASSUMPTION_MEANING_QUESTION,
                },
                role="judge",
                run_id=run_id,
            )
            decision = parse_decision(response)
        except Exception as exc:
            raise RuntimeError("assumption meaning check is unavailable") from exc
        if not isinstance(decision, NoulDecision):
            raise TypeError("assumption meaning check returned an unexpected decision")
        return {
            "passed": decision.probability >= 0.8,
            "checked": True,
            "score": decision.probability,
        }

    def _continue_clarification(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = str(state["prompt"])
        run_id = str(state["run_id"])
        metadata = (
            state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
        )
        stored_options = (metadata or {}).get("options", {})
        options = dict(stored_options) if isinstance(stored_options, Mapping) else {}
        run_settings = self._run_settings(options)
        assumptions = state.get("assumptions", [])
        stored_options = (metadata or {}).get("options", {})
        style = str(
            stored_options.get("improvement_style", "auto")
            if isinstance(stored_options, Mapping)
            else "auto"
        )
        context = _RunContext(
            prompt,
            run_id,
            (metadata or {}).get(
                "diagnosis", {"confirmed_gaps": [], "problem_sentences": []}
            ),
            assumptions,
            run_settings,
            _run_seed(prompt, (metadata or {}).get("options", {}).get("seed")),
            style,
        )
        return self._run_rounds(
            context,
            control=RunControl.from_options(options),
            options=options,
            started_perf=perf_counter(),
            started_at=utc_now(),
        )

    def resume(
        self,
        run_id: str,
        answers: dict[str, Any],
        *,
        progress: ProgressCallback | None = None,
    ) -> OptimizeResult:
        usage_before = self._usage_cost()
        started_perf = perf_counter()
        try:
            with self._progress_scope(progress, started_perf=started_perf):
                state = self._clarification.resume(run_id, answers)
        except UnknownRunError as exc:
            if self.store.get_run(run_id) is not None:
                raise RunNotPausedError(f"Run {run_id!r} is not paused") from exc
            raise RunNotFoundError(run_id) from exc
        return self._save_clarification_result(
            run_id, state, usage_before, started_perf
        )

    def validate_resume(self, run_id: str, answers: Mapping[str, Any]) -> None:
        """Validate a resume request before its background job is queued."""

        try:
            self._clarification.validate_answers(run_id, answers)
        except UnknownRunError as exc:
            if self.store.get_run(run_id) is not None:
                raise RunNotPausedError(f"Run {run_id!r} is not paused") from exc
            raise RunNotFoundError(run_id) from exc

    def _paused_record(
        self, run_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return the record, result, and resume context of a budget-paused run."""
        record = self.store.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        raw_result = record.get("result")
        result = dict(raw_result) if isinstance(raw_result, Mapping) else {}
        raw_report = result.get("report")
        report = raw_report if isinstance(raw_report, Mapping) else {}
        if report.get("status") != PAUSED_REPORT_STATUS:
            raise RunNotPausedError(f"Run {run_id!r} is not paused for approval")
        raw_context = record.get(RESUME_CONTEXT_KEY)
        if not isinstance(raw_context, Mapping):
            raise RunNotPausedError(f"Run {run_id!r} has no resumable state")
        return record, result, dict(raw_context)

    def validate_continue(
        self,
        run_id: str,
        time_limit_s: Any = None,
        spend_limit_usd: Any = None,
    ) -> None:
        """Validate an approval request before its background job is queued."""
        self._paused_record(run_id)
        RunControl.from_options(
            {"time_limit_s": time_limit_s, "spend_limit_usd": spend_limit_usd}
        )

    def continue_run(
        self,
        run_id: str,
        *,
        progress: ProgressCallback | None = None,
        time_limit_s: Any = None,
        spend_limit_usd: Any = None,
    ) -> OptimizeResult:
        """Resume a budget-paused run from its paused Round boundary.

        Approval continues without the previous limits unless new ones are
        supplied: spend only grows, so keeping the triggered limit would
        pause again at the very next boundary.
        """
        record, paused_result, saved = self._paused_record(run_id)
        self._run_log_start = len(self.gateway.decision_log)
        control = RunControl.from_options(
            {"time_limit_s": time_limit_s, "spend_limit_usd": spend_limit_usd}
        )
        options = dict(saved.get("options") or {})
        run_settings = self._run_settings(options)
        prompt = str(record.get("prompt") or "")
        saved_diagnosis = saved.get("diagnosis")
        diagnosis: Mapping[str, Any] = (
            dict(saved_diagnosis)
            if isinstance(saved_diagnosis, Mapping)
            else {"confirmed_gaps": [], "problem_sentences": []}
        )
        saved_assumptions = saved.get("assumptions")
        assumptions = (
            list(saved_assumptions) if isinstance(saved_assumptions, list) else []
        )
        raw_seed = saved.get("seed")
        seed = raw_seed if isinstance(raw_seed, int) else _run_seed(prompt, None)
        requested_style = str(
            options.get("improvement_style") or saved.get("improvement_style") or "auto"
        )
        understand = _understand_result_from_saved(saved.get("understand"))
        route = _route_result_from_saved(saved.get("route"))
        context = _RunContext(
            prompt,
            run_id,
            diagnosis,
            assumptions,
            run_settings,
            seed,
            requested_style,
            understand,
            route,
        )
        prior_history = _history_from_run(paused_result)
        tracker = RoundTracker.preload(
            [evidence.to_dict() for evidence in prior_history],
            str(paused_result.get("final_prompt") or prompt),
            bool(paused_result.get("original_kept", True)),
        )
        usage_before = self._usage_cost()
        cost_base = _mapping_total(record.get("cost")) - _mapping_total(usage_before)
        elapsed_base_ms = int(saved.get("elapsed_ms", 0) or 0)
        started_perf = perf_counter()
        with self._progress_scope(
            progress,
            started_perf=started_perf,
            cost_base=cost_base,
            elapsed_base_ms=elapsed_base_ms,
        ):
            result = dict(
                self._run_rounds(
                    context,
                    tracker=tracker,
                    control=control,
                    options=options,
                    prior_history=prior_history,
                    started_perf=started_perf,
                    started_at=str(record.get("created_at") or utc_now()),
                    cost_base=cost_base,
                    elapsed_base_ms=elapsed_base_ms,
                )
            )
        raw_timing = result.get("timing")
        timing = dict(raw_timing) if isinstance(raw_timing, Mapping) else {}
        timing["total_ms"] = int(timing.get("total_ms", 0) or 0) + elapsed_base_ms
        result["timing"] = timing
        result["cost"] = _add_usage_delta(
            dict(record.get("cost") or {}), usage_before, self._usage_cost()
        )
        previous_report = paused_result.get("report", {})
        previous_judgments = (
            previous_report.get("judgment_provenance", ())
            if isinstance(previous_report, Mapping)
            else ()
        )
        self._attach_current_run_jev(
            cast(OptimizeResult, result),
            [item for item in previous_judgments if isinstance(item, Mapping)],
            original_prompt=str(record.get("prompt") or ""),
        )
        return self._save_continued_run(record, cast(OptimizeResult, result))

    def stop_run(self, run_id: str) -> OptimizeResult:
        """Permanently stop a budget-paused run, keeping completed rounds."""
        record, paused_result, _saved = self._paused_record(run_id)
        timing = dict(paused_result.get("timing") or {})
        timing["finished_at"] = utc_now()
        stored_cost = record.get("cost")
        stopped = build_stopped_result(
            paused_result=paused_result,
            original_prompt=str(record.get("prompt") or ""),
            cost=dict(stored_cost)
            if isinstance(stored_cost, Mapping)
            else dict(paused_result.get("cost") or {}),
            timing=timing,
        )
        return self._save_continued_run(record, as_optimize_result(stopped))

    def _save_continued_run(
        self,
        record: Mapping[str, Any],
        result: OptimizeResult,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> OptimizeResult:
        resume_ctx = take_resume_context(cast(dict[str, Any], result))
        evidence = self._training_evidence(result, record)
        self._attach_jev_evidence(result, evidence)
        cleaned = {
            key: value for key, value in record.items() if key != RESUME_CONTEXT_KEY
        }
        saved: dict[str, Any] = {
            **cleaned,
            "result": result,
            "cost": result.get("cost", cleaned.get("cost", {})),
            "timing": result.get("timing", cleaned.get("timing", {})),
            **evidence,
        }
        if options is not None:
            saved["options"] = dict(options)
        if resume_ctx is not None:
            saved[RESUME_CONTEXT_KEY] = resume_ctx
        self.store.save_run(saved)
        return result

    def skip_clarification(
        self, run_id: str, *, progress: ProgressCallback | None = None
    ) -> OptimizeResult:
        usage_before = self._usage_cost()
        started_perf = perf_counter()
        try:
            with self._progress_scope(progress, started_perf=started_perf):
                state = self._clarification.skip(run_id)
        except UnknownRunError as exc:
            if self.store.get_run(run_id) is not None:
                raise RunNotPausedError(f"Run {run_id!r} is not paused") from exc
            raise RunNotFoundError(run_id) from exc
        return self._save_clarification_result(
            run_id, state, usage_before, started_perf
        )

    def _save_clarification_result(
        self,
        run_id: str,
        state: Mapping[str, Any],
        usage_before: Mapping[str, Any],
        started_perf: float,
    ) -> OptimizeResult:
        result = state.get("result")
        if not isinstance(result, Mapping):
            raise RunNotFoundError(run_id)
        result = dict(result)
        result["timing"] = _finished_timing(result.get("timing"), started_perf)
        resume_ctx = take_resume_context(result)
        record = self.store.get_run(run_id)
        if record is not None:
            result["cost"] = _add_usage_delta(
                dict(record.get("cost") or {}), usage_before, self._usage_cost()
            )
            evidence = self._training_evidence(result, record)
            self._attach_jev_evidence(cast(OptimizeResult, result), evidence)
            saved: dict[str, Any] = {
                **record,
                "result": result,
                "cost": result.get("cost", record.get("cost", {})),
                "timing": result.get("timing", record.get("timing", {})),
                **evidence,
            }
            if resume_ctx is not None:
                saved[RESUME_CONTEXT_KEY] = resume_ctx
            self.store.save_run(saved)
        return cast(OptimizeResult, result)

    def update_assumption(
        self, run_id: str, assumption: dict[str, Any]
    ) -> OptimizeResult:
        record = self.store.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        key = str(assumption.get("key", "")).strip()
        value = str(assumption.get("value", "")).strip()
        if not key or not value:
            raise ValueError("assumption key and value are required")

        result = cast(OptimizeResult, dict(record.get("result") or {}))
        if result.get("status") != "completed":
            raise ValueError("assumptions can only be edited on completed runs")
        report = dict(result.get("report") or {})
        assumptions = [
            dict(item)
            for item in report.get("assumptions", [])
            if isinstance(item, Mapping)
        ]
        previous = next(
            (item for item in assumptions if str(item.get("key", "")) == key), None
        )
        if previous is None:
            raise ValueError("assumption is not part of this run")
        old_value = str(previous.get("value", ""))
        final_prompt = str(result.get("final_prompt") or record.get("prompt") or "")
        usage_before = self._usage_cost()
        updated_prompt = _apply_assumption(
            final_prompt,
            key,
            old_value,
            value,
            previous.get("source"),
            sentence_protocol_version=self.sentence_diagnosis_version,
        )
        if updated_prompt is None:
            raise ValueError(
                "assumption value is not stated unambiguously in the final prompt"
            )
        if not updated_prompt or updated_prompt == final_prompt:
            raise ValueError("assumption edit did not update the final prompt")
        check = self._assumption_meaning_check(
            str(record["prompt"]), updated_prompt, run_id
        )
        if check.get("passed") is False:
            raise ValueError("edited assumption did not preserve the prompt's meaning")

        replacement = {"key": key, "value": value, "source": "user_edit"}
        for index, item in enumerate(assumptions):
            if str(item.get("key", "")) == key:
                assumptions[index] = replacement
                break
        report["assumptions"] = assumptions
        report["status"] = "edited"
        report.pop("convergence", None)
        report.pop("selection_evidence", None)
        report["summary"] = (
            "Assumption corrected; performance evidence is from the original optimization."
        )
        report["diff"] = prompt_diff(str(record["prompt"]), updated_prompt)
        report["final_prompt"] = updated_prompt
        report["assumption_check"] = check
        report["assumption_edit"] = {
            "key": key,
            "previous": old_value,
            "corrected": value,
            "previous_final_prompt": final_prompt,
        }
        result["final_prompt"] = updated_prompt
        result["original_kept"] = updated_prompt == str(record["prompt"])
        report = apply_outcome_fields(
            report,
            original_prompt=str(record.get("prompt") or ""),
            final_prompt=updated_prompt,
        )
        result["report"] = report
        result["cost"] = cast(
            CostBreakdown,
            _add_usage_delta(
                dict(result.get("cost") or record.get("cost") or {}),
                usage_before,
                self._usage_cost(),
            ),
        )
        self.store.save_run({**record, "result": result, "cost": result["cost"]})
        return result

    def _usage_cost(self) -> CostBreakdown:
        return cast(CostBreakdown, self.gateway.usage_report())

    def _needs_input_result(
        self,
        run_id: str,
        state: Mapping[str, Any],
        started_at: str,
        started_perf: float,
    ) -> OptimizeResult:
        questions = cast(list[dict[str, Any]], list(state.get("questions", [])))
        return OptimizeResult(
            status="needs_input",
            run_id=run_id,
            final_prompt=str(state.get("prompt", "")),
            original_kept=True,
            questions=questions,
            report={
                "status": "needs_input",
                "questions": questions,
                "assumptions": list(state.get("assumptions", [])),
                "diagnosis": {},
                "history": [],
            },
            cost=self._usage_cost(),
            timing={
                "total_ms": max(0, round((perf_counter() - started_perf) * 1000)),
                "started_at": started_at,
                "finished_at": utc_now(),
            },
        )

    def _save_result(
        self,
        prompt: str,
        options: dict[str, Any],
        result: OptimizeResult,
        created_at: str,
    ) -> None:
        resume_ctx = take_resume_context(cast(dict[str, Any], result))
        evidence = self._training_evidence(result)
        self._attach_jev_evidence(result, evidence)
        record: dict[str, Any] = {
            "run_id": result["run_id"],
            "created_at": created_at,
            "prompt": prompt,
            "options": options,
            "result": result,
            "cost": result["cost"],
            "timing": result["timing"],
            **evidence,
        }
        if resume_ctx is not None:
            record[RESUME_CONTEXT_KEY] = resume_ctx
        self.store.save_run(record)

    def _training_evidence(
        self, result: Mapping[str, Any], previous: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        report = (
            result.get("report") if isinstance(result.get("report"), Mapping) else {}
        )
        selection = (
            report.get("selection_evidence") if isinstance(report, Mapping) else {}
        )
        original_score = (
            selection.get("original_score") if isinstance(selection, Mapping) else None
        )
        old_answers = list((previous or {}).get("jev_answers") or [])
        current_answers = list(self.gateway.decision_log)
        answers = (
            current_answers
            if current_answers[: len(old_answers)] == old_answers
            else old_answers + current_answers
        )
        evidence: dict[str, Any] = {"jev_answers": answers}
        if isinstance(original_score, Mapping):
            evidence["original_weak_panel"] = original_score
            evidence["score_summaries"] = {"original": original_score}
        return evidence

    @staticmethod
    def _attach_jev_evidence(
        result: OptimizeResult, evidence: Mapping[str, Any]
    ) -> None:
        answers = evidence["jev_answers"]
        result["report"]["jev_answers"] = answers
        result["report"]["jev_snapshot"] = sorted(
            {answer["answered_by"] for answer in answers if "answered_by" in answer}
        )


# How a clarification appears in the returned prompt when its key alone would
# read as an internal marker.
_CLARIFICATION_LINE_LABELS = {"outside_reference": "Details"}
# Labels for the user's own answers only. Inferred lines keep their key, which
# recorded replays contain; answers never occur in recordings.
_ANSWER_LINE_LABELS = {"context": "Context"}


def _clarification_label(key: str, source: object = None) -> str:
    if source == "answer" and key in _ANSWER_LINE_LABELS:
        return _ANSWER_LINE_LABELS[key]
    return _CLARIFICATION_LINE_LABELS.get(key, key)


def _apply_assumption(
    prompt: str,
    key: str,
    old_value: str,
    value: str,
    source: object = None,
    *,
    sentence_protocol_version: int = SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
) -> str | None:
    label = _clarification_label(key, source)
    line = f"{label}: {value}"
    lines = prompt.splitlines()
    for index, existing in enumerate(lines):
        if existing.strip().casefold() == f"{label.casefold()}: {old_value.casefold()}":
            lines[index] = line
            return "\n".join(lines)
    if not old_value:
        return None
    # Match on the original string so offsets stay valid when case folding would
    # change its length (for example "ß"); underscores count as word characters.
    matches = list(
        re.finditer(rf"(?<!\w){re.escape(old_value)}(?!\w)", prompt, re.IGNORECASE)
    )
    if len(matches) != 1:
        return None
    start, end = matches[0].span()
    if not any(
        item.start <= start and end <= item.end
        for item in split_sentences(prompt, protocol_version=sentence_protocol_version)
    ):
        return None
    return prompt[:start] + value + prompt[end:]


def _prompt_with_assumptions(prompt: str, assumptions: Any) -> str:
    lines = []
    for item in assumptions:
        if not isinstance(item, Mapping) or item.get("source") not in {
            "answer",
            "inferred",
        }:
            continue
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if key and value:
            lines.append(f"{_clarification_label(key, item.get('source'))}: {value}")
    if not lines:
        return prompt
    return prompt.rstrip() + "\n\nClarifications:\n" + "\n".join(lines)


def _understand_result_from_saved(value: Any) -> UnderstandResult | None:
    if not isinstance(value, Mapping):
        return None
    requested = value.get("requested_style")
    applied = value.get("applied_style")
    task_type = value.get("task_type")
    if not all(isinstance(item, str) for item in (requested, applied, task_type)):
        return None
    raw_constraints = value.get("hard_constraints", ())
    constraints = (
        tuple(item for item in raw_constraints if isinstance(item, str))
        if isinstance(raw_constraints, list)
        else ()
    )
    raw_probes = value.get("probes", ())
    probes = (
        tuple(dict(item) for item in raw_probes if isinstance(item, Mapping))
        if isinstance(raw_probes, list)
        else ()
    )
    raw_provenance = value.get("provenance", {})
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
    inferred = value.get("inferred_style")
    screen = value.get("screen_embedded")
    return UnderstandResult(
        requested_style=requested,
        applied_style=applied,
        inferred_style=inferred if isinstance(inferred, str) else None,
        task_type=task_type,
        hard_constraints=constraints,
        exact_output=value.get("exact_output") is True,
        screen_embedded=screen if isinstance(screen, bool) else None,
        probes=probes,
        provenance=provenance,
    )


def _route_result_from_saved(value: Any) -> RouteResult | None:
    if not isinstance(value, Mapping):
        return None
    applied = value.get("applied_style")
    bundle = value.get("bundle")
    if not isinstance(applied, str) or not isinstance(bundle, str):
        return None
    raw_strategies = value.get("strategies", ())
    strategies = (
        tuple(item for item in raw_strategies if isinstance(item, str))
        if isinstance(raw_strategies, list)
        else ()
    )
    raw_provenance = value.get("provenance", {})
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
    find = value.get("find_selection")
    compatible = value.get("decide_compatible")
    impossible = value.get("impossible_reason")
    return RouteResult(
        applied_style=applied,
        bundle=bundle,
        strategies=strategies,
        find_selection=find if isinstance(find, str) else None,
        decide_compatible=compatible if isinstance(compatible, bool) else None,
        impossible_reason=impossible if isinstance(impossible, str) else None,
        provenance=provenance,
    )


def _mapping_total(report: Any) -> float:
    if not isinstance(report, Mapping):
        return 0.0
    total = report.get("total", 0.0)
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    return float(total)


def _add_usage_delta(
    previous: dict[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    updated = dict(previous)
    delta = max(0.0, float(after.get("total", 0.0)) - float(before.get("total", 0.0)))
    updated["total"] = float(previous.get("total", 0.0)) + delta
    previous_roles = previous.get("cost_by_role", previous.get("by_role", {}))
    roles = (
        dict(previous_roles)
        if isinstance(previous_roles, Mapping)
        and all(isinstance(v, (int, float)) for v in previous_roles.values())
        else {}
    )
    before_roles = before.get("cost_by_role", {})
    after_roles = after.get("cost_by_role", {})
    if isinstance(before_roles, Mapping) and isinstance(after_roles, Mapping):
        for role, amount in after_roles.items():
            roles[str(role)] = float(roles.get(str(role), 0.0)) + max(
                0.0, float(amount) - float(before_roles.get(role, 0.0))
            )
    updated["cost_by_role"] = roles
    return updated


def _finished_timing(timing: Any, started_perf: float) -> Any:
    updated = dict(timing) if isinstance(timing, Mapping) else {}
    updated["total_ms"] = max(0, round((perf_counter() - started_perf) * 1000))
    updated["finished_at"] = utc_now()
    return updated


def _safe_options(options: Mapping[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(k): clean(v)
                for k, v in value.items()
                if not any(
                    x in str(k).lower()
                    for x in ("key", "token", "secret", "password", "authorization")
                )
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(options)


def _run_seed(prompt: str, supplied: Any) -> int:
    if supplied is not None:
        if isinstance(supplied, bool) or not isinstance(supplied, int):
            raise ValueError("seed must be an integer")
        return supplied
    return int.from_bytes(sha256(prompt.encode("utf-8")).digest()[:4], "big")
