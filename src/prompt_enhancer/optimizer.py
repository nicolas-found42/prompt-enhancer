"""Public local optimization engine.

The facade is deliberately dependency-injected with a Gateway: production uses
HttpGateway; tests and local demos use ScriptedGateway or ReplayGateway.
The public result always includes the original prompt, evidence, cost, timing,
and a durable run identifier.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from . import jev_questions
from .catalog import DEFAULT_DEEP_WEAK_PANEL, DEFAULT_WEAK_PANEL, LiveModelCatalog
from .clarification import (
    ClarificationService,
    InMemoryClarificationRepository,
    RunNotPausedError,
    SQLiteClarificationRepository,
    UnknownRunError,
)
from .clarifier import Clarifier
from .config import Settings
from .diagnosis import (
    DEFAULT_RUBRIC,
    ConfirmedGap,
    Diagnoser,
    DiagnosisReport,
    DiagnosisRubric,
    GapImpact,
)
from .failures import describe_failure
from .gateway import (
    Gateway,
    GatewayConfig,
    HttpGateway,
    HttpTransport,
    ScriptedGateway,
    completion_text,
    writer_messages,
)
from .history import RunHistory
from .jev import NoulDecision, parse_decision
from .models import (
    CostBreakdown,
    OptimizeResult,
    Tier,
    new_run_id,
    utc_now,
)
from .repeat import RepeatCoordinator, RoundRequest, RoundRunner
from .rewrite import (
    CURRENT_WRITER_INSTRUCTION_VERSION,
    WRITER_INSTRUCTION_VERSIONS,
)
from .rounds import RoundOutcome, RoundPlan, prompt_diff, run_round
from .rubric_revisions import SQLiteRubricStore
from .settings import ModelDefaults, SettingsStore
from .store import RunStore
from .success_tests import (
    DEFAULT_FAITHFULNESS_THRESHOLD,
)

if TYPE_CHECKING:
    from .evaluation.calibration import CalibrationArtifact, DecisionPolicy


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
)


@dataclass(frozen=True, slots=True)
class _RunContext:
    prompt: str
    run_id: str
    diagnosis: Mapping[str, Any]
    assumptions: Any
    settings: Settings
    seed: int


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
    ) -> None:
        if writer_instruction_version not in WRITER_INSTRUCTION_VERSIONS:
            raise ValueError("unknown candidate writer instruction version")
        if not 0 <= faithfulness_threshold <= 1:
            raise ValueError("faithfulness threshold must be a probability")
        self.store = store or RunStore()
        self.config = config or Settings.from_env()
        self.diagnosis_rubric = diagnosis_rubric
        self.writer_instruction_version = writer_instruction_version
        self.faithfulness_threshold = faithfulness_threshold
        from .evaluation.calibration import DecisionPolicy

        selected_policy = (
            decision_policy if decision_policy is not None else calibration
        )
        self.decision_policy = (
            DecisionPolicy.from_artifact(selected_policy)
            if selected_policy is not None
            and not isinstance(selected_policy, DecisionPolicy)
            else selected_policy
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
            defaults = self.settings_store.load().defaults
            self.config.writer_model = defaults.writer
            self.config.strong_check_model = defaults.strong
            self.config.weak_models = defaults.weak
        self.gateway: Gateway = (
            gateway if gateway is not None else self._default_gateway()
        )
        self.history = RunHistory(self.store)
        self.rubric_store = rubric_store or (
            SQLiteRubricStore(self.store.path)
            if self.store.path != ":memory:"
            else None
        )
        self.repeat = RepeatCoordinator()
        self._progress: ProgressCallback | None = None
        self._round: dict[str, int] = {}
        self._clarification = ClarificationService(
            self._clarification_repository(),
            continuation=self._continue_clarification,
        )

    def get_model_settings(self) -> dict[str, Any]:
        return self.config.public_dict()

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
            or len(weak) < 3
            or any(not isinstance(item, str) or not item.strip() for item in weak)
            or len(set(weak)) != len(weak)
        ):
            raise ValueError(
                "weak_models must contain at least three distinct model IDs"
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

    def _run_settings(self, options: Mapping[str, Any], tier: str) -> Settings:
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
            if tier == "deep":
                selected_weak = tuple(
                    dict.fromkeys((*selected_weak, *DEFAULT_DEEP_WEAK_PANEL))
                )
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
        count = Tier.parse(tier).budget.models
        if len(set(selected_weak[:count])) != count:
            raise ValueError(f"weak panel for {tier} requires {count} distinct models")
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
    ) -> tuple[dict[str, Any], str, Settings, int]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        supplied_options = _safe_options(options or {})
        tier = Tier.parse(supplied_options.pop("tier", "standard")).value
        run_settings = self._run_settings(supplied_options, tier)
        run_seed = _run_seed(prompt, supplied_options.get("seed"))
        return supplied_options, tier, run_settings, run_seed

    @contextmanager
    def _progress_scope(self, progress: ProgressCallback | None) -> Iterator[None]:
        previous, previous_round = self._progress, self._round
        self._progress, self._round = progress, {}
        try:
            yield
        finally:
            self._progress, self._round = previous, previous_round

    def _stage(self, name: str) -> None:
        if self._progress is not None:
            self._progress(name, dict(self._round))

    def optimize(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> OptimizeResult:
        supplied_options, tier, run_settings, run_seed = self._prepare(prompt, options)
        run_id = run_id or new_run_id()
        started_perf = perf_counter()
        started_at = utc_now()
        self.gateway.new_run(run_id)

        try:
            with self._progress_scope(progress):
                result = self._optimize_started(
                    prompt,
                    supplied_options,
                    tier,
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
        self._save_result(prompt, tier, supplied_options, result, started_at)
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
        return OptimizeResult(
            status="failed",
            run_id=run_id,
            final_prompt=prompt,
            original_kept=True,
            report={
                "status": "cancelled" if cancelled else "failed",
                "summary": failure["hint"]
                if cancelled
                else "The run stopped before finishing; your original prompt was saved unchanged.",
                "error": failure["message"],
                "failure": failure,
                "diagnosis": {"confirmed_gaps": [], "problem_sentences": []},
                "assumptions": [],
            },
            cost=self._usage_cost(),
            timing={
                "total_ms": 0,
                "started_at": started_at or utc_now(),
                "finished_at": utc_now(),
            },
        )

    def _optimize_started(
        self,
        prompt: str,
        options: Mapping[str, Any],
        tier: str,
        run_settings: Settings,
        run_seed: int,
        run_id: str,
        started_at: str,
        started_perf: float,
    ) -> OptimizeResult:
        self._stage("diagnosing")
        diagnosis = self._diagnose(prompt)
        diagnosis_payload = (
            diagnosis.as_dict()
            if diagnosis is not None
            else {"confirmed_gaps": [], "problem_sentences": []}
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
                    "tier": tier,
                    "options": dict(options),
                    "diagnosis": diagnosis_payload,
                },
            )
            result = self._needs_input_result(
                run_id, state, tier, started_at, started_perf
            )
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
            ),
            tier,
            prior_failures=options.get("prior_round_failures", ()),
        )

    def _round_executor(self, context: _RunContext) -> RoundRunner:
        def execute(request: RoundRequest) -> RoundOutcome:
            self._round = {
                "round": request.round_number,
                "max_rounds": request.max_rounds,
            }
            plan = RoundPlan(
                prompt=context.prompt,
                working_prompt=_prompt_with_assumptions(
                    context.prompt, context.assumptions
                ),
                run_id=context.run_id,
                tier=request.tier,
                seed=context.seed,
                diagnosis=context.diagnosis,
                assumptions=context.assumptions,
                settings=context.settings,
                faithfulness_threshold=self.faithfulness_threshold,
                writer_instruction_version=self.writer_instruction_version,
                prior_failures=tuple(request.prior_failures),
            )
            return run_round(self.gateway, plan, on_stage=self._stage)

        return execute

    def _run_rounds(
        self,
        context: _RunContext,
        tier: str,
        *,
        prior_failures: Any = (),
    ) -> OptimizeResult:
        repeated = self.repeat.run(
            run_id=context.run_id,
            prompt=context.prompt,
            tier=tier,
            execute_round=self._round_executor(context),
            initial_failures=prior_failures,
        )
        return cast(OptimizeResult, repeated.as_payload())

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
        )
        active_rubric_version = (
            rubric.version_id if rubric is not None else "default-v1"
        )
        report = Diagnoser(
            self.gateway,
            rubric=diagnosis_rubric,
            decision_policy=self.decision_policy,
            rubric_version=active_rubric_version,
        ).diagnose(prompt)
        calibration_evidence = dict(report.calibration or {})
        if rubric is None:
            return report
        questions = [
            {
                "key": f"rubric:{item.question_id}",
                "model": self.config.judge_model,
                "type": item.response_type,
                "query": item.text,
                "state": {"prompt": prompt},
            }
            for item in rubric.questions
        ]
        if not questions:
            return replace(
                report,
                rubric_version=rubric.version_id,
                calibration=calibration_evidence or None,
            )
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
            if not isinstance(snapshot, str) or not snapshot:
                snapshot = getattr(self.gateway, "jev_model", None)
            from .evaluation.calibration import runtime_question_identity

            identity = runtime_question_identity(
                f"rubric:{item.question_id}",
                questions[index],
                family="rubric",
                rubric_version=rubric.version_id,
                snapshot=snapshot if isinstance(snapshot, str) else None,
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
        tier = str((metadata or {}).get("tier", "standard"))
        run_settings = self._run_settings((metadata or {}).get("options", {}), tier)
        assumptions = state.get("assumptions", [])
        context = _RunContext(
            prompt,
            run_id,
            (metadata or {}).get(
                "diagnosis", {"confirmed_gaps": [], "problem_sentences": []}
            ),
            assumptions,
            run_settings,
            _run_seed(prompt, (metadata or {}).get("options", {}).get("seed")),
        )
        return self._run_rounds(context, tier)

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
            with self._progress_scope(progress):
                state = self._clarification.resume(run_id, answers)
        except UnknownRunError as exc:
            if self.store.get_run(run_id) is not None:
                raise RunNotPausedError(f"Run {run_id!r} is not paused") from exc
            raise RunNotFoundError(run_id) from exc
        return self._save_clarification_result(
            run_id, state, usage_before, started_perf
        )

    def skip_clarification(
        self, run_id: str, *, progress: ProgressCallback | None = None
    ) -> OptimizeResult:
        usage_before = self._usage_cost()
        started_perf = perf_counter()
        try:
            with self._progress_scope(progress):
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
        record = self.store.get_run(run_id)
        if record is not None:
            result["cost"] = _add_usage_delta(
                dict(record.get("cost") or {}), usage_before, self._usage_cost()
            )
            evidence = self._training_evidence(result, record)
            self._attach_jev_evidence(cast(OptimizeResult, result), evidence)
            self.store.save_run(
                {
                    **record,
                    "result": result,
                    "cost": result.get("cost", record.get("cost", {})),
                    "timing": result.get("timing", record.get("timing", {})),
                    **evidence,
                }
            )
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
            final_prompt, key, old_value, value, previous.get("source")
        )
        if updated_prompt is None:
            try:
                response = self.gateway.chat(
                    self.config.writer_model,
                    writer_messages(
                        "Revise only the stated assumption in the final prompt. Preserve all other wording and return only the revised prompt.",
                        {
                            "original_prompt": record["prompt"],
                            "final_prompt": final_prompt,
                            "assumption": {
                                "key": key,
                                "previous": old_value,
                                "corrected": value,
                            },
                        },
                    ),
                    role="writer",
                    run_id=run_id,
                )
                updated_prompt = completion_text(response).strip()
            except Exception as exc:
                raise RuntimeError("assumption edit could not be applied") from exc
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

    def start_deep_pass(
        self, run_id: str, *, progress: ProgressCallback | None = None
    ) -> OptimizeResult:
        started_perf = perf_counter()
        with self._progress_scope(progress):
            result = self._start_deep_pass(run_id)
        result["timing"] = _finished_timing(result.get("timing"), started_perf)
        return result

    def _start_deep_pass(self, run_id: str) -> OptimizeResult:
        record = self.store.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        detail = self.history.get_run(run_id)
        assert detail is not None
        self.gateway.new_run(run_id)
        prior_cost = dict(record.get("cost") or {})
        deep_options = dict(record.get("options") or {})
        overrides = dict(deep_options.get("model_overrides") or {})
        if "weak" in overrides or "weak_models" in overrides:
            prior_weak = overrides.get("weak", overrides.get("weak_models"))
            chosen = [prior_weak] if isinstance(prior_weak, str) else list(prior_weak)
            base_slots = max(0, 3 - len(set(chosen)))
            base_fill = [model for model in DEFAULT_WEAK_PANEL if model not in chosen][
                :base_slots
            ]
            deep_extras = DEFAULT_DEEP_WEAK_PANEL[len(DEFAULT_WEAK_PANEL) :]
            overrides["weak"] = list(
                dict.fromkeys((*chosen, *base_fill, *deep_extras, *DEFAULT_WEAK_PANEL))
            )[:5]
            overrides.pop("weak_models", None)
            deep_options["model_overrides"] = overrides
        deep_options["tier"] = "deep"
        run_settings = self._run_settings(deep_options, "deep")
        prior_report = dict((record.get("result") or {}).get("report") or {})
        repeated = self.repeat.deep_pass(
            detail,
            self._round_executor(
                _RunContext(
                    str(record["prompt"]),
                    run_id,
                    prior_report.get("diagnosis", {}),
                    prior_report.get("assumptions", []),
                    run_settings,
                    _run_seed(
                        str(record["prompt"]), (record.get("options") or {}).get("seed")
                    ),
                )
            ),
        )
        result = cast(OptimizeResult, repeated.as_payload())
        result["cost"] = cast(
            CostBreakdown,
            _add_usage_delta(prior_cost, {"total": 0.0}, self._usage_cost()),
        )
        evidence = self._training_evidence(result, record)
        self._attach_jev_evidence(result, evidence)
        self.store.save_run(
            {
                **record,
                "result": result,
                "tier": "deep",
                "options": deep_options,
                "cost": result["cost"],
                **evidence,
            }
        )
        return result

    def _usage_cost(self) -> CostBreakdown:
        return cast(CostBreakdown, self.gateway.usage_report())

    def _needs_input_result(
        self,
        run_id: str,
        state: Mapping[str, Any],
        tier: str,
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
                "offer_deep": False,
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
        tier: str,
        options: dict[str, Any],
        result: OptimizeResult,
        created_at: str,
    ) -> None:
        evidence = self._training_evidence(result)
        self._attach_jev_evidence(result, evidence)
        self.store.save_run(
            {
                "run_id": result["run_id"],
                "created_at": created_at,
                "prompt": prompt,
                "tier": tier,
                "options": {**options, "tier": tier},
                "result": result,
                "cost": result["cost"],
                "timing": result["timing"],
                **evidence,
            }
        )

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
    prompt: str, key: str, old_value: str, value: str, source: object = None
) -> str | None:
    label = _clarification_label(key, source)
    line = f"{label}: {value}"
    lines = prompt.splitlines()
    for index, existing in enumerate(lines):
        if existing.strip().casefold() == f"{label.casefold()}: {old_value.casefold()}":
            lines[index] = line
            return "\n".join(lines)
    if old_value and prompt.count(old_value) == 1:
        return prompt.replace(old_value, value, 1)
    return None


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
