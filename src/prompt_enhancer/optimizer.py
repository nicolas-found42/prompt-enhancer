"""Public local optimization engine.

The facade is deliberately dependency-injected. Production can provide a
ModelGateway; tests and local demos can provide ScriptedGateway or ReplayGateway.
The public result always includes the original prompt, evidence, cost, timing,
and a durable run identifier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, cast

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
    DecisionGateway,
    Diagnoser,
    DiagnosisReport,
    DiagnosisRubric,
    GapImpact,
)
from .fidelity import check_candidate_fidelity
from .gateway import (
    GatewayConfig,
    HttpTransport,
    ModelGateway,
    ProviderError,
    ScriptedGateway,
)
from .grading import grade_panel_with_jev
from .history import RunHistory
from .jev import ChoiceDecision, NoulDecision, parse_decision
from .models import CostBreakdown, OptimizeResult, new_run_id, normalize_tier, utc_now
from .repeat import RepeatCoordinator, RoundRequest
from .rewrite import CandidateWriter
from .rewrite import ModelGateway as RewriteGateway
from .rewrite import _text as completion_text
from .rubric_revisions import SQLiteRubricStore
from .runner import PanelResult, run_candidates
from .selector import rank_candidates
from .settings import ModelDefaults, SettingsStore
from .store import RunStore
from .strategies import STRATEGY_LIBRARY, RewriteStrategy, search_strategies
from .strong_check import StrongCheckPolicy
from .success_tests import CompletionGateway, SuccessTestCompiler


class RunNotFoundError(KeyError):
    """Raised when a caller resumes or edits an unknown run."""


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
        gateway: Any | None = None,
        store: RunStore | None = None,
        config: Settings | None = None,
        rubric_store: SQLiteRubricStore | None = None,
        diagnosis_rubric: DiagnosisRubric = DEFAULT_RUBRIC,
    ) -> None:
        self.store = store or RunStore()
        self.config = config or Settings.from_env()
        self.diagnosis_rubric = diagnosis_rubric
        self.settings_store = (
            SettingsStore(
                Path(self.store.path).with_suffix(".settings.json"),
                defaults=ModelDefaults(
                    writer=self.config.writer_model,
                    strong=self.config.strong_check_model,
                    weak=self.config.weak_models,
                ),
            )
            if self.store.path != ":memory:" else None
        )
        if self.settings_store is not None:
            defaults = self.settings_store.load().defaults
            self.config.writer_model = defaults.writer
            self.config.strong_check_model = defaults.strong
            self.config.weak_models = defaults.weak
        self.gateway: Any = gateway if gateway is not None else self._default_gateway()
        self.history = RunHistory(self.store)
        self.rubric_store = rubric_store or (SQLiteRubricStore(self.store.path) if self.store.path != ":memory:" else None)
        self.repeat = RepeatCoordinator()
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
        if not isinstance(weak, (list, tuple)) or len(weak) < 3 or any(not isinstance(item, str) or not item.strip() for item in weak) or len(set(weak)) != len(weak):
            raise ValueError("weak_models must contain at least three distinct model IDs")
        selected = ModelDefaults(writer=writer.strip(), strong=strong.strip(), weak=tuple(weak))
        if self.settings_store is not None:
            self.settings_store.save(selected)
        self.config.writer_model = selected.writer
        self.config.strong_check_model = selected.strong
        self.config.weak_models = selected.weak
        return self.get_model_settings()

    def _default_gateway(self) -> Any:
        if not self.config.openrouter_api_key or not self.config.opencode_go_key:
            return ScriptedGateway()
        gateway_config = GatewayConfig.from_env()
        gateway_config.openrouter_api_key = self.config.openrouter_api_key
        gateway_config.go_api_key = self.config.opencode_go_key
        transport = HttpTransport()
        catalog = LiveModelCatalog(
            transport,
            go_url=gateway_config.go_models_url or f"{gateway_config.go_base_url}/models",
            openrouter_url=gateway_config.openrouter_models_url or f"{gateway_config.openrouter_base_url}/models",
            go_api_key=gateway_config.go_api_key,
            openrouter_api_key=gateway_config.openrouter_api_key,
        )
        return ModelGateway(transport, config=gateway_config, catalog=catalog)

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
        writer = overrides.get("writer", overrides.get("writer_model", self.config.writer_model))
        strong = overrides.get("strong", overrides.get("strong_check_model", self.config.strong_check_model))
        selected_weak = overrides.get("weak", overrides.get("weak_models"))
        if selected_weak is None:
            selected_weak = self.config.weak_models
            if tier == "deep":
                selected_weak = tuple(dict.fromkeys((*selected_weak, *DEFAULT_DEEP_WEAK_PANEL)))
        if not isinstance(writer, str) or not writer or not isinstance(strong, str) or not strong:
            raise ValueError("writer and strong model overrides must be model IDs")
        if isinstance(selected_weak, str):
            selected_weak = (selected_weak,)
        if not isinstance(selected_weak, (list, tuple)) or not selected_weak or any(not isinstance(item, str) or not item for item in selected_weak):
            raise ValueError("weak model overrides must be a non-empty list")
        count = {"fast": 2, "standard": 3, "deep": 5}[tier]
        if len(set(selected_weak[:count])) != count:
            raise ValueError(f"weak panel for {tier} requires {count} distinct models")
        return replace(self.config, writer_model=writer, strong_check_model=strong, weak_models=tuple(selected_weak[:count]))

    @staticmethod
    def _model_roles(settings: Settings) -> dict[str, Any]:
        return {
            "judge": settings.judge_model,
            "writer": settings.writer_model,
            "strong": settings.strong_check_model,
            "weak": list(settings.weak_models),
        }

    def optimize(
        self, prompt: str, options: dict[str, Any] | None = None
    ) -> OptimizeResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        supplied_options = _safe_options(options or {})
        tier = normalize_tier(supplied_options.pop("tier", "standard"))
        run_settings = self._run_settings(supplied_options, tier)
        run_seed = _run_seed(prompt, supplied_options.get("seed"))
        run_id = new_run_id()
        started_perf = perf_counter()
        started_at = utc_now()
        self.gateway.new_run(run_id)

        try:
            result = self._optimize_started(prompt, supplied_options, tier, run_settings, run_seed, run_id, started_at, started_perf)
        except ProviderError as exc:
            result = OptimizeResult(
                status="failed", run_id=run_id, final_prompt=prompt, original_kept=True,
                report={"status": "failed", "summary": "A model provider was unavailable; the original prompt was saved.", "error": str(exc), "diagnosis": {"confirmed_gaps": [], "problem_sentences": []}, "assumptions": []},
                cost=self._usage_cost(), timing={"total_ms": 0, "started_at": started_at, "finished_at": utc_now()},
            )
        result["timing"]["total_ms"] = max(0, round((perf_counter() - started_perf) * 1000))
        result["timing"]["started_at"] = started_at
        result["timing"]["finished_at"] = utc_now()
        self._save_result(prompt, tier, supplied_options, result, started_at)
        return result

    def _optimize_started(
        self, prompt: str, options: Mapping[str, Any], tier: str, run_settings: Settings,
        run_seed: int, run_id: str, started_at: str, started_perf: float,
    ) -> OptimizeResult:
        diagnosis = self._diagnose(prompt)
        diagnosis_payload = diagnosis.as_dict() if diagnosis is not None else {"confirmed_gaps": [], "problem_sentences": []}
        gaps = tuple(diagnosis.confirmed_gaps) if diagnosis is not None else ()
        plan = Clarifier(
            self.gateway,
            writer_model=run_settings.writer_model,
            judge_model=run_settings.judge_model,
        ).plan(prompt, gaps, allow_clarification=options.get("clarification_allowed", True), run_id=run_id)
        if plan.questions:
            state = self._clarification.start(
                run_id,
                prompt,
                plan,
                metadata={"tier": tier, "options": dict(options), "diagnosis": diagnosis_payload},
            )
            result = self._needs_input_result(run_id, state, tier, started_at, started_perf)
            result["report"]["models"] = self._model_roles(run_settings)
            result["report"]["diagnosis"] = diagnosis_payload
            return result
        return self._run_rounds(
            _RunContext(prompt, run_id, diagnosis_payload, [item.as_dict() for item in plan.assumptions], run_settings, run_seed),
            tier,
            prior_failures=options.get("prior_round_failures", ()),
        )

    def _round_executor(self, context: _RunContext) -> Any:
        def execute(request: RoundRequest) -> Mapping[str, Any]:
            result = self._run_optimization(context, request.tier.value, request.prior_failures)
            report = result["report"]
            failures = [
                {
                    "candidate_id": candidate["candidate_id"],
                    "strategy": candidate.get("strategy"),
                    "candidate_prompt": candidate.get("text"),
                    "reasons": candidate.get("rejection_reasons", []),
                    "weak_pass_rates": candidate.get("grade", {}).get("per_model", {}),
                }
                for candidate in report.get("candidates", [])
                if not candidate.get("selected")
            ]
            return {
                **result,
                "candidate_failures": failures,
                "selected_candidate_id": report.get("selection_evidence", {}).get("selected_candidate_id"),
                "evidence": {
                    key: report[key]
                    for key in ("diagnosis", "tests", "candidates", "per_model", "strong_check", "selection_evidence", "strategies")
                    if key in report
                },
                "continue_rounds": bool(failures) and result["original_kept"],
            }
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

    @staticmethod
    def _unchanged_report(
        *, status: str, summary: str, models: Mapping[str, Any],
        diagnosis: Mapping[str, Any], tests: Any, assumptions: Any,
        original: str, working: str, tier: str,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "models": dict(models),
            "summary": summary,
            "diagnosis": dict(diagnosis),
            "tests": tests,
            "candidates": [],
            "per_model": {},
            "assumptions": list(assumptions),
            "diff": _diff(original, working),
            "offer_deep": tier != "deep",
            "history": [],
        }

    def _run_optimization(
        self,
        context: _RunContext,
        tier: str,
        prior_failures: Any = (),
    ) -> OptimizeResult:
        prompt = context.prompt
        run_id = context.run_id
        diagnosis_payload = context.diagnosis
        assumptions = context.assumptions
        run_seed = context.seed
        selected_settings = context.settings
        models = self._model_roles(selected_settings)
        working_prompt = _prompt_with_assumptions(prompt, assumptions)
        try:
            compiled = SuccessTestCompiler(
                cast(CompletionGateway, self.gateway),
                writer_model=selected_settings.writer_model,
            ).compile(working_prompt)
        except (ValueError, TypeError) as exc:
            raise ProviderError("writer", selected_settings.writer_model, None, "invalid success-test response") from exc
        test_payload = [asdict(test) for test in compiled.tests]

        confirmed_gaps = diagnosis_payload.get("confirmed_gaps", [])
        if not test_payload or not confirmed_gaps:
            no_gaps = not confirmed_gaps
            summary = (
                "No confirmed gaps were found; the original request was returned unchanged."
                if no_gaps and test_payload else
                "No confirmed gaps were found; the original request was returned unchanged. No faithful success tests were established."
                if no_gaps else
                "No faithful success tests were established; the original request and any confirmed clarifications were returned without claiming an improvement."
            )
            report = self._unchanged_report(
                status="no_change" if no_gaps and test_payload else "unverified",
                summary=summary, models=models, diagnosis=diagnosis_payload,
                tests=test_payload, assumptions=assumptions, original=prompt,
                working=working_prompt, tier=tier,
            )
            return self._result(run_id, working_prompt, working_prompt == prompt, report)

        strategy_choice = self.gateway.decide(
            {"model": selected_settings.judge_model, "key": "strategy_choice", "type": "choice", "query": "Which rewrite strategy best addresses the diagnosed weakness?", "criteria": {**{item.name: item.description for item in STRATEGY_LIBRARY}, "none": "No rewrite strategy is suitable."}, "state": {"prompt": working_prompt, "diagnosis": diagnosis_payload, "prior_failures": list(prior_failures)}},
            role="judge", run_id=run_id,
        )
        parsed_strategy = parse_decision(strategy_choice)
        preferred_strategy = parsed_strategy.selected if isinstance(parsed_strategy, ChoiceDecision) else None

        def recheck_strategy(strategy: RewriteStrategy) -> dict[str, bool]:
            answer = self.gateway.decide(
                {"model": selected_settings.judge_model, "key": f"strategy_recheck:{strategy.name}", "type": "noul", "query": "Is this strategy appropriate for the prompt and diagnosed weakness without inventing requirements?", "state": {"prompt": working_prompt, "diagnosis": diagnosis_payload, "strategy": strategy.to_dict()}},
                role="judge", run_id=run_id,
            )
            decision = parse_decision(answer)
            return {"eligible": isinstance(decision, NoulDecision) and decision.probability >= 0.8}

        search = search_strategies(
            working_prompt,
            diagnosis_payload,
            tier,
            writer=CandidateWriter(
                cast(RewriteGateway, self.gateway),
                writer_model=selected_settings.writer_model,
            ),
            previous_failures=prior_failures,
            recheck=recheck_strategy,
            priority_strategy=preferred_strategy,
        )
        candidates = list(search.candidates)
        if not candidates:
            return self._result(
                run_id,
                working_prompt,
                working_prompt == prompt,
                self._unchanged_report(
                    status="no_change", summary="No candidate strategy was selected.",
                    models=models, diagnosis=diagnosis_payload, tests=test_payload,
                    assumptions=assumptions, original=prompt, working=working_prompt,
                    tier=tier,
                ),
            )

        panel = run_candidates(
            candidates,
            selected_settings.weak_models,
            self.gateway,
            original=working_prompt,
            budget=tier,
            run_seed=run_seed,
            run_id=run_id,
        )
        panel_grades, grading_answers = grade_panel_with_jev(
            panel, test_payload, self.gateway,
            judge_model=selected_settings.judge_model, run_id=run_id,
        )
        grades = [
            (candidate, panel_grades[candidate.candidate_id])
            for candidate in candidates
        ]
        original_grade = panel_grades["original"]
        ranking_candidates = [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text,
                "prompt": candidate.text,
                "strategy": candidate.strategy.name,
                "strategy_kind": candidate.strategy.kind,
                "grade": grade,
                "fidelity": (fidelity := check_candidate_fidelity(
                    self.gateway, working_prompt, candidate.text, diagnosis_payload,
                    candidate.strategy.name, run_id=run_id,
                    judge_model=selected_settings.judge_model,
                )).to_dict(),
                "eligible": fidelity.passed,
                "rejection_reasons": [] if fidelity.passed else ["candidate failed fidelity checks"],
            }
            for candidate, grade in grades
        ]
        strong = StrongCheckPolicy(selected_settings.strong_check_model).check(
            working_prompt,
            [candidate for candidate in ranking_candidates if candidate["eligible"]],
            test_payload,
            lambda candidate_prompt, _tests: self._strong_score(candidate_prompt, _tests, run_id, selected_settings),
        )
        ranking = rank_candidates(
            {"candidate_id": "original", "text": working_prompt, "grade": original_grade},
            ranking_candidates,
            original_grade=original_grade,
            strong_check=strong,
        )
        final_prompt = ranking.final_prompt
        original_kept = final_prompt == prompt
        report = {
            "status": "no_change" if original_kept else ("clarified" if ranking.original_kept else "improved"),
            "models": models,
            "summary": "No candidate beat the original." if original_kept else ("Clarifications were included; no candidate beat the clarified prompt." if ranking.original_kept else "Candidate selected after verification."),
            "diagnosis": diagnosis_payload,
            "tests": test_payload,
            "jev_answers": grading_answers,
            "candidates": [item.to_dict() for item in ranking.ranked],
            "per_model": {"ranking": ranking.to_dict(), "panel": panel.to_dict()},
            "assumptions": list(assumptions),
            "diff": _diff(prompt, final_prompt),
            "selection_evidence": ranking.to_dict(),
            "strong_check": strong.to_dict(),
            "strategies": search.to_dict(),
            "offer_deep": tier != "deep" and original_kept,
            "history": [],
        }
        return self._result(run_id, final_prompt, original_kept, report)

    def _diagnose(self, prompt: str) -> DiagnosisReport | None:
        rubric = None
        if self.rubric_store is not None:
            try:
                rubric = self.rubric_store.active_rubric()
            except RuntimeError:
                pass
        suppressed = (
            {item.question_id for item in rubric.questions} | set(rubric.disabled_default_question_ids)
            if rubric is not None else set()
        )
        diagnosis_rubric = replace(
            self.diagnosis_rubric,
            task_types=tuple(
                replace(task, checklist=tuple(item for item in task.checklist if item.key not in suppressed))
                for task in self.diagnosis_rubric.task_types
            ),
        )
        report = Diagnoser(cast(DecisionGateway, self.gateway), rubric=diagnosis_rubric).diagnose(prompt)
        if rubric is None:
            return report
        questions = [
            {"key": f"rubric:{item.question_id}", "model": self.config.judge_model, "type": item.response_type, "query": item.text, "state": {"prompt": prompt}}
            for item in rubric.questions
        ]
        if not questions:
            return replace(report, rubric_version=rubric.version_id)
        responses = self.gateway.jev_batch(questions)
        gaps = list(report.confirmed_gaps)
        default_impacts = {item.key: item.impact for task in DEFAULT_RUBRIC.task_types for item in task.checklist}
        for item, response in zip(rubric.questions, responses, strict=True):
            decision = parse_decision(response)
            if not isinstance(decision, NoulDecision):
                continue
            missing = decision.probability if item.missing_when == "yes" else 1.0 - decision.probability
            if missing >= item.threshold and decision.confidence >= diagnosis_rubric.confidence_threshold:
                gaps.append(ConfirmedGap(item.question_id, item.text, default_impacts.get(item.question_id, GapImpact.MEDIUM), missing, decision.confidence, item.threshold))
        return replace(report, confirmed_gaps=tuple(gaps), rubric_version=rubric.version_id)

    def _strong_score(self, prompt: str, tests: Any, run_id: str, settings: Settings) -> float:
        if not tests:
            raise ValueError("strong check requires success tests")
        output = completion_text(self.gateway.chat(
            settings.strong_check_model,
            [{"role": "user", "content": prompt}],
            role="strong_check",
            run_id=run_id,
        ))
        grades, _ = grade_panel_with_jev(
            [PanelResult("strong", settings.strong_check_model, 0, 0, output, prompt)],
            tests,
            self.gateway,
            judge_model=settings.judge_model,
            run_id=run_id,
        )
        return grades["strong"].sample_scores[0]
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
                    "question": "Does the updated prompt preserve the user's original meaning without contradictory instructions?",
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
        metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
        tier = str((metadata or {}).get("tier", "standard"))
        run_settings = self._run_settings((metadata or {}).get("options", {}), tier)
        assumptions = state.get("assumptions", [])
        context = _RunContext(
            prompt, run_id,
            (metadata or {}).get("diagnosis", {"confirmed_gaps": [], "problem_sentences": []}),
            assumptions, run_settings,
            _run_seed(prompt, (metadata or {}).get("options", {}).get("seed")),
        )
        return self._run_rounds(context, tier)

    def resume(self, run_id: str, answers: dict[str, Any]) -> OptimizeResult:
        usage_before = self._usage_cost()
        try:
            state = self._clarification.resume(run_id, answers)
        except UnknownRunError as exc:
            if self.store.get_run(run_id) is not None:
                raise RunNotPausedError(f"Run {run_id!r} is not paused") from exc
            raise RunNotFoundError(run_id) from exc
        return self._save_clarification_result(run_id, state, usage_before)

    def skip_clarification(self, run_id: str) -> OptimizeResult:
        usage_before = self._usage_cost()
        try:
            state = self._clarification.skip(run_id)
        except UnknownRunError as exc:
            if self.store.get_run(run_id) is not None:
                raise RunNotPausedError(f"Run {run_id!r} is not paused") from exc
            raise RunNotFoundError(run_id) from exc
        return self._save_clarification_result(run_id, state, usage_before)

    def _save_clarification_result(self, run_id: str, state: Mapping[str, Any], usage_before: Mapping[str, Any]) -> OptimizeResult:
        result = state.get("result")
        if not isinstance(result, Mapping):
            raise RunNotFoundError(run_id)
        record = self.store.get_run(run_id)
        if record is not None:
            result = dict(result)
            result["cost"] = _add_usage_delta(dict(record.get("cost") or {}), usage_before, self._usage_cost())
            evidence = self._training_evidence(result, record)
            result["report"]["jev_answers"] = evidence["jev_answers"]
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

    def update_assumption(self, run_id: str, assumption: dict[str, Any]) -> OptimizeResult:
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
        assumptions = [dict(item) for item in report.get("assumptions", []) if isinstance(item, Mapping)]
        previous = next((item for item in assumptions if str(item.get("key", "")) == key), None)
        if previous is None:
            raise ValueError("assumption is not part of this run")
        old_value = str(previous.get("value", ""))
        final_prompt = str(result.get("final_prompt") or record.get("prompt") or "")
        usage_before = self._usage_cost()
        updated_prompt = _apply_assumption(final_prompt, key, old_value, value)
        if updated_prompt is None:
            try:
                response = self.gateway.complete(
                    {
                        "model": self.config.writer_model,
                        "role": "writer",
                        "state": {
                            "original_prompt": record["prompt"],
                            "final_prompt": final_prompt,
                            "assumption": {"key": key, "previous": old_value, "corrected": value},
                        },
                        "instructions": "Revise only the stated assumption in the final prompt. Preserve all other wording and return only the revised prompt.",
                    },
                    run_id=run_id,
                )
                updated_prompt = completion_text(response).strip()
            except Exception as exc:
                raise RuntimeError("assumption edit could not be applied") from exc
        if not updated_prompt or updated_prompt == final_prompt:
            raise ValueError("assumption edit did not update the final prompt")
        check = self._assumption_meaning_check(str(record["prompt"]), updated_prompt, run_id)
        if check.get("passed") is False:
            raise ValueError("edited assumption did not preserve the prompt's meaning")

        replacement = {"key": key, "value": value, "source": "user_edit"}
        for index, item in enumerate(assumptions):
            if str(item.get("key", "")) == key:
                assumptions[index] = replacement
                break
        report["assumptions"] = assumptions
        report["status"] = "edited"
        report["summary"] = "Assumption corrected; performance evidence is from the original optimization."
        report["diff"] = _diff(str(record["prompt"]), updated_prompt)
        report["final_prompt"] = updated_prompt
        report["assumption_check"] = check
        report["assumption_edit"] = {"key": key, "previous": old_value, "corrected": value, "previous_final_prompt": final_prompt}
        result["final_prompt"] = updated_prompt
        result["original_kept"] = updated_prompt == str(record["prompt"])
        result["report"] = report
        result["cost"] = cast(CostBreakdown, _add_usage_delta(dict(result.get("cost") or record.get("cost") or {}), usage_before, self._usage_cost()))
        self.store.save_run({**record, "result": result, "cost": result["cost"]})
        return result

    def start_deep_pass(self, run_id: str) -> OptimizeResult:
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
            base_fill = [model for model in DEFAULT_WEAK_PANEL if model not in chosen][:base_slots]
            deep_extras = DEFAULT_DEEP_WEAK_PANEL[len(DEFAULT_WEAK_PANEL):]
            overrides["weak"] = list(dict.fromkeys((*chosen, *base_fill, *deep_extras, *DEFAULT_WEAK_PANEL)))[:5]
            overrides.pop("weak_models", None)
            deep_options["model_overrides"] = overrides
        deep_options["tier"] = "deep"
        run_settings = self._run_settings(deep_options, "deep")
        prior_report = dict((record.get("result") or {}).get("report") or {})
        repeated = self.repeat.deep_pass(
            detail,
            self._round_executor(_RunContext(
                str(record["prompt"]), run_id,
                prior_report.get("diagnosis", {}),
                prior_report.get("assumptions", []),
                run_settings,
                _run_seed(str(record["prompt"]), (record.get("options") or {}).get("seed")),
            )),
        )
        result = cast(OptimizeResult, repeated.as_payload())
        result["cost"] = cast(CostBreakdown, _add_usage_delta(prior_cost, {"total": 0.0}, self._usage_cost()))
        evidence = self._training_evidence(result, record)
        result["report"]["jev_answers"] = evidence["jev_answers"]
        self.store.save_run({**record, "result": result, "tier": "deep", "options": deep_options, "cost": result["cost"], **evidence})
        return result

    def _result(
        self,
        run_id: str,
        final_prompt: str,
        original_kept: bool,
        report: dict[str, Any],
    ) -> OptimizeResult:
        return OptimizeResult(
            status="completed",
            run_id=run_id,
            final_prompt=final_prompt,
            original_kept=original_kept,
            report=report,
            cost=self._usage_cost(),
            timing={"total_ms": 0, "started_at": utc_now(), "finished_at": utc_now()},
        )

    def _usage_cost(self) -> CostBreakdown:
        report = getattr(
            self.gateway, "usage_report", lambda: {"total": 0.0, "by_role": {}}
        )()
        return cast(CostBreakdown, report)

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
        result["report"]["jev_answers"] = evidence["jev_answers"]
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

    def _training_evidence(self, result: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
        report = result.get("report") if isinstance(result.get("report"), Mapping) else {}
        selection = report.get("selection_evidence") if isinstance(report, Mapping) else {}
        original_score = selection.get("original_score") if isinstance(selection, Mapping) else None
        old_answers = list((previous or {}).get("jev_answers") or [])
        current_answers = list(getattr(self.gateway, "decision_log", []))
        answers = current_answers if current_answers[:len(old_answers)] == old_answers else old_answers + current_answers
        evidence: dict[str, Any] = {"jev_answers": answers}
        if isinstance(original_score, Mapping):
            evidence["original_weak_panel"] = original_score
            evidence["score_summaries"] = {"original": original_score}
        return evidence


def _apply_assumption(prompt: str, key: str, old_value: str, value: str) -> str | None:
    line = f"{key}: {value}"
    lines = prompt.splitlines()
    for index, existing in enumerate(lines):
        if existing.strip().casefold() == f"{key.casefold()}: {old_value.casefold()}":
            lines[index] = line
            return "\n".join(lines)
    if old_value and prompt.count(old_value) == 1:
        return prompt.replace(old_value, value, 1)
    return None


def _prompt_with_assumptions(prompt: str, assumptions: Any) -> str:
    lines = []
    for item in assumptions:
        if not isinstance(item, Mapping) or item.get("source") not in {"answer", "inferred"}:
            continue
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if key and value:
            lines.append(f"{key}: {value}")
    if not lines:
        return prompt
    return prompt.rstrip() + "\n\nClarifications:\n" + "\n".join(lines)


def _add_usage_delta(previous: dict[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(previous)
    delta = max(0.0, float(after.get("total", 0.0)) - float(before.get("total", 0.0)))
    updated["total"] = float(previous.get("total", 0.0)) + delta
    previous_roles = previous.get("cost_by_role", previous.get("by_role", {}))
    roles = dict(previous_roles) if isinstance(previous_roles, Mapping) and all(isinstance(v, (int, float)) for v in previous_roles.values()) else {}
    before_roles = before.get("cost_by_role", {})
    after_roles = after.get("cost_by_role", {})
    if isinstance(before_roles, Mapping) and isinstance(after_roles, Mapping):
        for role, amount in after_roles.items():
            roles[str(role)] = float(roles.get(str(role), 0.0)) + max(0.0, float(amount) - float(before_roles.get(role, 0.0)))
    updated["cost_by_role"] = roles
    return updated

def _safe_options(options: Mapping[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): clean(v) for k, v in value.items() if not any(x in str(k).lower() for x in ("key", "token", "secret", "password", "authorization"))}
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


def _diff(original: str, final: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(original.splitlines(True), final.splitlines(True), fromfile="original", tofile="final"))
