"""Public local optimization engine.

The facade is deliberately dependency-injected. Production can provide a
ModelGateway; tests and local demos can provide ScriptedGateway or ReplayGateway.
The public result always includes the original prompt, evidence, cost, timing,
and a durable run identifier.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from time import perf_counter
from typing import Any, cast

from .clarification import (
    ClarificationService,
    GapAssessment,
    InMemoryClarificationRepository,
    SQLiteClarificationRepository,
    build_plan,
)
from .config import Settings
from .diagnosis import DecisionGateway, Diagnoser, DiagnosisReport
from .gateway import ScriptedGateway
from .grading import grade_candidate
from .history import RunHistory
from .models import CostBreakdown, OptimizeResult, new_run_id, normalize_tier, utc_now
from .repeat import RepeatCoordinator
from .rewrite import CandidateWriter
from .rewrite import ModelGateway as RewriteGateway
from .runner import run_candidates
from .selector import rank_candidates
from .store import RunStore
from .strategies import search_strategies
from .strong_check import StrongCheckPolicy, eligible_candidates
from .success_tests import CompiledSuccessTests, CompletionGateway, SuccessTestCompiler


class RunNotFoundError(KeyError):
    """Raised when a caller resumes or edits an unknown run."""


class PromptOptimizer:
    """Single public engine entry point for optimization and run lifecycle."""

    def __init__(
        self,
        gateway: Any | None = None,
        store: RunStore | None = None,
        config: Settings | None = None,
    ) -> None:
        self.gateway: Any = gateway or ScriptedGateway()
        self.store = store or RunStore()
        self.config = config or Settings()
        self.history = RunHistory(self.store)
        self.repeat = RepeatCoordinator()
        self._clarification = ClarificationService(
            self._clarification_repository(),
            continuation=self._continue_clarification,
        )

    def _clarification_repository(self) -> Any:
        if self.store.path == ":memory:":
            return InMemoryClarificationRepository()
        return SQLiteClarificationRepository(self.store.path)

    def optimize(
        self, prompt: str, options: dict[str, Any] | None = None
    ) -> OptimizeResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        supplied_options = _safe_options(options or {})
        tier = normalize_tier(supplied_options.pop("tier", "standard"))
        run_id = new_run_id()
        started_perf = perf_counter()
        started_at = utc_now()
        self.gateway.new_run(run_id)

        diagnosis = self._diagnose(prompt)
        diagnosis_payload = diagnosis.as_dict() if diagnosis is not None else {"confirmed_gaps": [], "problem_sentences": []}
        gaps = tuple(diagnosis.confirmed_gaps) if diagnosis is not None else ()
        plan = build_plan(
            (
                GapAssessment(
                    id=gap.key,
                    label=gap.label,
                    impact=gap.impact.value,
                    present=False,
                    confidence=gap.confidence,
                    value=None,
                )
                for gap in gaps
            ),
            allow_clarification=supplied_options.get("clarification_allowed", True),
        )
        if plan.questions:
            state = self._clarification.start(
                run_id,
                prompt,
                plan,
                metadata={"tier": tier, "options": supplied_options},
            )
            result = self._needs_input_result(run_id, state, tier, started_at, started_perf)
            self._save_result(prompt, tier, supplied_options, result, started_at)
            return result

        result = self._run_optimization(
            prompt,
            run_id,
            tier,
            diagnosis_payload,
            tests=(),
            assumptions=[item.as_dict() for item in plan.assumptions],
            prior_failures=supplied_options.get("prior_round_failures", ()),
        )
        result["timing"]["total_ms"] = max(0, round((perf_counter() - started_perf) * 1000))
        result["timing"]["started_at"] = started_at
        result["timing"]["finished_at"] = utc_now()
        self._save_result(prompt, tier, supplied_options, result, started_at)
        return result

    def _run_optimization(
        self,
        prompt: str,
        run_id: str,
        tier: str,
        diagnosis_payload: Mapping[str, Any],
        *,
        tests: Any = (),
        assumptions: Any = (),
        prior_failures: Any = (),
    ) -> OptimizeResult:
        # Compile tests only when a writer seam is available. The compiler is
        # fail-open for malformed model output and keeps all prompt text in state.
        compiled = CompiledSuccessTests((), (), ())
        if self.gateway is not None:
            with suppress(Exception):
                compiled = SuccessTestCompiler(
                    cast(CompletionGateway, self.gateway),
                    writer_model=self.config.writer_model,
                ).compile(prompt)
        test_payload = [asdict(test) for test in compiled.tests]

        if not test_payload:
            report = {
                "status": "no_change",
                "summary": "Prompt is already clear; no rewrite was needed.",
                "diagnosis": diagnosis_payload,
                "tests": [],
                "candidates": [],
                "per_model": {},
                "assumptions": list(assumptions),
                "diff": [],
                "offer_deep": tier != "deep",
                "history": [],
            }
            return self._result(run_id, prompt, True, report)

        search = search_strategies(
            prompt,
            diagnosis_payload,
            tier,
            writer=CandidateWriter(
                cast(RewriteGateway, self.gateway),
                writer_model=self.config.writer_model,
            ),
            previous_failures=prior_failures,
        )
        candidates = list(search.candidates)
        if not candidates:
            return self._result(
                run_id,
                prompt,
                True,
                {
                    "status": "no_change",
                    "summary": "No candidate strategy was selected.",
                    "diagnosis": diagnosis_payload,
                    "tests": test_payload,
                    "candidates": [],
                    "per_model": {},
                    "assumptions": list(assumptions),
                    "diff": [],
                    "offer_deep": tier != "deep",
                    "history": [],
                },
            )

        panel = run_candidates(
            candidates,
            self.config.weak_models,
            self.gateway,
            original=prompt,
            budget=tier,
            run_seed=hash((run_id, prompt)) & 0x7FFFFFFF,
            run_id=run_id,
        )
        judge = lambda request: self.gateway.decide(
            {"model": self.config.judge_model, "state": request.to_dict(), "question": "Does this output pass the success test?"},
            role="judge",
            run_id=run_id,
        )
        grades = [
            (candidate, grade_candidate(candidate, panel, judge, tests=test_payload))
            for candidate in candidates
        ]
        original_grade = grade_candidate(
            {"candidate_id": "original", "text": prompt}, panel, judge, tests=test_payload
        )
        ranking_candidates = [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text,
                "strategy": candidate.strategy.name,
                "strategy_kind": candidate.strategy.kind,
                "grade": grade,
            }
            for candidate, grade in grades
        ]
        strong = StrongCheckPolicy(self.config.strong_check_model).check(
            prompt,
            ranking_candidates,
            test_payload,
            lambda candidate_prompt, _tests: self._strong_score(candidate_prompt, _tests, run_id),
        )
        ranking_candidates = eligible_candidates(ranking_candidates, strong)
        ranking = rank_candidates(
            {"candidate_id": "original", "text": prompt, "grade": original_grade},
            ranking_candidates,
            original_grade=original_grade,
            strong_check=strong,
        )
        final_prompt = ranking.final_prompt
        original_kept = ranking.original_kept
        report = {
            "status": "no_change" if original_kept else "improved",
            "summary": "No candidate beat the original." if original_kept else "Candidate selected after verification.",
            "diagnosis": diagnosis_payload,
            "tests": test_payload,
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
        with suppress(Exception):
            return Diagnoser(cast(DecisionGateway, self.gateway)).diagnose(prompt)
        return None

    def _strong_score(self, prompt: str, tests: Any, run_id: str) -> float:
        output: Any = None
        with suppress(Exception):
            output = self.gateway.chat(
                self.config.strong_check_model,
                [{"role": "user", "content": prompt}],
                role="strong_check",
                run_id=run_id,
            )
        if output is None:
            return 0.0
        if not tests:
            return 0.0
        total = 0.0
        for test in tests:
            with suppress(Exception):
                value = self.gateway.decide(
                    {
                        "model": self.config.judge_model,
                        "state": {"prompt": prompt, "output": output, "test": test},
                        "question": test.get("question", "Did the output pass?"),
                    },
                    role="judge",
                    run_id=run_id,
                )
                if isinstance(value, (int, float)):
                    total += float(value)
                elif isinstance(value, Mapping):
                    total += float(
                        value.get("probability", value.get("score", 0.0)) or 0.0
                    )
        return total / len(tests)

    def _continue_clarification(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = str(state["prompt"])
        run_id = str(state["run_id"])
        metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
        tier = str((metadata or {}).get("tier", "standard"))
        assumptions = state.get("assumptions", [])
        return self._run_optimization(prompt, run_id, tier, {"confirmed_gaps": [], "problem_sentences": []}, assumptions=assumptions)

    def resume(self, run_id: str, answers: dict[str, Any]) -> OptimizeResult:
        try:
            state = self._clarification.resume(run_id, answers)
        except Exception as exc:
            if isinstance(exc, (KeyError, LookupError)):
                raise RunNotFoundError(run_id) from exc
            raise
        result = state.get("result")
        if not isinstance(result, Mapping):
            raise RunNotFoundError(run_id)
        record = self.store.get_run(run_id)
        if record is not None:
            result = dict(result)
            self.store.save_run(
                {
                    **record,
                    "result": result,
                    "cost": result.get("cost", record.get("cost", {})),
                    "timing": result.get("timing", record.get("timing", {})),
                }
            )
        return cast(OptimizeResult, result)

    def skip_clarification(self, run_id: str) -> OptimizeResult:
        state = self._clarification.skip(run_id)
        result = state.get("result")
        if not isinstance(result, Mapping):
            raise RunNotFoundError(run_id)
        return cast(OptimizeResult, result)

    def update_assumption(self, run_id: str, assumption: dict[str, Any]) -> OptimizeResult:
        record = self.store.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        result = cast(OptimizeResult, dict(record.get("result") or {}))
        report = dict(result.get("report") or {})
        assumptions = list(report.get("assumptions") or [])
        assumptions.append(dict(assumption))
        report["assumptions"] = assumptions
        result["report"] = report
        self.store.save_run({**record, "result": result})
        return result

    def start_deep_pass(self, run_id: str) -> OptimizeResult:
        record = self.store.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        result = cast(OptimizeResult, dict(record.get("result") or {}))
        report = dict(result.get("report") or {})
        if not report.get("offer_deep"):
            raise ValueError("Deep escalation is not available")
        report["offer_deep"] = False
        report["escalation"] = {"status": "accepted", "from_tier": record.get("tier"), "to_tier": "deep"}
        result["report"] = report
        result["status"] = "completed"
        self.store.save_run({**record, "result": result, "tier": "deep"})
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
            }
        )


def _safe_options(options: Mapping[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): clean(v) for k, v in value.items() if not any(x in str(k).lower() for x in ("key", "token", "secret", "password", "authorization"))}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(options)


def _diff(original: str, final: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(original.splitlines(True), final.splitlines(True), fromfile="original", tofile="final"))
