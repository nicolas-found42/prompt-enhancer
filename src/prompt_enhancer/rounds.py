"""The Round: one attempt to beat the prompt (see CONTEXT.md).

A round compiles success tests, chooses and writes rewrite strategies, runs
the original and every candidate on the weak panel, grades the outputs with
Jev, checks each candidate's fidelity, runs the strong check, and picks a
winner or keeps the prompt. ``run_round`` is the only entry point; the stage
modules behind it are its internals.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .config import Settings
from .diagnosis import model_diagnosis
from .fidelity import check_candidate_fidelity
from .gateway import Gateway, ProviderError, completion_text
from .grading import grade_panel_with_jev
from .jev import ChoiceDecision, NoulDecision, parse_decision
from .models import Tier, utc_now
from .rewrite import CandidateWriter
from .runner import PanelResult, PanelRunResult, run_candidates
from .selector import RankingResult, rank_candidates
from .strategies import (
    STRATEGY_LIBRARY,
    RewriteStrategy,
    StrategySearchResult,
    search_strategies,
)
from .strong_check import StrongCheckPolicy, StrongCheckReport
from .success_tests import SuccessTestCompiler

StageCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RoundPlan:
    """Everything one round needs."""

    prompt: str
    """The user's original prompt."""
    working_prompt: str
    """The prompt with confirmed clarifications applied; candidates rewrite this."""
    run_id: str
    tier: Tier
    seed: int
    diagnosis: Mapping[str, Any]
    assumptions: Sequence[Any]
    settings: Settings
    faithfulness_threshold: float
    writer_instruction_version: int
    prior_failures: tuple[str, ...] = ()
    """Summaries of the previous round's losing candidates."""


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """What one round decided, with the evidence for the run report."""

    plan: RoundPlan
    status: str
    summary: str
    final_prompt: str
    original_kept: bool
    cost: Mapping[str, Any]
    timing: Mapping[str, Any]
    tests: tuple[dict[str, Any], ...] = ()
    strategies: StrategySearchResult | None = None
    panel: PanelRunResult | None = None
    ranking: RankingResult | None = None
    strong_check: StrongCheckReport | None = None
    grading_answers: tuple[dict[str, Any], ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()

    @property
    def selected_candidate_id(self) -> str | None:
        return self.ranking.selected_candidate_id if self.ranking is not None else None

    def report(self) -> dict[str, Any]:
        """The run report for this round, in the shape the web app and history read."""
        plan = self.plan
        models = plan.settings.model_roles()
        if self.ranking is None:
            # Without a confirmed gap no strategy can run, so a Deep pass would
            # only repeat the diagnosis; it is not offered.
            return {
                "status": self.status,
                "models": models,
                "summary": self.summary,
                "diagnosis": dict(plan.diagnosis),
                "tests": list(self.tests),
                "candidates": [],
                "per_model": {},
                "assumptions": list(plan.assumptions),
                "diff": prompt_diff(plan.prompt, plan.working_prompt),
                "offer_deep": plan.tier != "deep" and bool(plan.diagnosis.get("confirmed_gaps")),
                "history": [],
            }
        assert self.panel is not None and self.strong_check is not None and self.strategies is not None
        return {
            "status": self.status,
            "models": models,
            "summary": self.summary,
            "diagnosis": plan.diagnosis,
            "tests": list(self.tests),
            "jev_answers": list(self.grading_answers),
            "candidates": list(self.candidates),
            "per_model": {"ranking": self.ranking.to_dict(), "panel": self.panel.to_dict()},
            "assumptions": list(plan.assumptions),
            "diff": prompt_diff(plan.prompt, self.final_prompt),
            "selection_evidence": self.ranking.to_dict(),
            "strong_check": self.strong_check.to_dict(),
            "strategies": self.strategies.to_dict(),
            "offer_deep": plan.tier != "deep" and self.original_kept,
            "history": [],
        }

    def payload(self) -> dict[str, Any]:
        """The public result of this round, with the repeat loop's evidence."""
        report = self.report()
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
            "status": "completed",
            "run_id": self.plan.run_id,
            "final_prompt": self.final_prompt,
            "original_kept": self.original_kept,
            "report": report,
            "cost": self.cost,
            "timing": self.timing,
            "candidate_failures": failures,
            "selected_candidate_id": report.get("selection_evidence", {}).get("selected_candidate_id"),
            "evidence": {
                key: report[key]
                for key in ("diagnosis", "tests", "candidates", "per_model", "strong_check", "selection_evidence", "strategies")
                if key in report
            },
            "continue_rounds": bool(failures) and self.original_kept,
        }


def run_round(gateway: Gateway, plan: RoundPlan, *, on_stage: StageCallback | None = None) -> RoundOutcome:
    """Run one round and return what it decided."""
    stage = on_stage or (lambda _name: None)
    settings = plan.settings
    working_prompt = plan.working_prompt
    model_view = model_diagnosis(plan.diagnosis)

    def ended(status: str, summary: str, tests: tuple[dict[str, Any], ...]) -> RoundOutcome:
        return RoundOutcome(
            plan=plan, status=status, summary=summary, final_prompt=working_prompt,
            original_kept=working_prompt == plan.prompt, tests=tests, **_spent(gateway),
        )

    stage("writing_tests")
    try:
        compiled = SuccessTestCompiler(
            gateway,
            writer_model=settings.writer_model,
            faithfulness_threshold=plan.faithfulness_threshold,
        ).compile(working_prompt)
    except (ValueError, TypeError) as exc:
        raise ProviderError("writer", settings.writer_model, None, "invalid success-test response", role="writer", kind="invalid_response") from exc
    tests = tuple(asdict(test) for test in compiled.tests)

    no_gaps = not plan.diagnosis.get("confirmed_gaps", [])
    if not tests or no_gaps:
        summary = (
            "No confirmed gaps were found; the original request was returned unchanged."
            if no_gaps and tests else
            "No confirmed gaps were found; the original request was returned unchanged. No faithful success tests were established."
            if no_gaps else
            "No faithful success tests were established; the original request and any confirmed clarifications were returned without claiming an improvement."
        )
        return ended("no_change" if no_gaps and tests else "unverified", summary, tests)

    stage("choosing_strategy")
    strategy_choice = gateway.decide(
        {"model": settings.judge_model, "key": "strategy_choice", "type": "choice", "query": "Which rewrite strategy best addresses the diagnosed weakness?", "criteria": {**{item.name: item.description for item in STRATEGY_LIBRARY}, "none": "No rewrite strategy is suitable."}, "state": {"prompt": working_prompt, "diagnosis": model_view, "prior_failures": list(plan.prior_failures)}},
        role="judge", run_id=plan.run_id,
    )
    parsed_strategy = parse_decision(strategy_choice)
    preferred_strategy = parsed_strategy.selected if isinstance(parsed_strategy, ChoiceDecision) else None

    def recheck_strategy(strategy: RewriteStrategy) -> dict[str, bool]:
        answer = gateway.decide(
            {"model": settings.judge_model, "key": f"strategy_recheck:{strategy.name}", "type": "noul", "query": "Is this strategy appropriate for the prompt and diagnosed weakness without inventing requirements?", "state": {"prompt": working_prompt, "diagnosis": model_view, "strategy": strategy.to_dict()}},
            role="judge", run_id=plan.run_id,
        )
        decision = parse_decision(answer)
        return {"eligible": isinstance(decision, NoulDecision) and decision.probability >= 0.8}

    stage("writing_candidates")
    search = search_strategies(
        working_prompt,
        model_view,
        plan.tier,
        writer=CandidateWriter(
            gateway,
            writer_model=settings.writer_model,
            instruction_version=plan.writer_instruction_version,
        ),
        previous_failures=plan.prior_failures,
        recheck=recheck_strategy,
        priority_strategy=preferred_strategy,
    )
    candidates = list(search.candidates)
    if not candidates:
        return ended("no_change", "No candidate strategy was selected.", tests)

    stage("running_weak_models")
    panel = run_candidates(
        candidates,
        settings.weak_models,
        gateway,
        original=working_prompt,
        budget=plan.tier,
        run_seed=plan.seed,
        run_id=plan.run_id,
    )
    stage("grading")
    panel_grades, grading_answers = grade_panel_with_jev(
        panel, list(tests), gateway,
        judge_model=settings.judge_model, run_id=plan.run_id,
    )
    original_grade = panel_grades["original"]
    stage("checking_fidelity")
    ranking_candidates = [
        {
            "candidate_id": candidate.candidate_id,
            "text": candidate.text,
            "prompt": candidate.text,
            "strategy": candidate.strategy.name,
            "strategy_kind": candidate.strategy.kind,
            "grade": panel_grades[candidate.candidate_id],
            "fidelity": (fidelity := check_candidate_fidelity(
                gateway, working_prompt, candidate.text, model_view,
                candidate.strategy.name, run_id=plan.run_id,
                judge_model=settings.judge_model,
            )).to_dict(),
            "eligible": fidelity.passed,
            "rejection_reasons": [] if fidelity.passed else ["candidate failed fidelity checks"],
            "metadata": {"fidelity": fidelity.to_dict()},
        }
        for candidate in candidates
    ]
    stage("strong_check")
    strong = StrongCheckPolicy(settings.strong_check_model).check(
        working_prompt,
        [candidate for candidate in ranking_candidates if candidate["eligible"]],
        list(tests),
        lambda candidate_prompt, _tests: _strong_score(gateway, candidate_prompt, _tests, plan),
    )
    ranking = rank_candidates(
        {"candidate_id": "original", "text": working_prompt, "grade": original_grade},
        ranking_candidates,
        original_grade=original_grade,
        strong_check=strong,
    )
    final_prompt = ranking.final_prompt
    original_kept = final_prompt == plan.prompt
    return RoundOutcome(
        plan=plan,
        status="no_change" if original_kept else ("clarified" if ranking.original_kept else "improved"),
        summary="No candidate beat the original." if original_kept else ("Clarifications were included; no candidate beat the clarified prompt." if ranking.original_kept else "Candidate selected after verification."),
        final_prompt=final_prompt,
        original_kept=original_kept,
        tests=tests,
        strategies=search,
        panel=panel,
        ranking=ranking,
        strong_check=strong,
        grading_answers=tuple(grading_answers),
        candidates=tuple(item.to_dict() for item in ranking.ranked),
        **_spent(gateway),
    )


def prompt_diff(original: str, final: str) -> str:
    """A unified diff from the original prompt to the returned one."""
    return "".join(difflib.unified_diff(original.splitlines(True), final.splitlines(True), fromfile="original", tofile="final"))


def _spent(gateway: Gateway) -> dict[str, Any]:
    finished = utc_now()
    return {"cost": gateway.usage_report(), "timing": {"total_ms": 0, "started_at": finished, "finished_at": finished}}


def _strong_score(gateway: Gateway, prompt: str, tests: Any, plan: RoundPlan) -> float:
    if not tests:
        raise ValueError("strong check requires success tests")
    settings = plan.settings
    output = completion_text(gateway.chat(
        settings.strong_check_model,
        [{"role": "user", "content": prompt}],
        role="strong_check",
        run_id=plan.run_id,
    ))
    grades, _ = grade_panel_with_jev(
        [PanelResult("strong", settings.strong_check_model, 0, 0, output, prompt)],
        tests,
        gateway,
        judge_model=settings.judge_model,
        run_id=plan.run_id,
    )
    return grades["strong"].sample_scores[0]


__all__ = ["RoundOutcome", "RoundPlan", "prompt_diff", "run_round"]
