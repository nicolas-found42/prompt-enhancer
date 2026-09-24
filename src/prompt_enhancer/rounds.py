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
from dataclasses import asdict, dataclass, field
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
from .selector import RankingCandidate, RankingResult, rank_candidates
from .strategies import (
    STRATEGY_LIBRARY,
    RewriteStrategy,
    StrategySearchResult,
    search_strategies,
)
from .strong_check import StrongCheckPolicy, StrongCheckReport
from .success_tests import SuccessTestCompiler

StageCallback = Callable[[str], None]


@dataclass(frozen=True)
class CandidateFailure:
    """Why a candidate lost a round.

    ``summary`` is what the next round's strategy choice and writer see; the
    other fields are evidence for the run report and history.
    """

    candidate_id: str
    strategy: str | None = None
    reasons: tuple[str, ...] = ()
    weak_pass_rates: Mapping[str, float] = field(default_factory=dict)
    strong_pass_rate: float | None = None
    mean_pass_rate: float | None = None
    worst_pass_rate: float | None = None
    sample_spread: float | None = None
    candidate_prompt: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateFailure:
        """Read the ``to_dict()`` shape; ``summary`` is derived, so it is ignored."""

        def number(name: str) -> float | None:
            item = value.get(name)
            return float(item) if isinstance(item, (int, float)) else None

        strategy = value.get("strategy")
        prompt = value.get("candidate_prompt")
        return cls(
            candidate_id=str(value.get("candidate_id") or "unknown"),
            strategy=str(strategy) if strategy is not None else None,
            reasons=tuple(
                str(reason) for reason in value.get("reasons") or () if reason
            ),
            weak_pass_rates={
                str(model): float(rate)
                for model, rate in (value.get("weak_pass_rates") or {}).items()
                if isinstance(rate, (int, float))
            },
            strong_pass_rate=number("strong_pass_rate"),
            mean_pass_rate=number("mean_pass_rate"),
            worst_pass_rate=number("worst_pass_rate"),
            sample_spread=number("sample_spread"),
            candidate_prompt=str(prompt) if prompt is not None else None,
        )

    @property
    def summary(self) -> str:
        identity = self.candidate_id
        if self.strategy:
            identity = f"{identity} ({self.strategy})"
        reasons = "; ".join(self.reasons) or "no qualifying improvement"
        if self.weak_pass_rates:
            rates = ", ".join(
                f"{model}={rate:.3f}"
                for model, rate in sorted(self.weak_pass_rates.items())
            )
            reasons = f"{reasons}; weak pass rates: {rates}"
        if self.strong_pass_rate is not None:
            reasons = f"{reasons}; strong pass rate={self.strong_pass_rate:.3f}"
        return f"{identity}: {reasons}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy": self.strategy,
            "reasons": list(self.reasons),
            "weak_pass_rates": dict(self.weak_pass_rates),
            "strong_pass_rate": self.strong_pass_rate,
            "mean_pass_rate": self.mean_pass_rate,
            "worst_pass_rate": self.worst_pass_rate,
            "sample_spread": self.sample_spread,
            "candidate_prompt": self.candidate_prompt,
            "summary": self.summary,
        }


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
    failures: tuple[CandidateFailure, ...] = ()
    """The candidates that lost this round; the next round is told why."""

    @property
    def continue_rounds(self) -> bool:
        """Another round can try to beat the prompt using these failures."""
        return bool(self.failures) and self.original_kept

    @property
    def selected_candidate_id(self) -> str | None:
        return self.ranking.selected_candidate_id if self.ranking is not None else None

    @property
    def selected_strategy(self) -> str | None:
        """The rewrite strategy of the winning candidate, if one was selected."""
        selected = self.ranking.selected if self.ranking is not None else None
        return selected.strategy if selected is not None else None

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
                "offer_deep": plan.tier != "deep"
                and bool(plan.diagnosis.get("confirmed_gaps")),
                "history": [],
            }
        assert (
            self.panel is not None
            and self.strong_check is not None
            and self.strategies is not None
        )
        return {
            "status": self.status,
            "models": models,
            "summary": self.summary,
            "diagnosis": plan.diagnosis,
            "tests": list(self.tests),
            "jev_answers": list(self.grading_answers),
            "candidates": list(self.candidates),
            "per_model": {
                "ranking": self.ranking.to_dict(),
                "panel": self.panel.to_dict(),
            },
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
        return {
            "status": "completed",
            "run_id": self.plan.run_id,
            "final_prompt": self.final_prompt,
            "original_kept": self.original_kept,
            "report": report,
            "cost": self.cost,
            "timing": self.timing,
            "candidate_failures": [
                {
                    "candidate_id": failure.candidate_id,
                    "strategy": failure.strategy,
                    "candidate_prompt": failure.candidate_prompt,
                    "reasons": list(failure.reasons),
                    "weak_pass_rates": dict(failure.weak_pass_rates),
                }
                for failure in self.failures
            ],
            "selected_candidate_id": self.selected_candidate_id,
            "evidence": self.evidence(report),
            "continue_rounds": self.continue_rounds,
        }

    def evidence(self, report: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The parts of the report kept in the run's round history."""
        report = self.report() if report is None else report
        return {
            key: report[key]
            for key in (
                "diagnosis",
                "tests",
                "candidates",
                "per_model",
                "strong_check",
                "selection_evidence",
                "strategies",
            )
            if key in report
        }


def run_round(
    gateway: Gateway, plan: RoundPlan, *, on_stage: StageCallback | None = None
) -> RoundOutcome:
    """Run one round and return what it decided."""
    stage = on_stage or (lambda _name: None)
    settings = plan.settings
    working_prompt = plan.working_prompt
    model_view = model_diagnosis(plan.diagnosis)

    def ended(
        status: str, summary: str, tests: tuple[dict[str, Any], ...]
    ) -> RoundOutcome:
        return RoundOutcome(
            plan=plan,
            status=status,
            summary=summary,
            final_prompt=working_prompt,
            original_kept=working_prompt == plan.prompt,
            tests=tests,
            **_spent(gateway),
        )

    stage("writing_tests")
    try:
        compiled = SuccessTestCompiler(
            gateway,
            writer_model=settings.writer_model,
            faithfulness_threshold=plan.faithfulness_threshold,
        ).compile(working_prompt)
    except (ValueError, TypeError) as exc:
        raise ProviderError(
            "writer",
            settings.writer_model,
            None,
            "invalid success-test response",
            role="writer",
            kind="invalid_response",
        ) from exc
    tests = tuple(asdict(test) for test in compiled.tests)

    no_gaps = not plan.diagnosis.get("confirmed_gaps", [])
    if not tests or no_gaps:
        summary = (
            "No confirmed gaps were found; the original request was returned unchanged."
            if no_gaps and tests
            else "No confirmed gaps were found; the original request was returned unchanged. No faithful success tests were established."
            if no_gaps
            else "No faithful success tests were established; the original request and any confirmed clarifications were returned without claiming an improvement."
        )
        return ended("no_change" if no_gaps and tests else "unverified", summary, tests)

    stage("choosing_strategy")
    strategy_choice = gateway.decide(
        {
            "model": settings.judge_model,
            "key": "strategy_choice",
            "type": "choice",
            "query": "Which rewrite strategy best addresses the diagnosed weakness?",
            "criteria": {
                **{item.name: item.description for item in STRATEGY_LIBRARY},
                "none": "No rewrite strategy is suitable.",
            },
            "state": {
                "prompt": working_prompt,
                "diagnosis": model_view,
                "prior_failures": list(plan.prior_failures),
            },
        },
        role="judge",
        run_id=plan.run_id,
    )
    parsed_strategy = parse_decision(strategy_choice)
    preferred_strategy = (
        parsed_strategy.selected
        if isinstance(parsed_strategy, ChoiceDecision)
        else None
    )

    def recheck_strategy(strategy: RewriteStrategy) -> dict[str, bool]:
        answer = gateway.decide(
            {
                "model": settings.judge_model,
                "key": f"strategy_recheck:{strategy.name}",
                "type": "noul",
                "query": "Is this strategy appropriate for the prompt and diagnosed weakness without inventing requirements?",
                "state": {
                    "prompt": working_prompt,
                    "diagnosis": model_view,
                    "strategy": strategy.to_dict(),
                },
            },
            role="judge",
            run_id=plan.run_id,
        )
        decision = parse_decision(answer)
        return {
            "eligible": isinstance(decision, NoulDecision)
            and decision.probability >= 0.8
        }

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
        panel.results,
        list(tests),
        gateway,
        judge_model=settings.judge_model,
        run_id=plan.run_id,
    )
    original_grade = panel_grades["original"]
    stage("checking_fidelity")
    ranking_candidates = [
        RankingCandidate(
            candidate_id=candidate.candidate_id,
            text=candidate.text,
            strategy=candidate.strategy.name,
            strategy_kind=candidate.strategy.kind,
            grade=panel_grades[candidate.candidate_id],
            eligible=(
                fidelity := check_candidate_fidelity(
                    gateway,
                    working_prompt,
                    candidate.text,
                    model_view,
                    candidate.strategy.name,
                    run_id=plan.run_id,
                    judge_model=settings.judge_model,
                )
            ).passed,
            rejection_reasons=()
            if fidelity.passed
            else ("candidate failed fidelity checks",),
            metadata={"fidelity": fidelity.to_dict()},
        )
        for candidate in candidates
    ]
    stage("strong_check")
    strong = StrongCheckPolicy(settings.strong_check_model).check(
        working_prompt,
        [candidate for candidate in ranking_candidates if candidate.eligible],
        list(tests),
        lambda candidate_prompt, _tests: _strong_score(
            gateway, candidate_prompt, _tests, plan
        ),
    )
    ranking = rank_candidates(
        RankingCandidate(
            "original", working_prompt, "original", "baseline", original_grade
        ),
        ranking_candidates,
        strong_check=strong,
    )
    final_prompt = ranking.final_prompt
    original_kept = final_prompt == plan.prompt
    return RoundOutcome(
        plan=plan,
        status="no_change"
        if original_kept
        else ("clarified" if ranking.original_kept else "improved"),
        summary="No candidate beat the original."
        if original_kept
        else (
            "Clarifications were included; no candidate beat the clarified prompt."
            if ranking.original_kept
            else "Candidate selected after verification."
        ),
        final_prompt=final_prompt,
        original_kept=original_kept,
        tests=tests,
        strategies=search,
        panel=panel,
        ranking=ranking,
        strong_check=strong,
        grading_answers=tuple(grading_answers),
        candidates=tuple(item.to_dict() for item in ranking.ranked),
        failures=tuple(
            CandidateFailure(
                candidate_id=item.candidate.candidate_id,
                strategy=item.candidate.strategy,
                reasons=tuple(
                    str(reason) for reason in item.rejection_reasons if reason
                ),
                weak_pass_rates=dict(item.candidate.grade.per_model)
                if item.candidate.grade is not None
                else {},
                candidate_prompt=item.candidate.text,
            )
            for item in ranking.ranked
            if not item.selected
        ),
        **_spent(gateway),
    )


def prompt_diff(original: str, final: str) -> str:
    """A unified diff from the original prompt to the returned one."""
    return "".join(
        difflib.unified_diff(
            original.splitlines(True),
            final.splitlines(True),
            fromfile="original",
            tofile="final",
        )
    )


def _spent(gateway: Gateway) -> dict[str, Any]:
    finished = utc_now()
    return {
        "cost": gateway.usage_report(),
        "timing": {"total_ms": 0, "started_at": finished, "finished_at": finished},
    }


def _strong_score(gateway: Gateway, prompt: str, tests: Any, plan: RoundPlan) -> float:
    if not tests:
        raise ValueError("strong check requires success tests")
    settings = plan.settings
    output = completion_text(
        gateway.chat(
            settings.strong_check_model,
            [{"role": "user", "content": prompt}],
            role="strong_check",
            run_id=plan.run_id,
        )
    )
    grades, _ = grade_panel_with_jev(
        [PanelResult("strong", settings.strong_check_model, 0, 0, output, prompt)],
        tests,
        gateway,
        judge_model=settings.judge_model,
        run_id=plan.run_id,
    )
    return grades["strong"].sample_scores[0]


__all__ = ["CandidateFailure", "RoundOutcome", "RoundPlan", "prompt_diff", "run_round"]
