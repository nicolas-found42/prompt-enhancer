"""The Round: one attempt to beat the prompt.

See docs/contexts/prompt-improvement/CONTEXT.md for its glossary.

A round compiles success tests, chooses and writes rewrite strategies, runs
the original and every candidate on the weak panel, grades the outputs with
Jev, checks each candidate's fidelity, runs the strong check, and picks a
winner or keeps the prompt. ``run_round`` is the only entry point; the stage
modules behind it are its internals.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from . import jev_questions
from .config import Settings
from .diagnosis import model_diagnosis
from .evaluation.calibration import DecisionPolicy
from .evaluation.order_bias import OrderBiasPolicy
from .failure_attribution import AttributionBudget, attribute_failed_pairs
from .fidelity import check_candidate_fidelity
from .gateway import Gateway, ProviderError, completion_text
from .grading import grade_panel_with_jev
from .grading_cascade import CascadeBudget
from .jev import ChoiceDecision, NoulDecision, parse_decision
from .lossless_restructuring import LosslessBuild, build_lossless_candidate
from .models import Tier, utc_now
from .rewrite import CandidateWriter
from .runner import PanelResult, PanelRunResult, run_candidates
from .selector import RankingCandidate, RankingResult, rank_candidates
from .strategies import (
    CURRENT_STRATEGY_LIBRARY,
    STRATEGY_LIBRARY,
    CandidateBatchRequest,
    CandidateDraft,
    RewriteStrategy,
    StrategyRejection,
    StrategySearchResult,
    search_strategies,
)
from .strong_check import StrongCheckPolicy, StrongCheckReport
from .success_tests import SuccessTestCompiler, SuccessTestScreenCache

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
    attributions: tuple[Mapping[str, Any], ...] = ()

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
            attributions=tuple(
                dict(item)
                for item in value.get("attributions") or ()
                if isinstance(item, Mapping)
            ),
        )

    @property
    def summary(self) -> str:
        identity = self.candidate_id
        if self.strategy:
            identity = f"{identity} ({self.strategy})"
        fidelity_reasons = [
            reason
            for reason in self.reasons
            if reason.startswith("fidelity ")
            or reason.startswith("whole-prompt meaning preservation")
        ]
        if fidelity_reasons:
            details = [
                _concise_fidelity_reason(reason) for reason in fidelity_reasons[:2]
            ]
            reasons = "prior fidelity evidence: " + "; ".join(details)
            if len(fidelity_reasons) > len(details):
                reasons += (
                    f"; {len(fidelity_reasons) - len(details)} more fidelity finding(s)"
                )
        else:
            reasons = "; ".join(self.reasons) or "no qualifying improvement"
        if self.weak_pass_rates:
            rates = ", ".join(
                f"{model}={rate:.3f}"
                for model, rate in sorted(self.weak_pass_rates.items())
            )
            reasons = f"{reasons}; weak pass rates: {rates}"
        if self.strong_pass_rate is not None:
            reasons = f"{reasons}; strong pass rate={self.strong_pass_rate:.3f}"
        supported = [
            item for item in self.attributions if item.get("status") == "supported"
        ]
        groups: dict[tuple[str, str, str, str], set[tuple[str, int]]] = {}
        for item in supported:
            key = (
                str(item.get("prompt_digest", "")),
                str(item.get("sentence_id", "")),
                str(item.get("sentence_text", "")),
                str(item.get("kind", "")),
            )
            groups.setdefault(key, set()).add(
                (str(item.get("model", "")), int(item.get("sample", 0)))
            )
        for (digest, sentence_id, sentence_text, kind), sources in sorted(
            groups.items()
        ):
            reasons += (
                f"; attribution hypothesis: source {self.candidate_id} "
                f"({digest[:12]}) {sentence_id} '{sentence_text[:120]}' "
                f"{kind} ({len(sources)} model/sample pair(s))"
            )
        return f"{identity}: {reasons}"

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        if self.attributions:
            result["attributions"] = [dict(item) for item in self.attributions]
        return result


def _concise_fidelity_reason(reason: str) -> str:
    if reason.startswith("fidelity rejected "):
        return "support was not verified for " + reason.removeprefix(
            "fidelity rejected "
        )
    if reason.startswith("fidelity confinement rejected "):
        return "edit was outside its authorized span: " + reason.removeprefix(
            "fidelity confinement rejected "
        )
    if reason.startswith("whole-prompt meaning preservation"):
        return "whole-prompt meaning preservation did not meet the policy threshold"
    return reason


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
    grading_policy: OrderBiasPolicy | None = None
    screen_cache: SuccessTestScreenCache | None = None
    decision_policy: DecisionPolicy | None = None


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
    lossless_restructuring: Mapping[str, Any] | None = None
    test_screening: Mapping[str, Any] | None = None
    grading_observation: Mapping[str, Any] | None = None
    output_screen: tuple[dict[str, Any], ...] | None = None
    grading_cascade: Mapping[str, Any] | None = None
    failure_attribution: Mapping[str, Any] | None = None

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
        grading_policies: list[dict[str, Any]] = []
        for answer in self.grading_answers:
            policy = answer.get("grading_policy")
            if isinstance(policy, Mapping) and dict(policy) not in grading_policies:
                grading_policies.append(dict(policy))
        if self.ranking is None:
            # Without a confirmed gap no strategy can run, so a Deep pass would
            # only repeat the diagnosis; it is not offered.
            report = {
                "status": self.status,
                "models": models,
                "summary": self.summary,
                "diagnosis": dict(plan.diagnosis),
                "tests": list(self.tests),
                "grading_policy": grading_policies,
                "candidates": [],
                "per_model": {},
                "assumptions": list(plan.assumptions),
                "diff": prompt_diff(plan.prompt, plan.working_prompt),
                "offer_deep": plan.tier != "deep"
                and bool(plan.diagnosis.get("confirmed_gaps")),
                "history": [],
            }
            if self.strategies is not None and self.lossless_restructuring is not None:
                report["strategies"] = self.strategies.to_dict()
            if self.lossless_restructuring is not None:
                report["lossless_restructuring"] = dict(self.lossless_restructuring)
            if self.test_screening is not None:
                report["test_screening"] = dict(self.test_screening)
            if self.grading_observation is not None:
                report["grading_observation"] = dict(self.grading_observation)
            if self.output_screen is not None:
                report["output_screen"] = list(self.output_screen)
            if self.grading_cascade is not None:
                report["grading_cascade"] = dict(self.grading_cascade)
            if self.failure_attribution is not None:
                report["failure_attribution"] = dict(self.failure_attribution)
            return report
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
            "grading_policy": grading_policies,
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
            **(
                {"test_screening": dict(self.test_screening)}
                if self.test_screening is not None
                else {}
            ),
            **(
                {"grading_observation": dict(self.grading_observation)}
                if self.grading_observation is not None
                else {}
            ),
            **(
                {"output_screen": list(self.output_screen)}
                if self.output_screen is not None
                else {}
            ),
            **(
                {"grading_cascade": dict(self.grading_cascade)}
                if self.grading_cascade is not None
                else {}
            ),
            **(
                {"failure_attribution": dict(self.failure_attribution)}
                if self.failure_attribution is not None
                else {}
            ),
            **(
                {"lossless_restructuring": dict(self.lossless_restructuring)}
                if self.lossless_restructuring is not None
                else {}
            ),
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
                    **(
                        {"attributions": [dict(item) for item in failure.attributions]}
                        if failure.attributions
                        else {}
                    ),
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
                "grading_policy",
                "strategies",
                "failure_attribution",
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
    screening_evidence: Mapping[str, Any] | None = None

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
            test_screening=screening_evidence,
            **_spent(gateway),
        )

    stage("writing_tests")
    try:
        compiled = SuccessTestCompiler(
            gateway,
            writer_model=settings.writer_model,
            faithfulness_threshold=plan.faithfulness_threshold,
            screen_protocol_version=2 if plan.writer_instruction_version >= 5 else 1,
            screen_cache=plan.screen_cache,
            decision_policy=plan.decision_policy,
            run_id=plan.run_id,
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
    tests = tuple(test.to_dict() for test in compiled.tests)
    if plan.writer_instruction_version >= 5:
        screening_evidence = compiled.as_dict()

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
    strategy_library = (
        CURRENT_STRATEGY_LIBRARY
        if plan.writer_instruction_version >= 4
        else STRATEGY_LIBRARY
    )
    strategy_choice = gateway.decide(
        {
            "model": settings.judge_model,
            "key": "strategy_choice",
            "type": "choice",
            "query": jev_questions.STRATEGY_CHOICE_QUESTION,
            "criteria": {
                **{item.name: item.description for item in strategy_library},
                "none": jev_questions.STRATEGY_NONE_DESCRIPTION,
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
                "query": jev_questions.STRATEGY_RECHECK_QUESTION,
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
    candidate_writer = CandidateWriter(
        gateway,
        writer_model=settings.writer_model,
        instruction_version=plan.writer_instruction_version,
    )
    lossless_build: LosslessBuild | None = None

    def write_selected(request: CandidateBatchRequest) -> Mapping[str, str]:
        nonlocal lossless_build
        ordinary = tuple(
            item for item in request.strategies if item.name != "restructure_lossless"
        )
        generated: dict[str, str] = {}
        if ordinary:
            generated.update(
                candidate_writer.generate_candidates(
                    replace(request, strategies=ordinary)
                )
            )
        if any(item.name == "restructure_lossless" for item in request.strategies):
            lossless_build = build_lossless_candidate(
                working_prompt,
                gateway,
                judge_model=settings.judge_model,
                run_id=plan.run_id,
            )
            generated["restructure_lossless"] = lossless_build.text or working_prompt
        return generated

    search = search_strategies(
        working_prompt,
        model_view,
        plan.tier,
        strategies=strategy_library,
        writer=write_selected,
        previous_failures=plan.prior_failures,
        recheck=recheck_strategy,
        priority_strategy=preferred_strategy,
    )
    if lossless_build is not None:
        kept: list[CandidateDraft] = []
        rejections = list(search.rejections)
        for candidate in search.candidates:
            if candidate.strategy.name != "restructure_lossless":
                kept.append(candidate)
            elif lossless_build.text is None:
                rejections.append(
                    StrategyRejection(
                        candidate.strategy.name,
                        lossless_build.decline_reason or "lossless proof unavailable",
                        int(candidate.metadata.get("rank_score", 0)),
                    )
                )
            else:
                kept.append(
                    replace(
                        candidate,
                        metadata={
                            **candidate.metadata,
                            "lossless_restructuring": dict(lossless_build.evidence),
                            "lossless_proof": dict(lossless_build.proof or {}),
                        },
                    )
                )
        search = replace(
            search,
            candidates=tuple(kept),
            selected_strategies=tuple(item.strategy for item in kept),
            rejections=tuple(rejections),
        )
    candidates = list(search.candidates)
    if not candidates:
        return replace(
            ended("no_change", "No candidate strategy was selected.", tests),
            strategies=search,
            lossless_restructuring=lossless_build.evidence
            if lossless_build is not None
            else None,
            test_screening=screening_evidence,
        )

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
    grading_observation: dict[str, Any] = {}
    cascade_observation: dict[str, Any] = {}
    pair_outcomes: list[dict[str, Any]] = []
    panel_grades, grading_answers = grade_panel_with_jev(
        panel.results,
        list(tests),
        gateway,
        judge_model=settings.judge_model,
        run_id=plan.run_id,
        grading_policy=plan.grading_policy,
        shared_state=plan.writer_instruction_version >= 5,
        output_screen=plan.writer_instruction_version >= 6,
        decision_policy=plan.decision_policy,
        cascade_budget=CascadeBudget.for_tier(
            plan.tier.value,
            pair_cap=settings.grading_cascade_pair_cap,
            dollar_cap=settings.grading_cascade_dollar_cap,
            judge_reservation_usd=settings.grading_confirmation_reservation_usd,
        )
        if plan.writer_instruction_version >= 7
        else None,
        cascade_observation=cascade_observation
        if plan.writer_instruction_version >= 7
        else None,
        cascade_strong_model=settings.strong_check_model,
        pair_outcomes_out=pair_outcomes
        if plan.writer_instruction_version >= 8
        else None,
        measurements=grading_observation
        if plan.writer_instruction_version >= 5
        else None,
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
                    plan.diagnosis,
                    candidate.strategy,
                    run_id=plan.run_id,
                    judge_model=settings.judge_model,
                    assumptions=plan.assumptions,
                    support_prompt=plan.prompt,
                    preservation_proof=candidate.metadata.get("lossless_proof"),
                )
            ).passed
            and panel_grades[candidate.candidate_id].ungradable_outputs == 0
            and original_grade.ungradable_outputs == 0
            and panel_grades[candidate.candidate_id].unresolved_screen_outputs == 0
            and original_grade.unresolved_screen_outputs == 0
            and panel_grades[candidate.candidate_id].unresolved_grade_outputs == 0
            and original_grade.unresolved_grade_outputs == 0,
            rejection_reasons=(
                (
                    ()
                    if fidelity.passed
                    else (
                        "candidate failed fidelity checks",
                        *fidelity.rejection_reasons,
                    )
                )
                + (
                    ("weak-panel grading was incomplete or oversized",)
                    if panel_grades[candidate.candidate_id].ungradable_outputs
                    or original_grade.ungradable_outputs
                    else ()
                )
                + (
                    ("weak-panel output screen was unresolved",)
                    if panel_grades[candidate.candidate_id].unresolved_screen_outputs
                    or original_grade.unresolved_screen_outputs
                    else ()
                )
                + (
                    ("weak-panel grade confirmation was unresolved",)
                    if panel_grades[candidate.candidate_id].unresolved_grade_outputs
                    or original_grade.unresolved_grade_outputs
                    else ()
                )
            ),
            metadata={
                "fidelity": fidelity.to_dict(),
                **(
                    {
                        "lossless_restructuring": candidate.metadata[
                            "lossless_restructuring"
                        ]
                    }
                    if "lossless_restructuring" in candidate.metadata
                    else {}
                ),
            },
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
    attribution_by_candidate: dict[str, tuple[dict[str, Any], ...]] = {}
    attribution_report: dict[str, Any] | None = None
    if plan.writer_instruction_version >= 8:
        grading_observation["pair_outcomes"] = pair_outcomes
        attribution_by_candidate, attribution_report = attribute_failed_pairs(
            panel.results,
            tests,
            pair_outcomes,
            {
                item.candidate.candidate_id
                for item in ranking.ranked
                if not item.selected
            },
            gateway,
            judge_model=settings.judge_model,
            run_id=plan.run_id,
            budget=AttributionBudget.for_tier(
                plan.tier,
                pair_cap=settings.attribution_pair_cap,
                dollar_cap=settings.attribution_dollar_cap,
            ),
            decision_policy=plan.decision_policy,
        )
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
        output_screen=tuple(
            answer["output_screen"]
            for answer in grading_answers
            if "output_screen" in answer
        )
        if plan.writer_instruction_version >= 6
        else None,
        grading_cascade=cascade_observation
        if plan.writer_instruction_version >= 7
        else None,
        failure_attribution=attribution_report,
        candidates=tuple(item.to_dict() for item in ranking.ranked),
        lossless_restructuring={
            **lossless_build.evidence,
            "selection_outcome": (
                "selected"
                if ranking.selected is not None
                and ranking.selected.strategy == "restructure_lossless"
                else "not_selected"
            ),
        }
        if lossless_build is not None
        else None,
        test_screening=screening_evidence,
        grading_observation=grading_observation
        if plan.writer_instruction_version >= 5
        else None,
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
                attributions=attribution_by_candidate.get(
                    item.candidate.candidate_id, ()
                ),
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
        grading_policy=plan.grading_policy,
        shared_state=plan.writer_instruction_version >= 5,
    )
    return grades["strong"].sample_scores[0]


__all__ = ["CandidateFailure", "RoundOutcome", "RoundPlan", "prompt_diff", "run_round"]
