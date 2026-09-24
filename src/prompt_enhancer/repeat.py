"""Repeat-round orchestration and Deep-pass escalation.

This module owns round policy and user-visible escalation state. Candidate writing,
evaluation, selection, persistence, and HTTP serialization stay behind injected
callbacks so the workflow can be replayed deterministically without provider keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast


class Tier(str, Enum):
    """Optimization effort tiers supported by the public workflow."""

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"

    @classmethod
    def parse(cls, value: Tier | str) -> Tier:
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(tier.value for tier in cls)
            raise ValueError(
                f"Unknown optimization tier {value!r}; expected one of: {choices}"
            ) from error

    @property
    def max_rounds(self) -> int:
        return _MAX_ROUNDS[self]


_MAX_ROUNDS: Mapping[Tier, int] = {
    Tier.FAST: 1,
    Tier.STANDARD: 2,
    Tier.DEEP: 3,
}


@dataclass(frozen=True)
class CandidateFailure:
    """Observable reasons a candidate lost in a completed round.

    ``summary`` is the stable writer-facing seam used by the candidate generator.
    The remaining fields preserve evidence for reports and run history.
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
    def from_value(
        cls, value: CandidateFailure | Mapping[str, Any] | str
    ) -> CandidateFailure:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(candidate_id="unknown", reasons=(value,))
        mapping = cast(Mapping[str, Any], value)
        candidate_id = str(
            mapping.get("candidate_id") or mapping.get("id") or "unknown"
        )
        strategy_value = mapping.get("strategy")
        strategy = str(strategy_value) if strategy_value is not None else None
        raw_reasons = mapping.get("reasons") or mapping.get("failure_reasons") or ()
        if isinstance(raw_reasons, str):
            reasons = (raw_reasons,)
        else:
            reasons = tuple(str(reason) for reason in raw_reasons if reason)
        prompt_value = mapping.get("prompt") or mapping.get("candidate_prompt")
        candidate_prompt = str(prompt_value) if prompt_value is not None else None
        rates_value = mapping.get("weak_pass_rates") or mapping.get("per_model") or {}
        weak_pass_rates = {
            str(model): float(rate)
            for model, rate in rates_value.items()
            if isinstance(rate, (int, float))
        }
        strong_value = mapping.get("strong_pass_rate")
        mean_value = mapping.get("mean_pass_rate")
        worst_value = mapping.get("worst_pass_rate")
        spread_value = mapping.get("sample_spread")
        failure = cls(
            candidate_id=candidate_id,
            strategy=strategy,
            reasons=reasons,
            weak_pass_rates=weak_pass_rates,
            strong_pass_rate=float(strong_value)
            if isinstance(strong_value, (int, float))
            else None,
            mean_pass_rate=float(mean_value)
            if isinstance(mean_value, (int, float))
            else None,
            worst_pass_rate=float(worst_value)
            if isinstance(worst_value, (int, float))
            else None,
            sample_spread=float(spread_value)
            if isinstance(spread_value, (int, float))
            else None,
            candidate_prompt=candidate_prompt,
        )
        explicit_summary = mapping.get("summary")
        if explicit_summary:
            return cls(
                candidate_id=failure.candidate_id,
                strategy=failure.strategy,
                reasons=(str(explicit_summary), *failure.reasons),
                weak_pass_rates=failure.weak_pass_rates,
                strong_pass_rate=failure.strong_pass_rate,
                mean_pass_rate=failure.mean_pass_rate,
                worst_pass_rate=failure.worst_pass_rate,
                sample_spread=failure.sample_spread,
                candidate_prompt=failure.candidate_prompt,
            )
        return failure

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


@dataclass(frozen=True)
class WorkflowContext:
    """Safe, credential-free state needed to continue a paused or prior run."""

    run_id: str
    original_prompt: str
    tier: Tier
    questions: tuple[str, ...] = ()
    answers: Mapping[str, str] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    @classmethod
    def from_run(cls, run: Mapping[str, Any]) -> WorkflowContext:
        report = _mapping_or_empty(run.get("report"))
        questions = run.get("questions") or report.get("questions") or ()
        answers = run.get("answers") or report.get("answers") or {}
        assumptions = run.get("assumptions") or report.get("assumptions") or ()
        original_prompt = str(run.get("original_prompt") or run.get("prompt") or "")
        return cls(
            run_id=str(run["run_id"]),
            original_prompt=original_prompt,
            tier=Tier.parse(run.get("tier") or report.get("tier") or Tier.STANDARD),
            questions=tuple(_safe_public_value(question) for question in questions),
            answers={
                str(key): _safe_public_value(value)
                for key, value in answers.items()
                if not _looks_like_secret(key)
            },
            assumptions=tuple(_safe_public_value(item) for item in assumptions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_prompt": self.original_prompt,
            "tier": self.tier.value,
            "questions": list(self.questions),
            "answers": dict(self.answers),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class RoundRequest:
    """Input to the injected round runner.

    Both structured failures and their stable summaries are present: report code
    can retain evidence, while a writer gateway can use the existing
    ``previous_failures`` string seam without knowing about orchestration types.
    """

    run_id: str
    round_number: int
    tier_round: int
    tier: Tier
    max_rounds: int
    workflow: WorkflowContext
    prior_round_failures: tuple[CandidateFailure, ...] = ()
    prior_failures: tuple[str, ...] = ()
    history: tuple[RoundEvidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "round_number": self.round_number,
            "tier_round": self.tier_round,
            "tier": self.tier.value,
            "max_rounds": self.max_rounds,
            "workflow": self.workflow.to_dict(),
            "prior_round_failures": [
                failure.to_dict() for failure in self.prior_round_failures
            ],
            "prior_failures": list(self.prior_failures),
            "history": [round_evidence.to_dict() for round_evidence in self.history],
        }


@dataclass(frozen=True)
class RoundEvidence:
    """User-visible evidence retained for one completed optimization round."""

    round_number: int
    tier_round: int
    tier: Tier
    max_rounds: int
    original_kept: bool
    status: str
    selected_candidate_id: str | None = None
    selected_strategy: str | None = None
    candidate_failures: tuple[CandidateFailure, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)
    continuation_requested: bool = False

    @classmethod
    def from_outcome(
        cls,
        *,
        round_number: int,
        tier_round: int,
        tier: Tier,
        outcome: Mapping[str, Any],
    ) -> RoundEvidence:
        report = _mapping_or_empty(outcome.get("report"))
        original_kept = _as_bool(
            outcome.get("original_kept", report.get("original_kept", False))
        )
        status = str(
            outcome.get("status")
            or report.get("status")
            or ("no_change" if original_kept else "improved")
        )
        failure_values = (
            outcome.get("candidate_failures")
            or outcome.get("prior_round_failures")
            or report.get("candidate_failures")
            or ()
        )
        failures = tuple(CandidateFailure.from_value(value) for value in failure_values)
        return cls(
            round_number=round_number,
            tier_round=tier_round,
            tier=tier,
            max_rounds=tier.max_rounds,
            original_kept=original_kept,
            status=status,
            selected_candidate_id=_optional_string(
                outcome.get("selected_candidate_id")
                or report.get("selected_candidate_id")
            ),
            selected_strategy=_optional_string(
                outcome.get("selected_strategy") or report.get("selected_strategy")
            ),
            candidate_failures=failures,
            evidence=_public_mapping(
                outcome.get("evidence") or report.get("evidence") or {}
            ),
            cost=_public_mapping(outcome.get("cost") or report.get("cost") or {}),
            timing=_public_mapping(outcome.get("timing") or report.get("timing") or {}),
            continuation_requested=_as_bool(
                outcome.get(
                    "continue_rounds",
                    outcome.get("continuation_requested", original_kept),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "tier_round": self.tier_round,
            "tier": self.tier.value,
            "max_rounds": self.max_rounds,
            "original_kept": self.original_kept,
            "status": self.status,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_strategy": self.selected_strategy,
            "candidate_failures": [
                failure.to_dict() for failure in self.candidate_failures
            ],
            "evidence": dict(self.evidence),
            "cost": dict(self.cost),
            "timing": dict(self.timing),
            "continuation_requested": self.continuation_requested,
        }


@dataclass(frozen=True)
class EscalationOffer:
    """Explicit choice to spend more effort after a lower-tier no-change result."""

    run_id: str
    from_tier: Tier
    to_tier: Tier
    offered_after_round: int
    source_max_rounds: int
    target_max_rounds: int
    expected_effort: str
    expected_cost_change: str
    expected_weak_model_evaluations: int
    expected_evaluation_multiplier: float
    state: str = "offered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "from_tier": self.from_tier.value,
            "to_tier": self.to_tier.value,
            "offered_after_round": self.offered_after_round,
            "source_max_rounds": self.source_max_rounds,
            "target_max_rounds": self.target_max_rounds,
            "expected_effort": self.expected_effort,
            "expected_cost_change": self.expected_cost_change,
            "expected_weak_model_evaluations": self.expected_weak_model_evaluations,
            "expected_evaluation_multiplier": self.expected_evaluation_multiplier,
            "state": self.state,
        }


@dataclass(frozen=True)
class EscalationState:
    """Lifecycle of the Deep pass for one user-visible run."""

    status: str
    run_id: str
    source_tier: Tier
    target_tier: Tier
    started_after_round: int | None = None
    completed_through_round: int | None = None
    final_original_kept: bool | None = None
    offer: EscalationOffer | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "source_tier": self.source_tier.value,
            "target_tier": self.target_tier.value,
            "started_after_round": self.started_after_round,
            "completed_through_round": self.completed_through_round,
            "final_original_kept": self.final_original_kept,
            "offer": self.offer.to_dict() if self.offer else None,
        }


@dataclass(frozen=True)
class RepeatResult:
    """Final round-runner payload augmented with repeat-round state."""

    run_id: str
    tier: Tier
    final_prompt: str
    original_kept: bool
    history: tuple[RoundEvidence, ...]
    outcome: Mapping[str, Any]
    offer_deep: EscalationOffer | None = None
    workflow: WorkflowContext | None = None
    escalation: EscalationState | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = dict(self.outcome)
        payload["run_id"] = self.run_id
        payload["final_prompt"] = self.final_prompt
        payload["original_kept"] = self.original_kept
        payload["tier"] = self.tier.value
        report = dict(_mapping_or_empty(payload.get("report")))
        report["history"] = [
            round_evidence.to_dict() for round_evidence in self.history
        ]
        report["round_history"] = report["history"]
        report["offer_deep"] = self.offer_deep.to_dict() if self.offer_deep else None
        report["escalation"] = self.escalation.to_dict() if self.escalation else None
        payload["report"] = report
        if self.workflow is not None:
            payload.update(
                {
                    "original_prompt": self.workflow.original_prompt,
                    "questions": list(self.workflow.questions),
                    "answers": dict(self.workflow.answers),
                    "assumptions": list(self.workflow.assumptions),
                }
            )
        return payload


class RoundRunner(Protocol):
    def __call__(self, request: RoundRequest) -> Mapping[str, Any]: ...


class RoundSink(Protocol):
    def __call__(
        self, request: RoundRequest, outcome: Mapping[str, Any], evidence: RoundEvidence
    ) -> None: ...


class RepeatCoordinator:
    """Run bounded repeat rounds or continue the same run with a Deep pass."""

    def run(
        self,
        *,
        run_id: str,
        prompt: str,
        tier: Tier | str,
        execute_round: RoundRunner,
        questions: Sequence[str] = (),
        answers: Mapping[str, str] | None = None,
        assumptions: Sequence[str] = (),
        initial_history: Sequence[RoundEvidence | Mapping[str, Any]] = (),
        initial_failures: Sequence[CandidateFailure | Mapping[str, Any] | str] = (),
        persist_round: RoundSink | None = None,
    ) -> RepeatResult:
        selected_tier = Tier.parse(tier)
        history = tuple(_round_evidence_from_value(value) for value in initial_history)
        failures = tuple(
            CandidateFailure.from_value(value) for value in initial_failures
        )
        workflow = WorkflowContext(
            run_id=run_id,
            original_prompt=prompt,
            tier=selected_tier,
            questions=tuple(str(question) for question in questions),
            answers=dict(answers or {}),
            assumptions=tuple(str(assumption) for assumption in assumptions),
        )
        outcome: Mapping[str, Any] = {}
        next_round_number = history[-1].round_number + 1 if history else 1
        tier_round = 1
        while tier_round <= selected_tier.max_rounds:
            request = RoundRequest(
                run_id=run_id,
                round_number=next_round_number,
                tier_round=tier_round,
                tier=selected_tier,
                max_rounds=selected_tier.max_rounds,
                workflow=workflow,
                prior_round_failures=failures,
                prior_failures=tuple(failure.summary for failure in failures),
                history=history,
            )
            outcome = execute_round(request)
            if not isinstance(outcome, Mapping):
                raise TypeError("Round runner must return a mapping")
            evidence = RoundEvidence.from_outcome(
                round_number=request.round_number,
                tier_round=request.tier_round,
                tier=selected_tier,
                outcome=outcome,
            )
            history = (*history, evidence)
            if persist_round is not None:
                persist_round(request, outcome, evidence)
            if not evidence.continuation_requested or evidence.original_kept is False:
                break
            failures = evidence.candidate_failures
            if not failures:
                break
            next_round_number += 1
            tier_round += 1

        final_prompt = str(
            outcome.get("final_prompt") or outcome.get("selected_prompt") or prompt
        )
        original_kept = history[-1].original_kept
        # A round runner may decline the offer (``report.offer_deep`` false) when
        # a Deep pass could not do anything the lower tier did not.
        deep_declined = _mapping_or_empty(outcome.get("report")).get("offer_deep") is False
        offer = _deep_offer(run_id, selected_tier, history) if original_kept and not deep_declined else None
        return RepeatResult(
            run_id=run_id,
            tier=selected_tier,
            final_prompt=final_prompt,
            original_kept=original_kept,
            history=history,
            outcome=outcome,
            workflow=workflow,
            offer_deep=offer,
        )

    def optimize(
        self,
        prompt: str,
        options: Mapping[str, Any],
        execute_round: RoundRunner,
        persist_round: RoundSink | None = None,
    ) -> RepeatResult:
        """Public credential-free facade for the injected optimizer round loop."""
        _reject_provider_secrets(options)
        run_id = options.get("run_id")
        if not run_id:
            raise ValueError(
                "options.run_id is required to retain repeat-round history"
            )
        report = _mapping_or_empty(options.get("report"))
        return self.run(
            run_id=str(run_id),
            prompt=prompt,
            tier=options.get("tier") or Tier.STANDARD,
            execute_round=execute_round,
            questions=options.get("questions") or (),
            answers=options.get("answers") or {},
            assumptions=options.get("assumptions") or (),
            initial_history=options.get("history") or report.get("history") or (),
            initial_failures=(
                options.get("prior_round_failures")
                or options.get("prior_failures")
                or ()
            ),
            persist_round=persist_round,
        )

    def deep_pass(
        self,
        run: Mapping[str, Any],
        execute_round: RoundRunner,
        persist_round: RoundSink | None = None,
    ) -> RepeatResult:
        """Accept the offered Deep pass without changing the public run ID."""
        return self.escalate(
            run=run,
            execute_round=execute_round,
            persist_round=persist_round,
        )

    def escalate(
        self,
        *,
        run: Mapping[str, Any],
        execute_round: RoundRunner,
        persist_round: RoundSink | None = None,
    ) -> RepeatResult:
        if not isinstance(run, Mapping) or not run.get("run_id"):
            raise ValueError(
                "A persisted run with a run_id is required for Deep escalation"
            )
        report = _mapping_or_empty(run.get("report"))
        original_kept = _as_bool(
            run.get("original_kept", report.get("original_kept", False))
        )
        source_tier = Tier.parse(run.get("tier") or report.get("tier") or Tier.STANDARD)
        if source_tier is Tier.DEEP:
            raise ValueError("This run has already used the Deep tier")
        if not original_kept:
            raise ValueError(
                "Deep escalation is only available when the original was retained"
            )
        existing_offer = report.get("offer_deep")
        if existing_offer is not None and not isinstance(existing_offer, Mapping):
            raise ValueError("Persisted Deep offer is malformed")
        offer = (
            EscalationOffer(
                run_id=str(run["run_id"]),
                from_tier=Tier.parse(existing_offer["from_tier"]),
                to_tier=Tier.DEEP,
                offered_after_round=int(existing_offer.get("offered_after_round", 0)),
                source_max_rounds=int(
                    existing_offer.get("source_max_rounds", source_tier.max_rounds)
                ),
                target_max_rounds=Tier.DEEP.max_rounds,
                expected_effort=str(existing_offer["expected_effort"]),
                expected_cost_change=str(existing_offer["expected_cost_change"]),
                expected_weak_model_evaluations=int(
                    existing_offer["expected_weak_model_evaluations"]
                ),
                expected_evaluation_multiplier=float(
                    existing_offer["expected_evaluation_multiplier"]
                ),
                state="accepted",
            )
            if existing_offer
            else _deep_offer(
                str(run["run_id"]),
                source_tier,
                _history_from_run(run),
            )
        )
        if offer is None:
            raise ValueError("Persisted run does not contain a valid Deep-pass offer")
        history = _history_from_run(run)
        workflow = WorkflowContext.from_run(run)
        outcome: Mapping[str, Any] = {}
        tier_round = 1
        next_round_number = history[-1].round_number + 1 if history else 1
        failures = history[-1].candidate_failures if history else ()
        while tier_round <= Tier.DEEP.max_rounds:
            request = RoundRequest(
                run_id=workflow.run_id,
                round_number=next_round_number,
                tier_round=tier_round,
                tier=Tier.DEEP,
                max_rounds=Tier.DEEP.max_rounds,
                workflow=workflow,
                prior_round_failures=failures,
                prior_failures=tuple(failure.summary for failure in failures),
                history=history,
            )
            outcome = execute_round(request)
            if not isinstance(outcome, Mapping):
                raise TypeError("Round runner must return a mapping")
            evidence = RoundEvidence.from_outcome(
                round_number=request.round_number,
                tier_round=request.tier_round,
                tier=Tier.DEEP,
                outcome=outcome,
            )
            history = (*history, evidence)
            if persist_round is not None:
                persist_round(request, outcome, evidence)
            if not evidence.continuation_requested or evidence.original_kept is False:
                break
            failures = evidence.candidate_failures
            if not failures:
                break
            next_round_number += 1
            tier_round += 1

        final_prompt = str(
            outcome.get("final_prompt")
            or outcome.get("selected_prompt")
            or workflow.original_prompt
        )
        final_original_kept = history[-1].original_kept
        escalation = EscalationState(
            status="completed",
            run_id=workflow.run_id,
            source_tier=offer.from_tier,
            target_tier=Tier.DEEP,
            started_after_round=offer.offered_after_round,
            completed_through_round=history[-1].round_number,
            final_original_kept=final_original_kept,
            offer=offer,
        )
        return RepeatResult(
            run_id=workflow.run_id,
            tier=Tier.DEEP,
            final_prompt=final_prompt,
            original_kept=final_original_kept,
            history=history,
            outcome=outcome,
            workflow=workflow,
            escalation=escalation,
        )


def _deep_offer(
    run_id: str,
    tier: Tier,
    history: Sequence[RoundEvidence],
) -> EscalationOffer | None:
    if tier is Tier.DEEP or not history or not history[-1].original_kept:
        return None
    source_evaluations = _tier_weak_evaluations(tier)
    target_evaluations = _tier_weak_evaluations(Tier.DEEP)
    return EscalationOffer(
        run_id=run_id,
        from_tier=tier,
        to_tier=Tier.DEEP,
        offered_after_round=history[-1].round_number,
        source_max_rounds=tier.max_rounds,
        target_max_rounds=Tier.DEEP.max_rounds,
        expected_effort=(
            "Up to 3 Deep rounds with 6 candidates, 5 weak models, and 3 samples "
            "per model instead of the lower-tier budget."
        ),
        expected_cost_change=(
            "Higher expected model cost than this run's lower tier; the exact amount "
            "depends on selected models and provider pricing and is shown after each round."
        ),
        expected_weak_model_evaluations=target_evaluations,
        expected_evaluation_multiplier=target_evaluations / source_evaluations,
    )


def _tier_weak_evaluations(tier: Tier) -> int:
    candidates, models, samples = {
        Tier.FAST: (3, 2, 1),
        Tier.STANDARD: (4, 3, 2),
        Tier.DEEP: (6, 5, 3),
    }[tier]
    return candidates * models * samples * tier.max_rounds


def _history_from_run(run: Mapping[str, Any]) -> tuple[RoundEvidence, ...]:
    report = _mapping_or_empty(run.get("report"))
    values = (
        report.get("history") or report.get("round_history") or run.get("history") or ()
    )
    return tuple(_round_evidence_from_value(value) for value in values)


def _round_evidence_from_value(
    value: RoundEvidence | Mapping[str, Any],
) -> RoundEvidence:
    if isinstance(value, RoundEvidence):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Round history entries must be mappings")
    raw_failures = value.get("candidate_failures") or ()
    return RoundEvidence(
        round_number=int(value.get("round_number") or value.get("round") or 1),
        tier_round=int(value.get("tier_round") or value.get("round") or 1),
        tier=Tier.parse(value.get("tier") or Tier.STANDARD),
        max_rounds=int(
            value.get("max_rounds")
            or Tier.parse(value.get("tier") or Tier.STANDARD).max_rounds
        ),
        original_kept=_as_bool(value.get("original_kept", False)),
        status=str(value.get("status") or "no_change"),
        selected_candidate_id=_optional_string(value.get("selected_candidate_id")),
        selected_strategy=_optional_string(value.get("selected_strategy")),
        candidate_failures=tuple(
            CandidateFailure.from_value(item) for item in raw_failures
        ),
        evidence=_public_mapping(value.get("evidence") or {}),
        cost=_public_mapping(value.get("cost") or {}),
        timing=_public_mapping(value.get("timing") or {}),
        continuation_requested=_as_bool(value.get("continuation_requested", False)),
    )


_SECRET_NAMES = {
    "key",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
}


def _looks_like_secret(name: object) -> bool:
    normalized = str(name).strip().lower().replace("-", "_")
    return normalized in _SECRET_NAMES or any(
        normalized.endswith(f"_{suffix}") for suffix in _SECRET_NAMES
    )


def _reject_provider_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _looks_like_secret(key):
                raise ValueError(
                    "Provider credentials are server configuration, not optimize options"
                )
            _reject_provider_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_provider_secrets(item)


def _safe_public_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("text")
            or value.get("question")
            or value.get("label")
            or value.get("value")
            or ""
        )
    return str(value)


def _public_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item for key, item in value.items() if not _looks_like_secret(key)
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
