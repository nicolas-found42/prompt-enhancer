"""Repeat-round orchestration and Deep-pass escalation.

This module owns round policy and user-visible escalation state. Candidate writing,
evaluation, selection, persistence, and HTTP serialization stay behind injected
callbacks so the workflow can be replayed deterministically without provider keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from .models import Tier
from .rounds import CandidateFailure, RoundOutcome


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
        outcome: RoundOutcome,
    ) -> RoundEvidence:
        return cls(
            round_number=round_number,
            tier_round=tier_round,
            tier=tier,
            max_rounds=tier.max_rounds,
            original_kept=outcome.original_kept,
            # A round that returns an outcome has completed.
            status="completed",
            selected_candidate_id=outcome.selected_candidate_id,
            selected_strategy=outcome.selected_strategy,
            candidate_failures=outcome.failures,
            evidence=_public_mapping(outcome.evidence()),
            cost=_public_mapping(outcome.cost),
            timing=_public_mapping(outcome.timing),
            continuation_requested=outcome.continue_rounds,
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
    outcome: RoundOutcome
    offer_deep: EscalationOffer | None = None
    workflow: WorkflowContext | None = None
    escalation: EscalationState | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = self.outcome.payload()
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
    def __call__(self, request: RoundRequest) -> RoundOutcome: ...


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
        initial_failures: Sequence[CandidateFailure | Mapping[str, Any] | str] = (),
    ) -> RepeatResult:
        selected_tier = Tier.parse(tier)
        workflow = WorkflowContext(
            run_id=run_id,
            original_prompt=prompt,
            tier=selected_tier,
            questions=tuple(str(question) for question in questions),
            answers=dict(answers or {}),
            assumptions=tuple(str(assumption) for assumption in assumptions),
        )
        history, outcome = _run_tier(
            execute_round,
            workflow,
            selected_tier,
            history=(),
            failures=tuple(_supplied_failure(value) for value in initial_failures),
        )
        original_kept = history[-1].original_kept
        # A round may decline the offer (``report.offer_deep`` false) when a
        # Deep pass could not do anything the lower tier did not.
        deep_declined = outcome.report().get("offer_deep") is False
        offer = (
            _deep_offer(run_id, selected_tier, history)
            if original_kept and not deep_declined
            else None
        )
        return RepeatResult(
            run_id=run_id,
            tier=selected_tier,
            final_prompt=outcome.final_prompt or prompt,
            original_kept=original_kept,
            history=history,
            outcome=outcome,
            workflow=workflow,
            offer_deep=offer,
        )

    def deep_pass(
        self, run: Mapping[str, Any], execute_round: RoundRunner
    ) -> RepeatResult:
        """Accept the offered Deep pass without changing the public run ID."""
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
        prior_history = _history_from_run(run)
        workflow = WorkflowContext.from_run(run)
        history, outcome = _run_tier(
            execute_round,
            workflow,
            Tier.DEEP,
            history=prior_history,
            failures=prior_history[-1].candidate_failures if prior_history else (),
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
            final_prompt=outcome.final_prompt or workflow.original_prompt,
            original_kept=final_original_kept,
            history=history,
            outcome=outcome,
            workflow=workflow,
            escalation=escalation,
        )


def _run_tier(
    execute_round: RoundRunner,
    workflow: WorkflowContext,
    tier: Tier,
    *,
    history: tuple[RoundEvidence, ...],
    failures: tuple[CandidateFailure, ...],
) -> tuple[tuple[RoundEvidence, ...], RoundOutcome]:
    """Run up to the tier's round limit, feeding each round the last one's failures."""
    next_round_number = history[-1].round_number + 1 if history else 1
    tier_round = 1
    while True:
        request = RoundRequest(
            run_id=workflow.run_id,
            round_number=next_round_number,
            tier_round=tier_round,
            tier=tier,
            max_rounds=tier.max_rounds,
            workflow=workflow,
            prior_round_failures=failures,
            prior_failures=tuple(failure.summary for failure in failures),
            history=history,
        )
        outcome = execute_round(request)
        evidence = RoundEvidence.from_outcome(
            round_number=request.round_number,
            tier_round=request.tier_round,
            tier=tier,
            outcome=outcome,
        )
        history = (*history, evidence)
        failures = evidence.candidate_failures
        if (
            tier_round >= tier.max_rounds
            or not evidence.continuation_requested
            or not evidence.original_kept
            or not failures
        ):
            return history, outcome
        next_round_number += 1
        tier_round += 1


def _supplied_failure(
    value: CandidateFailure | Mapping[str, Any] | str,
) -> CandidateFailure:
    """Read a failure passed in with the ``prior_round_failures`` option."""
    if isinstance(value, CandidateFailure):
        return value
    if isinstance(value, str):
        return CandidateFailure(candidate_id="unknown", reasons=(value,))
    return CandidateFailure.from_dict(value)


def _deep_offer(
    run_id: str,
    tier: Tier,
    history: Sequence[RoundEvidence],
) -> EscalationOffer | None:
    if tier is Tier.DEEP or not history or not history[-1].original_kept:
        return None
    source_evaluations = tier.weak_model_evaluations
    target_evaluations = Tier.DEEP.weak_model_evaluations
    deep = Tier.DEEP.budget
    return EscalationOffer(
        run_id=run_id,
        from_tier=tier,
        to_tier=Tier.DEEP,
        offered_after_round=history[-1].round_number,
        source_max_rounds=tier.max_rounds,
        target_max_rounds=Tier.DEEP.max_rounds,
        expected_effort=(
            f"Up to {deep.max_rounds} Deep rounds with {deep.candidates} candidates, "
            f"{deep.models} weak models, and {deep.samples} samples per model "
            "instead of the lower-tier budget."
        ),
        expected_cost_change=(
            "Higher expected model cost than this run's lower tier; the exact amount "
            "depends on selected models and provider pricing and is shown after each round."
        ),
        expected_weak_model_evaluations=target_evaluations,
        expected_evaluation_multiplier=target_evaluations / source_evaluations,
    )


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
            CandidateFailure.from_dict(item) for item in raw_failures
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
