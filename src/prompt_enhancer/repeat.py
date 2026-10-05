"""Uncapped repeat-round orchestration for one optimization run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from .convergence import convergence_summary, mean_score
from .improve import failure_unverified
from .rounds import CandidateFailure, RoundOutcome


@dataclass(frozen=True)
class WorkflowContext:
    """Safe, credential-free state needed to continue a paused run."""

    run_id: str
    original_prompt: str
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
            "questions": list(self.questions),
            "answers": dict(self.answers),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class RoundRequest:
    """Input to the injected round runner."""

    run_id: str
    round_number: int
    workflow: WorkflowContext
    prior_round_failures: tuple[CandidateFailure, ...] = ()
    prior_failures: tuple[str, ...] = ()
    history: tuple[RoundEvidence, ...] = ()

    @property
    def prior_vector(self) -> Mapping[str, Any] | None:
        """The previous completed round's best score vector, for convergence."""
        return self.history[-1].score_vector if self.history else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "round_number": self.round_number,
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
    original_kept: bool
    status: str
    selected_candidate_id: str | None = None
    selected_strategy: str | None = None
    candidate_failures: tuple[CandidateFailure, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)
    continuation_requested: bool = False
    final_prompt: str | None = None
    score_vector: Mapping[str, Any] | None = None

    @classmethod
    def from_outcome(cls, *, round_number: int, outcome: RoundOutcome) -> RoundEvidence:
        return cls(
            round_number=round_number,
            original_kept=outcome.original_kept,
            status=outcome.status,
            final_prompt=outcome.final_prompt,
            selected_candidate_id=outcome.selected_candidate_id,
            selected_strategy=outcome.selected_strategy,
            candidate_failures=outcome.failures,
            evidence=_public_mapping(outcome.evidence()),
            cost=_public_mapping(outcome.cost),
            timing=_public_mapping(outcome.timing),
            continuation_requested=outcome.continue_rounds,
            score_vector=dict(outcome.convergence)
            if outcome.convergence is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "original_kept": self.original_kept,
            "status": self.status,
            "final_prompt": self.final_prompt,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_strategy": self.selected_strategy,
            "candidate_failures": [
                failure.to_dict() for failure in self.candidate_failures
            ],
            "evidence": dict(self.evidence),
            "cost": dict(self.cost),
            "timing": dict(self.timing),
            "continuation_requested": self.continuation_requested,
            "convergence": dict(self.score_vector) if self.score_vector else None,
        }


@dataclass(frozen=True)
class RepeatResult:
    """Final round-runner payload augmented with repeat-round state."""

    run_id: str
    final_prompt: str
    original_kept: bool
    history: tuple[RoundEvidence, ...]
    outcome: RoundOutcome
    workflow: WorkflowContext | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = self.outcome.payload()
        payload["run_id"] = self.run_id
        payload["final_prompt"] = self.final_prompt
        payload["original_kept"] = self.original_kept
        if self.original_kept and not self.outcome.converged:
            payload["status"] = "failed"
            if not _mapping_or_empty(
                _mapping_or_empty(payload.get("report")).get("failure")
            ):
                payload.setdefault("failure", failure_unverified())
        report = dict(_mapping_or_empty(payload.get("report")))
        report["history"] = [
            round_evidence.to_dict() for round_evidence in self.history
        ]
        report["round_history"] = report["history"]
        payload["report"] = report
        if self.outcome.converged:
            payload["converged"] = True
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
    """Retry until convergence, an external stop, or an impossible Route."""

    def run(
        self,
        *,
        run_id: str,
        prompt: str,
        execute_round: RoundRunner,
        questions: Sequence[str] = (),
        answers: Mapping[str, str] | None = None,
        assumptions: Sequence[str] = (),
        initial_failures: Sequence[CandidateFailure | Mapping[str, Any] | str] = (),
        prior_history: tuple[RoundEvidence, ...] = (),
    ) -> RepeatResult:
        workflow = WorkflowContext(
            run_id=run_id,
            original_prompt=prompt,
            questions=tuple(str(question) for question in questions),
            answers=dict(answers or {}),
            assumptions=tuple(str(assumption) for assumption in assumptions),
        )
        supplied = tuple(_supplied_failure(value) for value in initial_failures)
        if not supplied and prior_history:
            supplied = prior_history[-1].candidate_failures
        history, outcome, best_prompt = _run_rounds(
            execute_round,
            workflow,
            history=prior_history,
            failures=supplied,
        )
        return RepeatResult(
            run_id=run_id,
            final_prompt=best_prompt,
            original_kept=best_prompt == workflow.original_prompt,
            history=history,
            outcome=outcome,
            workflow=workflow,
        )


def _run_rounds(
    execute_round: RoundRunner,
    workflow: WorkflowContext,
    *,
    history: tuple[RoundEvidence, ...],
    failures: tuple[CandidateFailure, ...],
) -> tuple[tuple[RoundEvidence, ...], RoundOutcome, str]:
    """Run rounds without an attempt cap until evidence or control stops them."""
    next_round_number = history[-1].round_number + 1 if history else 1
    best_prompt, best_score = _best_prompt_from_history(
        history, workflow.original_prompt
    )
    best_outcome: RoundOutcome | None = None
    stored_best = next(
        (
            entry
            for entry in reversed(history)
            if entry.final_prompt == best_prompt
            and entry.score_vector
            and entry.score_vector.get("selected") is True
            and entry.score_vector.get("passed") is True
            and mean_score(entry.score_vector.get("scores", {})) == best_score
        ),
        None,
    )
    best_source_round = stored_best.round_number if stored_best else None
    best_vector = stored_best.score_vector if stored_best else None
    while True:
        request = RoundRequest(
            run_id=workflow.run_id,
            round_number=next_round_number,
            workflow=workflow,
            prior_round_failures=failures,
            prior_failures=tuple(failure.summary for failure in failures),
            history=history,
        )
        outcome = execute_round(request)
        if best_outcome is None and stored_best is not None:
            saved_vector = dict(stored_best.score_vector or {})
            best_outcome = replace(
                outcome,
                final_prompt=best_prompt,
                original_kept=best_prompt == workflow.original_prompt,
                summary=convergence_summary(
                    saved_vector.get("scores", {}),
                    saved_vector.get("floors", {}),
                    gain=saved_vector.get("gain"),
                ),
                convergence=saved_vector,
                restored_evidence=stored_best.evidence,
                tests=tuple(stored_best.evidence.get("tests", ())),
                candidates=tuple(stored_best.evidence.get("candidates", ())),
                failures=stored_best.candidate_failures,
                cost=stored_best.cost,
                timing=stored_best.timing,
                reported_failure=None,
            )
        evidence = RoundEvidence.from_outcome(
            round_number=request.round_number, outcome=outcome
        )
        vector = evidence.score_vector or {}
        scores = vector.get("scores") if isinstance(vector, Mapping) else None
        if (
            isinstance(scores, Mapping)
            and vector.get("selected") is True
            and vector.get("passed") is True
        ):
            score = mean_score(scores)
            if score >= best_score:
                best_score = score
                best_prompt = outcome.final_prompt
                best_outcome = outcome
                best_source_round = request.round_number
                best_vector = dict(vector)
        elif (
            outcome.status == "clarified"
            and outcome.final_prompt != workflow.original_prompt
        ):
            best_prompt = outcome.final_prompt
        history = (*history, evidence)
        failures = evidence.candidate_failures
        if not evidence.continuation_requested:
            if outcome.converged and best_outcome is not None:
                terminal_vector = dict(vector)
                accepted_vector = dict(best_vector or {})
                accepted_vector.update(
                    {
                        "status": "converged",
                        "gain": terminal_vector.get("gain"),
                        "source_round": best_source_round,
                        "selected_candidate_id": best_outcome.selected_candidate_id,
                        "terminal_scores": terminal_vector.get("scores"),
                        "terminal_gain": terminal_vector.get("gain"),
                    }
                )
                outcome = replace(
                    best_outcome, status=outcome.status, convergence=accepted_vector
                )
            return history, outcome, best_prompt
        next_round_number += 1


def _best_prompt_from_history(
    history: Sequence[RoundEvidence], original_prompt: str
) -> tuple[str, float]:
    """Recover the highest-scoring floor-passing selected prompt in history."""
    best_prompt = original_prompt
    best_score = -1.0
    for evidence in history:
        if (
            evidence.status == "clarified"
            and evidence.final_prompt
            and evidence.final_prompt != original_prompt
        ):
            best_prompt = evidence.final_prompt
        vector = evidence.score_vector
        scores = vector.get("scores") if isinstance(vector, Mapping) else None
        if (
            evidence.selected_candidate_id is None
            or not isinstance(vector, Mapping)
            or not isinstance(scores, Mapping)
            or vector.get("passed") is not True
        ):
            continue
        selection = evidence.evidence.get("selection_evidence")
        selection = selection if isinstance(selection, Mapping) else {}
        candidate = selection.get("selected_candidate")
        if not isinstance(candidate, Mapping):
            continue
        prompt = candidate.get("text") or candidate.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            continue
        score = mean_score(scores)
        if score >= best_score:
            best_prompt = prompt
            best_score = score
    return best_prompt, best_score


def _supplied_failure(
    value: CandidateFailure | Mapping[str, Any] | str,
) -> CandidateFailure:
    if isinstance(value, CandidateFailure):
        return value
    if isinstance(value, str):
        return CandidateFailure(candidate_id="unknown", reasons=(value,))
    return CandidateFailure.from_dict(value)


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
    convergence = value.get("convergence")
    return RoundEvidence(
        round_number=int(value.get("round_number") or value.get("round") or 1),
        original_kept=_as_bool(value.get("original_kept", False)),
        status=str(value.get("status") or "no_change"),
        final_prompt=str(value["final_prompt"])
        if value.get("final_prompt") is not None
        else None,
        selected_candidate_id=_optional_string(value.get("selected_candidate_id")),
        selected_strategy=_optional_string(value.get("selected_strategy")),
        candidate_failures=tuple(
            CandidateFailure.from_dict(item) for item in raw_failures
        ),
        evidence=_public_mapping(value.get("evidence") or {}),
        cost=_public_mapping(value.get("cost") or {}),
        timing=_public_mapping(value.get("timing") or {}),
        continuation_requested=_as_bool(value.get("continuation_requested", False)),
        score_vector=dict(convergence) if isinstance(convergence, Mapping) else None,
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
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
