"""Auditable rubric revision workflows.

The runtime does not start revisions implicitly. NEW, REVISED, and DROPPED
proposals retain manual decisions. The explicit REWORD path uses sealed
validation and an automatic, recorded adoption policy.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .catalog import DEFAULT_GO_WRITER, JEV_MODEL
from .gateway import Gateway, completion_text, writer_messages


class RevisionKind(StrEnum):
    NEW = "new"
    REVISED = "revised"
    DROPPED = "dropped"
    REWORD = "reword"


class RecommendedDecision(StrEnum):
    ADOPT = "adopt"
    HOLD = "hold"
    REJECT = "reject"


class MaintainerDecisionKind(StrEnum):
    ADOPT = "adopt"
    REJECT = "reject"


class RegressionRisk(StrEnum):
    NONE = "none"
    ELEVATED = "elevated"


@dataclass(frozen=True, slots=True)
class RubricQuestion:
    """A reusable Jev decision question in a diagnosis rubric."""

    question_id: str
    text: str
    response_type: str = "noul"
    threshold: float = 0.5
    missing_when: str = "no"
    question_version: int = 1
    calibration_snapshot: str | None = None
    calibration_policy_version: str | None = None
    calibration_artifact: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.text.strip():
            raise ValueError("question text must not be empty")
        if not self.response_type.strip():
            raise ValueError("response_type must not be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.missing_when not in {"yes", "no"}:
            raise ValueError("missing_when must be yes or no")
        if self.question_version < 1:
            raise ValueError("question_version must be positive")
        if (self.calibration_snapshot is None) != (
            self.calibration_policy_version is None
        ):
            raise ValueError(
                "calibration snapshot and policy version must appear together"
            )
        if (
            self.calibration_snapshot is not None
            and self.calibration_snapshot != JEV_MODEL
        ):
            raise ValueError(
                "rubric calibration snapshot differs from the pinned Jev model"
            )
        if self.calibration_artifact is not None:
            questions = self.calibration_artifact.get("questions")
            result = (
                questions.get(f"rubric:{self.question_id}")
                if isinstance(questions, Mapping)
                else None
            )
            identity = result.get("identity") if isinstance(result, Mapping) else None
            if (
                not isinstance(result, Mapping)
                or not isinstance(identity, Mapping)
                or result.get("verdict") not in {"gate", "gate-above-confidence"}
                or result.get("threshold") != self.threshold
                or identity.get("question") != self.text
                or identity.get("primitive") != self.response_type
                or identity.get("event_mapping")
                != {
                    "polarity": "positive" if self.missing_when == "yes" else "negative"
                }
                or identity.get("answering_snapshot") != self.calibration_snapshot
                or identity.get("policy_version") != self.calibration_policy_version
            ):
                raise ValueError(
                    "rubric calibration artifact does not match its question"
                )


@dataclass(frozen=True, slots=True)
class RubricVersion:
    """An immutable runtime rubric version."""

    version_id: str
    questions: tuple[RubricQuestion, ...]
    parent_version_id: str | None = None
    created_at: str | None = None
    disabled_default_question_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.version_id.strip():
            raise ValueError("version_id must not be empty")
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric question IDs must be unique")
        object.__setattr__(
            self,
            "questions",
            tuple(sorted(self.questions, key=lambda item: item.question_id)),
        )
        object.__setattr__(
            self,
            "disabled_default_question_ids",
            tuple(sorted(set(self.disabled_default_question_ids))),
        )

    def question(self, question_id: str) -> RubricQuestion | None:
        return next(
            (item for item in self.questions if item.question_id == question_id), None
        )

    def apply(
        self,
        change: RubricQuestionChange,
        *,
        version_id: str,
        created_at: str,
    ) -> RubricVersion:
        """Return the next immutable version after applying one change."""

        if change.after is not None and change.after.calibration_artifact is not None:
            artifact = change.after.calibration_artifact
            entry = artifact["questions"][f"rubric:{change.after.question_id}"]
            if entry["identity"].get("rubric_version") != version_id:
                raise ValueError(
                    "calibration artifact targets a different rubric version"
                )

        questions = {question.question_id: question for question in self.questions}
        disabled_defaults = set(self.disabled_default_question_ids)
        if change.kind is RevisionKind.NEW:
            if change.after is None or change.after.question_id in questions:
                raise StaleProposalError("a new question must have a new question ID")
            questions[change.after.question_id] = change.after
            disabled_defaults.discard(change.after.question_id)
        elif change.kind in {RevisionKind.REVISED, RevisionKind.REWORD}:
            if (
                change.after is None
                or questions.get(change.question_id) != change.before
            ):
                raise StaleProposalError(
                    "the revised question is not the question in this rubric"
                )
            questions[change.question_id] = change.after
            disabled_defaults.discard(change.question_id)
        else:
            if (
                change.after is not None
                or questions.get(change.question_id) != change.before
            ):
                raise StaleProposalError(
                    "the dropped question is not the question in this rubric"
                )
            del questions[change.question_id]
            disabled_defaults.add(change.question_id)

        return RubricVersion(
            version_id=version_id,
            questions=tuple(questions.values()),
            parent_version_id=self.version_id,
            created_at=created_at,
            disabled_default_question_ids=tuple(disabled_defaults),
        )


@dataclass(frozen=True, slots=True)
class RubricQuestionChange:
    """A proposed new, revised, or dropped rubric question."""

    kind: RevisionKind
    question_id: str
    before: RubricQuestion | None
    after: RubricQuestion | None
    rationale: str

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.rationale.strip():
            raise ValueError("revision rationale must not be empty")
        if self.kind is RevisionKind.NEW:
            if (
                self.before is not None
                or self.after is None
                or self.after.question_id != self.question_id
            ):
                raise ValueError("a new question needs only its proposed definition")
        elif self.kind in {RevisionKind.REVISED, RevisionKind.REWORD}:
            if (
                self.before is None
                or self.after is None
                or self.before.question_id != self.question_id
                or self.after.question_id != self.question_id
                or self.before == self.after
            ):
                raise ValueError(
                    "a revised question needs two different definitions with the same ID"
                )
            if self.kind is RevisionKind.REWORD and (
                self.before.response_type != self.after.response_type
                or self.before.missing_when != self.after.missing_when
                or self.before.text == self.after.text
            ):
                raise ValueError("rewording must preserve the question's judgment")
        elif (
            self.before is None
            or self.after is not None
            or self.before.question_id != self.question_id
        ):
            raise ValueError("a dropped question needs only its previous definition")

    @property
    def signature(self) -> tuple[str, str, str | None, str | None, str | None]:
        return (
            self.kind.value,
            self.question_id,
            _json_dump(self.before) if self.before is not None else None,
            _json_dump(self.after) if self.after is not None else None,
            self.rationale,
        )


@dataclass(frozen=True, slots=True)
class MeasuredError:
    """A predictor/evaluation error that may motivate a rubric change."""

    error_id: str
    evaluation_case_ids: tuple[str, ...]
    current_question_id: str | None
    proposed_question: RubricQuestion | None
    observation: str
    weight: float = 1.0
    predictor_artifact: str | None = None

    def __post_init__(self) -> None:
        if not self.error_id.strip():
            raise ValueError("error_id must not be empty")
        if not self.evaluation_case_ids:
            raise ValueError("a measured error must cite at least one evaluation case")
        if not self.observation.strip():
            raise ValueError("measured error observation must not be empty")
        if self.weight <= 0:
            raise ValueError("measured error weight must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationSet:
    """Identity of the fixed, replay-backed evaluation set used for a workflow."""

    identifier: str
    input_digest: str
    case_ids: tuple[str, ...]
    replay_artifact: str

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.input_digest.strip():
            raise ValueError("evaluation set identifier and digest must not be empty")
        if not self.case_ids:
            raise ValueError("evaluation set must contain cases")
        if not self.replay_artifact.strip():
            raise ValueError("offline evaluation requires a replay artifact")


class MeasuredErrorSource(Protocol):
    """Adapter seam for failure-predictor reports and evaluation evidence."""

    def errors(self, evaluation_set: EvaluationSet) -> Sequence[MeasuredError]: ...


class RevisionProposer(Protocol):
    def propose(
        self,
        rubric: RubricVersion,
        errors: Sequence[MeasuredError],
    ) -> Sequence[RubricQuestionChange]: ...


class RevisionEvaluator(Protocol):
    """Adapter seam for the replay-backed evaluation harness."""

    def evaluate_baseline(
        self,
        rubric: RubricVersion,
        evaluation_set: EvaluationSet,
    ) -> EvaluationMetrics: ...

    def evaluate_revision(
        self,
        rubric: RubricVersion,
        change: RubricQuestionChange,
        evaluation_set: EvaluationSet,
        baseline: EvaluationMetrics,
    ) -> EvaluationMetrics: ...


class MeasuredErrorProposer:
    """Deterministic offline proposer for measured writer suggestions."""

    def propose(
        self,
        rubric: RubricVersion,
        errors: Sequence[MeasuredError],
    ) -> Sequence[RubricQuestionChange]:
        changes: list[RubricQuestionChange] = []
        for error in errors:
            current_id = error.current_question_id
            proposed = error.proposed_question
            current = (
                rubric.question(current_id)
                if current_id
                else (rubric.question(proposed.question_id) if proposed else None)
            )
            if current is None:
                if proposed is None:
                    continue
                changes.append(
                    RubricQuestionChange(
                        kind=RevisionKind.NEW,
                        question_id=proposed.question_id,
                        before=None,
                        after=proposed,
                        rationale=error.observation,
                    )
                )
            elif proposed is None:
                changes.append(
                    RubricQuestionChange(
                        kind=RevisionKind.DROPPED,
                        question_id=current.question_id,
                        before=current,
                        after=None,
                        rationale=error.observation,
                    )
                )
            elif proposed.question_id == current.question_id and proposed != current:
                changes.append(
                    RubricQuestionChange(
                        kind=RevisionKind.REVISED,
                        question_id=current.question_id,
                        before=current,
                        after=proposed,
                        rationale=error.observation,
                    )
                )
        return changes


class WriterRevisionProposer:
    """Ask a writer to turn measured errors into candidate Jev questions."""

    def __init__(
        self, gateway: Gateway, *, writer_model: str = DEFAULT_GO_WRITER
    ) -> None:
        self.gateway = gateway
        self.writer_model = writer_model
        self._evidence: dict[
            tuple[str, str, str | None, str | None, str | None],
            tuple[MeasuredError, ...],
        ] = {}

    def propose(
        self, rubric: RubricVersion, errors: Sequence[MeasuredError]
    ) -> Sequence[RubricQuestionChange]:
        instructions = (
            "Propose at most one new, revised, or dropped Jev diagnosis question per measured error. "
            'Return JSON only as {"suggestions":[{"error_id":"...","action":"new|revised|dropped",'
            '"question":{"question_id":"...","text":"...","response_type":"noul",'
            '"threshold":0.8,"missing_when":"yes|no"}}]}. '
            "For dropped, omit question. Use only errors and rubric in state as evidence."
        )
        state = {
            "rubric": [asdict(question) for question in rubric.questions],
            "errors": [asdict(error) for error in errors],
        }
        response = self.gateway.chat(
            self.writer_model, writer_messages(instructions, state), role="writer"
        )
        payload = json.loads(completion_text(response))
        suggestions = (
            payload.get("suggestions") if isinstance(payload, Mapping) else None
        )
        if not isinstance(suggestions, list):
            raise TypeError("writer revision response must contain suggestions")
        by_id = {error.error_id: error for error in errors}
        proposed_errors: list[MeasuredError] = []
        for item in suggestions:
            if not isinstance(item, Mapping):
                raise TypeError("each writer suggestion must be an object")
            error_id = str(item.get("error_id", ""))
            if error_id not in by_id:
                raise ValueError("writer suggestion cites an unknown measured error")
            error = by_id[error_id]
            action = item.get("action")
            if action not in {"new", "revised", "dropped"}:
                raise ValueError("writer suggestion has an unsupported action")
            current = (
                rubric.question(error.current_question_id)
                if error.current_question_id
                else None
            )
            if (
                action == "new"
                and current is not None
                or action != "new"
                and current is None
            ):
                raise ValueError("writer suggestion does not match the current rubric")
            question_data = item.get("question")
            if action == "dropped":
                if question_data is not None:
                    raise ValueError("dropped questions must not include a replacement")
                question = None
            else:
                if not isinstance(question_data, Mapping):
                    raise ValueError("writer suggestion requires a question")
                question = RubricQuestion(**question_data)
                if question.response_type != "noul":
                    raise ValueError("diagnosis revision questions must use noul")
            proposed_errors.append(replace(error, proposed_question=question))
        changes = tuple(MeasuredErrorProposer().propose(rubric, proposed_errors))
        self._evidence = {
            change.signature: tuple(
                error
                for error in proposed_errors
                if error.current_question_id == change.question_id
                or error.proposed_question is not None
                and error.proposed_question.question_id == change.question_id
            )
            for change in changes
        }
        return changes

    def evidence_for(self, change: RubricQuestionChange) -> tuple[MeasuredError, ...]:
        return self._evidence.get(change.signature, ())


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    precision: float
    recall: float
    total_cost_usd: float
    regression_count: int
    case_count: int
    evaluated_case_ids: tuple[str, ...] = ()
    regressed_case_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.precision <= 1.0:
            raise ValueError("precision must be between 0 and 1")
        if not 0.0 <= self.recall <= 1.0:
            raise ValueError("recall must be between 0 and 1")
        if self.total_cost_usd < 0:
            raise ValueError("total cost must not be negative")
        if self.regression_count < 0 or self.regression_count > self.case_count:
            raise ValueError("regression count is outside the evaluated case set")
        if self.case_count <= 0:
            raise ValueError("evaluation must include at least one case")
        if self.evaluated_case_ids and len(self.evaluated_case_ids) != self.case_count:
            raise ValueError("case ids do not match the evaluated case count")
        if self.evaluated_case_ids and not set(self.regressed_case_ids) <= set(
            self.evaluated_case_ids
        ):
            raise ValueError("regression ids must belong to the evaluated case set")

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def regression_rate(self) -> float:
        return self.regression_count / self.case_count


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    max_cost_increase_usd: float = 0.01

    def __post_init__(self) -> None:
        if self.max_cost_increase_usd < 0:
            raise ValueError("maximum cost increase must not be negative")


@dataclass(frozen=True, slots=True)
class RevisionEvaluation:
    """Required evidence presented before a maintainer can make a decision."""

    baseline: EvaluationMetrics
    candidate: EvaluationMetrics
    precision_delta: float
    recall_delta: float
    f1_delta: float
    cost_delta_usd: float
    baseline_regression_rate: float
    candidate_regression_rate: float
    additional_regressions: int
    regression_risk: RegressionRisk
    recommended_decision: RecommendedDecision
    reasons: tuple[str, ...]
    policy: EvaluationPolicy

    @classmethod
    def compare(
        cls,
        baseline: EvaluationMetrics,
        candidate: EvaluationMetrics,
        policy: EvaluationPolicy,
    ) -> RevisionEvaluation:
        if baseline.case_count != candidate.case_count:
            raise ValueError(
                "baseline and candidate must use the same evaluation cases"
            )
        if (
            baseline.evaluated_case_ids
            and candidate.evaluated_case_ids
            and set(baseline.evaluated_case_ids) != set(candidate.evaluated_case_ids)
        ):
            raise ValueError("baseline and candidate evaluated different case ids")
        precision_delta = candidate.precision - baseline.precision
        recall_delta = candidate.recall - baseline.recall
        f1_delta = candidate.f1 - baseline.f1
        cost_delta = round(candidate.total_cost_usd - baseline.total_cost_usd, 12)
        additional_regressions = (
            len(set(candidate.regressed_case_ids) - set(baseline.regressed_case_ids))
            if baseline.evaluated_case_ids and candidate.evaluated_case_ids
            else candidate.regression_count - baseline.regression_count
        )
        risk = (
            RegressionRisk.ELEVATED
            if additional_regressions > 0
            else RegressionRisk.NONE
        )
        reasons: list[str] = []
        if f1_delta < 0:
            reasons.append("diagnosis F1 regressed")
        if cost_delta > policy.max_cost_increase_usd:
            reasons.append("cost increase exceeds the offline policy")
        if additional_regressions > 0:
            reasons.append("the candidate introduced additional regressions")
        if not reasons and f1_delta == 0:
            reasons.append("the revision produced no measured diagnosis improvement")
        if not reasons:
            reasons.append("diagnosis improved within the cost and regression policy")

        if f1_delta < 0:
            recommendation = RecommendedDecision.REJECT
        elif (
            additional_regressions > 0
            or cost_delta > policy.max_cost_increase_usd
            or f1_delta == 0
        ):
            recommendation = RecommendedDecision.HOLD
        else:
            recommendation = RecommendedDecision.ADOPT

        return cls(
            baseline=baseline,
            candidate=candidate,
            precision_delta=precision_delta,
            recall_delta=recall_delta,
            f1_delta=f1_delta,
            cost_delta_usd=cost_delta,
            baseline_regression_rate=baseline.regression_rate,
            candidate_regression_rate=candidate.regression_rate,
            additional_regressions=additional_regressions,
            regression_risk=risk,
            recommended_decision=recommendation,
            reasons=tuple(reasons),
            policy=policy,
        )


@dataclass(frozen=True, slots=True)
class RevisionProposal:
    proposal_id: str
    workflow_id: str
    base_rubric_version_id: str
    change: RubricQuestionChange
    evidence: tuple[MeasuredError, ...]
    evaluation: RevisionEvaluation | None
    created_at: str


@dataclass(frozen=True, slots=True)
class RevisionWorkflow:
    workflow_id: str
    evaluation_set_id: str
    evaluation_input_digest: str
    replay_artifact: str
    base_rubric_version_id: str
    evidence_artifact_ids: tuple[str, ...]
    proposals: tuple[RevisionProposal, ...]
    previous_workflow_id: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class RubricVersionChange:
    from_version_id: str
    to_version_id: str
    added_question_ids: tuple[str, ...]
    revised_question_ids: tuple[str, ...]
    dropped_question_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowComparison:
    previous_workflow_id: str
    current_workflow_id: str
    rubric_change: RubricVersionChange
    shared_proposal_signatures: tuple[
        tuple[str, str, str | None, str | None, str | None], ...
    ]
    current_only_proposal_signatures: tuple[
        tuple[str, str, str | None, str | None, str | None], ...
    ]
    previous_only_proposal_signatures: tuple[
        tuple[str, str, str | None, str | None, str | None], ...
    ]
    current_mean_candidate_f1: float | None
    previous_mean_candidate_f1: float | None
    mean_candidate_f1_delta: float | None
    current_mean_candidate_cost_usd: float | None
    previous_mean_candidate_cost_usd: float | None
    mean_candidate_cost_delta_usd: float | None
    current_mean_candidate_regression_rate: float | None
    previous_mean_candidate_regression_rate: float | None
    mean_candidate_regression_rate_delta: float | None


@dataclass(frozen=True, slots=True)
class RevisionWorkflowResult:
    workflow: RevisionWorkflow
    comparison: WorkflowComparison | None


@dataclass(frozen=True, slots=True)
class RevisionDecision:
    decision_id: str
    proposal_id: str
    maintainer: str
    decision: MaintainerDecisionKind
    rationale: str
    proposal: RevisionProposal
    resulting_rubric: RubricVersion
    decided_at: str
    actor_type: str = "human"
    automatic_evidence: Mapping[str, Any] = field(default_factory=dict)


class StaleProposalError(ValueError):
    """Raised when a proposal no longer applies to the active rubric."""


def _utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(
        asdict(value) if is_dataclass(value) and not isinstance(value, type) else value,
        sort_keys=True,
        separators=(",", ":"),
    )


def _question_from_dict(value: dict[str, Any]) -> RubricQuestion:
    return RubricQuestion(**value)


def _change_from_dict(value: dict[str, Any]) -> RubricQuestionChange:
    before = (
        _question_from_dict(value["before"]) if value["before"] is not None else None
    )
    after = _question_from_dict(value["after"]) if value["after"] is not None else None
    return RubricQuestionChange(
        kind=RevisionKind(value["kind"]),
        question_id=value["question_id"],
        before=before,
        after=after,
        rationale=value["rationale"],
    )


def _error_from_dict(value: dict[str, Any]) -> MeasuredError:
    proposed = (
        _question_from_dict(value["proposed_question"])
        if value["proposed_question"] is not None
        else None
    )
    return MeasuredError(
        error_id=value["error_id"],
        evaluation_case_ids=tuple(value["evaluation_case_ids"]),
        current_question_id=value["current_question_id"],
        proposed_question=proposed,
        observation=value["observation"],
        weight=value["weight"],
        predictor_artifact=value["predictor_artifact"],
    )


def _metrics_from_dict(value: Mapping[str, Any]) -> EvaluationMetrics:
    return EvaluationMetrics(**value)


def _evaluation_from_dict(value: Mapping[str, Any]) -> RevisionEvaluation:
    return RevisionEvaluation(
        baseline=_metrics_from_dict(value["baseline"]),
        candidate=_metrics_from_dict(value["candidate"]),
        precision_delta=value["precision_delta"],
        recall_delta=value["recall_delta"],
        f1_delta=value["f1_delta"],
        cost_delta_usd=value["cost_delta_usd"],
        baseline_regression_rate=value["baseline_regression_rate"],
        candidate_regression_rate=value["candidate_regression_rate"],
        additional_regressions=value["additional_regressions"],
        regression_risk=RegressionRisk(value["regression_risk"]),
        recommended_decision=RecommendedDecision(value["recommended_decision"]),
        reasons=tuple(value["reasons"]),
        policy=EvaluationPolicy(**value["policy"]),
    )


def _proposal_from_dict(value: Mapping[str, Any]) -> RevisionProposal:
    evaluation = (
        _evaluation_from_dict(value["evaluation"])
        if value["evaluation"] is not None
        else None
    )
    return RevisionProposal(
        proposal_id=value["proposal_id"],
        workflow_id=value["workflow_id"],
        base_rubric_version_id=value["base_rubric_version_id"],
        change=_change_from_dict(value["change"]),
        evidence=tuple(_error_from_dict(item) for item in value["evidence"]),
        evaluation=evaluation,
        created_at=value["created_at"],
    )


def _rubric_from_dict(value: Mapping[str, Any]) -> RubricVersion:
    return RubricVersion(
        version_id=value["version_id"],
        questions=tuple(_question_from_dict(item) for item in value["questions"]),
        parent_version_id=value["parent_version_id"],
        created_at=value["created_at"],
        disabled_default_question_ids=tuple(
            value.get("disabled_default_question_ids", ())
        ),
    )


def _workflow_from_dict(
    value: Mapping[str, Any], proposals: Sequence[RevisionProposal]
) -> RevisionWorkflow:
    return RevisionWorkflow(
        workflow_id=value["workflow_id"],
        evaluation_set_id=value["evaluation_set_id"],
        evaluation_input_digest=value["evaluation_input_digest"],
        replay_artifact=value["replay_artifact"],
        base_rubric_version_id=value["base_rubric_version_id"],
        evidence_artifact_ids=tuple(value["evidence_artifact_ids"]),
        proposals=tuple(proposals),
        previous_workflow_id=value["previous_workflow_id"],
        created_at=value["created_at"],
    )


def _decision_from_dict(value: Mapping[str, Any]) -> RevisionDecision:
    return RevisionDecision(
        decision_id=value["decision_id"],
        proposal_id=value["proposal_id"],
        maintainer=value["maintainer"],
        decision=MaintainerDecisionKind(value["decision"]),
        rationale=value["rationale"],
        proposal=_proposal_from_dict(value["proposal"]),
        resulting_rubric=_rubric_from_dict(value["resulting_rubric"]),
        decided_at=value["decided_at"],
        actor_type=value.get("actor_type", "human"),
        automatic_evidence=value.get("automatic_evidence", {}),
    )


def _rubric_comparison(
    before: RubricVersion, after: RubricVersion
) -> RubricVersionChange:
    before_questions = {question.question_id: question for question in before.questions}
    after_questions = {question.question_id: question for question in after.questions}
    added = tuple(sorted(after_questions.keys() - before_questions.keys()))
    dropped = tuple(sorted(before_questions.keys() - after_questions.keys()))
    revised = tuple(
        sorted(
            question_id
            for question_id in before_questions.keys() & after_questions.keys()
            if before_questions[question_id] != after_questions[question_id]
        )
    )
    return RubricVersionChange(
        from_version_id=before.version_id,
        to_version_id=after.version_id,
        added_question_ids=added,
        revised_question_ids=revised,
        dropped_question_ids=dropped,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


class SQLiteRubricStore:
    """Durable proposal, evidence, decision, and rubric-version audit store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rubric_versions (
                    version_id TEXT PRIMARY KEY,
                    parent_version_id TEXT,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_rubric (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version_id TEXT NOT NULL REFERENCES rubric_versions(version_id)
                );
                CREATE TABLE IF NOT EXISTS revision_sequences (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS revision_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS revision_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL REFERENCES revision_workflows(workflow_id),
                    ordinal INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE (workflow_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS maintainer_decisions (
                    decision_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL UNIQUE REFERENCES revision_proposals(proposal_id),
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reword_holdouts (
                    digest TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reword_holdout_groups (
                    group_id TEXT PRIMARY KEY,
                    digest TEXT NOT NULL REFERENCES reword_holdouts(digest)
                );
                CREATE TABLE IF NOT EXISTS reword_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    holdout_digest TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reword_evaluations (
                    input_digest TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS automatic_reword_decisions (
                    decision_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                """
            )

    def initialize(self, rubric: RubricVersion) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO rubric_versions VALUES (?, ?, ?)",
                (rubric.version_id, rubric.parent_version_id, _json_dump(rubric)),
            )
            connection.execute(
                "INSERT OR IGNORE INTO active_rubric(singleton, version_id) VALUES (1, ?)",
                (rubric.version_id,),
            )

    def _next_id(self, connection: sqlite3.Connection, name: str, prefix: str) -> str:
        connection.execute(
            "INSERT OR IGNORE INTO revision_sequences(name, value) VALUES (?, 0)",
            (name,),
        )
        connection.execute(
            "UPDATE revision_sequences SET value = value + 1 WHERE name = ?",
            (name,),
        )
        value = connection.execute(
            "SELECT value FROM revision_sequences WHERE name = ?",
            (name,),
        ).fetchone()[0]
        return f"{prefix}-{value}"

    def active_rubric(self) -> RubricVersion:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT rv.payload FROM active_rubric ar JOIN rubric_versions rv ON rv.version_id = ar.version_id"
            ).fetchone()
        if row is None:
            raise RuntimeError("rubric store has not been initialized")
        return _rubric_from_dict(json.loads(row[0]))

    def get_rubric(self, version_id: str) -> RubricVersion:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM rubric_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown rubric version: {version_id}")
        return _rubric_from_dict(json.loads(row[0]))

    def save_workflow(self, workflow: RevisionWorkflow) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO revision_workflows VALUES (?, ?)",
                (workflow.workflow_id, _json_dump(workflow)),
            )
            for ordinal, proposal in enumerate(workflow.proposals):
                connection.execute(
                    "INSERT INTO revision_proposals VALUES (?, ?, ?, ?)",
                    (
                        proposal.proposal_id,
                        workflow.workflow_id,
                        ordinal,
                        _json_dump(proposal),
                    ),
                )

    def get_workflow(self, workflow_id: str) -> RevisionWorkflow:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM revision_workflows WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown revision workflow: {workflow_id}")
            proposal_rows = connection.execute(
                "SELECT payload FROM revision_proposals WHERE workflow_id = ? ORDER BY ordinal",
                (workflow_id,),
            ).fetchall()
        return _workflow_from_dict(
            json.loads(row[0]),
            [_proposal_from_dict(json.loads(item[0])) for item in proposal_rows],
        )

    def get_proposal(self, proposal_id: str) -> RevisionProposal:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM revision_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown rubric proposal: {proposal_id}")
        return _proposal_from_dict(json.loads(row[0]))

    def record_decision(
        self,
        decision: RevisionDecision,
        adopted_rubric: RubricVersion | None = None,
    ) -> None:
        with self._connect() as connection:
            if decision.decision is MaintainerDecisionKind.ADOPT:
                if (
                    adopted_rubric is None
                    or adopted_rubric.version_id != decision.resulting_rubric.version_id
                ):
                    raise ValueError("adoption requires its resulting rubric")
                active = connection.execute(
                    "SELECT version_id FROM active_rubric WHERE singleton = 1"
                ).fetchone()
                if (
                    active is None
                    or active[0] != decision.proposal.base_rubric_version_id
                ):
                    raise StaleProposalError(
                        "the active rubric changed after evaluation"
                    )
                connection.execute(
                    "INSERT INTO rubric_versions VALUES (?, ?, ?)",
                    (
                        adopted_rubric.version_id,
                        adopted_rubric.parent_version_id,
                        _json_dump(adopted_rubric),
                    ),
                )
                connection.execute(
                    "UPDATE active_rubric SET version_id = ? WHERE singleton = 1",
                    (adopted_rubric.version_id,),
                )
            elif adopted_rubric is not None:
                raise ValueError("rejection cannot change the active rubric")
            connection.execute(
                "INSERT INTO maintainer_decisions VALUES (?, ?, ?)",
                (decision.decision_id, decision.proposal_id, _json_dump(decision)),
            )

    def get_reword_attempt(self, attempt_id: str) -> Mapping[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM reword_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def get_reword_evaluation(self, input_digest: str) -> Mapping[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM reword_evaluations WHERE input_digest = ?",
                (input_digest,),
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def cache_reword_evaluation(
        self, input_digest: str, payload: Mapping[str, Any]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO reword_evaluations VALUES (?, ?)",
                (input_digest, _json_dump(payload)),
            )

    def holdout_consumed(self, digest: str, groups: Sequence[str] = ()) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM reword_holdouts WHERE digest = ?", (digest,)
            ).fetchone()
            if row is not None:
                return True
            if groups:
                placeholders = ",".join("?" for _ in groups)
                row = connection.execute(
                    f"SELECT 1 FROM reword_holdout_groups WHERE group_id IN ({placeholders}) LIMIT 1",
                    tuple(groups),
                ).fetchone()
        return row is not None

    def record_reword_attempt(
        self,
        attempt_id: str,
        holdout_digest: str,
        payload: Mapping[str, Any],
        *,
        holdout_groups: Sequence[str] = (),
        adopted_rubric: RubricVersion | None = None,
        automated_decision: RevisionDecision | None = None,
        consume_holdout: bool = True,
    ) -> None:
        """Consume a sealed holdout once and compare-and-swap any adopted rubric."""
        with self._connect() as connection:
            if (adopted_rubric is None) != (automated_decision is None):
                raise ValueError("automated adoption requires its decision record")
            if consume_holdout:
                connection.execute(
                    "INSERT INTO reword_holdouts VALUES (?, ?)",
                    (holdout_digest, _json_dump(payload)),
                )
                for group_id in sorted(set(holdout_groups)):
                    claimed = connection.execute(
                        "INSERT OR IGNORE INTO reword_holdout_groups VALUES (?, ?)",
                        (group_id, holdout_digest),
                    )
                    if claimed.rowcount != 1:
                        raise StaleProposalError("final group was already consumed")
            if adopted_rubric is not None:
                active = connection.execute(
                    "SELECT version_id FROM active_rubric WHERE singleton = 1"
                ).fetchone()
                if active is None or active[0] != adopted_rubric.parent_version_id:
                    raise StaleProposalError(
                        "the active rubric changed after evaluation"
                    )
                connection.execute(
                    "INSERT INTO rubric_versions VALUES (?, ?, ?)",
                    (
                        adopted_rubric.version_id,
                        adopted_rubric.parent_version_id,
                        _json_dump(adopted_rubric),
                    ),
                )
                swapped = connection.execute(
                    "UPDATE active_rubric SET version_id = ? WHERE singleton = 1 AND version_id = ?",
                    (adopted_rubric.version_id, adopted_rubric.parent_version_id),
                )
                if swapped.rowcount != 1:
                    raise StaleProposalError(
                        "the active rubric changed after evaluation"
                    )
                assert automated_decision is not None
                connection.execute(
                    "INSERT INTO automatic_reword_decisions VALUES (?, ?)",
                    (automated_decision.decision_id, _json_dump(automated_decision)),
                )
            connection.execute(
                "INSERT INTO reword_attempts VALUES (?, ?, ?)",
                (attempt_id, holdout_digest, _json_dump(payload)),
            )

    def rollback_reword(self, version_id: str) -> RubricVersion:
        """Restore the immediate parent of the active automated version."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM rubric_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            active = connection.execute(
                "SELECT version_id FROM active_rubric WHERE singleton = 1"
            ).fetchone()
            if row is None or active is None or active[0] != version_id:
                raise StaleProposalError("only the active version can be rolled back")
            automated = connection.execute(
                "SELECT 1 FROM reword_attempts WHERE json_extract(payload, '$.adopted_version_id') = ? AND json_extract(payload, '$.status') = 'adopted'",
                (version_id,),
            ).fetchone()
            if automated is None:
                raise StaleProposalError(
                    "only an automated rewording can be rolled back"
                )
            current = _rubric_from_dict(json.loads(row[0]))
            if current.parent_version_id is None:
                raise StaleProposalError("the active rubric has no parent")
            parent = self.get_rubric(current.parent_version_id)
            restored = connection.execute(
                "UPDATE active_rubric SET version_id = ? WHERE singleton = 1 AND version_id = ?",
                (parent.version_id, version_id),
            )
            if restored.rowcount != 1:
                raise StaleProposalError("only the active version can be rolled back")
        return parent

    def list_decisions(self) -> tuple[RevisionDecision, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM maintainer_decisions ORDER BY decision_id"
            ).fetchall()
            automated = connection.execute(
                "SELECT payload FROM automatic_reword_decisions ORDER BY decision_id"
            ).fetchall()
        return tuple(
            sorted(
                (
                    _decision_from_dict(json.loads(row[0]))
                    for row in (*rows, *automated)
                ),
                key=lambda decision: decision.decision_id,
            )
        )


class RubricRevisionService:
    """Coordinates proposal, replay evaluation, maintainer review, and adoption."""

    def __init__(
        self,
        store: SQLiteRubricStore,
        error_source: MeasuredErrorSource,
        evaluator: RevisionEvaluator,
        *,
        proposer: RevisionProposer | None = None,
        writer_gateway: Gateway | None = None,
        policy: EvaluationPolicy | None = None,
        clock: Callable[[], str] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.store = store
        self.error_source = error_source
        self.evaluator = evaluator
        self.proposer = proposer or (
            WriterRevisionProposer(writer_gateway)
            if writer_gateway is not None
            else MeasuredErrorProposer()
        )
        self.policy = policy or EvaluationPolicy()
        self.clock = clock or _utc_now
        self.id_factory = id_factory or (lambda: str(uuid4()))
        self._proposals: dict[str, RevisionProposal] = {}

    def active_rubric(self) -> RubricVersion:
        """Runtime integration seam passed to the diagnoser factory."""

        return self.store.active_rubric()

    def reword(
        self,
        gateway: Gateway,
        question_id: str,
        dataset: Mapping[str, Any],
        *,
        attempt_id: str,
        policy: Any | None = None,
        decision_policy: Any | None = None,
    ) -> Mapping[str, Any]:
        """Run one noninteractive, bounded REWORD attempt on an active question."""
        from .reword_optimization import optimize_reword

        return optimize_reword(
            self.store,
            gateway,
            question_id,
            dataset,
            attempt_id=attempt_id,
            policy=policy,
            decision_policy=decision_policy,
        )

    def propose(
        self,
        evaluation_set: EvaluationSet,
        *,
        previous_workflow_id: str | None = None,
    ) -> RevisionWorkflowResult:
        if previous_workflow_id is not None:
            self.store.get_workflow(previous_workflow_id)
        rubric = self.active_rubric()
        errors = tuple(self.error_source.errors(evaluation_set))
        self._validate_errors(errors)
        changes = tuple(self.proposer.propose(rubric, errors))
        self._validate_changes(changes)
        baseline = self.evaluator.evaluate_baseline(rubric, evaluation_set)
        workflow_id = self.id_factory()
        created_at = self.clock()
        evidence_by_change: dict[
            tuple[str, str, str | None, str | None, str | None], list[MeasuredError]
        ] = {}
        for error in errors:
            key = self._error_change_key(rubric, error)
            if key is not None:
                evidence_by_change.setdefault(key, []).append(error)
        writer_evidence = getattr(self.proposer, "evidence_for", None)
        if callable(writer_evidence):
            for change in changes:
                existing = evidence_by_change.setdefault(change.signature, [])
                for error in writer_evidence(change):
                    if error.error_id not in {item.error_id for item in existing}:
                        existing.append(error)

        proposals: list[RevisionProposal] = []
        for change in changes:
            candidate = self.evaluator.evaluate_revision(
                rubric,
                change,
                evaluation_set,
                baseline,
            )
            evaluation = RevisionEvaluation.compare(baseline, candidate, self.policy)
            proposals.append(
                RevisionProposal(
                    proposal_id=self.id_factory(),
                    workflow_id=workflow_id,
                    base_rubric_version_id=rubric.version_id,
                    change=change,
                    evidence=tuple(evidence_by_change.get(change.signature, ())),
                    evaluation=evaluation,
                    created_at=created_at,
                )
            )
        for proposal in proposals:
            self._proposals[proposal.proposal_id] = proposal

        workflow = RevisionWorkflow(
            workflow_id=workflow_id,
            evaluation_set_id=evaluation_set.identifier,
            evaluation_input_digest=evaluation_set.input_digest,
            replay_artifact=evaluation_set.replay_artifact,
            base_rubric_version_id=rubric.version_id,
            evidence_artifact_ids=tuple(
                sorted(
                    {
                        error.predictor_artifact
                        for error in errors
                        if error.predictor_artifact
                    }
                )
            ),
            proposals=tuple(proposals),
            previous_workflow_id=previous_workflow_id,
            created_at=created_at,
        )
        self.store.save_workflow(workflow)
        comparison = (
            self.compare_workflows(previous_workflow_id, workflow.workflow_id)
            if previous_workflow_id is not None
            else None
        )
        return RevisionWorkflowResult(workflow=workflow, comparison=comparison)

    def rerun(
        self,
        evaluation_set: EvaluationSet,
        previous_workflow_id: str,
    ) -> RevisionWorkflowResult:
        return self.propose(evaluation_set, previous_workflow_id=previous_workflow_id)

    def compare_workflows(
        self,
        previous_workflow_id: str,
        current_workflow_id: str,
    ) -> WorkflowComparison:
        previous = self.store.get_workflow(previous_workflow_id)
        current = self.store.get_workflow(current_workflow_id)
        previous_rubric = self.store.get_rubric(previous.base_rubric_version_id)
        current_rubric = self.store.get_rubric(current.base_rubric_version_id)
        previous_signatures = {
            proposal.change.signature for proposal in previous.proposals
        }
        current_signatures = {
            proposal.change.signature for proposal in current.proposals
        }
        current_f1 = [
            proposal.evaluation.candidate.f1
            for proposal in current.proposals
            if proposal.evaluation is not None
        ]
        previous_f1 = [
            proposal.evaluation.candidate.f1
            for proposal in previous.proposals
            if proposal.evaluation is not None
        ]
        current_cost = [
            proposal.evaluation.candidate.total_cost_usd
            for proposal in current.proposals
            if proposal.evaluation is not None
        ]
        previous_cost = [
            proposal.evaluation.candidate.total_cost_usd
            for proposal in previous.proposals
            if proposal.evaluation is not None
        ]
        current_regression = [
            proposal.evaluation.candidate.regression_rate
            for proposal in current.proposals
            if proposal.evaluation is not None
        ]
        previous_regression = [
            proposal.evaluation.candidate.regression_rate
            for proposal in previous.proposals
            if proposal.evaluation is not None
        ]
        current_mean_f1 = _mean(current_f1)
        previous_mean_f1 = _mean(previous_f1)
        current_mean_cost = _mean(current_cost)
        previous_mean_cost = _mean(previous_cost)
        current_mean_regression = _mean(current_regression)
        previous_mean_regression = _mean(previous_regression)
        return WorkflowComparison(
            previous_workflow_id=previous.workflow_id,
            current_workflow_id=current.workflow_id,
            rubric_change=_rubric_comparison(previous_rubric, current_rubric),
            shared_proposal_signatures=tuple(
                sorted(previous_signatures & current_signatures, key=_json_dump)
            ),
            current_only_proposal_signatures=tuple(
                sorted(current_signatures - previous_signatures, key=_json_dump)
            ),
            previous_only_proposal_signatures=tuple(
                sorted(previous_signatures - current_signatures, key=_json_dump)
            ),
            current_mean_candidate_f1=current_mean_f1,
            previous_mean_candidate_f1=previous_mean_f1,
            mean_candidate_f1_delta=(
                current_mean_f1 - previous_mean_f1
                if current_mean_f1 is not None and previous_mean_f1 is not None
                else None
            ),
            current_mean_candidate_cost_usd=current_mean_cost,
            previous_mean_candidate_cost_usd=previous_mean_cost,
            mean_candidate_cost_delta_usd=(
                round(current_mean_cost - previous_mean_cost, 12)
                if current_mean_cost is not None and previous_mean_cost is not None
                else None
            ),
            current_mean_candidate_regression_rate=current_mean_regression,
            previous_mean_candidate_regression_rate=previous_mean_regression,
            mean_candidate_regression_rate_delta=(
                current_mean_regression - previous_mean_regression
                if current_mean_regression is not None
                and previous_mean_regression is not None
                else None
            ),
        )

    def adopt(
        self,
        proposal_id: str,
        *,
        maintainer: str,
        rationale: str,
    ) -> RevisionDecision:
        return self._decide(
            proposal_id,
            MaintainerDecisionKind.ADOPT,
            maintainer=maintainer,
            rationale=rationale,
        )

    def reject(
        self,
        proposal_id: str,
        *,
        maintainer: str,
        rationale: str,
    ) -> RevisionDecision:
        return self._decide(
            proposal_id,
            MaintainerDecisionKind.REJECT,
            maintainer=maintainer,
            rationale=rationale,
        )

    def list_decisions(self) -> tuple[RevisionDecision, ...]:
        return self.store.list_decisions()

    def _decide(
        self,
        proposal_id: str,
        decision: MaintainerDecisionKind,
        *,
        maintainer: str,
        rationale: str,
    ) -> RevisionDecision:
        if not maintainer.strip():
            raise ValueError("a maintainer identity is required")
        if not rationale.strip():
            raise ValueError("a decision rationale is required")
        proposal = self._proposals.get(proposal_id) or self.store.get_proposal(
            proposal_id
        )
        if proposal.evaluation is None:
            raise RuntimeError("a proposal must be evaluated before maintainer review")
        active = self.active_rubric()
        if decision is MaintainerDecisionKind.ADOPT:
            if active.version_id != proposal.base_rubric_version_id:
                raise StaleProposalError(
                    "the active rubric changed after this proposal was evaluated"
                )
            resulting = active.apply(
                proposal.change,
                version_id=f"rubric-from-{proposal.proposal_id}",
                created_at=self.clock(),
            )
        else:
            resulting = active
        record = RevisionDecision(
            decision_id=self.id_factory(),
            proposal_id=proposal.proposal_id,
            maintainer=maintainer,
            decision=decision,
            rationale=rationale,
            proposal=proposal,
            resulting_rubric=resulting,
            decided_at=self.clock(),
        )
        self.store.record_decision(
            record, resulting if decision is MaintainerDecisionKind.ADOPT else None
        )
        return record

    @staticmethod
    def _validate_errors(errors: Sequence[MeasuredError]) -> None:
        ids = [error.error_id for error in errors]
        if len(ids) != len(set(ids)):
            raise ValueError("measured error IDs must be unique")

    @staticmethod
    def _validate_changes(changes: Sequence[RubricQuestionChange]) -> None:
        ids = [change.question_id for change in changes]
        if len(ids) != len(set(ids)):
            raise ValueError("one workflow may propose at most one change per question")

    @staticmethod
    def _error_change_key(
        rubric: RubricVersion,
        error: MeasuredError,
    ) -> tuple[str, str, str | None, str | None, str | None] | None:
        current_id = error.current_question_id
        proposed = error.proposed_question
        current = (
            rubric.question(current_id)
            if current_id
            else (rubric.question(proposed.question_id) if proposed else None)
        )
        if current is None:
            if proposed is None:
                return None
            return RubricQuestionChange(
                kind=RevisionKind.NEW,
                question_id=proposed.question_id,
                before=None,
                after=proposed,
                rationale=error.observation,
            ).signature
        if proposed is None:
            return RubricQuestionChange(
                kind=RevisionKind.DROPPED,
                question_id=current.question_id,
                before=current,
                after=None,
                rationale=error.observation,
            ).signature
        if proposed.question_id == current.question_id and proposed != current:
            return RubricQuestionChange(
                kind=RevisionKind.REVISED,
                question_id=current.question_id,
                before=current,
                after=proposed,
                rationale=error.observation,
            ).signature
        return None


class HarnessRevisionEvaluator:
    """Adapter for the issue-11 ``EvaluationHarness`` public seam.

    ``engine_factory`` receives the baseline or candidate rubric and must return
    an object exposing ``run(dataset, replay_path=...)``.  Its report must expose
    ``to_dict()`` (or already be a mapping).  The constructor requires a replay
    path, so normal use cannot silently make live provider calls.
    """

    def __init__(
        self,
        engine_factory: Callable[[RubricVersion], Any],
        dataset: Any,
        replay_path: str,
        *,
        options: Any | None = None,
    ) -> None:
        if not replay_path.strip():
            raise ValueError("offline rubric evaluation requires a replay path")
        self.engine_factory = engine_factory
        self.dataset = dataset
        self.replay_path = replay_path
        self.options = options
        self._baseline_cache: dict[tuple[str, str], EvaluationMetrics] = {}

    def evaluate_baseline(
        self,
        rubric: RubricVersion,
        evaluation_set: EvaluationSet,
    ) -> EvaluationMetrics:
        cache_key = (rubric.version_id, evaluation_set.input_digest)
        if cache_key not in self._baseline_cache:
            self._baseline_cache[cache_key] = self._metrics_from_engine(
                self.engine_factory(rubric), evaluation_set
            )
        return self._baseline_cache[cache_key]

    def evaluate_revision(
        self,
        rubric: RubricVersion,
        change: RubricQuestionChange,
        evaluation_set: EvaluationSet,
        baseline: EvaluationMetrics,
    ) -> EvaluationMetrics:
        del baseline
        candidate = rubric.apply(
            change,
            version_id=f"candidate-{change.kind.value}-{change.question_id}",
            created_at="offline-evaluation",
        )
        return self._metrics_from_engine(self.engine_factory(candidate), evaluation_set)

    def _metrics_from_engine(
        self, engine: Any, evaluation_set: EvaluationSet
    ) -> EvaluationMetrics:
        if self.options is None:
            report = engine.run(self.dataset, replay_path=self.replay_path)
        else:
            report = engine.run(
                self.dataset,
                options=self.options,
                replay_path=self.replay_path,
            )
        payload = report.to_dict() if hasattr(report, "to_dict") else report
        metrics = self.metrics_from_report(payload)
        if not metrics.evaluated_case_ids or set(metrics.evaluated_case_ids) != set(
            evaluation_set.case_ids
        ):
            raise ValueError(
                "evaluation report does not cover the fixed evaluation case set"
            )
        return metrics

    @staticmethod
    def metrics_from_report(payload: Mapping[str, Any]) -> EvaluationMetrics:
        diagnosis = payload.get("diagnosis", {})
        per_gap = diagnosis.get("per_gap", {}) if isinstance(diagnosis, Mapping) else {}
        metric_rows = [row for row in per_gap.values() if isinstance(row, Mapping)]
        if not metric_rows:
            raise ValueError("evaluation report has no diagnosis quality metrics")
        precision = sum(float(row["precision"]) for row in metric_rows) / len(
            metric_rows
        )
        recall = sum(float(row["recall"]) for row in metric_rows) / len(metric_rows)
        cost = payload.get("cost", {})
        if not isinstance(cost, Mapping) or "total" not in cost:
            raise ValueError("evaluation report has no total cost")
        improvement = payload.get("improvement", {})
        if not isinstance(improvement, Mapping) or "regressed" not in improvement:
            raise ValueError("evaluation report has no regression count")
        regression_count = int(improvement["regressed"])
        cases = payload.get("cases", [])
        if (
            isinstance(cases, Sequence)
            and not isinstance(cases, (str, bytes))
            and cases
        ):
            if any(
                isinstance(case, Mapping) and case.get("status") in {"failed", "error"}
                for case in cases
            ):
                raise ValueError("evaluation report includes failed cases")
            case_count = len(cases)
            case_ids = tuple(
                str(case["case_id"]) for case in cases if isinstance(case, Mapping)
            )
            if len(case_ids) != case_count or len(set(case_ids)) != case_count:
                raise ValueError("evaluation report needs unique case ids")
            regressed_case_ids = tuple(
                str(case["case_id"])
                for case in cases
                if isinstance(case, Mapping) and case.get("outcome") == "regressed"
            )
        else:
            case_ids = ()
            regressed_case_ids = ()
            case_count = (
                int(improvement.get("improved", 0))
                + int(improvement.get("unchanged", 0))
                + regression_count
            )
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            total_cost_usd=float(cost["total"]),
            regression_count=regression_count,
            case_count=case_count,
            evaluated_case_ids=case_ids,
            regressed_case_ids=regressed_case_ids,
        )


__all__ = [
    "EvaluationMetrics",
    "EvaluationPolicy",
    "EvaluationSet",
    "HarnessRevisionEvaluator",
    "MaintainerDecisionKind",
    "MeasuredError",
    "MeasuredErrorProposer",
    "MeasuredErrorSource",
    "RecommendedDecision",
    "RegressionRisk",
    "RevisionDecision",
    "RevisionEvaluation",
    "RevisionEvaluator",
    "RevisionKind",
    "RevisionProposal",
    "RevisionProposer",
    "RevisionWorkflow",
    "RevisionWorkflowResult",
    "RubricQuestion",
    "RubricQuestionChange",
    "RubricRevisionService",
    "RubricVersion",
    "RubricVersionChange",
    "SQLiteRubricStore",
    "StaleProposalError",
    "WorkflowComparison",
]
