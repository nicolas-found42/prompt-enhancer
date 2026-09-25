"""Jev-backed prompt diagnosis.

This module owns the public diagnosis value objects.  All user text is carried in
request ``state``; the decision questions and writer instructions are static.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .evaluation.calibration import DecisionPolicy, PolicyDecision
from . import jev_questions
from .gateway import Gateway
from .jev import (
    ChoiceDecision,
    JevDecision,
    JevResponseError,
    NoulDecision,
    parse_decision,
)


class GapImpact(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProblemKind(StrEnum):
    VAGUENESS = "vagueness"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    CONTRADICTION = "contradiction"
    EMBEDDED_INSTRUCTION = "embedded_instruction"


@dataclass(frozen=True, slots=True)
class ChecklistItem:
    key: str
    label: str
    impact: GapImpact
    question: str | Mapping[str, Any] | Sequence[Any] | None = None


@dataclass(frozen=True, slots=True)
class TaskType:
    key: str
    label: str
    checklist: tuple[ChecklistItem, ...]


@dataclass(frozen=True, slots=True)
class Sentence:
    id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ConfirmedGap:
    key: str
    label: str
    impact: GapImpact
    missing_probability: float
    confidence: float
    threshold: float


@dataclass(frozen=True, slots=True)
class PossibleGap:
    """A checklist item Jev leaned towards but not past its confirmation cutoff.

    It never drives clarification or rewriting; it only lets the result tell the
    user what may still be missing instead of claiming nothing is.
    """

    key: str
    label: str
    missing_probability: float
    threshold: float
    sentence: str | None = None


@dataclass(frozen=True, slots=True)
class ProblemSentence:
    sentence: Sentence
    kind: ProblemKind
    probability: float
    confidence: float
    threshold: float

    @property
    def sentence_id(self) -> str:
        return self.sentence.id


@dataclass(frozen=True, slots=True)
class DiagnosisReport:
    task_type: str
    task_type_label: str
    task_type_confidence: float
    confirmed_gaps: tuple[ConfirmedGap, ...]
    problem_sentences: tuple[ProblemSentence, ...]
    rubric_version: str | None = None
    possible_gaps: tuple[PossibleGap, ...] = ()
    calibration: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.calibration is None:
            result.pop("calibration", None)
        else:
            result["calibration"] = dict(self.calibration)
        result["confirmed_gaps"] = [
            asdict(gap) | {"impact": gap.impact.value} for gap in self.confirmed_gaps
        ]
        result["problem_sentences"] = [
            {
                "sentence": asdict(problem.sentence),
                "sentence_id": problem.sentence.id,
                "kind": problem.kind.value,
                "probability": problem.probability,
                "confidence": problem.confidence,
                "threshold": problem.threshold,
            }
            for problem in self.problem_sentences
        ]
        result["possible_gaps"] = [asdict(gap) for gap in self.possible_gaps]
        return result


@dataclass(frozen=True, slots=True)
class DiagnosisRubric:
    task_types: tuple[TaskType, ...]
    default_task_type: str = "general"
    confidence_threshold: float = 0.8
    gap_threshold: float = 0.9
    gap_thresholds: Mapping[str, float] = field(default_factory=dict)
    problem_threshold: float = 0.9
    pointer_threshold: float = 0.8
    uncertainty_margin: float = 0.1
    # Lower bounds for reporting a near-miss gap as a hint. Only the listed
    # checklist keys are hinted; confirmation still uses the gap thresholds.
    hint_thresholds: Mapping[str, float] = field(default_factory=dict)

    def gap_threshold_for(self, question_id: str) -> float:
        return self.gap_thresholds.get(question_id, self.gap_threshold)


_GENERAL_CHECKLIST = (
    ChecklistItem("goal", "goal", GapImpact.HIGH),
    # High impact, so an unknown context is asked about: rewrites cannot supply
    # it and failed fidelity in every recorded attempt (see the 2026-09-24
    # recalibration).
    ChecklistItem("context", "relevant context", GapImpact.HIGH),
    ChecklistItem("constraints", "constraints", GapImpact.MEDIUM),
    ChecklistItem("output_format", "output format", GapImpact.LOW),
    ChecklistItem("done_criteria", "done criteria", GapImpact.HIGH),
    # References to material the prompt never includes ("like last time",
    # "the thing about the warranty") cannot be inferred by any writer, so
    # they are asked about rather than assumed.
    ChecklistItem(
        "outside_reference",
        "details only you know",
        GapImpact.HIGH,
        question=jev_questions.OUTSIDE_REFERENCE_GAP_QUESTION,
    ),
)
_WRITING_CHECKLIST = _GENERAL_CHECKLIST
_ANALYSIS_CHECKLIST = _GENERAL_CHECKLIST + (
    ChecklistItem("sources", "source basis", GapImpact.HIGH),
)
_CODING_CHECKLIST = _GENERAL_CHECKLIST + (
    ChecklistItem("language", "language or runtime", GapImpact.HIGH),
    ChecklistItem("tests", "test expectations", GapImpact.MEDIUM),
)
_PLANNING_CHECKLIST = _GENERAL_CHECKLIST + (
    ChecklistItem("time_horizon", "time horizon", GapImpact.MEDIUM),
)
_CHAT_CHECKLIST = _GENERAL_CHECKLIST[:2]

_TASK_TREE = {
    "communication": ("writing", "chat"),
    "investigation": ("analysis", "research"),
    "execution": ("coding", "planning"),
}

DEFAULT_RUBRIC = DiagnosisRubric(
    task_types=(
        TaskType("general", "General", _GENERAL_CHECKLIST),
        TaskType("writing", "Writing", _WRITING_CHECKLIST),
        TaskType("analysis", "Analysis", _ANALYSIS_CHECKLIST),
        TaskType("coding", "Coding", _CODING_CHECKLIST),
        TaskType("research", "Research", _ANALYSIS_CHECKLIST),
        TaskType("planning", "Planning", _PLANNING_CHECKLIST),
        TaskType("chat", "Chat", _CHAT_CHECKLIST),
    ),
    # Cutoffs selected on training data as the lowest with precision >= 0.5; see
    # docs/gap-cutoff-recalibration-2026-09-24.md. outside_reference has no
    # positive labels; 0.80 is backed only by 0/101 false flags.
    gap_thresholds={"context": 0.83, "language": 0.77, "outside_reference": 0.8},
    hint_thresholds={"outside_reference": 0.75},
)


# Checklist items added after replay bundles began recording their checklist.
# A bundle without ``checklist_keys`` replays without these questions, because
# its strict recordings never saw them.
HISTORICAL_CHECKLIST_EXCLUSIONS = ("outside_reference",)


# Impacts that differ from the ones replay bundles saw before bundles recorded
# ``checklist_impacts``. A bundle without the field replays with these.
HISTORICAL_CHECKLIST_IMPACTS: Mapping[str, str] = {"context": GapImpact.MEDIUM.value}


def checklist_impacts(rubric: DiagnosisRubric) -> dict[str, str]:
    return {
        item.key: item.impact.value
        for task in rubric.task_types
        for item in task.checklist
    }


def with_impacts(
    rubric: DiagnosisRubric, impacts: Mapping[str, str]
) -> DiagnosisRubric:
    """Override the impact of the named checklist items, as a replay recorded them."""
    return replace(
        rubric,
        task_types=tuple(
            replace(
                task,
                checklist=tuple(
                    replace(item, impact=GapImpact(impacts[item.key]))
                    if item.key in impacts
                    else item
                    for item in task.checklist
                ),
            )
            for task in rubric.task_types
        ),
    )


def checklist_keys(rubric: DiagnosisRubric) -> tuple[str, ...]:
    return tuple(
        sorted({item.key for task in rubric.task_types for item in task.checklist})
    )


def restrict_checklist(rubric: DiagnosisRubric, keys: Iterable[str]) -> DiagnosisRubric:
    allowed = set(keys)
    return replace(
        rubric,
        task_types=tuple(
            replace(
                task,
                checklist=tuple(item for item in task.checklist if item.key in allowed),
            )
            for task in rubric.task_types
        ),
    )


def gap_question(item: ChecklistItem) -> str | Mapping[str, Any] | Sequence[Any]:
    if item.question is not None:
        return item.question
    return jev_questions.gap_question(item.label)


def default_gap_question(key: str) -> str:
    item = next(
        (
            item
            for task in DEFAULT_RUBRIC.task_types
            for item in task.checklist
            if item.key == key
        ),
        None,
    )
    if item is None:
        raise KeyError(key)
    return cast(str, gap_question(item))


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’\)\]]*)(?=\s+|$)|\n{2,}")
_PROBLEM_QUESTIONS = {
    ProblemKind(key): value for key, value in jev_questions.PROBLEM_QUESTIONS.items()
}
POINTER_CALIBRATION_CRITERIA = "sentence-id-options-with-none"


def split_sentences(prompt: str) -> tuple[Sentence, ...]:
    """Split text while preserving exact offsets and deterministic IDs."""

    if not prompt.strip():
        return ()
    sentences: list[Sentence] = []
    position = 0
    for match in _SENTENCE_BOUNDARY.finditer(prompt):
        end = match.start()
        _append_sentence(sentences, prompt[position:end], position, position + end)
        position = end
    _append_sentence(sentences, prompt[position:], position, len(prompt))
    return tuple(sentences)


def _append_sentence(result: list[Sentence], text: str, start: int, end: int) -> None:
    stripped = text.strip()
    if not stripped:
        return
    left_trimmed = len(text) - len(text.lstrip())
    actual_start = start + left_trimmed
    actual_end = actual_start + len(stripped)
    result.append(
        Sentence(
            id=f"s{len(result) + 1:04d}",
            text=stripped,
            start=actual_start,
            end=actual_end,
        )
    )


def _request(
    question: str | Mapping[str, Any] | Sequence[Any],
    state: Mapping[str, Any],
    **decision: Any,
) -> dict[str, Any]:
    return {
        "model": "typesafe/jev-1.13",
        "query": question,
        "state": dict(state),
        **decision,
    }


def _request_key(request: Mapping[str, Any]) -> str:
    return str(request.get("key", request.get("query", "")))


@dataclass(frozen=True, slots=True)
class _DecisionObservation:
    request: Mapping[str, Any]
    raw_answer: Any
    decision: JevDecision
    answered_by: str | None


class Diagnoser:
    """Run diagnosis through a replaceable, deterministic-capable gateway."""

    def __init__(
        self,
        gateway: Gateway,
        *,
        rubric: DiagnosisRubric | Callable[[], DiagnosisRubric] = DEFAULT_RUBRIC,
        decision_policy: DecisionPolicy | None = None,
        rubric_version: str | None = "default-v1",
    ) -> None:
        self.gateway = gateway
        self._rubric = rubric
        self.decision_policy = decision_policy
        self.rubric_version = rubric_version
        self._calibration_evidence: dict[str, Any] = {}

    @property
    def rubric(self) -> DiagnosisRubric:
        if isinstance(self._rubric, DiagnosisRubric):
            return self._rubric
        return self._rubric()

    def _observe(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> tuple[_DecisionObservation, ...]:
        log = getattr(self.gateway, "decision_log", ())
        before = len(log) if isinstance(log, Sequence) else 0
        raw_responses = list(self.gateway.decide_batch(requests))
        entries = list(log)[before:] if isinstance(log, Sequence) else []
        try:
            decisions = tuple(parse_decision(response) for response in raw_responses)
        except JevResponseError:
            # An unusable audit is fail-open: it must not invent a defect.
            return ()
        observations: list[_DecisionObservation] = []
        for index, (request, raw_answer, decision) in enumerate(
            zip(requests, raw_responses, decisions, strict=True)
        ):
            entry = entries[index] if index < len(entries) else {}
            answered_by = (
                entry.get("answered_by") if isinstance(entry, Mapping) else None
            )
            observations.append(
                _DecisionObservation(
                    request=dict(request),
                    raw_answer=raw_answer,
                    decision=decision,
                    answered_by=answered_by if isinstance(answered_by, str) else None,
                )
            )
        return tuple(observations)

    def _decide(self, requests: Sequence[Mapping[str, Any]]) -> tuple[JevDecision, ...]:
        return tuple(observation.decision for observation in self._observe(requests))

    def _policy_decision(
        self,
        *,
        question_id: str,
        request: Mapping[str, Any],
        observation: _DecisionObservation,
        family: str,
        event_mapping: Mapping[str, Any] | None = None,
        criteria_descriptor: str | None = None,
    ) -> PolicyDecision | None:
        if self.decision_policy is None:
            return None
        from .evaluation.calibration import runtime_question_identity

        identity = runtime_question_identity(
            question_id,
            request,
            family=family,
            rubric_version=self.rubric_version,
            snapshot=observation.answered_by,
            policy_version=self.decision_policy.policy_version,
        )
        if criteria_descriptor is not None:
            identity = replace(identity, criteria=criteria_descriptor)
        if event_mapping is not None:
            identity = replace(identity, event_mapping=dict(event_mapping))
        decision = self.decision_policy.apply(
            question_id=question_id,
            identity=identity,
            decision=observation.decision,
            raw_answer=observation.raw_answer,
            snapshot=observation.answered_by,
        )
        self._calibration_evidence[question_id] = {
            "disposition": decision.disposition,
            "verdict": decision.verdict,
            "reason": decision.reason,
            "threshold": decision.threshold,
            "predicate": dict(decision.predicate),
            "event_probability": decision.evidence.get("event_probability"),
            "fit": decision.evidence.get("fit"),
        }
        return decision

    def diagnose(self, prompt: str) -> DiagnosisReport:
        self._calibration_evidence = {}
        rubric = self.rubric
        state = {"prompt": prompt}
        unknown = "unknown"
        task_options = {
            "general": jev_questions.GENERAL_TASK_DESCRIPTION,
            **{
                group: jev_questions.task_branch_description(children)
                for group, children in _TASK_TREE.items()
            },
            unknown: jev_questions.UNKNOWN_TASK_DESCRIPTION,
        }
        task_request = _request(
            jev_questions.TASK_TYPE_QUESTION,
            state,
            type="choice",
            options=task_options,
            key="task_type",
        )
        task_results = self._decide((task_request,))
        selected = str(rubric.default_task_type)
        task_confidence = 0.0
        if task_results and isinstance(task_results[0], ChoiceDecision):
            task_result = task_results[0]
            if task_result.selected in {task.key for task in rubric.task_types}:
                selected = task_result.selected
                task_confidence = task_result.confidence
            elif task_result.selected in _TASK_TREE:
                branch = task_result.selected
                children = _TASK_TREE[branch]
                leaf_request = _request(
                    jev_questions.task_leaf_question(branch),
                    state,
                    type="choice",
                    options={
                        **{
                            child: next(
                                task.label
                                for task in rubric.task_types
                                if task.key == child
                            )
                            for child in children
                        },
                        unknown: jev_questions.UNKNOWN_LEAF_DESCRIPTION,
                    },
                    key=f"task_type:{branch}",
                )
                leaf_results = self._decide((leaf_request,))
                if (
                    leaf_results
                    and isinstance(leaf_results[0], ChoiceDecision)
                    and leaf_results[0].selected in children
                ):
                    selected = leaf_results[0].selected
                    task_confidence = min(
                        task_result.confidence, leaf_results[0].confidence
                    )

        task = next(
            (item for item in rubric.task_types if item.key == selected),
            rubric.task_types[0],
        )
        gaps, near_misses, sentences = self._diagnose_gaps(prompt, state, task, rubric)
        problems, pointed = self._diagnose_sentences(prompt, state, sentences, rubric)
        # Quote the sentence Jev pointed at, preferring an unresolved reference.
        suspect = pointed.get(ProblemKind.UNRESOLVED_REFERENCE) or pointed.get(
            ProblemKind.VAGUENESS
        )
        return DiagnosisReport(
            task_type=task.key,
            task_type_label=task.label,
            task_type_confidence=task_confidence,
            confirmed_gaps=gaps,
            problem_sentences=problems,
            possible_gaps=tuple(
                replace(gap, sentence=suspect.text if suspect else None)
                for gap in near_misses
            ),
            calibration=dict(self._calibration_evidence) or None,
        )

    def _diagnose_gaps(
        self,
        prompt: str,
        state: Mapping[str, Any],
        task: TaskType,
        rubric: DiagnosisRubric,
    ) -> tuple[tuple[ConfirmedGap, ...], tuple[PossibleGap, ...], tuple[Sentence, ...]]:
        del prompt  # The caller's exact text is already isolated in state.
        requests = [
            _request(
                gap_question(item),
                state,
                type="noul",
                key=f"gap:{item.key}",
            )
            for item in task.checklist
        ]
        observations = self._observe(requests)
        gaps: list[ConfirmedGap] = []
        near_misses: list[PossibleGap] = []
        for item, observation in zip(task.checklist, observations, strict=False):
            response = observation.decision
            if not isinstance(response, NoulDecision):
                continue
            policy_decision = self._policy_decision(
                question_id=f"gap:{item.key}",
                request=observation.request,
                observation=observation,
                family="gap",
            )
            threshold = (
                policy_decision.threshold
                if policy_decision is not None
                and policy_decision.may_gate
                and policy_decision.threshold is not None
                else rubric.gap_threshold_for(item.key)
            )
            if policy_decision is not None and not policy_decision.is_legacy:
                confident_missing = policy_decision.may_gate
            else:
                confident_missing = (
                    response.probability >= threshold
                    and response.confidence >= rubric.confidence_threshold
                    and abs(response.probability - 0.5) >= rubric.uncertainty_margin
                )
            if confident_missing:
                reported_probability = (
                    policy_decision.evidence.get(
                        "event_probability", response.probability
                    )
                    if policy_decision is not None and policy_decision.may_gate
                    else response.probability
                )
                gaps.append(
                    ConfirmedGap(
                        key=item.key,
                        label=item.label,
                        impact=item.impact,
                        missing_probability=reported_probability,
                        confidence=response.confidence,
                        threshold=threshold,
                    )
                )
            elif policy_decision is None or policy_decision.is_legacy:
                if (
                    rubric.hint_thresholds.get(item.key, 1.0)
                    <= response.probability
                    < threshold
                ):
                    near_misses.append(
                        PossibleGap(
                            key=item.key,
                            label=item.label,
                            missing_probability=response.probability,
                            threshold=threshold,
                        )
                    )
        return tuple(gaps), tuple(near_misses), split_sentences(str(state["prompt"]))

    def _diagnose_sentences(
        self,
        prompt: str,
        state: Mapping[str, Any],
        sentences: tuple[Sentence, ...],
        rubric: DiagnosisRubric,
    ) -> tuple[tuple[ProblemSentence, ...], dict[ProblemKind, Sentence]]:
        del prompt
        if not sentences:
            return (), {}
        sentence_state = {
            **state,
            "sentences": [{"id": item.id, "text": item.text} for item in sentences],
        }
        pointer_requests = []
        pointer_kinds = []
        for window_index, start in enumerate(range(0, len(sentences), 254)):
            window = sentences[start : start + 254]
            window_state = {
                **sentence_state,
                "sentences": [{"id": item.id, "text": item.text} for item in window],
            }
            for kind in _PROBLEM_QUESTIONS:
                pointer_requests.append(
                    _request(
                        jev_questions.sentence_pointer_question(
                            kind.value.replace("_", " ")
                        ),
                        window_state,
                        type="choice",
                        options=[item.id for item in window] + ["none"],
                        key=f"pointer:{kind.value}:{window_index}",
                    )
                )
                pointer_kinds.append(kind)
        pointer_observations = self._observe(pointer_requests)
        selected: list[tuple[ProblemKind, Sentence]] = []
        for kind, observation in zip(pointer_kinds, pointer_observations, strict=False):
            pointer = observation.decision
            if not isinstance(pointer, ChoiceDecision) or pointer.selected == "none":
                continue
            policy_decision = self._policy_decision(
                question_id=f"pointer:{kind.value}",
                request=observation.request,
                observation=observation,
                family="pointer",
                event_mapping={"selected_correctness": True},
                criteria_descriptor=POINTER_CALIBRATION_CRITERIA,
            )
            if policy_decision is not None and policy_decision.disposition == "abstain":
                continue
            if (
                policy_decision is None or not policy_decision.may_gate
            ) and pointer.confidence < rubric.pointer_threshold:
                continue
            sentence = next(
                (item for item in sentences if item.id == pointer.selected), None
            )
            if sentence is not None:
                selected.append((kind, sentence))
        pointed = {kind: sentence for kind, sentence in reversed(selected)}
        if not selected:
            return (), pointed

        checks = [
            _request(
                _PROBLEM_QUESTIONS[kind],
                {**sentence_state, "selected_sentence_id": sentence.id},
                type="noul",
                key=f"problem:{kind.value}:{sentence.id}",
            )
            for kind, sentence in selected
        ]
        observations = self._observe(checks)
        problems: list[ProblemSentence] = []
        for (kind, sentence), observation in zip(selected, observations, strict=False):
            result = observation.decision
            if not isinstance(result, NoulDecision):
                continue
            policy_decision = self._policy_decision(
                question_id=f"problem:{kind.value}",
                request=observation.request,
                observation=observation,
                family="problem",
            )
            threshold = (
                policy_decision.threshold
                if policy_decision is not None
                and policy_decision.may_gate
                and policy_decision.threshold is not None
                else rubric.problem_threshold
            )
            if policy_decision is not None and not policy_decision.is_legacy:
                confident = policy_decision.may_gate
            else:
                confident = (
                    result.probability >= threshold
                    and result.confidence >= rubric.confidence_threshold
                    and abs(result.probability - 0.5) >= rubric.uncertainty_margin
                )
            if confident:
                problems.append(
                    ProblemSentence(
                        sentence=sentence,
                        kind=kind,
                        probability=result.probability,
                        confidence=result.confidence,
                        threshold=threshold,
                    )
                )
        return tuple(problems), pointed


def model_diagnosis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The diagnosis as sent to models: without the user-facing possible gaps.

    Near-miss hints must never steer a strategy, candidate, or fidelity decision,
    and leaving them out keeps recorded replays exact.
    """
    return {
        key: value
        for key, value in payload.items()
        if key not in {"possible_gaps", "calibration"}
    }


def diagnosis_from_dict(value: Mapping[str, Any]) -> DiagnosisReport:
    """Deserialize a persisted report without leaking internal parsing details."""

    gaps = tuple(
        ConfirmedGap(
            key=str(item["key"]),
            label=str(item["label"]),
            impact=GapImpact(item["impact"]),
            missing_probability=float(item["missing_probability"]),
            confidence=float(item["confidence"]),
            threshold=float(item["threshold"]),
        )
        for item in value.get("confirmed_gaps", ())
    )
    problems = tuple(
        ProblemSentence(
            sentence=Sentence(
                id=str(item["sentence_id"]),
                text=str(item["sentence"]["text"]),
                start=int(item["sentence"]["start"]),
                end=int(item["sentence"]["end"]),
            ),
            kind=ProblemKind(item["kind"]),
            probability=float(item["probability"]),
            confidence=float(item["confidence"]),
            threshold=float(item["threshold"]),
        )
        for item in value.get("problem_sentences", ())
    )
    possible = tuple(
        PossibleGap(
            key=str(item["key"]),
            label=str(item["label"]),
            missing_probability=float(item["missing_probability"]),
            threshold=float(item["threshold"]),
            sentence=None if item.get("sentence") is None else str(item["sentence"]),
        )
        for item in value.get("possible_gaps", ())
    )
    calibration_value = value.get("calibration")
    calibration = (
        dict(calibration_value) if isinstance(calibration_value, Mapping) else None
    )
    return DiagnosisReport(
        task_type=str(value["task_type"]),
        task_type_label=str(value["task_type_label"]),
        task_type_confidence=float(value["task_type_confidence"]),
        confirmed_gaps=gaps,
        problem_sentences=problems,
        possible_gaps=possible,
        calibration=calibration,
    )


def _state_json(value: Mapping[str, Any]) -> str:
    # Kept as a helper for gateway adapters that require a JSON string.
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
