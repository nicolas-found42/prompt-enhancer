"""Jev-backed prompt diagnosis.

This module owns the public diagnosis value objects.  All user text is carried in
request ``state``; the decision questions and writer instructions are static.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .evaluation.calibration import DecisionPolicy, PolicyDecision
from . import jev_questions
from .gateway import Gateway, HttpGateway, ProviderError
from .jev import (
    ChoiceDecision,
    JevDecision,
    JevResponseError,
    NoulDecision,
    batch_decision_payload,
    parse_decision,
)

SENTENCE_DIAGNOSIS_PROTOCOL_VERSION = 2
HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION = 1
SENTENCE_EXISTENCE_QUESTION_VERSION = 1
HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION = 1
TASK_TAXONOMY_PROTOCOL_VERSION = 2
DEFAULT_TASK_BRANCH_MARGIN = 0.15
DEFAULT_TASK_BEAM_WIDTH = 2
DEFAULT_TASK_TYPE_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_EXISTENCE_THRESHOLD = 0.8
DEFAULT_EXISTENCE_THRESHOLD_VERSION = "existence-cutoffs-v1"
MAX_DIAGNOSIS_INPUT_CHARACTERS = 20_000
MAX_DIAGNOSIS_REQUEST_BYTES = 96_000
MAX_DIAGNOSIS_QUESTIONS_PER_REQUEST = 40
MAX_DIAGNOSIS_PROVIDER_REQUESTS = 8


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
    description: str = ""
    scope: str = ""


@dataclass(frozen=True, slots=True)
class TaskBranch:
    key: str
    label: str
    description: str
    scope: str
    children: tuple[str, ...]
    checklist: tuple[ChecklistItem, ...] | None = None


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
    sentence_protocol_version: int = HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION
    sentence_evidence: tuple[Mapping[str, Any], ...] = ()
    task_type_path: tuple[Mapping[str, Any], ...] = ()
    task_type_fallback_reason: str | None = None
    effective_checklist: tuple[Mapping[str, Any], ...] = ()
    taxonomy_evidence: Mapping[str, Any] | None = None
    request_evidence: Mapping[str, Any] | None = None

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
        result["sentence_evidence"] = [dict(item) for item in self.sentence_evidence]
        result["task_type_path"] = [dict(item) for item in self.task_type_path]
        result["effective_checklist"] = [
            dict(item) for item in self.effective_checklist
        ]
        if self.taxonomy_evidence is None:
            result.pop("taxonomy_evidence", None)
        else:
            result["taxonomy_evidence"] = dict(self.taxonomy_evidence)
        if self.request_evidence is None:
            result.pop("request_evidence", None)
        else:
            result["request_evidence"] = dict(self.request_evidence)
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
    existence_thresholds: Mapping[str, float] = field(default_factory=dict)
    existence_threshold_version: str = DEFAULT_EXISTENCE_THRESHOLD_VERSION
    task_branches: tuple[TaskBranch, ...] = ()
    task_beam_width: int = DEFAULT_TASK_BEAM_WIDTH
    task_branch_margin: float = DEFAULT_TASK_BRANCH_MARGIN
    task_type_confidence_threshold: float = DEFAULT_TASK_TYPE_CONFIDENCE_THRESHOLD

    def __post_init__(self) -> None:
        if self.task_beam_width < 1:
            raise ValueError("task_beam_width must be positive")
        if not 0.0 <= self.task_branch_margin <= 1.0:
            raise ValueError("task_branch_margin must be between 0 and 1")
        if not 0.0 <= self.task_type_confidence_threshold <= 1.0:
            raise ValueError("task_type_confidence_threshold must be between 0 and 1")

    def gap_threshold_for(self, question_id: str) -> float:
        return self.gap_thresholds.get(question_id, self.gap_threshold)

    def existence_threshold_for(self, kind: ProblemKind | str) -> float:
        key = kind.value if isinstance(kind, ProblemKind) else str(kind)
        return self.existence_thresholds.get(key, DEFAULT_EXISTENCE_THRESHOLD)


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

_DEFAULT_TASK_BRANCHES = (
    TaskBranch(
        "communication",
        "Communication",
        "Create, revise, or exchange language with another person.",
        "Drafting, editing, summarizing, formatting, and conversational help.",
        ("writing", "chat"),
    ),
    TaskBranch(
        "investigation",
        "Investigation",
        "Interpret existing information or find and synthesize information about a topic.",
        "Analysis, evidence-based explanations, research, and source discovery.",
        ("analysis", "research"),
    ),
    TaskBranch(
        "execution",
        "Execution",
        "Produce a working implementation or organize work toward a goal.",
        "Software implementation, debugging, planning, scheduling, and coordination.",
        ("coding", "planning"),
    ),
)
_HISTORICAL_TASK_TREE = {
    "communication": ("writing", "chat"),
    "investigation": ("analysis", "research"),
    "execution": ("coding", "planning"),
}

DEFAULT_RUBRIC = DiagnosisRubric(
    task_types=(
        TaskType("general", "General", _GENERAL_CHECKLIST),
        TaskType(
            "writing",
            "Writing",
            _WRITING_CHECKLIST,
            "Create or improve written material.",
            "Drafting, editing, summarizing, and formatting text.",
        ),
        TaskType(
            "analysis",
            "Analysis",
            _ANALYSIS_CHECKLIST,
            "Interpret information, compare evidence, or explain causes.",
            "Reasoning about information supplied in the prompt or otherwise available.",
        ),
        TaskType(
            "coding",
            "Coding",
            _CODING_CHECKLIST,
            "Design, implement, debug, or explain software.",
            "Source code, software behavior, and technical execution.",
        ),
        TaskType(
            "research",
            "Research",
            _ANALYSIS_CHECKLIST,
            "Find or synthesize information about a topic.",
            "Information discovery, source-based synthesis, and research summaries.",
        ),
        TaskType(
            "planning",
            "Planning",
            _PLANNING_CHECKLIST,
            "Organize future work or steps toward a goal.",
            "Plans, schedules, sequencing, and coordination.",
        ),
        TaskType(
            "chat",
            "Chat",
            _CHAT_CHECKLIST,
            "Have a conversational exchange or answer a direct question.",
            "Interactive help and concise conversational responses.",
        ),
    ),
    task_branches=_DEFAULT_TASK_BRANCHES,
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
    decision: JevDecision | None
    answered_by: str | None


@dataclass(frozen=True, slots=True)
class _TaskSelection:
    task: TaskType
    confidence: float
    path: tuple[Mapping[str, Any], ...]
    fallback_reason: str | None
    evidence: Mapping[str, Any]


def _active_task_branches(rubric: DiagnosisRubric) -> tuple[TaskBranch, ...]:
    tasks = {task.key for task in rubric.task_types}
    branches = rubric.task_branches or _DEFAULT_TASK_BRANCHES
    return tuple(
        replace(branch, children=tuple(key for key in branch.children if key in tasks))
        for branch in branches
        if any(key in tasks for key in branch.children)
    )


def _task_description(task: TaskType) -> str:
    description = task.description or f"Perform a {task.label.lower()} task."
    scope = task.scope or ", ".join(item.label for item in task.checklist)
    return f"{task.label}: {description} Intended scope: {scope}."


def _checklist_item_identity(item: ChecklistItem) -> str:
    return json.dumps(
        {
            "key": item.key,
            "label": item.label,
            "impact": item.impact.value,
            "question": item.question,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parent_checklist(
    branch: TaskBranch, task_by_key: Mapping[str, TaskType]
) -> tuple[tuple[ChecklistItem, ...], bool]:
    if branch.checklist is not None:
        return branch.checklist, False
    descendants = [task_by_key[key] for key in branch.children if key in task_by_key]
    if not descendants:
        return (), True
    if len(descendants) == 1:
        return descendants[0].checklist, False
    first = descendants[0].checklist
    shared = {
        _checklist_item_identity(item)
        for task in descendants[1:]
        for item in task.checklist
    }
    for task in descendants[1:]:
        shared.intersection_update(
            _checklist_item_identity(item) for item in task.checklist
        )
    intersection = tuple(
        item for item in first if _checklist_item_identity(item) in shared
    )
    return intersection, not intersection


def _checklist_payload(items: Sequence[ChecklistItem]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "key": item.key,
            "label": item.label,
            "impact": item.impact.value,
            "question": item.question,
        }
        for item in items
    )


class Diagnoser:
    """Run diagnosis through a replaceable, deterministic-capable gateway."""

    def __init__(
        self,
        gateway: Gateway,
        *,
        rubric: DiagnosisRubric | Callable[[], DiagnosisRubric] = DEFAULT_RUBRIC,
        decision_policy: DecisionPolicy | None = None,
        rubric_version: str | None = "default-v1",
        sentence_protocol_version: int = SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
        task_taxonomy_version: int = TASK_TAXONOMY_PROTOCOL_VERSION,
        speculative_fanout: bool = True,
        record_request_evidence: bool | None = None,
        additional_requests: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if sentence_protocol_version not in {
            HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
            SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
        }:
            raise ValueError("unsupported sentence diagnosis protocol version")
        self.gateway = gateway
        self._rubric = rubric
        self.decision_policy = decision_policy
        self.rubric_version = rubric_version
        self.sentence_protocol_version = sentence_protocol_version
        if task_taxonomy_version not in {
            HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
            TASK_TAXONOMY_PROTOCOL_VERSION,
        }:
            raise ValueError("unsupported task taxonomy protocol version")
        self.task_taxonomy_version = task_taxonomy_version
        self._calibration_evidence: dict[str, Any] = {}
        self.speculative_fanout = speculative_fanout
        self.record_request_evidence = (
            speculative_fanout
            if record_request_evidence is None
            else record_request_evidence
        )
        self.additional_requests = tuple(
            dict(request) for request in additional_requests
        )
        self._prefetched: dict[str, _DecisionObservation] | None = None
        self._bounded_fallback = False
        self._provider_requests = 0
        self._request_latencies_ms: list[float] = []
        self._incomplete = False
        self._fallback_reason: str | None = None
        self._dispatch_blocked = False

    @property
    def rubric(self) -> DiagnosisRubric:
        if isinstance(self._rubric, DiagnosisRubric):
            return self._rubric
        return self._rubric()

    def _observe(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> tuple[_DecisionObservation, ...]:
        if self._prefetched is not None:
            selected: list[_DecisionObservation] = []
            for request in requests:
                observation = self._prefetched.get(self._request_identity(request))
                if observation is None and str(request.get("key", "")).startswith(
                    "problem:"
                ):
                    return self._observe_direct(requests)
                if observation is None:
                    self._incomplete = True
                    observation = _DecisionObservation(dict(request), None, None, None)
                elif observation.raw_answer is None:
                    self._incomplete = True
                selected.append(observation)
            return tuple(selected)
        return self._observe_direct(requests)

    @staticmethod
    def _request_identity(request: Mapping[str, Any]) -> str:
        return json.dumps(request, ensure_ascii=False, sort_keys=True)

    def _observe_direct(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> tuple[_DecisionObservation, ...]:
        if not requests:
            return ()
        if self._bounded_fallback:
            observations: list[_DecisionObservation] = []
            chunk: list[Mapping[str, Any]] = []
            for request in requests:
                proposal = [*chunk, request]
                if chunk and not self._request_fits(proposal):
                    observations.extend(self._dispatch(chunk))
                    chunk = []
                if self._request_fits([request]):
                    chunk.append(request)
                else:
                    self._incomplete = True
                    observations.append(
                        _DecisionObservation(dict(request), None, None, None)
                    )
            if chunk:
                observations.extend(self._dispatch(chunk))
            return tuple(observations)
        return self._dispatch(requests)

    def _request_fits(self, requests: Sequence[Mapping[str, Any]]) -> bool:
        if len(requests) > MAX_DIAGNOSIS_QUESTIONS_PER_REQUEST:
            return False
        try:
            _, envelope = batch_decision_payload(requests, model=self.gateway.jev_model)
        except ValueError:
            return False
        return (
            len(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))
            <= MAX_DIAGNOSIS_REQUEST_BYTES
        )

    def _dispatch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> tuple[_DecisionObservation, ...]:
        if self._dispatch_blocked:
            self._incomplete = True
            return tuple(
                _DecisionObservation(dict(request), None, None, None)
                for request in requests
            )
        gateway = getattr(self.gateway, "gateway", self.gateway)
        attempt_reservation = (
            max(0, gateway.config.max_retries) + 1
            if isinstance(gateway, HttpGateway)
            else 1
        )
        if (
            self._bounded_fallback
            and self._provider_requests + attempt_reservation
            > MAX_DIAGNOSIS_PROVIDER_REQUESTS
        ):
            self._incomplete = True
            return tuple(
                _DecisionObservation(dict(request), None, None, None)
                for request in requests
            )
        log = getattr(self.gateway, "decision_log", ())
        before = len(log) if isinstance(log, Sequence) else 0
        transport_before = self._transport_attempt_count()
        started = perf_counter()
        self._provider_requests += 1
        try:
            raw_responses = list(self.gateway.decide_batch(requests))
        except ProviderError:
            if not (
                self.speculative_fanout
                and self.sentence_protocol_version >= 2
                and self.task_taxonomy_version >= 2
            ):
                raise
            self._incomplete = True
            self._dispatch_blocked = True
            self._fallback_reason = "diagnosis_provider_error"
            raw_responses = []
        finally:
            transport_after = self._transport_attempt_count()
            if transport_before is not None and transport_after is not None:
                self._provider_requests += max(
                    0, transport_after - transport_before - 1
                )
        self._request_latencies_ms.append((perf_counter() - started) * 1000)
        entries = list(log)[before:] if isinstance(log, Sequence) else []
        observations: list[_DecisionObservation] = []
        for index, request in enumerate(requests):
            raw_answer = raw_responses[index] if index < len(raw_responses) else None
            try:
                decision = parse_decision(raw_answer)
            except JevResponseError:
                # A malformed answer invalidates its own decision only. Other
                # independent questions in the same batch remain usable.
                decision = None
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

    def _transport_attempt_count(self) -> int | None:
        gateway = getattr(self.gateway, "gateway", self.gateway)
        if not isinstance(gateway, HttpGateway):
            return None
        return gateway.transport_attempts_by_role.get("judge", 0)

    def _decide(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> tuple[JevDecision | None, ...]:
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

    def _classify_task_v1(
        self, prompt: str, state: Mapping[str, Any], rubric: DiagnosisRubric
    ) -> _TaskSelection:
        task_by_key = {task.key: task for task in rubric.task_types}
        general = task_by_key.get("general") or task_by_key.get(
            rubric.default_task_type
        )
        if general is None:
            general = rubric.task_types[0]
        options = {
            "general": jev_questions.GENERAL_TASK_DESCRIPTION,
            **{
                key: "Contains " + ", ".join(children) + " requests."
                for key, children in _HISTORICAL_TASK_TREE.items()
            },
            "unknown": jev_questions.UNKNOWN_TASK_DESCRIPTION,
        }
        request = _request(
            jev_questions.TASK_TYPE_QUESTION,
            state,
            type="choice",
            options=options,
            key="task_type",
        )
        root_observation = self._observe((request,))[0]
        root_decision = root_observation.decision
        root_selected = (
            root_decision.selected
            if isinstance(root_decision, ChoiceDecision)
            else None
        )
        root_confidence = (
            root_decision.confidence
            if isinstance(root_decision, ChoiceDecision)
            else 0.0
        )
        root_evidence = self._role_evidence(
            root_observation,
            accepted=isinstance(root_decision, ChoiceDecision),
            reason=(
                "supported"
                if isinstance(root_decision, ChoiceDecision)
                else "root_missing_or_malformed"
            ),
            question_id="task_type",
            family="task_type",
        )
        root_entry = {
            "key": root_selected,
            "label": root_selected,
            "probability": (
                root_decision.probabilities.get(root_selected, 0.0)
                if isinstance(root_decision, ChoiceDecision) and root_selected
                else 0.0
            ),
            "confidence": root_confidence,
            "evidence": root_evidence,
        }
        selected = rubric.default_task_type
        confidence = 0.0
        fallback_reason: str | None = None
        path: tuple[Mapping[str, Any], ...] = (root_entry,)
        if isinstance(root_decision, ChoiceDecision):
            if root_selected in task_by_key:
                selected = root_selected
                confidence = root_confidence
            elif root_selected in _HISTORICAL_TASK_TREE:
                children = _HISTORICAL_TASK_TREE[root_selected]
                leaf_request = _request(
                    jev_questions.task_leaf_question(root_selected),
                    state,
                    type="choice",
                    options={
                        child: task_by_key[child].label
                        for child in children
                        if child in task_by_key
                    }
                    | {"unknown": jev_questions.UNKNOWN_LEAF_DESCRIPTION},
                    key=f"task_type:{root_selected}",
                )
                leaf_observation = self._observe((leaf_request,))[0]
                leaf_decision = leaf_observation.decision
                if (
                    isinstance(leaf_decision, ChoiceDecision)
                    and leaf_decision.selected in children
                    and leaf_decision.selected in task_by_key
                ):
                    selected = leaf_decision.selected
                    confidence = min(root_confidence, leaf_decision.confidence)
                    leaf_evidence = self._role_evidence(
                        leaf_observation,
                        accepted=True,
                        reason="supported",
                        question_id=f"task_type:{root_selected}",
                        family="task_type",
                    )
                    path = (
                        root_entry,
                        {
                            "key": selected,
                            "label": task_by_key[selected].label,
                            "probability": leaf_decision.probabilities.get(
                                selected, 0.0
                            ),
                            "confidence": leaf_decision.confidence,
                            "evidence": leaf_evidence,
                        },
                    )
                elif not isinstance(leaf_decision, ChoiceDecision):
                    fallback_reason = "leaf_missing_or_malformed"
                elif leaf_decision.selected == "unknown":
                    fallback_reason = "leaf_unknown"
                else:
                    fallback_reason = "leaf_unsupported_option"
            elif root_selected == "unknown":
                fallback_reason = "root_unknown"
            else:
                fallback_reason = "root_unsupported_option"
        else:
            fallback_reason = "root_missing_or_malformed"
        task = task_by_key.get(selected, general)
        return _TaskSelection(
            task=task,
            confidence=confidence,
            path=path,
            fallback_reason=fallback_reason,
            evidence={
                "protocol_version": HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
                "root": root_evidence,
                "explored_paths": [],
                "selected_path_score": None,
                "checklist_source": "historical_leaf",
                "provider_requests": 1 + int(root_selected in _HISTORICAL_TASK_TREE),
                "provider_request_delta_vs_legacy": 0,
            },
        )

    def _classify_task(
        self, prompt: str, state: Mapping[str, Any], rubric: DiagnosisRubric
    ) -> _TaskSelection:
        if self.task_taxonomy_version == HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION:
            return self._classify_task_v1(prompt, state, rubric)
        task_by_key = {task.key: task for task in rubric.task_types}
        general = task_by_key.get("general") or task_by_key.get(
            rubric.default_task_type
        )
        if general is None:
            general = rubric.task_types[0]
        branches = _active_task_branches(rubric)
        branch_by_key = {branch.key: branch for branch in branches}
        tasks_in_branches = {key for branch in branches for key in branch.children}
        options: dict[str, str] = {"general": jev_questions.GENERAL_TASK_DESCRIPTION}
        for branch in branches:
            children = [task_by_key[key] for key in branch.children]
            options[branch.key] = jev_questions.task_branch_description(
                label=branch.label,
                description=branch.description,
                scope=branch.scope,
                children=[_task_description(task) for task in children],
            )
        direct_tasks = tuple(
            task
            for task in rubric.task_types
            if task.key not in tasks_in_branches and task.key != general.key
        )
        options.update({task.key: _task_description(task) for task in direct_tasks})
        options["unknown"] = jev_questions.UNKNOWN_TASK_DESCRIPTION
        request = _request(
            jev_questions.TASK_TYPE_QUESTION,
            state,
            type="choice",
            options=options,
            key="task_type",
            question_schema={"protocol": TASK_TAXONOMY_PROTOCOL_VERSION, "question": 1},
        )
        root_observation = self._observe((request,))[0]
        root_decision = root_observation.decision
        root_policy = (
            self._policy_decision(
                question_id="task_type",
                request=request,
                observation=root_observation,
                family="task_type",
                event_mapping={"selected_correctness": True},
            )
            if isinstance(root_decision, ChoiceDecision)
            else None
        )
        root_threshold = rubric.task_type_confidence_threshold
        if root_policy is not None and root_policy.threshold is not None:
            root_threshold = root_policy.threshold

        def root_supported() -> bool:
            if not isinstance(root_decision, ChoiceDecision):
                return False
            if root_policy is not None and not root_policy.is_legacy:
                return root_policy.may_gate
            return root_decision.confidence >= rubric.task_type_confidence_threshold

        selected_root = (
            root_decision.selected
            if isinstance(root_decision, ChoiceDecision)
            else None
        )
        if not isinstance(root_decision, ChoiceDecision):
            root_reason = "root_missing_or_malformed"
        elif selected_root not in options:
            root_reason = "root_unsupported_option"
        elif selected_root == "unknown":
            root_reason = "root_unknown"
        elif not root_supported():
            root_reason = (
                "root_calibration_unavailable"
                if root_policy is not None and not root_policy.is_legacy
                else "root_below_confidence"
            )
        else:
            root_reason = None

        root_evidence = self._role_evidence(
            root_observation,
            threshold=root_threshold,
            accepted=root_reason is None,
            reason="supported" if root_reason is None else root_reason,
            question_id="task_type",
            family="task_type",
            event_mapping={"selected_correctness": True},
        )
        root_probability = (
            root_decision.probabilities.get(selected_root, 0.0)
            if isinstance(root_decision, ChoiceDecision) and selected_root is not None
            else 0.0
        )
        root_path_entry = {
            "key": selected_root,
            "label": (
                branch_by_key[selected_root].label
                if selected_root in branch_by_key
                else task_by_key[selected_root].label
                if selected_root in task_by_key
                else selected_root
            ),
            "probability": root_probability,
            "confidence": (
                root_decision.confidence
                if isinstance(root_decision, ChoiceDecision)
                else 0.0
            ),
            "evidence": root_evidence,
        }

        def general_selection(reason: str | None) -> _TaskSelection:
            return _TaskSelection(
                task=general,
                confidence=(
                    root_decision.confidence
                    if isinstance(root_decision, ChoiceDecision)
                    and selected_root == "general"
                    and reason is None
                    else root_probability
                ),
                path=(root_path_entry,),
                fallback_reason=reason,
                evidence={
                    "protocol_version": TASK_TAXONOMY_PROTOCOL_VERSION,
                    "root": root_evidence,
                    "explored_paths": [],
                    "selected_path_score": None,
                    "checklist_source": "general",
                    "provider_requests": 1,
                    "provider_request_delta_vs_legacy": -int(
                        selected_root in _HISTORICAL_TASK_TREE
                    ),
                },
            )

        if root_reason is not None:
            return general_selection(root_reason)
        if not isinstance(root_decision, ChoiceDecision):
            return general_selection("root_missing_or_malformed")
        if selected_root == general.key:
            return general_selection(None)

        if selected_root in task_by_key and selected_root not in branch_by_key:
            selected_task = task_by_key[selected_root]
            calibrated = root_policy is not None and not root_policy.is_legacy
            accepted = (
                root_policy.may_gate
                if calibrated and root_policy is not None
                else root_decision.confidence >= rubric.task_type_confidence_threshold
            )
            if root_probability <= 0 or not accepted:
                return general_selection("leaf_below_confidence")
            return _TaskSelection(
                task=selected_task,
                confidence=root_probability,
                path=(root_path_entry,),
                fallback_reason=None,
                evidence={
                    "protocol_version": TASK_TAXONOMY_PROTOCOL_VERSION,
                    "root": root_evidence,
                    "explored_paths": [],
                    "selected_path_score": root_probability,
                    "checklist_source": "leaf",
                    "provider_requests": 1,
                    "provider_request_delta_vs_legacy": 0,
                },
            )

        if selected_root not in branch_by_key:
            return general_selection("root_unsupported_option")
        if root_probability <= 0:
            return general_selection("root_branch_probability_missing")

        branch_probabilities = [
            (branch, root_decision.probabilities.get(branch.key, 0.0), index)
            for index, branch in enumerate(branches)
            if root_decision.probabilities.get(branch.key, 0.0) > 0
        ]
        branch_probabilities.sort(key=lambda item: (-item[1], item[2]))
        if not branch_probabilities:
            return general_selection("root_no_supported_branch")
        selected_branch = next(
            (item for item in branch_probabilities if item[0].key == selected_root),
            None,
        )
        if selected_branch is None:
            return general_selection("root_branch_probability_missing")
        beam = [branch_probabilities[0]]
        if (
            len(branch_probabilities) > 1
            and branch_probabilities[0][1] - branch_probabilities[1][1]
            < rubric.task_branch_margin
        ):
            beam = branch_probabilities[: rubric.task_beam_width]

        requests: list[Mapping[str, Any]] = []
        for branch, _probability, _index in beam:
            requests.append(
                _request(
                    jev_questions.task_leaf_question(branch.key),
                    state,
                    type="choice",
                    options={
                        task_key: _task_description(task_by_key[task_key])
                        for task_key in branch.children
                    }
                    | {"unknown": jev_questions.UNKNOWN_LEAF_DESCRIPTION},
                    key=f"task_type:{branch.key}",
                    question_schema={
                        "protocol": TASK_TAXONOMY_PROTOCOL_VERSION,
                        "question": 2,
                        "branch": branch.key,
                    },
                )
            )
        leaf_observations = self._observe(requests)
        explored: list[dict[str, Any]] = []
        supported: list[
            tuple[float, int, int, TaskType, tuple[Mapping[str, Any], ...]]
        ] = []
        for (branch, branch_probability, branch_index), observation in zip(
            beam, leaf_observations, strict=True
        ):
            decision = observation.decision
            leaf_policy = (
                self._policy_decision(
                    question_id=f"task_type:{branch.key}",
                    request=observation.request,
                    observation=observation,
                    family="task_type",
                    event_mapping={"selected_correctness": True},
                )
                if isinstance(decision, ChoiceDecision)
                else None
            )
            chosen = decision.selected if isinstance(decision, ChoiceDecision) else None
            leaf_probability = (
                decision.probabilities.get(chosen, 0.0)
                if isinstance(decision, ChoiceDecision) and chosen is not None
                else 0.0
            )
            if not isinstance(decision, ChoiceDecision):
                leaf_reason = "leaf_missing_or_malformed"
            elif chosen == "unknown":
                leaf_reason = "leaf_unknown"
            elif chosen not in branch.children:
                leaf_reason = "leaf_unsupported_option"
            elif leaf_probability <= 0:
                leaf_reason = "leaf_probability_missing"
            elif leaf_policy is not None and not leaf_policy.is_legacy:
                leaf_reason = (
                    None if leaf_policy.may_gate else "leaf_calibration_unavailable"
                )
            elif decision.confidence < rubric.task_type_confidence_threshold:
                leaf_reason = "leaf_below_confidence"
            else:
                leaf_reason = None
            leaf_threshold = (
                leaf_policy.threshold
                if leaf_policy is not None and leaf_policy.threshold is not None
                else rubric.task_type_confidence_threshold
            )
            leaf_evidence = self._role_evidence(
                observation,
                threshold=leaf_threshold,
                accepted=leaf_reason is None,
                reason="supported" if leaf_reason is None else leaf_reason,
                question_id=f"task_type:{branch.key}",
                family="task_type",
                event_mapping={"selected_correctness": True},
            )
            branch_path = {
                "key": branch.key,
                "label": branch.label,
                "probability": branch_probability,
                "confidence": root_decision.confidence,
                "evidence": root_evidence,
            }
            leaf_path = {
                "key": chosen,
                "label": task_by_key[chosen].label if chosen in task_by_key else chosen,
                "probability": leaf_probability,
                "confidence": decision.confidence
                if isinstance(decision, ChoiceDecision)
                else 0.0,
                "evidence": leaf_evidence,
            }
            path_score = math.sqrt(branch_probability * leaf_probability)
            explored.append(
                {
                    "branch": branch.key,
                    "branch_probability": branch_probability,
                    "leaf": chosen,
                    "leaf_probability": leaf_probability,
                    "leaf_confidence": (
                        decision.confidence
                        if isinstance(decision, ChoiceDecision)
                        else None
                    ),
                    "path_score": path_score,
                    "accepted": leaf_reason is None,
                    "reason": leaf_reason,
                    "evidence": leaf_evidence,
                }
            )
            if leaf_reason is None and chosen in task_by_key:
                supported.append(
                    (
                        path_score,
                        branch_index,
                        branch.children.index(chosen),
                        task_by_key[chosen],
                        (branch_path, leaf_path),
                    )
                )

        if supported:
            supported.sort(key=lambda item: (-item[0], item[1], item[2]))
            score, _branch_index, _leaf_index, selected_task, path = supported[0]
            selection = _TaskSelection(
                task=selected_task,
                confidence=score,
                path=path,
                fallback_reason=None,
                evidence={
                    "protocol_version": TASK_TAXONOMY_PROTOCOL_VERSION,
                    "root": root_evidence,
                    "explored_paths": explored,
                    "selected_path_score": score,
                    "checklist_source": "leaf",
                    "provider_requests": 2,
                    "provider_request_delta_vs_legacy": 0,
                },
            )
            return selection

        parent_branch, parent_probability, _parent_index = branch_probabilities[0]
        checklist, empty_intersection = _parent_checklist(parent_branch, task_by_key)
        checklist_source = (
            "explicit_parent"
            if parent_branch.checklist is not None
            else "parent_intersection"
        )
        checklist_fallback_reason = None
        if empty_intersection:
            checklist = general.checklist
            checklist_source = "general_after_empty_parent_intersection"
            checklist_fallback_reason = "parent_intersection_empty"
        parent = TaskType(
            key=parent_branch.key,
            label=parent_branch.label,
            checklist=checklist,
            description=parent_branch.description,
            scope=parent_branch.scope,
        )
        branch_path = {
            "key": parent_branch.key,
            "label": parent_branch.label,
            "probability": parent_probability,
            "confidence": root_decision.confidence,
            "evidence": root_evidence,
        }
        failed_reasons = [item["reason"] for item in explored if item["reason"]]
        fallback_reason = failed_reasons[0] if failed_reasons else "leaf_unsupported"
        return _TaskSelection(
            task=parent,
            confidence=parent_probability,
            path=(branch_path,),
            fallback_reason=fallback_reason,
            evidence={
                "protocol_version": TASK_TAXONOMY_PROTOCOL_VERSION,
                "root": root_evidence,
                "explored_paths": explored,
                "selected_path_score": parent_probability,
                "checklist_source": checklist_source,
                "checklist_fallback_reason": checklist_fallback_reason,
                "provider_requests": 2,
                "provider_request_delta_vs_legacy": 0,
            },
        )

    def _speculative_requests(
        self, prompt: str, rubric: DiagnosisRubric
    ) -> tuple[tuple[Mapping[str, Any], str], ...]:
        """Plan existing question meanings before the selected path is known."""
        state = {"prompt": prompt}
        task_by_key = {task.key: task for task in rubric.task_types}
        general = task_by_key.get("general") or task_by_key.get(
            rubric.default_task_type
        )
        if general is None:
            general = rubric.task_types[0]
        branches = _active_task_branches(rubric)
        branched = {key for branch in branches for key in branch.children}
        options: dict[str, str] = {"general": jev_questions.GENERAL_TASK_DESCRIPTION}
        for branch in branches:
            options[branch.key] = jev_questions.task_branch_description(
                label=branch.label,
                description=branch.description,
                scope=branch.scope,
                children=[
                    _task_description(task_by_key[key]) for key in branch.children
                ],
            )
        options.update(
            {
                task.key: _task_description(task)
                for task in rubric.task_types
                if task.key not in branched and task.key != general.key
            }
        )
        options["unknown"] = jev_questions.UNKNOWN_TASK_DESCRIPTION
        planned: list[Mapping[str, Any]] = [
            _request(
                jev_questions.TASK_TYPE_QUESTION,
                state,
                type="choice",
                options=options,
                key="task_type",
                question_schema={
                    "protocol": TASK_TAXONOMY_PROTOCOL_VERSION,
                    "question": 1,
                },
            )
        ]
        for branch in branches:
            planned.append(
                _request(
                    jev_questions.task_leaf_question(branch.key),
                    state,
                    type="choice",
                    options={
                        task_key: _task_description(task_by_key[task_key])
                        for task_key in branch.children
                    }
                    | {"unknown": jev_questions.UNKNOWN_LEAF_DESCRIPTION},
                    key=f"task_type:{branch.key}",
                    question_schema={
                        "protocol": TASK_TAXONOMY_PROTOCOL_VERSION,
                        "question": 2,
                        "branch": branch.key,
                    },
                )
            )
        checklist_items = [
            item for task in rubric.task_types for item in task.checklist
        ] + [item for branch in branches for item in (branch.checklist or ())]
        for item in checklist_items:
            planned.append(
                _request(gap_question(item), state, type="noul", key=f"gap:{item.key}")
            )
        sentences = split_sentences(prompt)
        for window_index, start in enumerate(range(0, len(sentences), 254)):
            window = sentences[start : start + 254]
            window_state = {
                **state,
                "sentences": [{"id": item.id, "text": item.text} for item in window],
                "candidate_sentence_ids": [item.id for item in window],
                "sentence_window": {
                    "protocol_version": self.sentence_protocol_version,
                    "index": window_index,
                    "first_sentence_id": window[0].id,
                    "last_sentence_id": window[-1].id,
                },
            }
            for kind in _PROBLEM_QUESTIONS:
                planned.append(
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
                planned.append(
                    _request(
                        jev_questions.sentence_existence_question(
                            kind.value.replace("_", " ")
                        ),
                        window_state,
                        type="noul",
                        key=f"existence:{kind.value}:{window_index}",
                        question_schema={
                            "protocol": self.sentence_protocol_version,
                            "question": SENTENCE_EXISTENCE_QUESTION_VERSION,
                        },
                    )
                )
        planned.extend(self.additional_requests)
        seen: set[str] = set()
        used_keys: set[str] = set()
        unique: list[tuple[Mapping[str, Any], str]] = []
        for request in planned:
            identity = self._request_identity(request)
            if identity in seen:
                continue
            seen.add(identity)
            dispatched = dict(request)
            key = str(dispatched["key"])
            if key in used_keys:
                suffix = 1
                while f"{key}:fanout:{suffix}" in used_keys:
                    suffix += 1
                dispatched["key"] = f"{key}:fanout:{suffix}"
            used_keys.add(str(dispatched["key"]))
            unique.append((dispatched, identity))
        return tuple(unique)

    def observe_additional(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> tuple[_DecisionObservation, ...]:
        return self._observe(requests)

    def _prefetch_diagnosis(self, prompt: str, rubric: DiagnosisRubric) -> None:
        planned = self._speculative_requests(prompt, rubric)
        requests = [request for request, _identity in planned]
        if not self._request_fits(requests):
            self._bounded_fallback = True
            self._fallback_reason = "speculative_request_exceeds_provider_limits"
            return
        observations = self._dispatch(requests)
        self._prefetched = {}
        for (_request, identity), observation in zip(
            planned, observations, strict=True
        ):
            original = json.loads(identity)
            self._prefetched[identity] = replace(observation, request=original)

    def diagnose(self, prompt: str) -> DiagnosisReport:
        self._calibration_evidence = {}
        self._provider_requests = 0
        self._request_latencies_ms = []
        self._incomplete = False
        self._fallback_reason = None
        self._prefetched = None
        self._bounded_fallback = False
        self._dispatch_blocked = False
        rubric = self.rubric
        if (
            self.speculative_fanout
            and self.sentence_protocol_version >= 2
            and self.task_taxonomy_version >= 2
        ):
            if len(prompt) > MAX_DIAGNOSIS_INPUT_CHARACTERS:
                general = next(
                    (
                        task
                        for task in rubric.task_types
                        if task.key == rubric.default_task_type
                    ),
                    rubric.task_types[0],
                )
                return DiagnosisReport(
                    task_type=general.key,
                    task_type_label=general.label,
                    task_type_confidence=0.0,
                    confirmed_gaps=(),
                    problem_sentences=(),
                    rubric_version=self.rubric_version,
                    request_evidence={
                        "protocol": "diagnosis-fanout-v1",
                        "complete": False,
                        "mode": "input_cap_hold",
                        "reason": "draft_character_cap_exceeded",
                        "provider_requests": 0,
                        "request_latencies_ms": [],
                        "latency_source": "unavailable",
                        "input_characters": len(prompt),
                        "question_count": 0,
                    },
                )
            self._prefetch_diagnosis(prompt, rubric)
        state = {"prompt": prompt}
        classification_started = perf_counter()
        selection = self._classify_task(prompt, state, rubric)
        taxonomy_evidence = {
            **dict(selection.evidence),
            "effective_task_type": selection.task.key,
            "fallback_reason": selection.fallback_reason,
            "effective_checklist": list(_checklist_payload(selection.task.checklist)),
            "classification_latency_ms": (perf_counter() - classification_started)
            * 1000,
        }
        task = selection.task
        gaps, near_misses, sentences = self._diagnose_gaps(prompt, state, task, rubric)
        problems, pointed, sentence_evidence = self._diagnose_sentences(
            prompt, state, sentences, rubric
        )
        # Quote the sentence Jev pointed at, preferring an unresolved reference.
        suspect = pointed.get(ProblemKind.UNRESOLVED_REFERENCE) or pointed.get(
            ProblemKind.VAGUENESS
        )
        return DiagnosisReport(
            task_type=task.key,
            task_type_label=task.label,
            task_type_confidence=selection.confidence,
            confirmed_gaps=gaps,
            problem_sentences=problems,
            possible_gaps=tuple(
                replace(gap, sentence=suspect.text if suspect else None)
                for gap in near_misses
            ),
            calibration=dict(self._calibration_evidence) or None,
            rubric_version=self.rubric_version,
            sentence_protocol_version=self.sentence_protocol_version,
            sentence_evidence=sentence_evidence,
            task_type_path=selection.path,
            task_type_fallback_reason=selection.fallback_reason,
            effective_checklist=_checklist_payload(task.checklist),
            taxonomy_evidence=taxonomy_evidence,
            request_evidence=self.request_evidence(prompt)
            if self.record_request_evidence
            else None,
        )

    def request_evidence(self, prompt: str) -> dict[str, Any]:
        return {
            "protocol": "diagnosis-fanout-v1",
            "complete": not self._incomplete,
            "mode": "bounded_sequential_fallback"
            if self._bounded_fallback
            else "speculative_fanout"
            if self._prefetched is not None
            else "sequential",
            "reason": self._fallback_reason
            if self._fallback_reason is not None
            else "required_evidence_missing"
            if self._incomplete
            else None,
            "provider_requests": self._provider_requests,
            "request_latencies_ms": list(self._request_latencies_ms),
            "latency_source": "measured_provider"
            if isinstance(getattr(self.gateway, "gateway", self.gateway), HttpGateway)
            else "deterministic_or_replay",
            "input_characters": len(prompt),
            "question_count": len(self._prefetched or {}),
        }

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
    ) -> tuple[
        tuple[ProblemSentence, ...],
        dict[ProblemKind, Sentence],
        tuple[Mapping[str, Any], ...],
    ]:
        if (
            self.sentence_protocol_version
            == HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION
        ):
            problems, pointed = self._diagnose_sentences_v1(
                prompt, state, sentences, rubric
            )
            return problems, pointed, ()
        return self._diagnose_sentences_v2(prompt, state, sentences, rubric)

    def _diagnose_sentences_v1(
        self,
        prompt: str,
        state: Mapping[str, Any],
        sentences: tuple[Sentence, ...],
        rubric: DiagnosisRubric,
    ) -> tuple[tuple[ProblemSentence, ...], dict[ProblemKind, Sentence]]:
        """Replay the historical pointer-then-confirmation protocol exactly."""

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

    def _diagnose_sentences_v2(
        self,
        prompt: str,
        state: Mapping[str, Any],
        sentences: tuple[Sentence, ...],
        rubric: DiagnosisRubric,
    ) -> tuple[
        tuple[ProblemSentence, ...],
        dict[ProblemKind, Sentence],
        tuple[Mapping[str, Any], ...],
    ]:
        del prompt
        if not sentences:
            return (), {}, ()

        pairs: list[dict[str, Any]] = []
        requests: list[Mapping[str, Any]] = []
        for window_index, start in enumerate(range(0, len(sentences), 254)):
            window = sentences[start : start + 254]
            sentence_values = [{"id": item.id, "text": item.text} for item in window]
            window_state = {
                **state,
                "sentences": sentence_values,
                "candidate_sentence_ids": [item.id for item in window],
                "sentence_window": {
                    "protocol_version": self.sentence_protocol_version,
                    "index": window_index,
                    "first_sentence_id": window[0].id,
                    "last_sentence_id": window[-1].id,
                },
            }
            for kind in _PROBLEM_QUESTIONS:
                pointer_request = _request(
                    jev_questions.sentence_pointer_question(
                        kind.value.replace("_", " ")
                    ),
                    window_state,
                    type="choice",
                    options=[item.id for item in window] + ["none"],
                    key=f"pointer:{kind.value}:{window_index}",
                )
                existence_request = _request(
                    jev_questions.sentence_existence_question(
                        kind.value.replace("_", " ")
                    ),
                    window_state,
                    type="noul",
                    key=f"existence:{kind.value}:{window_index}",
                    question_schema={
                        "protocol": self.sentence_protocol_version,
                        "question": SENTENCE_EXISTENCE_QUESTION_VERSION,
                    },
                )
                pairs.append(
                    {
                        "kind": kind,
                        "window_index": window_index,
                        "window": window,
                        "candidate_sentence_ids": tuple(item.id for item in window),
                        "pointer_request": pointer_request,
                        "existence_request": existence_request,
                    }
                )
                requests.extend((pointer_request, existence_request))

        # Each pair's pointer Choice and existence Noul share one batch request.
        observations = self._observe(requests)
        evidence: list[dict[str, Any]] = []
        selected: dict[tuple[ProblemKind, str], dict[str, Any]] = {}
        for index, pair in enumerate(pairs):
            kind = pair["kind"]
            pointer_observation = (
                observations[index * 2] if index * 2 < len(observations) else None
            )
            existence_observation = (
                observations[index * 2 + 1]
                if index * 2 + 1 < len(observations)
                else None
            )
            record: dict[str, Any] = {
                "protocol_version": self.sentence_protocol_version,
                "kind": kind.value,
                "window_index": pair["window_index"],
                "candidate_sentence_ids": list(pair["candidate_sentence_ids"]),
                "candidate_sentences": [
                    {"id": item.id, "text": item.text} for item in pair["window"]
                ],
                "existence": {},
                "pointer": {},
                "confirmation": {
                    "requested": False,
                    "accepted": False,
                    "reason": "existence_not_supported",
                },
            }
            evidence.append(record)
            record["pointer"] = self._role_evidence(
                pointer_observation,
                accepted=False,
                reason="existence_not_supported",
                question_id=f"pointer:{kind.value}",
                family="pointer",
                event_mapping={"selected_correctness": True},
                criteria_descriptor=POINTER_CALIBRATION_CRITERIA,
            )

            existence = (
                existence_observation.decision
                if existence_observation is not None
                else None
            )
            existence_policy = None
            existence_threshold = rubric.existence_threshold_for(kind)
            if (
                isinstance(existence, NoulDecision)
                and existence_observation is not None
            ):
                existence_policy = self._policy_decision(
                    question_id=f"existence:{kind.value}",
                    request=existence_observation.request,
                    observation=existence_observation,
                    family="existence",
                )
                if (
                    existence_policy is not None
                    and existence_policy.threshold is not None
                ):
                    existence_threshold = existence_policy.threshold
            existence_calibrated = (
                existence_policy is None
                or existence_policy.is_legacy
                or existence_policy.may_gate
            )
            existence_supported = (
                isinstance(existence, NoulDecision)
                and existence.probability >= existence_threshold
                and existence_calibrated
            )
            record["existence"] = self._role_evidence(
                existence_observation,
                threshold=existence_threshold,
                accepted=existence_supported,
                reason=(
                    "supported"
                    if existence_supported
                    else "below_threshold"
                    if isinstance(existence, NoulDecision)
                    else "missing_or_malformed_answer"
                ),
                question_id=f"existence:{kind.value}",
                family="existence",
            )
            record["existence"]["question_version"] = (
                SENTENCE_EXISTENCE_QUESTION_VERSION
            )
            record["existence"]["rubric_threshold_version"] = (
                rubric.existence_threshold_version
            )
            if existence_policy is not None:
                record["existence"]["calibration"] = self._policy_evidence(
                    existence_policy
                )
            if not existence_supported:
                continue
            record["confirmation"] = {
                "requested": False,
                "accepted": False,
                "reason": "pointer_not_accepted",
            }
            if pointer_observation is None:
                continue

            pointer = pointer_observation.decision
            if not isinstance(pointer, ChoiceDecision):
                record["pointer"] = self._role_evidence(
                    pointer_observation,
                    accepted=False,
                    reason="missing_or_malformed_answer",
                    question_id=f"pointer:{kind.value}",
                    family="pointer",
                    event_mapping={"selected_correctness": True},
                    criteria_descriptor=POINTER_CALIBRATION_CRITERIA,
                )
                continue
            if pointer.selected == "none":
                record["pointer"] = self._role_evidence(
                    pointer_observation,
                    accepted=False,
                    reason="none_selected",
                    question_id=f"pointer:{kind.value}",
                    family="pointer",
                    event_mapping={"selected_correctness": True},
                    criteria_descriptor=POINTER_CALIBRATION_CRITERIA,
                )
                continue
            window_ids = set(pair["candidate_sentence_ids"])
            if pointer.selected not in window_ids:
                record["pointer"] = self._role_evidence(
                    pointer_observation,
                    accepted=False,
                    reason="selected_id_outside_window",
                    question_id=f"pointer:{kind.value}",
                    family="pointer",
                    event_mapping={"selected_correctness": True},
                    criteria_descriptor=POINTER_CALIBRATION_CRITERIA,
                )
                continue

            pointer_policy = self._policy_decision(
                question_id=f"pointer:{kind.value}",
                request=pointer_observation.request,
                observation=pointer_observation,
                family="pointer",
                event_mapping={"selected_correctness": True},
                criteria_descriptor=POINTER_CALIBRATION_CRITERIA,
            )
            pointer_supported = (
                pointer_policy is None
                or pointer_policy.is_legacy
                or pointer_policy.disposition != "abstain"
            )
            if pointer_policy is not None and pointer_policy.disposition == "abstain":
                pointer_supported = False
            if (
                pointer_policy is None or not pointer_policy.may_gate
            ) and pointer.confidence < rubric.pointer_threshold:
                pointer_supported = False
            sentence = next(
                item for item in pair["window"] if item.id == pointer.selected
            )
            record["pointer"] = self._role_evidence(
                pointer_observation,
                threshold=rubric.pointer_threshold,
                accepted=pointer_supported,
                reason="supported" if pointer_supported else "below_threshold",
                question_id=f"pointer:{kind.value}",
                family="pointer",
                event_mapping={"selected_correctness": True},
                criteria_descriptor=POINTER_CALIBRATION_CRITERIA,
            )
            if pointer_policy is not None:
                record["pointer"]["calibration"] = self._policy_evidence(pointer_policy)
            if not pointer_supported:
                continue
            pair_key = (kind, sentence.id)
            if pair_key in selected:
                record["pointer"]["reason"] = "duplicate_pair"
                record["pointer"]["deduplicated_to_window"] = selected[pair_key][
                    "window_index"
                ]
                continue
            selected[pair_key] = {
                "kind": kind,
                "sentence": sentence,
                "window_index": pair["window_index"],
                "evidence": record,
            }
            record["confirmation"] = {
                "requested": True,
                "accepted": False,
                "reason": "awaiting_confirmation",
            }

        pointed: dict[ProblemKind, Sentence] = {}
        confirmation_requests: list[Mapping[str, Any]] = []
        confirmation_pairs: list[dict[str, Any]] = []
        for selected_pair in selected.values():
            kind = selected_pair["kind"]
            sentence = selected_pair["sentence"]
            pointed.setdefault(kind, sentence)
            request = _request(
                _PROBLEM_QUESTIONS[kind],
                {
                    **state,
                    "sentences": [{"id": sentence.id, "text": sentence.text}],
                    "candidate_sentence_ids": [sentence.id],
                    "selected_sentence_id": sentence.id,
                    "sentence_window": {
                        "protocol_version": self.sentence_protocol_version,
                        "index": selected_pair["window_index"],
                    },
                },
                type="noul",
                key=f"problem:{kind.value}:{sentence.id}",
            )
            confirmation_requests.append(request)
            confirmation_pairs.append({**selected_pair, "request": request})

        confirmations = self._observe(confirmation_requests)
        problems: list[ProblemSentence] = []
        for index, selected_pair in enumerate(confirmation_pairs):
            observation = confirmations[index] if index < len(confirmations) else None
            result = observation.decision if observation is not None else None
            kind = selected_pair["kind"]
            sentence = selected_pair["sentence"]
            policy_decision = None
            threshold = rubric.problem_threshold
            if isinstance(result, NoulDecision) and observation is not None:
                policy_decision = self._policy_decision(
                    question_id=f"problem:{kind.value}",
                    request=observation.request,
                    observation=observation,
                    family="problem",
                )
                if (
                    policy_decision is not None
                    and policy_decision.may_gate
                    and policy_decision.threshold is not None
                ):
                    threshold = policy_decision.threshold
            if policy_decision is not None and not policy_decision.is_legacy:
                confident = policy_decision.may_gate
            else:
                confident = (
                    isinstance(result, NoulDecision)
                    and result.probability >= threshold
                    and result.confidence >= rubric.confidence_threshold
                    and abs(result.probability - 0.5) >= rubric.uncertainty_margin
                )
            selected_pair["evidence"]["confirmation"] = {
                **self._role_evidence(
                    observation,
                    threshold=threshold,
                    accepted=confident,
                    reason=(
                        "supported"
                        if confident
                        else "below_threshold"
                        if isinstance(result, NoulDecision)
                        else "missing_or_malformed_answer"
                    ),
                    question_id=f"problem:{kind.value}",
                    family="problem",
                ),
                "requested": True,
            }
            if policy_decision is not None:
                selected_pair["evidence"]["confirmation"]["calibration"] = (
                    self._policy_evidence(policy_decision)
                )
            if confident and isinstance(result, NoulDecision):
                problems.append(
                    ProblemSentence(
                        sentence=sentence,
                        kind=kind,
                        probability=result.probability,
                        confidence=result.confidence,
                        threshold=threshold,
                    )
                )
        return tuple(problems), pointed, tuple(evidence)

    def _role_evidence(
        self,
        observation: _DecisionObservation | None,
        *,
        accepted: bool,
        reason: str,
        question_id: str,
        family: str,
        threshold: float | None = None,
        event_mapping: Mapping[str, Any] | None = None,
        criteria_descriptor: str | None = None,
    ) -> dict[str, Any]:
        request = observation.request if observation is not None else {}
        raw = observation.raw_answer if observation is not None else None
        decision = observation.decision if observation is not None else None
        identity: Mapping[str, Any] | None = None
        if observation is not None:
            from .evaluation.calibration import (
                DEFAULT_POLICY_VERSION,
                runtime_question_identity,
            )

            identity_value = runtime_question_identity(
                question_id,
                request,
                family=family,
                rubric_version=self.rubric_version,
                snapshot=observation.answered_by,
                policy_version=(
                    self.decision_policy.policy_version
                    if self.decision_policy is not None
                    else DEFAULT_POLICY_VERSION
                ),
            )
            if criteria_descriptor is not None:
                identity_value = replace(identity_value, criteria=criteria_descriptor)
            if event_mapping is not None:
                identity_value = replace(
                    identity_value, event_mapping=dict(event_mapping)
                )
            identity = identity_value.to_dict()
        parsed: dict[str, Any] = {}
        if isinstance(decision, NoulDecision):
            parsed = {
                "type": "noul",
                "probability": decision.probability,
                "confidence": decision.confidence,
            }
        elif isinstance(decision, ChoiceDecision):
            parsed = {
                "type": "choice",
                "selected": decision.selected,
                "probabilities": decision.probabilities,
                "confidence": decision.confidence,
            }
        evidence = {
            "question_id": question_id,
            "question": request.get("query", request.get("question")),
            "question_schema": request.get("question_schema"),
            "raw_answer": raw,
            "parsed_answer": parsed or None,
            "snapshot": observation.answered_by if observation is not None else None,
            "identity": identity,
            "threshold": threshold,
            "accepted": accepted,
            "reason": reason,
        }
        if question_id in self._calibration_evidence:
            evidence["calibration"] = self._calibration_evidence[question_id]
        return evidence

    @staticmethod
    def _policy_evidence(decision: PolicyDecision) -> dict[str, Any]:
        return {
            "disposition": decision.disposition,
            "verdict": decision.verdict,
            "reason": decision.reason,
            "threshold": decision.threshold,
            "predicate": dict(decision.predicate),
            "event_probability": decision.evidence.get("event_probability"),
            "fit": decision.evidence.get("fit"),
        }


def model_diagnosis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The diagnosis as sent to models: without the user-facing possible gaps.

    Near-miss hints must never steer a strategy, candidate, or fidelity decision,
    and leaving them out keeps recorded replays exact.
    """
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "possible_gaps",
            "calibration",
            "sentence_evidence",
            "sentence_protocol_version",
            "task_type_path",
            "task_type_fallback_reason",
            "effective_checklist",
            "taxonomy_evidence",
            "request_evidence",
        }
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
        rubric_version=(
            str(value["rubric_version"])
            if value.get("rubric_version") is not None
            else None
        ),
        sentence_protocol_version=int(
            value.get(
                "sentence_protocol_version",
                HISTORICAL_SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
            )
        ),
        sentence_evidence=tuple(
            dict(item)
            for item in value.get("sentence_evidence", ())
            if isinstance(item, Mapping)
        ),
        task_type_path=tuple(
            dict(item)
            for item in value.get("task_type_path", ())
            if isinstance(item, Mapping)
        ),
        task_type_fallback_reason=(
            str(value["task_type_fallback_reason"])
            if value.get("task_type_fallback_reason") is not None
            else None
        ),
        effective_checklist=tuple(
            dict(item)
            for item in value.get("effective_checklist", ())
            if isinstance(item, Mapping)
        ),
        taxonomy_evidence=(
            dict(value["taxonomy_evidence"])
            if isinstance(value.get("taxonomy_evidence"), Mapping)
            else None
        ),
        request_evidence=(
            dict(value["request_evidence"])
            if isinstance(value.get("request_evidence"), Mapping)
            else None
        ),
    )


def _state_json(value: Mapping[str, Any]) -> str:
    # Kept as a helper for gateway adapters that require a JSON string.
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
