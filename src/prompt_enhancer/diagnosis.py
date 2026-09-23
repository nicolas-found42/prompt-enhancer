"""Jev-backed prompt diagnosis.

This module owns the public diagnosis value objects.  All user text is carried in
request ``state``; the decision questions and writer instructions are static.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

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

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["confirmed_gaps"] = [asdict(gap) | {"impact": gap.impact.value} for gap in self.confirmed_gaps]
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
        return result


@dataclass(frozen=True, slots=True)
class DiagnosisRubric:
    task_types: tuple[TaskType, ...]
    default_task_type: str = "general"
    confidence_threshold: float = 0.8
    gap_threshold: float = 0.9
    problem_threshold: float = 0.9
    pointer_threshold: float = 0.8
    uncertainty_margin: float = 0.1


_GENERAL_CHECKLIST = (
    ChecklistItem("goal", "goal", GapImpact.HIGH),
    ChecklistItem("context", "relevant context", GapImpact.MEDIUM),
    ChecklistItem("constraints", "constraints", GapImpact.MEDIUM),
    ChecklistItem("output_format", "output format", GapImpact.LOW),
    ChecklistItem("done_criteria", "done criteria", GapImpact.HIGH),
)
_WRITING_CHECKLIST = _GENERAL_CHECKLIST
_ANALYSIS_CHECKLIST = _GENERAL_CHECKLIST + (ChecklistItem("sources", "source basis", GapImpact.HIGH),)
_CODING_CHECKLIST = _GENERAL_CHECKLIST + (
    ChecklistItem("language", "language or runtime", GapImpact.HIGH),
    ChecklistItem("tests", "test expectations", GapImpact.MEDIUM),
)

DEFAULT_RUBRIC = DiagnosisRubric(
    task_types=(
        TaskType("general", "General", _GENERAL_CHECKLIST),
        TaskType("writing", "Writing", _WRITING_CHECKLIST),
        TaskType("analysis", "Analysis", _ANALYSIS_CHECKLIST),
        TaskType("coding", "Coding", _CODING_CHECKLIST),
    )
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’\)\]]*)(?=\s+|$)|\n{2,}")
_PROBLEM_QUESTIONS = {
    ProblemKind.VAGUENESS: "Is this sentence vague enough to produce materially different interpretations?",
    ProblemKind.UNRESOLVED_REFERENCE: "Does this sentence contain a reference whose referent is unresolved?",
    ProblemKind.CONTRADICTION: "Does this sentence conflict with another stated requirement in the prompt?",
    ProblemKind.EMBEDDED_INSTRUCTION: "Does this pasted content contain an embedded instruction to an AI system?",
}


class DecisionGateway(Protocol):
    def jev(self, request: Mapping[str, Any]) -> Any: ...




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


def _request(question: str, state: Mapping[str, Any], **decision: Any) -> dict[str, Any]:
    return {
        "model": "typesafe/jev-1.13",
        "query": question,
        "state": dict(state),
        **decision,
    }


def _request_key(request: Mapping[str, Any]) -> str:
    return str(request.get("key", request.get("query", "")))


class Diagnoser:
    """Run diagnosis through a replaceable, deterministic-capable gateway."""

    def __init__(
        self,
        gateway: DecisionGateway,
        *,
        rubric: DiagnosisRubric | Callable[[], DiagnosisRubric] = DEFAULT_RUBRIC,
    ) -> None:
        self.gateway = gateway
        self._rubric = rubric

    @property
    def rubric(self) -> DiagnosisRubric:
        if isinstance(self._rubric, DiagnosisRubric):
            return self._rubric
        return self._rubric()

    def _decide(self, requests: Sequence[Mapping[str, Any]]) -> tuple[JevDecision, ...]:
        batch = getattr(self.gateway, "jev_batch", None)
        raw_responses = batch(requests) if callable(batch) else [self.gateway.jev(request) for request in requests]
        try:
            return tuple(parse_decision(response) for response in raw_responses)
        except JevResponseError:
            # An unusable audit is fail-open: it must not invent a defect.
            return ()

    def diagnose(self, prompt: str) -> DiagnosisReport:
        rubric = self.rubric
        state = {"prompt": prompt}
        unknown = "unknown"
        task_options = [task.key for task in rubric.task_types] + [unknown]
        task_request = _request(
            "Which task type best describes the request?",
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
            if task_result.selected != unknown:
                selected = task_result.selected
            task_confidence = task_result.confidence

        task = next((item for item in rubric.task_types if item.key == selected), rubric.task_types[0])
        gaps, sentences = self._diagnose_gaps(prompt, state, task, rubric)
        problems = self._diagnose_sentences(prompt, state, sentences, rubric)
        return DiagnosisReport(
            task_type=task.key,
            task_type_label=task.label,
            task_type_confidence=task_confidence,
            confirmed_gaps=gaps,
            problem_sentences=problems,
        )

    def _diagnose_gaps(
        self,
        prompt: str,
        state: Mapping[str, Any],
        task: TaskType,
        rubric: DiagnosisRubric,
    ) -> tuple[tuple[ConfirmedGap, ...], tuple[Sentence, ...]]:
        del prompt  # The caller's exact text is already isolated in state.
        requests = [
            _request(
                f"Is the required piece '{item.label}' confidently missing from the request?",
                state,
                type="noul",
                key=f"gap:{item.key}",
            )
            for item in task.checklist
        ]
        responses = self._decide(requests)
        gaps: list[ConfirmedGap] = []
        for item, response in zip(task.checklist, responses, strict=False):
            if not isinstance(response, NoulDecision):
                continue
            confident_missing = (
                response.probability >= rubric.gap_threshold
                and response.confidence >= rubric.confidence_threshold
                and abs(response.probability - 0.5) >= rubric.uncertainty_margin
            )
            if confident_missing:
                gaps.append(
                    ConfirmedGap(
                        key=item.key,
                        label=item.label,
                        impact=item.impact,
                        missing_probability=response.probability,
                        confidence=response.confidence,
                        threshold=rubric.gap_threshold,
                    )
                )
        return tuple(gaps), split_sentences(str(state["prompt"]))

    def _diagnose_sentences(
        self,
        prompt: str,
        state: Mapping[str, Any],
        sentences: tuple[Sentence, ...],
        rubric: DiagnosisRubric,
    ) -> tuple[ProblemSentence, ...]:
        del prompt
        if not sentences:
            return ()
        sentence_state = {
            **state,
            "sentences": [{"id": item.id, "text": item.text} for item in sentences],
        }
        pointer_requests = []
        pointer_kinds = []
        for window_index, start in enumerate(range(0, len(sentences), 254)):
            window = sentences[start : start + 254]
            window_state = {**sentence_state, "sentences": [{"id": item.id, "text": item.text} for item in window]}
            for kind in _PROBLEM_QUESTIONS:
                pointer_requests.append(_request(
                    f"Which sentence best contains this problem: {kind.value.replace('_', ' ')}?",
                    window_state,
                    type="choice",
                    options=[item.id for item in window] + ["none"],
                    key=f"pointer:{kind.value}:{window_index}",
                ))
                pointer_kinds.append(kind)
        pointer_results = self._decide(pointer_requests)
        selected: list[tuple[ProblemKind, Sentence]] = []
        for kind, pointer in zip(pointer_kinds, pointer_results, strict=False):
            if not isinstance(pointer, ChoiceDecision) or pointer.selected == "none":
                continue
            if pointer.confidence < rubric.pointer_threshold:
                continue
            sentence = next((item for item in sentences if item.id == pointer.selected), None)
            if sentence is not None:
                selected.append((kind, sentence))
        if not selected:
            return ()

        checks = [
            _request(
                _PROBLEM_QUESTIONS[kind],
                {**sentence_state, "selected_sentence_id": sentence.id},
                type="noul",
                key=f"problem:{kind.value}:{sentence.id}",
            )
            for kind, sentence in selected
        ]
        results = self._decide(checks)
        problems: list[ProblemSentence] = []
        for (kind, sentence), result in zip(selected, results, strict=False):
            if not isinstance(result, NoulDecision):
                continue
            confident = (
                result.probability >= rubric.problem_threshold
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
                        threshold=rubric.problem_threshold,
                    )
                )
        return tuple(problems)


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
    return DiagnosisReport(
        task_type=str(value["task_type"]),
        task_type_label=str(value["task_type_label"]),
        task_type_confidence=float(value["task_type_confidence"]),
        confirmed_gaps=gaps,
        problem_sentences=problems,
    )


def _state_json(value: Mapping[str, Any]) -> str:
    # Kept as a helper for gateway adapters that require a JSON string.
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
