"""Deterministic clarification planning and paused-run persistence.

The rest of the engine only needs to turn a diagnosis into :class:`GapAssessment`
values, call :func:`build_plan`, and then use :class:`ClarificationService` to
pause or continue a run.  Keeping this logic independent of the gateway makes
clarification replayable and keeps the HTTP and web surfaces thin.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Protocol,
)

INFERENCE_CONFIDENCE = 0.8
OTHER_VALUE = "other"
OTHER_LABEL = "Other"


class ClarificationError(ValueError):
    """Base class for invalid clarification requests."""


class UnknownRunError(ClarificationError):
    """The requested run is not present in the clarification store."""


class InvalidAnswerError(ClarificationError):
    """An answer does not match a question or is missing required text."""


class RunNotPausedError(ClarificationError):
    """A run cannot be resumed or skipped because it is not paused."""


@dataclass(frozen=True)
class ClarificationOption:
    """A selectable answer. ``likely`` identifies the preselected option."""

    value: str
    label: str
    likely: bool = False
    other: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "preselected": self.likely,
            "other": self.other,
        }


@dataclass(frozen=True)
class ClarificationQuestion:
    """A batched question shown while a run is paused."""

    id: str
    prompt: str
    options: tuple[ClarificationOption, ...]
    default_answer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "options": [option.as_dict() for option in self.options],
            "default_answer": self.default_answer,
            "default": self.default_answer,
            "allow_other": True,
            "other_value": OTHER_VALUE,
        }

    @property
    def other_option(self) -> ClarificationOption | None:
        return next((option for option in self.options if option.other), None)


@dataclass(frozen=True)
class Assumption:
    """An assumption visible in the eventual report."""

    key: str
    value: str
    source: str
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "key": self.key,
            "value": self.value,
            "source": self.source,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass(frozen=True)
class GapAssessment:
    """A confirmed gap and any values proposed for it.

    ``present`` is tri-state: ``True`` means the prompt already contains the
    value, ``False`` means it is missing, and ``None`` means the value is
    unknown.  ``inferred`` distinguishes a generated assumption from a direct
    user-provided value.
    """

    id: str
    label: str
    impact: str
    present: bool | None
    confidence: float = 1.0
    value: str | None = None
    question: str | None = None
    options: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    inferred: bool = False

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() in {"high", "critical"}


@dataclass(frozen=True)
class ClarificationPlan:
    """Questions to ask and assumptions to carry into the run."""

    questions: tuple[ClarificationQuestion, ...]
    assumptions: tuple[Assumption, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "questions": [question.as_dict() for question in self.questions],
            "assumptions": [assumption.as_dict() for assumption in self.assumptions],
        }


def _option_from_mapping(option: Mapping[str, Any]) -> ClarificationOption:
    value = str(option.get("value", "")).strip()
    label = str(option.get("label", value)).strip()
    if not value:
        raise ClarificationError("Clarification options need a non-empty value")
    return ClarificationOption(
        value=value,
        label=label or value,
        likely=bool(option.get("likely", option.get("preselected", False))),
        other=bool(option.get("other", False)) or value == OTHER_VALUE,
    )


def _make_question(gap: GapAssessment) -> ClarificationQuestion:
    options: list[ClarificationOption] = []
    for raw_option in gap.options:
        option = _option_from_mapping(raw_option)
        if option.other:
            continue
        if option.value != OTHER_VALUE:
            options.append(option)

    # A question must still be useful when a writer did not return candidates.
    # "No additional detail" is intentionally an ordinary answer, not a hidden
    # default, and can be edited later as an assumption.
    if not options:
        options.append(ClarificationOption("not_specified", "No additional detail", likely=True))
    elif not any(option.likely for option in options):
        options[0] = ClarificationOption(options[0].value, options[0].label, likely=True)

    other_exists = any(option.other for option in options)
    if not other_exists:
        options.append(ClarificationOption(OTHER_VALUE, OTHER_LABEL, other=True))

    default = next(option.value for option in options if option.likely)
    return ClarificationQuestion(
        id=gap.id,
        prompt=gap.question or f"What should be used for {gap.label}?",
        options=tuple(options),
        default_answer=default,
    )


def build_plan(
    assessments: Iterable[GapAssessment],
    *,
    allow_clarification: bool = True,
) -> ClarificationPlan:
    """Plan clarification deterministically from diagnosis assessments.

    Confident inferred values are always visible assumptions. Unknown low-impact
    values are recorded as skipped assumptions and do not block. Only unknown
    high-impact values create questions, and only while clarification is
    enabled.
    """

    assumptions: list[Assumption] = []
    questions: list[ClarificationQuestion] = []
    for gap in assessments:
        if gap.present is True:
            continue
        has_value = gap.value is not None and str(gap.value).strip() != ""
        confident_inference = has_value and gap.inferred and gap.confidence >= INFERENCE_CONFIDENCE
        if confident_inference:
            assumptions.append(
                Assumption(
                    key=gap.id,
                    value=str(gap.value),
                    source="inferred",
                    confidence=gap.confidence,
                )
            )
            continue
        if not gap.is_high_impact:
            assumptions.append(
                Assumption(
                    key=gap.id,
                    value=str(gap.value or "Not specified"),
                    source="skipped_low_impact",
                    confidence=gap.confidence,
                )
            )
            continue
        if allow_clarification:
            questions.append(_make_question(gap))
        else:
            assumptions.append(
                Assumption(
                    key=gap.id,
                    value=str(gap.value or "Not specified"),
                    source="skipped_clarification",
                    confidence=gap.confidence,
                )
            )

    return ClarificationPlan(tuple(questions), tuple(assumptions))


def _answer_parts(answer: Any) -> tuple[str, str | None]:
    if isinstance(answer, Mapping):
        value = answer.get("value", answer.get("other"))
        text = answer.get("text")
        if value is None and "other" in answer:
            value = OTHER_VALUE
            text = answer.get("other", text)
        return str(value or ""), None if text is None else str(text)
    return str(answer), None


def _selected_answer(question: ClarificationQuestion, answer: Any) -> str:
    value, text = _answer_parts(answer)
    matching = [option for option in question.options if option.value == value]
    if not matching and value == OTHER_VALUE:
        matching = [option for option in question.options if option.other]
    if not matching:
        raise InvalidAnswerError(f"Answer for {question.id!r} is not one of its options")
    if matching[0].other:
        if text is None and value != OTHER_VALUE:
            text = value
        if text is None or not text.strip():
            raise InvalidAnswerError(f"Answer for {question.id!r} requires other text")
        return text.strip()
    return matching[0].label


def apply_answers(
    plan_or_state: ClarificationPlan | Mapping[str, Any],
    answers: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply all answers to a plan and return a serializable completed state."""

    if isinstance(plan_or_state, ClarificationPlan):
        state: dict[str, Any] = {
            "status": "needs_input" if plan_or_state.questions else "completed",
            "questions": [question.as_dict() for question in plan_or_state.questions],
            "assumptions": [assumption.as_dict() for assumption in plan_or_state.assumptions],
        }
    else:
        state = json.loads(json.dumps(dict(plan_or_state)))
    questions = tuple(_question_from_dict(item) for item in state.get("questions", []))
    answered: dict[str, str] = {}
    for question in questions:
        if question.id not in answers:
            raise InvalidAnswerError(f"Missing answer for {question.id!r}")
        answered[question.id] = _selected_answer(question, answers[question.id])

    assumptions = {item["key"]: dict(item) for item in state.get("assumptions", [])}
    for question in questions:
        assumptions[question.id] = {
            "key": question.id,
            "value": answered[question.id],
            "source": "answer",
        }
    state["assumptions"] = list(assumptions.values())
    state["questions"] = []
    state["status"] = "completed"
    state["answers"] = answered
    return state


def skip_questions(plan_or_state: ClarificationPlan | Mapping[str, Any]) -> dict[str, Any]:
    """Complete a paused plan using each question's preselected likely answer."""

    if isinstance(plan_or_state, ClarificationPlan):
        state: dict[str, Any] = {
            "status": "needs_input" if plan_or_state.questions else "completed",
            "questions": [question.as_dict() for question in plan_or_state.questions],
            "assumptions": [assumption.as_dict() for assumption in plan_or_state.assumptions],
        }
    else:
        state = json.loads(json.dumps(dict(plan_or_state)))
    assumptions = {item["key"]: dict(item) for item in state.get("assumptions", [])}
    for raw_question in state.get("questions", []):
        question = _question_from_dict(raw_question)
        default = next(
            option.label for option in question.options if option.value == question.default_answer
        )
        assumptions[question.id] = {
            "key": question.id,
            "value": default,
            "source": "skipped_clarification",
        }
    state["assumptions"] = list(assumptions.values())
    state["questions"] = []
    state["status"] = "completed"
    return state


def _question_from_dict(raw: Mapping[str, Any]) -> ClarificationQuestion:
    options = tuple(_option_from_mapping(item) for item in raw.get("options", []))
    default = str(raw.get("default_answer", ""))
    if not any(option.value == default for option in options):
        raise ClarificationError("Clarification question has an invalid default answer")
    return ClarificationQuestion(
        id=str(raw["id"]),
        prompt=str(raw.get("prompt", "")),
        options=options,
        default_answer=default,
    )


class ClarificationRepository(Protocol):
    """Persistence seam used by :class:`ClarificationService`."""

    def load(self, run_id: str) -> Mapping[str, Any] | None:
        ...

    def save(self, run_id: str, state: Mapping[str, Any]) -> None:
        ...


class InMemoryClarificationRepository:
    """Small deterministic repository useful for tests and embedding."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def load(self, run_id: str) -> Mapping[str, Any] | None:
        state = self._runs.get(run_id)
        return json.loads(json.dumps(state)) if state is not None else None

    def save(self, run_id: str, state: Mapping[str, Any]) -> None:
        self._runs[run_id] = json.loads(json.dumps(dict(state)))


class SQLiteClarificationRepository:
    """Persist paused clarification state without coupling it to the run store."""

    def __init__(self, database: str | Path) -> None:
        self._database = str(database)
        with sqlite3.connect(self._database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clarification_runs ("
                "run_id TEXT PRIMARY KEY, state TEXT NOT NULL)"
            )

    def load(self, run_id: str) -> Mapping[str, Any] | None:
        with sqlite3.connect(self._database) as connection:
            row = connection.execute(
                "SELECT state FROM clarification_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def save(self, run_id: str, state: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(state), sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self._database) as connection:
            connection.execute(
                "INSERT INTO clarification_runs (run_id, state) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET state = excluded.state",
                (run_id, payload),
            )


Continuation = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class ClarificationService:
    """Pause and resume the same run around a clarification plan.

    ``continuation`` is called only after all questions have been answered or
    explicitly skipped. Its returned mapping is stored as ``result`` and is
    returned by ``resume``/``skip``. The engine facade can therefore inject its
    normal optimization continuation without this module knowing about gateways.
    """

    def __init__(
        self,
        repository: ClarificationRepository,
        *,
        continuation: Continuation | None = None,
    ) -> None:
        self.repository = repository
        self.continuation = continuation

    def start(
        self,
        run_id: str,
        prompt: str,
        plan: ClarificationPlan,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a paused state, or return the existing state for ``run_id``."""

        existing = self.repository.load(run_id)
        if existing is not None:
            return dict(existing)
        state: dict[str, Any] = {
            "run_id": run_id,
            "prompt": prompt,
            "status": "needs_input" if plan.questions else "completed",
            "questions": [question.as_dict() for question in plan.questions],
            "assumptions": [assumption.as_dict() for assumption in plan.assumptions],
            "metadata": dict(metadata or {}),
        }
        if not plan.questions and self.continuation is not None:
            state["result"] = dict(self.continuation(state))
        self.repository.save(run_id, state)
        return json.loads(json.dumps(state))

    def _load_paused(self, run_id: str) -> dict[str, Any]:
        state = self.repository.load(run_id)
        if state is None:
            raise UnknownRunError(f"Unknown run {run_id!r}")
        if state.get("status") != "needs_input":
            raise RunNotPausedError(f"Run {run_id!r} is not paused")
        return json.loads(json.dumps(dict(state)))

    def _finish(self, run_id: str, state: dict[str, Any]) -> dict[str, Any]:
        if self.continuation is not None:
            state["result"] = dict(self.continuation(state))
        self.repository.save(run_id, state)
        return json.loads(json.dumps(state))

    def resume(self, run_id: str, answers: Mapping[str, Any]) -> dict[str, Any]:
        state = self._load_paused(run_id)
        return self._finish(run_id, apply_answers(state, answers))

    def skip(self, run_id: str) -> dict[str, Any]:
        state = self._load_paused(run_id)
        return self._finish(run_id, skip_questions(state))


__all__ = [
    "INFERENCE_CONFIDENCE",
    "OTHER_LABEL",
    "OTHER_VALUE",
    "Assumption",
    "ClarificationError",
    "ClarificationOption",
    "ClarificationPlan",
    "ClarificationQuestion",
    "ClarificationService",
    "GapAssessment",
    "InMemoryClarificationRepository",
    "InvalidAnswerError",
    "RunNotPausedError",
    "SQLiteClarificationRepository",
    "UnknownRunError",
    "apply_answers",
    "build_plan",
    "skip_questions",
]
