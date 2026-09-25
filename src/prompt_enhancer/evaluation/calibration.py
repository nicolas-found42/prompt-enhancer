"""Offline Jev calibration, verdict artifacts, and runtime decision policy.

The calibration path deliberately sits next to the evaluation harness instead of
inside :class:`~prompt_enhancer.gateway.Gateway`.  A Gateway returns raw answers;
callers decide how a malformed answer is handled and how a decision may affect
behavior.  This module keeps the raw answer, normalizes the three Jev
primitives, fits and evaluates a source-group-disjoint artifact, and exposes a
small policy interface for runtime consumers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from ..jev import (
    ChoiceDecision,
    JevResponseError,
    NoulDecision,
    ScoreDecision,
    parse_decision,
)

# canonical JSON is implemented locally so artifact identities never depend on
# mutable dataset helpers.

CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_REPORT_KIND = "calibration-report"
CALIBRATION_ARTIFACT_KIND = "calibration-artifact"
DEFAULT_POLICY_VERSION = "issue-50-v1"
DEFAULT_BOOTSTRAP_SEED = 1729
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
VERDICT_CATALOG = (
    "too-few-examples",
    "gate",
    "gate-above-confidence",
    "ranker",
    "unusable",
)


class CalibrationError(ValueError):
    """Raised when calibration input cannot be represented safely."""


class Primitive(StrEnum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class Verdict(StrEnum):
    TOO_FEW_EXAMPLES = "too-few-examples"
    GATE = "gate"
    GATE_ABOVE_CONFIDENCE = "gate-above-confidence"
    RANKER = "ranker"
    UNUSABLE = "unusable"


class Disposition(StrEnum):
    LEGACY = "legacy"
    GATE = "gate"
    GATE_ABOVE_CONFIDENCE = "gate-above-confidence"
    RANKER = "ranker"
    ABSTAIN = "abstain"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _probability(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise CalibrationError(f"{field_name} must be a probability")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{field_name} must be a probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise CalibrationError(f"{field_name} must be between 0 and 1")
    return result


def _jsonable(value: object) -> object:
    """Convert local value objects into deterministic JSON-compatible values."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{name} must be an object")
    return value


def _first(mapping: Mapping[str, Any], *names: str, default: object = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _class_key(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _class_value(key: str, original: Sequence[object] | None = None) -> object:
    if original is not None:
        for value in original:
            if _class_key(value) == key:
                return value
    return key


def _label_parts(value: object) -> tuple[bool | None, str | None]:
    """Return a binary label and an optional semantic class label."""

    if value is None:
        return None, None
    if isinstance(value, bool):
        return value, ("yes" if value else "no")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            result = bool(value)
            return result, ("yes" if result else "no")
        return None, str(value)
    if isinstance(value, Mapping):
        candidate = _first(
            value,
            "class",
            "expected_class",
            "option",
            "selected",
            "value",
            "label",
            "outcome",
            "expected",
            "correct",
        )
        if candidate is not None:
            return _label_parts(candidate)
        return None, None
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {
            "yes",
            "true",
            "present",
            "positive",
            "missing",
            "defect",
            "pass",
            "correct",
        }:
            return True, value.strip()
        if text in {"no", "false", "absent", "negative", "clean", "fail", "incorrect"}:
            return False, value.strip()
        return None, value.strip()
    return None, None


def _label_binary(value: object) -> bool | None:
    return _label_parts(value)[0]


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Declarative mapping from one Jev primitive to a labeled event."""

    primitive: str = "noul"
    classes: tuple[object, ...] = ()
    expected_class: object | None = None
    boundary: object | None = None
    positive_classes: tuple[object, ...] = ()
    polarity: str = "positive"
    criteria: object | None = None

    def __post_init__(self) -> None:
        primitive = str(self.primitive).lower()
        if primitive not in {item.value for item in Primitive}:
            raise CalibrationError(
                f"unsupported calibration primitive {self.primitive!r}"
            )
        object.__setattr__(self, "primitive", primitive)
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "positive_classes", tuple(self.positive_classes))
        polarity = str(self.polarity).lower()
        if polarity not in {"positive", "negative", "yes", "no", "true", "false"}:
            raise CalibrationError(f"unsupported event polarity {self.polarity!r}")
        object.__setattr__(self, "polarity", polarity)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> EventSpec:
        value = value or {}
        primitive = _first(value, "primitive", "type", "response_type", default="noul")
        criteria = _first(value, "criteria", "options", "levels", default=None)
        classes: tuple[object, ...]
        if isinstance(criteria, Mapping):
            classes = tuple(criteria.keys())
        elif isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
            classes = tuple(criteria)
        else:
            classes = ()
        expected = _first(
            value,
            "expected_class",
            "expected_option",
            "target_class",
            "positive_class",
            default=None,
        )
        boundary = _first(
            value, "boundary", "semantic_boundary", "cutoff", default=None
        )
        positives = _first(value, "positive_classes", "at_or_above", default=())
        if not isinstance(positives, Sequence) or isinstance(positives, (str, bytes)):
            positives = (positives,)
        return cls(
            primitive=str(primitive),
            classes=classes,
            expected_class=expected,
            boundary=boundary,
            positive_classes=tuple(positives),
            polarity=str(
                _first(value, "polarity", "event_polarity", default="positive")
            ),
            criteria=criteria,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "classes": [_jsonable(item) for item in self.classes],
            "expected_class": _jsonable(self.expected_class),
            "boundary": _jsonable(self.boundary),
            "positive_classes": [_jsonable(item) for item in self.positive_classes],
            "polarity": self.polarity,
            "criteria": _jsonable(self.criteria),
        }

    @classmethod
    def from_identity(cls, identity: QuestionIdentity) -> EventSpec:
        mapping = dict(identity.event_mapping)
        mapping.setdefault("primitive", identity.primitive)
        mapping.setdefault("criteria", identity.criteria)
        return cls.from_mapping(mapping)


@dataclass(frozen=True, slots=True)
class QuestionIdentity:
    """Complete identity for a reusable question and its calibration artifact."""

    question_id: str
    question: Any
    primitive: str = "noul"
    criteria: Any = None
    question_schema: Any = None
    event_mapping: Mapping[str, Any] = field(default_factory=dict)
    family: str = ""
    schema_version: int = 1
    rubric_version: str | None = None
    answering_snapshot: str | None = None
    policy_version: str = DEFAULT_POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.question_id, str) or not self.question_id.strip():
            raise CalibrationError("question_id must be a non-empty string")
        if not isinstance(self.primitive, str) or not self.primitive.strip():
            raise CalibrationError("primitive must be a non-empty string")
        if not isinstance(self.event_mapping, Mapping):
            raise CalibrationError("event_mapping must be an object")
        object.__setattr__(self, "event_mapping", dict(self.event_mapping))
        object.__setattr__(self, "primitive", self.primitive.lower())
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "policy_version", str(self.policy_version))

    @property
    def question_digest(self) -> str:
        return _digest(
            {
                "question_id": self.question_id,
                "question": _jsonable(self.question),
                "primitive": self.primitive,
                "criteria": _jsonable(self.criteria),
                "question_schema": _jsonable(self.question_schema),
                "event_mapping": _jsonable(self.event_mapping),
                "family": self.family,
                "schema_version": self.schema_version,
                "rubric_version": self.rubric_version,
            }
        )

    @property
    def identity_digest(self) -> str:
        return _digest(
            {
                "question_digest": self.question_digest,
                "answering_snapshot": self.answering_snapshot,
                "policy_version": self.policy_version,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": _jsonable(self.question),
            "primitive": self.primitive,
            "criteria": _jsonable(self.criteria),
            "question_schema": _jsonable(self.question_schema),
            "event_mapping": _jsonable(self.event_mapping),
            "family": self.family,
            "schema_version": self.schema_version,
            "rubric_version": self.rubric_version,
            "answering_snapshot": self.answering_snapshot,
            "policy_version": self.policy_version,
            "question_digest": self.question_digest,
            "identity_digest": self.identity_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QuestionIdentity:
        data = _as_mapping(value, name="question identity")
        return cls(
            question_id=str(_first(data, "question_id", "id", default="")),
            question=_first(
                data, "question", "query", "text", "instructions", default=""
            ),
            primitive=str(
                _first(data, "primitive", "type", "response_type", default="noul")
            ),
            criteria=_first(data, "criteria", "options", "levels", default=None),
            question_schema=_first(data, "question_schema", "schema", default=None),
            event_mapping=_as_mapping(
                _first(data, "event_mapping", "event", default={}),
                name="event_mapping",
            ),
            family=str(_first(data, "family", "question_family", default="")),
            schema_version=int(_first(data, "schema_version", default=1)),
            rubric_version=_first(data, "rubric_version", default=None),
            answering_snapshot=_first(
                data, "answering_snapshot", "snapshot", "answered_by", default=None
            ),
            policy_version=str(
                _first(data, "policy_version", default=DEFAULT_POLICY_VERSION)
            ),
        )

    def matches(
        self,
        other: QuestionIdentity,
        *,
        snapshot: str | None = None,
        require_snapshot: bool = True,
    ) -> bool:
        if self.question_id != other.question_id:
            return False
        if self.primitive != other.primitive:
            return False
        if _canonical(_jsonable(self.question)) != _canonical(
            _jsonable(other.question)
        ):
            return False
        if _canonical(_jsonable(self.criteria)) != _canonical(
            _jsonable(other.criteria)
        ):
            return False
        if _canonical(_jsonable(self.question_schema)) != _canonical(
            _jsonable(other.question_schema)
        ):
            return False
        if _canonical(_jsonable(self.event_mapping)) != _canonical(
            _jsonable(other.event_mapping)
        ):
            return False
        if self.family != other.family:
            return False
        if self.schema_version != other.schema_version:
            return False
        if self.rubric_version != other.rubric_version:
            return False
        if self.policy_version != other.policy_version:
            return False
        if require_snapshot:
            expected = snapshot if snapshot is not None else self.answering_snapshot
            return self.answering_snapshot == expected
        return True


def _state_is_empty(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return True
    if isinstance(value, Mapping):
        return all(_state_is_empty(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_state_is_empty(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """One independently labeled raw answer, repeat, or state-blind control."""

    event_id: str
    source_group: str
    example_id: str
    identity: QuestionIdentity
    label: object | None
    raw_answer: Any
    provenance: str
    answering_snapshot: str | None = None
    repeat_index: int = 0
    control: bool = False
    request_id: str | None = None
    answer_id: str | None = None
    partition: str | None = None
    state: Any = None
    label_present: bool = True

    def __post_init__(self) -> None:
        for name in ("event_id", "source_group", "example_id", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CalibrationError(f"calibration {name} must be a non-empty string")
        if not isinstance(self.identity, QuestionIdentity):
            raise CalibrationError("identity must be a QuestionIdentity")
        if not isinstance(self.repeat_index, int) or isinstance(
            self.repeat_index, bool
        ):
            raise CalibrationError("repeat_index must be an integer")
        if self.repeat_index < 0:
            raise CalibrationError("repeat_index must be non-negative")
        if not isinstance(self.control, bool):
            raise CalibrationError("control must be a boolean")
        if not isinstance(self.label_present, bool):
            raise CalibrationError("label_present must be a boolean")
        if self.control and not _state_is_empty(self.state):
            raise CalibrationError("state-blind control events must have empty state")
        if self.answer_id is None:
            identity_payload = {
                "question_digest": self.identity.question_digest,
                "source_group": self.source_group,
                "example_id": self.example_id,
                "control": self.control,
                "request_id": self.request_id,
                "state": _jsonable(self.state),
                "raw_answer": _jsonable(self.raw_answer),
            }
            if self.raw_answer is None:
                identity_payload["event_id"] = self.event_id
                identity_payload["repeat_index"] = self.repeat_index
            object.__setattr__(self, "answer_id", _digest(identity_payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_group": self.source_group,
            "example_id": self.example_id,
            "identity": self.identity.to_dict(),
            "label": _jsonable(self.label),
            "raw_answer": _jsonable(self.raw_answer),
            "provenance": self.provenance,
            "answering_snapshot": self.answering_snapshot,
            "repeat_index": self.repeat_index,
            "control": self.control,
            "request_id": self.request_id,
            "answer_id": self.answer_id,
            "partition": self.partition,
            "state": _jsonable(self.state),
            "label_present": self.label_present,
        }


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """A parsed event with explicit unavailable reasons."""

    observation: CalibrationObservation
    usable: bool
    probability: float | None
    distribution: Mapping[str, float]
    expected_class: str | None = None
    predicted_class: str | None = None
    unavailable_reason: str | None = None
    distribution_unavailable_reason: str | None = None
    derived_margin: float | None = None
    provider_confidence: float | None = None

    @property
    def label(self) -> bool | None:
        raw_label = self.observation.label
        binary = _label_binary(raw_label)
        if binary is not None:
            return binary
        identity = self.observation.identity
        if identity.primitive == "choice" and identity.event_mapping.get(
            "selected_correctness"
        ):
            return self.predicted_class == self.expected_class
        if identity.primitive == "score":
            _, expected_class = _label_parts(raw_label)
            boundary = identity.event_mapping.get(
                "boundary", identity.event_mapping.get("semantic_boundary")
            )
            if expected_class is not None and boundary is not None:
                try:
                    return float(expected_class) >= float(boundary)
                except (TypeError, ValueError):
                    return None
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.observation.event_id,
            "source_group": self.observation.source_group,
            "example_id": self.observation.example_id,
            "answer_id": self.observation.answer_id,
            "repeat_index": self.observation.repeat_index,
            "control": self.observation.control,
            "label": _jsonable(self.observation.label),
            "label_present": self.observation.label_present,
            "provenance": self.observation.provenance,
            "answering_snapshot": self.observation.answering_snapshot,
            "raw_answer": _jsonable(self.observation.raw_answer),
            "usable": self.usable,
            "probability": self.probability,
            "distribution": dict(self.distribution),
            "expected_class": self.expected_class,
            "predicted_class": self.predicted_class,
            "unavailable_reason": self.unavailable_reason,
            "distribution_unavailable_reason": self.distribution_unavailable_reason,
            "derived_margin": self.derived_margin,
            "provider_confidence": self.provider_confidence,
        }


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """Deterministic threshold selection on the calibration partition."""

    name: str = "f0.5-grid-v1"
    grid_start: float = 0.01
    grid_end: float = 0.99
    grid_step: float = 0.01
    tie_break: str = "higher"

    def candidates(self) -> tuple[float, ...]:
        if self.grid_step <= 0:
            raise CalibrationError("threshold grid step must be positive")
        values: list[float] = []
        current = self.grid_start
        while current <= self.grid_end + 1e-12:
            values.append(round(current, 10))
            current += self.grid_step
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerdictPolicy:
    """Provisional, versioned defaults from issue #50.

    These values are deliberately explicit and are not claims about Jev's real
    world accuracy.  Callers can provide a lower policy for a known-answer
    fixture without changing the persisted policy version.
    """

    minimum_evaluation_groups: int = 30
    minimum_positive_examples: int = 5
    minimum_negative_examples: int = 5
    precision_floor: float = 0.90
    recall_floor: float = 0.50
    coverage_floor: float = 0.90
    minimum_control_groups: int = 1
    minimum_repeat_examples: int = 1
    require_control: bool = True
    require_repeats: bool = True
    require_brier_better_than_control: bool = True
    ranker_auc_lower_bound: float = 0.5
    policy_version: str = DEFAULT_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationBudget:
    """Bounded experiment limits and accounting for optional live capture."""

    max_source_examples: int = 100
    max_repeats: int = 3
    max_question_evaluations: int = 5000
    budget_usd: float | None = None
    spent_usd: float = 0.0
    retries: int = 0
    stopped_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("max_source_examples", "max_repeats", "max_question_evaluations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CalibrationError(f"{name} must be a positive integer")
        if self.budget_usd is not None:
            if not math.isfinite(self.budget_usd) or self.budget_usd <= 0:
                raise CalibrationError("budget_usd must be finite and positive")
        if self.spent_usd < 0:
            raise CalibrationError("spent_usd must be non-negative")

    def can_spend(self, amount: float) -> bool:
        if amount < 0 or not math.isfinite(amount):
            return False
        return self.budget_usd is None or self.spent_usd + amount <= self.budget_usd

    def record(self, amount: float, *, retries: int = 0) -> CalibrationBudget:
        if not self.can_spend(amount):
            return replace(
                self,
                stopped_reason="budget-exceeded",
            )
        return replace(
            self,
            spent_usd=self.spent_usd + amount,
            retries=self.retries + max(0, retries),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    identity: QuestionIdentity
    metrics: Mapping[str, Any]
    control_metrics: Mapping[str, Any]
    threshold: float | None
    predicate: Mapping[str, Any]
    fit: Mapping[str, Any]
    partitions: Mapping[str, str]
    verdict: str
    verdict_components: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    evidence: Mapping[str, Any]
    input_digest: str
    status: str = "complete"
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "kind": "calibration-question",
            "name": self.name,
            "identity": self.identity.to_dict(),
            "metrics": _jsonable(self.metrics),
            "control_metrics": _jsonable(self.control_metrics),
            "threshold": self.threshold,
            "predicate": _jsonable(self.predicate),
            "fit": _jsonable(self.fit),
            "partitions": dict(sorted(self.partitions.items())),
            "verdict": self.verdict,
            "verdict_components": _jsonable(self.verdict_components),
            "uncertainty": _jsonable(self.uncertainty),
            "evidence": _jsonable(self.evidence),
            "input_digest": self.input_digest,
            "status": self.status,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def artifact(self) -> CalibrationArtifact:
        return CalibrationArtifact(
            name=self.name or self.identity.question_id,
            input_digest=self.input_digest,
            questions={self.identity.question_id: self.to_dict()},
            status=self.status,
        )


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """Persisted, replayable per-question calibration verdicts."""

    name: str
    input_digest: str
    questions: Mapping[str, Mapping[str, Any]]
    schema_version: int = CALIBRATION_SCHEMA_VERSION
    kind: str = CALIBRATION_ARTIFACT_KIND
    status: str = "complete"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        questions = {
            str(key): _jsonable(value) for key, value in sorted(self.questions.items())
        }
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "name": self.name,
            "input_digest": self.input_digest,
            "status": self.status,
            "questions": questions,
            "verdicts": {
                key: value.get("verdict")
                for key, value in questions.items()
                if isinstance(value, Mapping)
            },
            "verdict_catalog": list(VERDICT_CATALOG),
            "metadata": _jsonable(dict(self.metadata)),
        }
        if len(questions) == 1:
            question_id, question = next(iter(questions.items()))
            if isinstance(question, Mapping):
                result["question_id"] = question_id
                result["verdict"] = question.get("verdict")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationArtifact:
        data = _as_mapping(value, name="calibration artifact")
        raw_questions = data.get("questions", data.get("results", {}))
        questions: dict[str, Mapping[str, Any]] = {}
        if isinstance(raw_questions, Mapping):
            for key, item in raw_questions.items():
                if isinstance(item, Mapping):
                    questions[str(key)] = dict(item)
        elif isinstance(raw_questions, Sequence) and not isinstance(
            raw_questions, (str, bytes)
        ):
            for index, item in enumerate(raw_questions):
                if not isinstance(item, Mapping):
                    raise CalibrationError(
                        "calibration question results must be objects"
                    )
                identity = item.get("identity", {})
                key = (
                    str(identity.get("question_id", f"question-{index}"))
                    if isinstance(identity, Mapping)
                    else f"question-{index}"
                )
                questions[key] = dict(item)
        else:
            raise CalibrationError(
                "calibration artifact questions must be an object or list"
            )
        return cls(
            name=str(data.get("name", "calibration")),
            input_digest=str(data.get("input_digest", _digest(data))),
            questions=questions,
            schema_version=int(data.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
            kind=str(data.get("kind", CALIBRATION_ARTIFACT_KIND)),
            status=str(data.get("status", "complete")),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationArtifact:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CalibrationError(
                f"could not load calibration artifact {source}: {exc}"
            ) from exc
        return cls.from_dict(_as_mapping(raw, name="calibration artifact"))


@dataclass(frozen=True, slots=True)
class CalibrationManifest:
    name: str
    events: tuple[CalibrationObservation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CALIBRATION_SCHEMA_VERSION
    input_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "name": self.name,
            "metadata": _jsonable(dict(self.metadata)),
            "events": [event.to_dict() for event in self.events],
        }
        return {**payload, "input_digest": self.input_digest or _digest(payload)}


def _identity_from_event(
    value: Mapping[str, Any], *, position: int
) -> QuestionIdentity:
    nested = value.get("question")
    if isinstance(nested, Mapping):
        question_data = dict(nested)
    else:
        question_data = {
            "question_id": _first(
                value,
                "question_id",
                "logical_question_id",
                default=f"question-{position}",
            ),
            "question": nested
            if nested is not None
            else _first(value, "query", "text", "instructions", default=""),
            "primitive": _first(
                value, "primitive", "type", "response_type", default="noul"
            ),
            "criteria": _first(value, "criteria", "options", "levels", default=None),
            "question_schema": _first(value, "question_schema", "schema", default=None),
            "event_mapping": _first(
                value, "event_mapping", "event", "event_spec", default={}
            ),
            "family": _first(value, "family", "question_family", default=""),
            "schema_version": _first(value, "schema_version", default=1),
            "rubric_version": _first(value, "rubric_version", default=None),
            "answering_snapshot": _first(
                value,
                "answering_snapshot",
                "snapshot",
                "answered_by",
                "model_snapshot",
                default=None,
            ),
        }
    if "question_id" not in question_data:
        question_data["question_id"] = _first(
            value, "question_id", "logical_question_id", default=f"question-{position}"
        )
    if "question" not in question_data:
        question_data["question"] = _first(
            value, "query", "text", "instructions", default=""
        )
    if "primitive" not in question_data:
        question_data["primitive"] = _first(
            value, "primitive", "type", "response_type", default="noul"
        )
    if "criteria" not in question_data:
        question_data["criteria"] = _first(
            value, "criteria", "options", "levels", default=None
        )
    if "question_schema" not in question_data:
        question_data["question_schema"] = _first(
            value, "question_schema", "schema", default=None
        )
    if "event_mapping" not in question_data:
        question_data["event_mapping"] = _first(
            value, "event_mapping", "event", "event_spec", default={}
        )
    for field_name, aliases, default in (
        ("family", ("family", "question_family"), ""),
        ("schema_version", ("schema_version",), 1),
        ("rubric_version", ("rubric_version",), None),
        (
            "answering_snapshot",
            ("answering_snapshot", "snapshot", "answered_by", "model_snapshot"),
            None,
        ),
    ):
        if field_name not in question_data:
            question_data[field_name] = _first(value, *aliases, default=default)
    if not isinstance(question_data.get("event_mapping"), Mapping):
        question_data["event_mapping"] = {}
    return QuestionIdentity.from_dict(question_data)


def _observation_from_event(
    value: Mapping[str, Any], *, position: int, default_snapshot: str | None
) -> CalibrationObservation:
    event_id = str(
        _first(value, "event_id", "record_id", "id", default=f"event-{position}")
    )
    identity = _identity_from_event(value, position=position)
    label_present = bool(
        _first(value, "labels_present", "label_present", default=None) is not False
        and any(
            key in value
            for key in (
                "label",
                "expected",
                "expected_outcome",
                "expected_class",
                "expected_option",
                "outcome",
                "correct",
            )
        )
    )
    label = _first(
        value,
        "label",
        "expected",
        "expected_outcome",
        "expected_class",
        "expected_option",
        "outcome",
        "correct",
        default=None,
    )
    raw_answer = _first(
        value, "answer", "raw_answer", "response", "decision", default=None
    )
    provenance = str(
        _first(
            value,
            "label_provenance",
            "provenance",
            "source_provenance",
            default="unspecified",
        )
    )
    repeat_index = _first(value, "repeat_index", "repeat", "run_index", default=0)
    if isinstance(repeat_index, Mapping):
        repeat_index = _first(repeat_index, "index", "repeat_index", default=0)
    if not isinstance(repeat_index, int) or isinstance(repeat_index, bool):
        raise CalibrationError(f"event {position} repeat_index must be an integer")
    control_value = _first(
        value,
        "control",
        "state_blind",
        "state_blind_control",
        default=str(value.get("arm", "")).casefold() in {"control", "state-blind"},
    )
    if not isinstance(control_value, bool):
        raise CalibrationError(f"event {position} control must be a boolean")
    return CalibrationObservation(
        event_id=event_id,
        source_group=str(
            _first(value, "source_group", "group", "conversation_id", default="")
        ),
        example_id=str(
            _first(value, "example_id", "case_id", "source_id", default=event_id)
        ),
        identity=identity,
        label=label,
        raw_answer=raw_answer,
        provenance=provenance,
        answering_snapshot=_first(
            value,
            "answering_snapshot",
            "snapshot",
            "answered_by",
            "model_snapshot",
            default=default_snapshot or identity.answering_snapshot,
        ),
        repeat_index=repeat_index,
        control=bool(control_value),
        request_id=_first(value, "request_id", "recording_id", default=None),
        answer_id=_first(value, "answer_id", "response_id", default=None),
        partition=_first(value, "partition", "split", default=None),
        state=_first(value, "state", "input_state", "context", default=None),
        label_present=label_present,
    )


def _manifest_events(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = _first(
        value, "events", "records", "observations", "answers", "cases", default=None
    )
    if raw is None and isinstance(value.get("questions"), Sequence):
        flattened: list[Mapping[str, Any]] = []
        for question in cast(Sequence[Any], value["questions"]):
            if not isinstance(question, Mapping):
                raise CalibrationError("calibration questions must be objects")
            nested = _first(question, "events", "records", "answers", default=[])
            if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes)):
                raise CalibrationError("calibration question events must be an array")
            for event in nested:
                if not isinstance(event, Mapping):
                    raise CalibrationError("calibration events must be objects")
                merged = dict(event)
                for key in (
                    "question_id",
                    "question",
                    "primitive",
                    "criteria",
                    "event_mapping",
                    "family",
                    "rubric_version",
                    "answering_snapshot",
                    "question_schema",
                    "state",
                ):
                    if key not in merged and key in question:
                        merged[key] = question[key]
                flattened.append(merged)
        return flattened
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise CalibrationError("calibration input requires an events array")
    result: list[Mapping[str, Any]] = []
    for event in raw:
        if not isinstance(event, Mapping):
            raise CalibrationError("calibration events must be objects")
        result.append(event)
    return result


def manifest_from_dict(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, name: str | None = None
) -> CalibrationManifest:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        data: Mapping[str, Any] = {"events": list(value)}
    else:
        data = _as_mapping(value, name="calibration input")
    events = _manifest_events(data)
    if not events:
        raise CalibrationError("calibration input requires at least one event")
    metadata = dict(_as_mapping(data.get("metadata", {}), name="calibration metadata"))
    default_snapshot = _first(
        data,
        "answering_snapshot",
        "snapshot",
        "model_snapshot",
        default=metadata.get("answering_snapshot"),
    )
    observations = tuple(
        _observation_from_event(
            event, position=index, default_snapshot=cast(str | None, default_snapshot)
        )
        for index, event in enumerate(events, start=1)
    )
    seen_events: set[str] = set()
    seen_answers: set[tuple[str, str, str, str]] = set()
    seen_repeats: set[tuple[str, str, str, bool, int]] = set()
    seen_requests: set[tuple[str, str, str, str]] = set()
    for observation in observations:
        if observation.event_id in seen_events:
            raise CalibrationError(
                f"duplicate calibration event id {observation.event_id!r}"
            )
        seen_events.add(observation.event_id)
        repeat_key = (
            observation.identity.question_id,
            observation.source_group,
            observation.example_id,
            observation.control,
            observation.repeat_index,
        )
        if repeat_key in seen_repeats:
            raise CalibrationError(
                "duplicate repeat index/answer identity for a question/example/arm"
            )
        seen_repeats.add(repeat_key)
        key = (
            observation.identity.question_id,
            observation.source_group,
            observation.example_id,
            observation.answer_id or observation.request_id or "",
        )
        if key in seen_answers:
            raise CalibrationError(
                "duplicate answer identity: a cached response cannot count as an independent repeat"
            )
        seen_answers.add(key)
        if observation.request_id:
            request_key = (
                observation.identity.question_id,
                observation.source_group,
                observation.example_id,
                observation.request_id,
            )
            if request_key in seen_requests:
                raise CalibrationError(
                    "duplicate request identity: repeated requests must be distinct"
                )
            seen_requests.add(request_key)
    schema_version = data.get("schema_version", CALIBRATION_SCHEMA_VERSION)
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
    ):
        raise CalibrationError("calibration schema_version must be a positive integer")
    manifest = CalibrationManifest(
        name=str(name or data.get("name", "calibration")),
        events=observations,
        metadata=metadata,
        schema_version=schema_version,
    )
    return replace(manifest, input_digest=manifest.to_dict()["input_digest"])


def load_calibration_manifest(
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> CalibrationManifest:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CalibrationError(
                f"could not load calibration input {path}: {exc}"
            ) from exc
        return manifest_from_dict(raw, name=path.stem)
    return manifest_from_dict(source)


def _explicit_partitions(
    observations: Sequence[CalibrationObservation],
) -> dict[str, str] | None:
    explicit = {
        observation.source_group: observation.partition
        for observation in observations
        if observation.partition is not None
    }
    if not explicit:
        return None
    if any(
        value not in {"fit", "calibration", "evaluation"} for value in explicit.values()
    ):
        raise CalibrationError("partition must be fit, calibration, or evaluation")
    for observation in observations:
        value = explicit.get(observation.source_group)
        if value is not None and observation.partition not in {None, value}:
            raise CalibrationError("a source group cannot span calibration partitions")
    return explicit


def partition_groups(
    groups: Iterable[str], *, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> dict[str, str]:
    """Assign complete source groups to stable, disjoint fit/calibration/eval sets."""

    unique = sorted({str(group) for group in groups})
    if not unique:
        return {}
    assignments: dict[str, str] = {}
    for group in unique:
        bucket = (
            int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
        )
        assignments[group] = (
            "fit" if bucket < 60 else "calibration" if bucket < 80 else "evaluation"
        )
    if len(unique) >= 3:
        for partition in ("fit", "calibration", "evaluation"):
            if any(value == partition for value in assignments.values()):
                continue
            donor = next(
                group
                for group in sorted(
                    unique,
                    key=lambda item: hashlib.sha256(
                        f"{seed}:{item}".encode()
                    ).hexdigest(),
                )
                if assignments[group] != partition
            )
            assignments[donor] = partition
    return assignments


def _partition_observations(
    observations: Sequence[CalibrationObservation], *, seed: int
) -> dict[str, str]:
    explicit = _explicit_partitions(observations)
    groups = partition_groups(
        (observation.source_group for observation in observations), seed=seed
    )
    if explicit:
        groups.update(explicit)
    return groups


def _raw_confidence(raw: Any) -> float | None:
    if not isinstance(raw, Mapping):
        return None
    body = raw
    for name in ("noul", "choice", "score", "answer", "result", "data"):
        nested = body.get(name)
        if isinstance(nested, Mapping):
            body = nested
            break
    value = _first(body, "confidence", "certainty", default=None)
    if value is None:
        return None
    try:
        return _probability(value, field_name="provider confidence")
    except CalibrationError:
        return None


def _declared_classes(spec: EventSpec, decision: Any) -> tuple[object, ...]:
    if spec.classes:
        return spec.classes
    if isinstance(decision, ChoiceDecision):
        return tuple(option.value for option in decision.options)
    if isinstance(decision, ScoreDecision):
        return tuple(option.level for option in decision.levels)
    return ("no", "yes")


def _distribution_map(
    spec: EventSpec, decision: Any
) -> tuple[dict[str, float], tuple[object, ...], str | None]:
    classes = _declared_classes(spec, decision)
    if isinstance(decision, NoulDecision):
        values = {"no": 1.0 - decision.probability, "yes": decision.probability}
        missing = [item for item in classes if _class_key(item) not in values]
        if missing:
            return values, classes, "missing_class_coverage"
        return values, classes, None
    if isinstance(decision, ChoiceDecision):
        values = {str(option.value): option.probability for option in decision.options}
        missing = [item for item in classes if _class_key(item) not in values]
        if missing:
            return values, classes, "missing_class_coverage"
        return values, classes, None
    if isinstance(decision, ScoreDecision):
        values = {str(option.level): option.probability for option in decision.levels}
        missing = [item for item in classes if _class_key(item) not in values]
        if missing:
            return values, classes, "missing_class_coverage"
        return values, classes, None
    return {}, classes, "unsupported_decision"


def _noul_probability(decision: NoulDecision, spec: EventSpec) -> float:
    if spec.polarity in {"negative", "no", "false"}:
        return 1.0 - decision.probability
    if spec.expected_class is not None and _class_key(
        spec.expected_class
    ).casefold() in {
        "no",
        "false",
        "negative",
    }:
        return 1.0 - decision.probability
    return decision.probability


def _score_boundary(spec: EventSpec) -> object | None:
    if spec.boundary is not None:
        return spec.boundary
    if spec.positive_classes:
        return spec.positive_classes[0]
    return spec.expected_class


def _score_probability(decision: ScoreDecision, spec: EventSpec) -> float | None:
    boundary = _score_boundary(spec)
    if boundary is None:
        return None
    classes = _declared_classes(spec, decision)
    try:
        numeric_boundary = float(cast(Any, boundary))
    except (TypeError, ValueError):
        return None
    total = 0.0
    for value in classes:
        try:
            numeric = float(cast(Any, value))
        except (TypeError, ValueError):
            continue
        if numeric >= numeric_boundary:
            total += decision.probabilities.get(str(value), 0.0)
    return total


def normalize_event(
    observation: CalibrationObservation, spec: EventSpec | None = None
) -> NormalizedEvent:
    """Parse a raw answer into a generic labeled event without losing reasons."""

    identity = observation.identity
    event_spec = spec or EventSpec.from_identity(identity)
    if observation.raw_answer is None:
        return NormalizedEvent(
            observation=observation,
            usable=False,
            probability=None,
            distribution={},
            unavailable_reason="missing_answer",
        )
    try:
        decision = parse_decision(observation.raw_answer)
    except (JevResponseError, TypeError, ValueError):
        return NormalizedEvent(
            observation=observation,
            usable=False,
            probability=None,
            distribution={},
            unavailable_reason="malformed_answer",
        )
    if event_spec.primitive != getattr(
        decision, "__class__", type(decision)
    ).__name__.lower().replace("decision", ""):
        # The parser's concrete classes are the source of truth; a mismatched
        # declaration is an unavailable event rather than a guessed mapping.
        expected_type = {
            "noul": NoulDecision,
            "choice": ChoiceDecision,
            "score": ScoreDecision,
        }.get(event_spec.primitive)
        if expected_type is None or not isinstance(decision, expected_type):
            return NormalizedEvent(
                observation=observation,
                usable=False,
                probability=None,
                distribution={},
                unavailable_reason="primitive_mismatch",
            )
    distribution, classes, distribution_reason = _distribution_map(event_spec, decision)
    provider_confidence = _raw_confidence(observation.raw_answer)
    label, semantic_label = _label_parts(observation.label)
    expected_class = semantic_label
    derived_margin: float | None = None
    probability: float | None = None
    predicted_class: str | None = None

    if isinstance(decision, NoulDecision):
        probability = _noul_probability(decision, event_spec)
        derived_margin = abs(2.0 * decision.probability - 1.0)
        predicted_class = "yes" if decision.probability >= 0.5 else "no"
        if expected_class is None:
            expected_class = (
                "yes" if label is True else "no" if label is False else None
            )
    elif isinstance(decision, ChoiceDecision):
        predicted_class = decision.selected
        if event_spec.expected_class is not None:
            expected_class = _class_key(event_spec.expected_class)
        if expected_class is None:
            return NormalizedEvent(
                observation=observation,
                usable=False,
                probability=None,
                distribution=distribution,
                expected_class=None,
                predicted_class=predicted_class,
                unavailable_reason="missing_expected_class",
                distribution_unavailable_reason=distribution_reason,
                derived_margin=derived_margin,
                provider_confidence=provider_confidence,
            )
        probability = distribution.get(expected_class)
        if probability is None:
            return NormalizedEvent(
                observation=observation,
                usable=False,
                probability=None,
                distribution=distribution,
                expected_class=expected_class,
                predicted_class=predicted_class,
                unavailable_reason="missing_expected_class",
                distribution_unavailable_reason=distribution_reason,
                derived_margin=derived_margin,
                provider_confidence=provider_confidence,
            )
    elif isinstance(decision, ScoreDecision):
        predicted_class = str(decision.level)
        probability = _score_probability(decision, event_spec)
        if probability is None:
            return NormalizedEvent(
                observation=observation,
                usable=False,
                probability=None,
                distribution=distribution,
                expected_class=expected_class,
                predicted_class=predicted_class,
                unavailable_reason="missing_semantic_boundary",
                distribution_unavailable_reason=distribution_reason,
                derived_margin=derived_margin,
                provider_confidence=provider_confidence,
            )
        derived_margin = abs(2.0 * probability - 1.0)
        if expected_class is None and event_spec.expected_class is not None:
            expected_class = _class_key(event_spec.expected_class)
    else:
        return NormalizedEvent(
            observation=observation,
            usable=False,
            probability=None,
            distribution=distribution,
            unavailable_reason="unsupported_decision",
        )
    return NormalizedEvent(
        observation=observation,
        usable=True,
        probability=probability,
        distribution=distribution,
        expected_class=expected_class,
        predicted_class=predicted_class,
        distribution_unavailable_reason=distribution_reason,
        derived_margin=derived_margin,
        provider_confidence=provider_confidence,
    )


def _valid_binary(events: Sequence[NormalizedEvent]) -> list[NormalizedEvent]:
    return [
        event
        for event in events
        if event.usable and event.probability is not None and event.label is not None
    ]


def _confusion(
    events: Sequence[NormalizedEvent], threshold: float
) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for event in _valid_binary(events):
        probability = cast(float, event.probability)
        predicted = probability >= threshold
        expected = bool(event.label)
        if predicted and expected:
            tp += 1
        elif predicted:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _f05(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def _roc_auc(events: Sequence[NormalizedEvent]) -> float | None:
    rows = [
        (cast(float, event.probability), bool(event.label))
        for event in _valid_binary(events)
    ]
    positives = sum(label for _, label in rows)
    negatives = len(rows) - positives
    if not rows or positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(rows), key=lambda item: item[1][0])
    ranks = [0.0] * len(rows)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1][0] == ordered[index][1][0]:
            end += 1
        average = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average
        index = end
    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, rows, strict=True) if label
    )
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _reliability(
    events: Sequence[NormalizedEvent], bins: int = 10
) -> tuple[float | None, list[dict[str, Any]]]:
    valid = _valid_binary(events)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            event
            for event in valid
            if (
                cast(float, event.probability) >= lower
                and (
                    cast(float, event.probability) < upper
                    or index == bins - 1
                    and cast(float, event.probability) <= upper
                )
            )
        ]
        count = len(selected)
        result.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_prediction": (
                    math.fsum(cast(float, event.probability) for event in selected)
                    / count
                    if count
                    else None
                ),
                "observed_rate": (
                    math.fsum(bool(event.label) for event in selected) / count
                    if count
                    else None
                ),
                "absolute_gap": (
                    abs(
                        math.fsum(cast(float, event.probability) for event in selected)
                        / count
                        - math.fsum(bool(event.label) for event in selected) / count
                    )
                    if count
                    else None
                ),
            }
        )
    if not valid:
        return None, result
    ece = math.fsum(
        (item["count"] / len(valid)) * item["absolute_gap"]
        for item in result
        if item["count"] and item["absolute_gap"] is not None
    )
    return ece, result


def _repeat_noise(events: Sequence[NormalizedEvent]) -> dict[str, Any]:
    grouped: dict[str, list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        if (
            not event.observation.control
            and event.usable
            and event.probability is not None
        ):
            grouped[event.observation.example_id].append(event)
    ranges: list[float] = []
    brier_ranges: list[float] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        probabilities = [cast(float, event.probability) for event in group]
        ranges.append(max(probabilities) - min(probabilities))
        brier = [
            (cast(float, event.probability) - bool(event.label)) ** 2
            for event in group
            if event.label is not None
        ]
        if len(brier) >= 2:
            brier_ranges.append(max(brier) - min(brier))
    return {
        "examples": sum(len(group) >= 2 for group in grouped.values()),
        "largest_within_example": max(ranges) if ranges else None,
        "mean_within_example": math.fsum(ranges) / len(ranges) if ranges else None,
        "brier_largest_within_example": max(brier_ranges) if brier_ranges else None,
        "brier_mean_within_example": math.fsum(brier_ranges) / len(brier_ranges)
        if brier_ranges
        else None,
    }


def _distribution_brier(
    events: Sequence[NormalizedEvent],
) -> tuple[float | None, str | None]:
    usable = [event for event in events if event.usable]
    if not usable:
        return None, "no_usable_events"
    losses: list[float] = []
    for event in usable:
        if event.distribution_unavailable_reason:
            return None, event.distribution_unavailable_reason
        if (
            event.expected_class is None
            or event.expected_class not in event.distribution
        ):
            return None, "missing_expected_class_for_distribution_brier"
        classes = set(event.distribution) | {event.expected_class}
        losses.append(
            math.fsum(
                (
                    event.distribution.get(class_name, 0.0)
                    - (1.0 if class_name == event.expected_class else 0.0)
                )
                ** 2
                for class_name in classes
            )
        )
    return math.fsum(losses) / len(losses), None


def compute_metrics(
    events: Sequence[NormalizedEvent], threshold: float | None
) -> dict[str, Any]:
    """Compute reason-bearing per-question metrics for a normalized event set."""

    total = len(events)
    valid = _valid_binary(events)
    labels = [bool(event.label) for event in valid]
    positive = sum(labels)
    negative = len(labels) - positive
    reasons: dict[str, str] = {}
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f05: float | None = None
    brier: float | None = None
    auc: float | None = None
    ece, reliability = _reliability(valid)
    distribution_brier, distribution_brier_reason = _distribution_brier(events)
    if valid:
        if threshold is None:
            reasons["classification"] = "threshold_unavailable"
        else:
            tp, fp, fn, tn = _confusion(valid, threshold)
            accuracy = (tp + tn) / len(valid)
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f05 = _f05(precision, recall)
            if precision is None:
                reasons["precision"] = "no_predicted_positive_events"
            if recall is None:
                reasons["recall"] = "no_positive_events"
        brier = math.fsum(
            (cast(float, event.probability) - bool(event.label)) ** 2 for event in valid
        ) / len(valid)
        auc = _roc_auc(valid)
    else:
        reasons["metrics"] = "no_usable_labeled_events"
    if auc is None:
        reasons["roc_auc"] = "missing_class_coverage_or_no_observations"
    if ece is None:
        reasons["ece"] = "no_usable_labeled_events"
    crossing = None
    if threshold is not None and valid:
        crossing = sum(
            cast(float, event.probability) >= threshold for event in valid
        ) / len(valid)
    elif threshold is None:
        reasons["threshold_crossing_fraction"] = "threshold_unavailable"
    unavailable = total - len(valid)
    missing_answers = sum(
        event.unavailable_reason
        in {"missing_answer", "malformed_answer", "primitive_mismatch"}
        for event in events
    )
    missing_labels = sum(event.label is None for event in events)
    provenance = Counter(event.observation.provenance for event in events)
    repeat = _repeat_noise(events)
    return {
        "support": {
            "events": total,
            "usable_labeled_events": len(valid),
            "source_groups": len({event.observation.source_group for event in events}),
            "examples": len({event.observation.example_id for event in events}),
            "positive_examples": positive,
            "negative_examples": negative,
            "controls": sum(event.observation.control for event in events),
            "repeat_examples": repeat["examples"],
        },
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f0_5": f05,
        "brier": brier,
        "distribution_brier": distribution_brier,
        "brier_distribution": distribution_brier,
        "distribution_brier_convention": "sum_over_declared_classes",
        "distribution_brier_reason": distribution_brier_reason,
        "roc_auc": auc,
        "ece": ece,
        "reliability": reliability,
        "threshold_crossing_fraction": crossing,
        "missing_answer_rate": missing_answers / total if total else None,
        "missing_answer_count": missing_answers,
        "missing_label_rate": missing_labels / total if total else None,
        "unavailable_rate": unavailable / total if total else None,
        "label_provenance": dict(sorted(provenance.items())),
        "repeat_spread": repeat,
        "reasons": reasons,
    }


def select_threshold(
    events: Sequence[NormalizedEvent], policy: ThresholdPolicy | None = None
) -> float | None:
    policy = policy or ThresholdPolicy()
    valid = _valid_binary(events)
    if not valid or not any(event.label for event in valid):
        return None
    candidates: list[tuple[float, float | None]] = []
    for threshold in policy.candidates():
        tp, fp, fn, _ = _confusion(valid, threshold)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        candidates.append((threshold, _f05(precision, recall)))
    available = [
        (threshold, score) for threshold, score in candidates if score is not None
    ]
    if not available:
        return None
    best = max(score for _, score in available)
    winners = [threshold for threshold, score in available if score == best]
    return max(winners) if policy.tie_break == "higher" else min(winners)


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-9), 1.0 - 1e-9)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def apply_temperature(probability: float, temperature: float) -> float:
    if temperature <= 0:
        raise CalibrationError("temperature must be positive")
    return _sigmoid(_logit(probability) / temperature)


def fit_temperature(
    events: Sequence[NormalizedEvent], *, grid: Sequence[float] | None = None
) -> float | None:
    valid = _valid_binary(events)
    if not valid:
        return None
    candidates = tuple(grid or (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0))
    scored: list[tuple[float, float]] = []
    for temperature in candidates:
        if temperature <= 0:
            continue
        loss = math.fsum(
            (
                apply_temperature(cast(float, event.probability), temperature)
                - bool(event.label)
            )
            ** 2
            for event in valid
        ) / len(valid)
        scored.append((loss, temperature))
    if not scored:
        return None
    best_loss = min(loss for loss, _ in scored)
    return max(temperature for loss, temperature in scored if loss == best_loss)


def _bootstrap_summary(
    events: Sequence[NormalizedEvent], *, seed: int, resamples: int
) -> dict[str, Any]:
    grouped: dict[str, list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        if not event.observation.control:
            grouped[event.observation.source_group].append(event)
    groups = sorted(grouped)
    if not groups or resamples <= 0:
        return {
            "method": "source_group_bootstrap",
            "seed": seed,
            "resamples": resamples,
            "valid_resamples": 0,
            "unavailable_replicates": resamples,
            "roc_auc": None,
            "brier": None,
            "ece": None,
        }
    rng = random.Random(seed)
    auc_values: list[float] = []
    brier_values: list[float] = []
    ece_values: list[float] = []
    for _ in range(resamples):
        selected = [rng.choice(groups) for _ in groups]
        sample = [event for group in selected for event in grouped[group]]
        auc = _roc_auc(sample)
        valid = _valid_binary(sample)
        if valid:
            if auc is not None:
                auc_values.append(auc)
            brier_values.append(
                math.fsum(
                    (cast(float, event.probability) - bool(event.label)) ** 2
                    for event in valid
                )
                / len(valid)
            )
            ece, _ = _reliability(valid)
            if ece is not None:
                ece_values.append(ece)

    def summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"mean": None, "stddev": None, "lower_95": None, "upper_95": None}
        ordered = sorted(values)
        return {
            "mean": math.fsum(values) / len(values),
            "stddev": math.sqrt(
                math.fsum(
                    (value - math.fsum(values) / len(values)) ** 2 for value in values
                )
                / len(values)
            ),
            "lower_95": ordered[max(0, math.ceil(0.05 * len(ordered)) - 1)],
            "upper_95": ordered[
                min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
            ],
        }

    return {
        "method": "source_group_bootstrap",
        "seed": seed,
        "resamples": resamples,
        "valid_resamples": len(auc_values),
        "unavailable_replicates": resamples - len(auc_values),
        "roc_auc": summary(auc_values),
        "brier": summary(brier_values),
        "ece": summary(ece_values),
    }


def _gate_components(
    metrics: Mapping[str, Any],
    control_metrics: Mapping[str, Any],
    *,
    policy: VerdictPolicy,
    events: Sequence[NormalizedEvent],
    threshold: float | None,
) -> dict[str, Any]:
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    coverage = (
        metrics["support"]["usable_labeled_events"] / metrics["support"]["events"]
        if metrics["support"]["events"]
        else 0.0
    )
    brier = metrics.get("brier")
    control_brier = control_metrics.get("brier")
    straddle = False
    if threshold is not None:
        grouped: dict[str, list[float]] = defaultdict(list)
        for event in events:
            if (
                not event.observation.control
                and event.usable
                and event.probability is not None
            ):
                grouped[event.observation.example_id].append(event.probability)
        straddle = any(
            min(values) < threshold <= max(values)
            for values in grouped.values()
            if len(values) >= 2
        )
    components = {
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "brier": brier,
        "control_brier": control_brier,
        "brier_better_than_control": (
            brier is not None and control_brier is not None and brier < control_brier
        ),
        "no_repeat_range_straddles_cutoff": not straddle,
        "threshold": threshold,
        "passes": False,
    }
    components["passes"] = bool(
        precision is not None
        and recall is not None
        and precision >= policy.precision_floor
        and recall >= policy.recall_floor
        and coverage >= policy.coverage_floor
        and (
            not policy.require_brier_better_than_control
            or components["brier_better_than_control"]
        )
        and (
            not policy.require_repeats
            or metrics["support"]["repeat_examples"] >= policy.minimum_repeat_examples
        )
        and (
            not policy.require_control
            or metrics["support"]["controls"] >= policy.minimum_control_groups
        )
        and (
            not policy.require_repeats or components["no_repeat_range_straddles_cutoff"]
        )
        and threshold is not None
    )
    return components


def _select_confidence_predicate(
    calibration_events: Sequence[NormalizedEvent],
    *,
    threshold: float,
    policy: VerdictPolicy,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for margin in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        subset = [
            event
            for event in calibration_events
            if event.usable
            and event.probability is not None
            and event.probability >= threshold
            and (event.derived_margin or 0.0) >= margin
            and event.label is not None
        ]
        if not subset:
            continue
        controls = [event for event in calibration_events if event.observation.control]
        metrics = compute_metrics(subset, threshold)
        metrics = dict(metrics)
        metrics["support"] = dict(metrics["support"])
        metrics["support"]["controls"] = len(controls)
        components = _gate_components(
            metrics,
            compute_metrics(controls, threshold),
            policy=policy,
            events=subset,
            threshold=threshold,
        )
        if components["passes"]:
            candidates.append(
                {
                    "probability_gte": threshold,
                    "margin_gte": margin,
                    "support": metrics["support"],
                    "coverage": metrics["support"]["usable_labeled_events"]
                    / max(1, metrics["support"]["events"]),
                }
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["coverage"], -item["margin_gte"]))


def _too_few(metrics: Mapping[str, Any], policy: VerdictPolicy) -> bool:
    support = metrics["support"]
    return (
        support["source_groups"] < policy.minimum_evaluation_groups
        or support["positive_examples"] < policy.minimum_positive_examples
        or support["negative_examples"] < policy.minimum_negative_examples
    )


def _fit_events(events: Sequence[NormalizedEvent], *, mode: str) -> dict[str, Any]:
    normalized_mode = str(mode).lower()
    if normalized_mode in {"none", "no-fit", "no_fit"}:
        return {"mode": "none", "temperature": 1.0, "objective": None}
    if normalized_mode not in {"temperature", "scalar-temperature"}:
        raise CalibrationError(f"unsupported calibration fit mode {mode!r}")
    temperature = fit_temperature(events)
    return {
        "mode": "temperature",
        "temperature": temperature,
        "objective": "binary_brier_on_fit_partition",
        "unavailable_reason": None
        if temperature is not None
        else "no_usable_fit_events",
    }


def _apply_fit(
    events: Sequence[NormalizedEvent], fit: Mapping[str, Any]
) -> list[NormalizedEvent]:
    temperature = fit.get("temperature")
    if fit.get("mode") == "none" or temperature is None:
        return list(events)
    result: list[NormalizedEvent] = []
    for event in events:
        if event.probability is None:
            result.append(event)
            continue
        transformed = apply_temperature(event.probability, float(temperature))
        distribution = dict(event.distribution)
        if event.observation.identity.primitive == "noul":
            distribution = {"no": 1.0 - transformed, "yes": transformed}
        result.append(
            replace(
                event,
                probability=transformed,
                distribution=distribution,
            )
        )
    return result


def calibrate_question(
    identity: QuestionIdentity,
    observations: Iterable[CalibrationObservation],
    *,
    verdict_policy: VerdictPolicy | None = None,
    threshold_policy: ThresholdPolicy | None = None,
    fit_mode: str = "none",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    name: str = "",
) -> CalibrationResult:
    """Fit, freeze, and independently evaluate one question's calibration policy."""

    policy = verdict_policy or VerdictPolicy()
    threshold_policy = threshold_policy or ThresholdPolicy()
    raw_observations = tuple(observations)
    if not raw_observations:
        raise CalibrationError("calibration requires at least one observation")
    if any(observation.identity != identity for observation in raw_observations):
        raise CalibrationError(
            "all observations must use the supplied question identity"
        )
    answer_keys: set[tuple[str, str, str, str]] = set()
    request_keys: set[tuple[str, str, str, str]] = set()
    for observation in raw_observations:
        answer_key = (
            observation.identity.question_id,
            observation.source_group,
            observation.example_id,
            observation.answer_id or observation.request_id or "",
        )
        if answer_key in answer_keys:
            raise CalibrationError(
                "duplicate answer identity: a cached response cannot count as an independent repeat"
            )
        answer_keys.add(answer_key)
        if observation.request_id:
            request_key = (
                observation.identity.question_id,
                observation.source_group,
                observation.example_id,
                observation.request_id,
            )
            if request_key in request_keys:
                raise CalibrationError(
                    "duplicate request identity: repeated calibration requests must be distinct"
                )
            request_keys.add(request_key)
    normalized = [normalize_event(observation) for observation in raw_observations]
    snapshot_values = {
        observation.answering_snapshot for observation in raw_observations
    }
    snapshot_match = identity.answering_snapshot is not None and snapshot_values == {
        identity.answering_snapshot
    }
    partition_map = _partition_observations(raw_observations, seed=seed)
    fit_source = [
        event
        for event in normalized
        if partition_map[event.observation.source_group] == "fit"
    ]
    calibration_source = [
        event
        for event in normalized
        if partition_map[event.observation.source_group] == "calibration"
    ]
    evaluation_source = [
        event
        for event in normalized
        if partition_map[event.observation.source_group] == "evaluation"
    ]
    fit = _fit_events(fit_source, mode=fit_mode)
    calibration_events = _apply_fit(calibration_source, fit)
    evaluation_events = _apply_fit(evaluation_source, fit)
    threshold = select_threshold(calibration_events, threshold_policy)
    control_events = [event for event in evaluation_events if event.observation.control]
    primary_evaluation_events = [
        event for event in evaluation_events if not event.observation.control
    ]
    control_metrics = compute_metrics(control_events, threshold)
    metrics = compute_metrics(primary_evaluation_events, threshold)
    metrics = dict(metrics)
    metrics["support"] = dict(metrics["support"])
    metrics["support"]["controls"] = len(control_events)
    metrics["support"]["control_source_groups"] = len(
        {event.observation.source_group for event in control_events}
    )
    full_components = _gate_components(
        metrics,
        control_metrics,
        policy=policy,
        events=primary_evaluation_events,
        threshold=threshold,
    )
    full_components["snapshot_match"] = snapshot_match
    if not snapshot_match:
        full_components["passes"] = False
    predicate: Mapping[str, Any] = {}
    verdict: str
    components: dict[str, Any]
    if _too_few(metrics, policy):
        verdict = Verdict.TOO_FEW_EXAMPLES.value
        components = {
            "reason": "insufficient_independent_evaluation_support",
            "minimum_evaluation_groups": policy.minimum_evaluation_groups,
            "minimum_positive_examples": policy.minimum_positive_examples,
            "minimum_negative_examples": policy.minimum_negative_examples,
            "observed": metrics["support"],
            "gate_components": full_components,
        }
    elif not snapshot_match:
        verdict = Verdict.UNUSABLE.value
        components = {
            "reason": "answering_snapshot_missing_or_mismatched",
            "observed_snapshots": sorted(
                snapshot for snapshot in snapshot_values if snapshot is not None
            ),
            "expected_snapshot": identity.answering_snapshot,
            "gate_components": full_components,
        }
    elif full_components["passes"]:
        verdict = Verdict.GATE.value
        components = {
            "reason": "all_gate_conditions_hold",
            "gate_components": full_components,
        }
    else:
        subset = (
            _select_confidence_predicate(
                calibration_events,
                threshold=threshold or 0.0,
                policy=policy,
            )
            if threshold is not None
            else None
        )
        if subset is not None:
            verdict = Verdict.GATE_ABOVE_CONFIDENCE.value
            predicate = subset
            components = {
                "reason": "gate_conditions_hold_on_frozen_calibration_subset",
                "subset": subset,
                "gate_components": full_components,
            }
        else:
            bootstrap = _bootstrap_summary(
                primary_evaluation_events,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            lower_bound = bootstrap.get("roc_auc", {}).get("lower_95")
            if lower_bound is not None and lower_bound > policy.ranker_auc_lower_bound:
                verdict = Verdict.RANKER.value
                components = {
                    "reason": "bootstrap_auc_lower_bound_exceeds_chance",
                    "auc_lower_95": lower_bound,
                    "gate_components": full_components,
                }
            else:
                verdict = Verdict.UNUSABLE.value
                components = {
                    "reason": "gate_conditions_failed_and_ranker_bound_not_met",
                    "gate_components": full_components,
                }
    ece_bootstrap = _bootstrap_summary(
        primary_evaluation_events,
        seed=bootstrap_seed + 1,
        resamples=bootstrap_resamples,
    )
    ece_summary = ece_bootstrap.get("ece") or {}
    uncertainty = {
        "bootstrap": _bootstrap_summary(
            primary_evaluation_events,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        ),
        "repeat_prediction_noise": metrics["repeat_spread"],
        "brier_repeat_noise": {
            "unit": "Brier_loss",
            "largest_within_example": metrics["repeat_spread"].get(
                "brier_largest_within_example"
            ),
            "mean_within_example": metrics["repeat_spread"].get(
                "brier_mean_within_example"
            ),
        },
        "ece_sampling_noise": {
            "method": "source_group_bootstrap",
            "value": ece_summary.get("stddev"),
        },
    }
    unavailable_reason_set = {
        reason
        for event in normalized
        for reason in (
            event.unavailable_reason,
            event.distribution_unavailable_reason,
        )
        if reason
    }
    if not snapshot_match:
        unavailable_reason_set.add("answering_snapshot_missing_or_mismatched")
    unavailable_reasons = sorted(unavailable_reason_set)
    status = "partial" if unavailable_reasons else "complete"
    input_digest = _digest(
        {
            "identity": identity.to_dict(),
            "events": [observation.to_dict() for observation in raw_observations],
        }
    )
    return CalibrationResult(
        identity=identity,
        metrics=metrics,
        control_metrics=control_metrics,
        threshold=threshold,
        predicate=predicate,
        fit=fit,
        partitions=partition_map,
        verdict=verdict,
        verdict_components=components,
        uncertainty=uncertainty,
        evidence={
            "partitions": {
                name: sorted(
                    group for group, value in partition_map.items() if value == name
                )
                for name in ("fit", "calibration", "evaluation")
            },
            "unavailable_reasons": unavailable_reasons,
            "label_provenance": metrics["label_provenance"],
            "raw_answer_count": len(raw_observations),
            "observations": [observation.to_dict() for observation in raw_observations],
            "control_count": sum(event.observation.control for event in normalized),
            "repeat_spread": metrics["repeat_spread"],
        },
        input_digest=input_digest,
        status=status,
        name=name or identity.question_id,
    )


def _budget_slice(
    observations: Sequence[CalibrationObservation], budget: CalibrationBudget | None
) -> tuple[list[CalibrationObservation], dict[str, Any]]:
    if budget is None:
        return list(observations), {"applied": False}
    groups = sorted({observation.source_group for observation in observations})
    selected_groups = set(groups[: budget.max_source_examples])
    selected = [
        observation
        for observation in observations
        if observation.source_group in selected_groups
    ]
    grouped_repeats: dict[tuple[str, str, str, bool], list[CalibrationObservation]] = (
        defaultdict(list)
    )
    for observation in selected:
        grouped_repeats[
            (
                observation.identity.question_id,
                observation.source_group,
                observation.example_id,
                observation.control,
            )
        ].append(observation)
    bounded: list[CalibrationObservation] = []
    dropped_repeats = 0
    for repeat_group in grouped_repeats.values():
        ordered = sorted(repeat_group, key=lambda item: item.repeat_index)
        bounded.extend(ordered[: budget.max_repeats])
        dropped_repeats += max(0, len(ordered) - budget.max_repeats)
    partial = (
        len(selected_groups) < len(groups)
        or dropped_repeats > 0
        or len(bounded) > budget.max_question_evaluations
    )
    if len(bounded) > budget.max_question_evaluations:
        bounded = bounded[: budget.max_question_evaluations]
    selected = bounded
    return selected, {
        "applied": True,
        "limits": budget.to_dict(),
        "observed_source_groups": len(groups),
        "selected_source_groups": len(selected_groups),
        "selected_events": len(selected),
        "dropped_repeats": dropped_repeats,
        "distinct_answer_identities": len(
            {observation.answer_id for observation in selected if observation.answer_id}
        ),
        "partial": partial,
        "stopped_reason": "bounded-input" if partial else None,
    }


def calibrate_manifest(
    manifest: CalibrationManifest,
    *,
    verdict_policy: VerdictPolicy | None = None,
    threshold_policy: ThresholdPolicy | None = None,
    fit_mode: str = "none",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    budget: CalibrationBudget | None = None,
) -> tuple[CalibrationArtifact, dict[str, Any]]:
    """Calibrate every question in a manifest and return artifact plus report."""

    observations, budget_report = _budget_slice(manifest.events, budget)
    grouped: dict[str, list[CalibrationObservation]] = defaultdict(list)
    identities: dict[str, QuestionIdentity] = {}
    for observation in observations:
        key = observation.identity.identity_digest
        grouped[key].append(observation)
        identities[key] = observation.identity
    results: dict[str, CalibrationResult] = {}
    for key, group in sorted(grouped.items()):
        identity = identities[key]
        result = calibrate_question(
            identity,
            group,
            verdict_policy=verdict_policy,
            threshold_policy=threshold_policy,
            fit_mode=fit_mode,
            seed=seed,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
            name=identity.question_id,
        )
        output_key = identity.question_id
        if output_key in results:
            output_key = f"{output_key}@{key[:12]}"
        results[output_key] = result
    questions = {key: result.to_dict() for key, result in sorted(results.items())}
    status = (
        "partial"
        if any(result.status != "complete" for result in results.values())
        else "complete"
    )
    if budget_report.get("partial"):
        status = "partial"
    artifact = CalibrationArtifact(
        name=manifest.name,
        input_digest=manifest.input_digest,
        questions=questions,
        status=status,
        metadata={
            "schema_version": manifest.schema_version,
            "source_event_count": len(manifest.events),
            "calibrated_event_count": len(observations),
            "seed": seed,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "fit_mode": fit_mode,
            "verdict_policy": (verdict_policy or VerdictPolicy()).to_dict(),
            "threshold_policy": (threshold_policy or ThresholdPolicy()).to_dict(),
            "budget": budget_report,
        },
    )
    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": CALIBRATION_REPORT_KIND,
        "name": manifest.name,
        "input_digest": manifest.input_digest,
        "status": status,
        "questions": questions,
        "verdict_catalog": list(VERDICT_CATALOG),
        "budget": budget_report,
        "metadata": dict(artifact.metadata),
    }
    return artifact, report


def run_calibration(
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> tuple[CalibrationArtifact, dict[str, Any]]:
    return calibrate_manifest(load_calibration_manifest(source), **kwargs)


# Backwards-friendly short name for callers that use the command as a function.
calibrate = run_calibration


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    disposition: str
    threshold: float | None = None
    predicate: Mapping[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    reason: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_legacy(self) -> bool:
        return self.disposition == Disposition.LEGACY.value

    @property
    def may_gate(self) -> bool:
        return self.disposition in {
            Disposition.GATE.value,
            Disposition.GATE_ABOVE_CONFIDENCE.value,
        }


class DecisionPolicy:
    """Resolve persisted calibration verdicts for one runtime question.

    The distinction between no candidate and an incompatible candidate is
    intentional: no artifact preserves the named legacy policy, while an
    artifact with a stale identity or snapshot must abstain.
    """

    def __init__(
        self,
        artifact: CalibrationArtifact | Mapping[str, Any] | str | Path | None = None,
        *,
        artifacts: Iterable[CalibrationArtifact | Mapping[str, Any] | str | Path]
        | None = None,
        policy_version: str = DEFAULT_POLICY_VERSION,
    ) -> None:
        self.policy_version = policy_version
        self._questions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        sources: list[object] = []
        if artifact is not None:
            sources.append(artifact)
        if artifacts is not None:
            sources.extend(artifacts)
        for source in sources:
            loaded = (
                CalibrationArtifact.load(source)
                if isinstance(source, (str, Path))
                else CalibrationArtifact.from_dict(source)
                if isinstance(source, Mapping)
                else source
            )
            if not isinstance(loaded, CalibrationArtifact):
                raise CalibrationError("DecisionPolicy accepts calibration artifacts")
            for key, value in loaded.questions.items():
                self._questions[str(key)].append(dict(value))

    @classmethod
    def from_artifact(
        cls, artifact: CalibrationArtifact | Mapping[str, Any] | str | Path
    ) -> DecisionPolicy:
        return cls(artifact)

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._questions))

    def has_candidate(self, question_id: str) -> bool:
        return question_id in self._questions

    def _identity_matches(
        self,
        question: Mapping[str, Any],
        identity: QuestionIdentity | None,
        snapshot: str | None,
    ) -> bool:
        if identity is None:
            return False
        raw_identity = question.get("identity")
        if not isinstance(raw_identity, Mapping):
            return False
        try:
            stored = QuestionIdentity.from_dict(raw_identity)
        except CalibrationError:
            return False
        actual_snapshot = snapshot or identity.answering_snapshot
        return (
            stored.question_id == identity.question_id
            and stored.answering_snapshot == actual_snapshot
            and stored.identity_digest == identity.identity_digest
        )

    def resolve(
        self,
        question_id: str,
        *,
        identity: QuestionIdentity | None = None,
        snapshot: str | None = None,
    ) -> PolicyDecision:
        candidates = self._questions.get(question_id, [])
        if not candidates:
            return PolicyDecision(
                Disposition.LEGACY.value, reason="no_calibration_artifact"
            )
        matching = [
            candidate
            for candidate in candidates
            if self._identity_matches(candidate, identity, snapshot)
        ]
        if not matching:
            return PolicyDecision(
                Disposition.ABSTAIN.value,
                reason="calibration_identity_or_snapshot_mismatch",
                evidence={"question_id": question_id},
            )
        question = matching[0]
        verdict = str(question.get("verdict", ""))
        threshold = question.get("threshold")
        try:
            threshold_value = (
                None
                if threshold is None
                else _probability(threshold, field_name="threshold")
            )
        except CalibrationError:
            return PolicyDecision(
                Disposition.ABSTAIN.value,
                verdict=verdict,
                reason="invalid_calibration_threshold",
            )
        predicate = question.get("predicate", {})
        if not isinstance(predicate, Mapping):
            predicate = {}
        if (
            verdict in {Verdict.GATE.value, Verdict.GATE_ABOVE_CONFIDENCE.value}
            and threshold_value is None
        ):
            return PolicyDecision(
                Disposition.ABSTAIN.value,
                verdict=verdict,
                reason="gate_without_threshold",
                predicate=dict(predicate),
            )
        disposition = {
            Verdict.GATE.value: Disposition.GATE.value,
            Verdict.GATE_ABOVE_CONFIDENCE.value: Disposition.GATE_ABOVE_CONFIDENCE.value,
            Verdict.RANKER.value: Disposition.RANKER.value,
            Verdict.UNUSABLE.value: Disposition.ABSTAIN.value,
            Verdict.TOO_FEW_EXAMPLES.value: Disposition.ABSTAIN.value,
        }.get(verdict, Disposition.ABSTAIN.value)
        return PolicyDecision(
            disposition=disposition,
            threshold=threshold_value,
            predicate=dict(predicate),
            verdict=verdict,
            reason="calibration_artifact",
            evidence={
                "verdict_components": question.get("verdict_components", {}),
                "metrics": question.get("metrics", {}),
            },
        )

    @staticmethod
    def _event_probability(decision: Any, identity: QuestionIdentity) -> float | None:
        if isinstance(decision, NoulDecision):
            mapping = identity.event_mapping
            if str(mapping.get("polarity", "positive")).lower() in {
                "negative",
                "no",
                "false",
            }:
                return 1.0 - decision.probability
            return decision.probability
        if isinstance(decision, ChoiceDecision):
            mapping = identity.event_mapping
            target = mapping.get("expected_class", mapping.get("expected_option"))
            if target is not None:
                return decision.probabilities.get(str(target))
            if mapping.get("selected_correctness"):
                return decision.probabilities.get(decision.selected)
            return None
        if isinstance(decision, ScoreDecision):
            return _score_probability(decision, EventSpec.from_identity(identity))
        return None

    @staticmethod
    def _margin(decision: Any) -> float | None:
        if isinstance(decision, NoulDecision):
            return abs(2.0 * decision.probability - 1.0)
        if isinstance(decision, (ChoiceDecision, ScoreDecision)):
            probabilities = list(decision.probabilities.values())
            return max(probabilities) if probabilities else None
        return None

    def apply(
        self,
        *,
        question_id: str,
        identity: QuestionIdentity,
        decision: Any,
        raw_answer: Any = None,
        snapshot: str | None = None,
    ) -> PolicyDecision:
        resolved = self.resolve(question_id, identity=identity, snapshot=snapshot)
        if resolved.is_legacy or resolved.disposition == Disposition.RANKER.value:
            return resolved
        if resolved.disposition == Disposition.ABSTAIN.value:
            return resolved
        probability = self._event_probability(decision, identity)
        if probability is None:
            return replace(
                resolved,
                disposition=Disposition.ABSTAIN.value,
                reason="calibration_event_probability_unavailable",
            )
        if resolved.threshold is not None and probability < resolved.threshold:
            return replace(
                resolved,
                disposition=Disposition.ABSTAIN.value,
                reason="probability_below_calibrated_threshold",
            )
        predicate = resolved.predicate
        if predicate:
            checks: list[bool] = []
            for name, actual in (
                ("probability_gte", probability),
                ("margin_gte", self._margin(decision)),
                ("confidence_gte", _raw_confidence(raw_answer)),
            ):
                if name not in predicate or actual is None:
                    continue
                try:
                    checks.append(actual >= float(predicate[name]))
                except (TypeError, ValueError):
                    return replace(
                        resolved,
                        disposition=Disposition.ABSTAIN.value,
                        reason="invalid_calibration_predicate",
                    )
            if not checks or not all(checks):
                return replace(
                    resolved,
                    disposition=Disposition.ABSTAIN.value,
                    reason="frozen_calibration_predicate_failed",
                )
        return resolved


CalibrationPolicy = DecisionPolicy


def runtime_question_identity(
    question_id: str,
    request: Mapping[str, Any],
    *,
    family: str = "",
    rubric_version: str | None = None,
    snapshot: str | None = None,
) -> QuestionIdentity:
    """Build the runtime identity without changing the Gateway request shape."""

    question = _first(request, "query", "question", "instructions", default="")
    primitive = str(_first(request, "type", "primitive", default="noul")).lower()
    criteria = _first(request, "criteria", "options", "levels", default=None)
    if criteria is None and primitive == "noul":
        criteria = ("no", "yes")
    event_mapping = _first(request, "event_mapping", "event", default=None)
    if event_mapping is None:
        event_mapping = {"positive_class": "yes"} if primitive == "noul" else {}
    return QuestionIdentity(
        question_id=question_id,
        question=question,
        primitive=str(_first(request, "type", "primitive", default="noul")),
        criteria=criteria,
        question_schema=_first(request, "question_schema", "schema", default=None),
        event_mapping=event_mapping,
        family=family,
        rubric_version=rubric_version,
        answering_snapshot=snapshot
        or getattr(request.get("_calibration_gateway", None), "jev_model", None),
    )


__all__ = [
    "CALIBRATION_ARTIFACT_KIND",
    "CALIBRATION_REPORT_KIND",
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_POLICY_VERSION",
    "VERDICT_CATALOG",
    "CalibrationArtifact",
    "CalibrationBudget",
    "CalibrationError",
    "CalibrationManifest",
    "CalibrationObservation",
    "CalibrationPolicy",
    "CalibrationResult",
    "DecisionPolicy",
    "Disposition",
    "EventSpec",
    "NormalizedEvent",
    "PolicyDecision",
    "Primitive",
    "QuestionIdentity",
    "ThresholdPolicy",
    "Verdict",
    "VerdictPolicy",
    "apply_temperature",
    "calibrate",
    "calibrate_manifest",
    "calibrate_question",
    "compute_metrics",
    "fit_temperature",
    "group_bootstrap",
    "load_calibration_manifest",
    "manifest_from_dict",
    "normalize_event",
    "partition_groups",
    "run_calibration",
    "runtime_question_identity",
    "select_threshold",
]


def group_bootstrap(
    events: Sequence[NormalizedEvent],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = 1000,
) -> dict[str, Any]:
    """Public spelling for the source-group bootstrap summary."""

    return _bootstrap_summary(events, seed=seed, resamples=resamples)
