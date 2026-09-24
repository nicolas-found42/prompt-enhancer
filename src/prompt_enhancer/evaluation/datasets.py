"""Dataset loading for maintainer prompt evaluations.

The harness deliberately accepts a small, provider-neutral JSON format so real,
synthetic, and hand-labeled examples can live together.  Only ``prompt`` and
``id`` are required; ground-truth diagnosis annotations are optional.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOURCE_REAL = "real"
SOURCE_SYNTHETIC = "synthetic"
SOURCE_HAND_LABELED = "hand_labeled"
DATASET_SOURCES = frozenset(
    {SOURCE_REAL, SOURCE_SYNTHETIC, SOURCE_HAND_LABELED}
)

_GAP_KEYS = ("gap_type", "type", "kind", "name", "key", "id")
_GAP_FIELDS = (
    "expected_gaps",
    "known_gaps",
    "known_defects",
    "planted_defects",
    "annotated_gaps",
    "gaps",
    "labels",
)
_CASE_METADATA_EXCLUSIONS = frozenset(
    {
        *_GAP_FIELDS,
        "id",
        "prompt",
        "text",
        "source",
        "kind",
        "notes",
        "evaluation_notes",
        "labels_present",
    }
)

_GAP_TYPE_ALIASES = {
    "missing_context": "context",
    "missing_output_format": "output_format",
    "indices_&_ranges": "indices_ranges",
    "ordering_&_atomicity": "ordering_atomicity",
    "string_&_localization": "string_localization",
    "definition_of_done": "done_criteria",
    "success_criteria": "done_criteria",
    "done_criterion": "done_criteria",
    "missing_done_criteria": "done_criteria",
    "unresolved_references": "unresolved_reference",
    "vague": "vagueness",
    "vague_sentence": "vagueness",
    "sentence_vagueness": "vagueness",
    "contradiction": "contradiction",
    "contradictory": "contradiction",
    "contradictions": "contradiction",
    "embedded_instruction": "embedded_instructions",
    "prompt_injection": "embedded_instructions",
    "instructions_in_pasted_content": "embedded_instructions",
}


class DatasetError(ValueError):
    """Raised when evaluation data cannot be interpreted safely."""


def normalize_gap_type(value: object) -> str:
    """Return a stable, human-readable key for a gap annotation."""

    if not isinstance(value, str) or not value.strip():
        raise DatasetError("gap types must be non-empty strings")
    text = value.strip().lower().replace("-", " ")
    text = "_".join(text.split())
    text = _GAP_TYPE_ALIASES.get(text, text.replace(" ", "_"))
    # Checklist keys are often namespaced (``output_format.missing``).
    for separator in (".", "/", ":"):
        if separator in text:
            candidate = text.rsplit(separator, 1)[-1]
            if candidate:
                text = candidate
    return text


def _gap_type_from_mapping(value: Mapping[str, Any]) -> str | None:
    for key in _GAP_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return normalize_gap_type(candidate)

    nested = value.get("checklist_item") or value.get("gap")
    if isinstance(nested, Mapping):
        return _gap_type_from_mapping(nested)
    if isinstance(nested, str) and nested.strip():
        return normalize_gap_type(nested)
    return None


def _annotation_gaps(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {normalize_gap_type(value)}
    if isinstance(value, Mapping):
        if value.get("present") is False:
            return set()
        gap_type = _gap_type_from_mapping(value)
        return {gap_type} if gap_type else set()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        gaps: set[str] = set()
        for item in value:
            if isinstance(item, Mapping) and item.get("present") is False:
                continue
            if isinstance(item, str) and item.strip():
                gaps.add(normalize_gap_type(item))
            elif isinstance(item, Mapping):
                gap_type = _gap_type_from_mapping(item)
                if gap_type:
                    gaps.add(gap_type)
        return gaps
    raise DatasetError(
        "gap annotations must be strings, objects, or lists of strings/objects"
    )


def _expected_gaps(case: Mapping[str, Any]) -> tuple[str, ...]:
    gaps: set[str] = set()
    for field_name in _GAP_FIELDS:
        if field_name not in case:
            continue
        value = case[field_name]
        if field_name == "labels" and isinstance(value, Mapping):
            gaps.update(
                normalize_gap_type(key)
                for key, present in value.items()
                if present is True
            )
        else:
            gaps.update(_annotation_gaps(value))
    return tuple(sorted(gaps))


def _normalize_source(value: object, *, has_labels: bool) -> str:
    if value is None:
        return SOURCE_HAND_LABELED if has_labels else SOURCE_REAL
    if not isinstance(value, str):
        raise DatasetError("dataset case source must be a string")
    source = value.strip().lower().replace("-", "_")
    aliases = {
        "planted": SOURCE_SYNTHETIC,
        "synthetic_planted_defect": SOURCE_SYNTHETIC,
        "hand_labeled": SOURCE_HAND_LABELED,
        "handlabels": SOURCE_HAND_LABELED,
        "user": SOURCE_REAL,
    }
    source = aliases.get(source, source)
    if source not in DATASET_SOURCES:
        choices = ", ".join(sorted(DATASET_SOURCES))
        raise DatasetError(f"unknown dataset source {value!r}; expected {choices}")
    return source


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One prompt and its optional diagnosis ground truth."""

    id: str
    prompt: str
    source: str
    expected_gaps: tuple[str, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    labels_present: bool = False

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, position: int, dataset_name: str
    ) -> EvaluationCase:
        prompt = value.get("prompt", value.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetError(
                f"{dataset_name} case {position} requires non-empty prompt text"
            )
        case_id = value.get("id", f"case-{position:04d}")
        if not isinstance(case_id, str) or not case_id.strip():
            raise DatasetError(
                f"{dataset_name} case {position} id must be a non-empty string"
            )
        expected = _expected_gaps(value)
        notes = value.get("evaluation_notes", value.get("notes"))
        if notes is not None and not isinstance(notes, str):
            raise DatasetError(
                f"{dataset_name} case {case_id!r} notes must be a string"
            )
        nested_metadata = value.get("metadata", {})
        if not isinstance(nested_metadata, Mapping):
            raise DatasetError(f"{dataset_name} case {case_id!r} metadata must be an object")
        metadata = dict(nested_metadata)
        for key, item in value.items():
            if key in _CASE_METADATA_EXCLUSIONS or key == "metadata":
                continue
            if key in metadata and metadata[key] != item:
                raise DatasetError(f"{dataset_name} case {case_id!r} has conflicting metadata for {key!r}")
            metadata[key] = item
        return cls(
            id=case_id,
            prompt=prompt,
            source=_normalize_source(value.get("source"), has_labels=bool(expected)),
            expected_gaps=expected,
            notes=notes,
            metadata=metadata,
            labels_present=bool(value.get("labels_present", any(field in value for field in _GAP_FIELDS))),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "source": self.source,
            "labels_present": self.labels_present,
        }
        if self.expected_gaps:
            value["expected_gaps"] = list(self.expected_gaps)
        if self.notes is not None:
            value["evaluation_notes"] = self.notes
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value


@dataclass(frozen=True, slots=True)
class Dataset:
    """A deterministic collection of evaluation cases."""

    name: str
    cases: tuple[EvaluationCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, name: str | None = None
    ) -> Dataset:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw_cases: object = value
            raw_metadata: Mapping[str, Any] = {}
            schema_version = 1
            dataset_name = name or "evaluation"
        elif isinstance(value, Mapping):
            raw_cases = value.get("cases", value.get("examples", value.get("prompts")))
            raw_metadata = value.get("metadata", {})
            schema_version = value.get("schema_version", 1)
            dataset_name = name or value.get("name") or "evaluation"
            if not isinstance(dataset_name, str) or not dataset_name.strip():
                raise DatasetError("dataset name must be a non-empty string")
            if not isinstance(raw_metadata, Mapping):
                raise DatasetError("dataset metadata must be an object")
            if not isinstance(schema_version, int) or schema_version < 1:
                raise DatasetError("dataset schema_version must be a positive integer")
        else:
            raise DatasetError("dataset JSON must be an object or a list of cases")

        if not isinstance(raw_cases, Sequence) or isinstance(
            raw_cases, (str, bytes)
        ):
            raise DatasetError("dataset requires a cases array")
        if not raw_cases:
            raise DatasetError("dataset requires at least one case")

        cases: list[EvaluationCase] = []
        seen_ids: set[str] = set()
        for position, raw_case in enumerate(raw_cases, start=1):
            if not isinstance(raw_case, Mapping):
                raise DatasetError(
                    f"{dataset_name} case {position} must be an object"
                )
            case = EvaluationCase.from_dict(
                raw_case, position=position, dataset_name=dataset_name
            )
            if case.id in seen_ids:
                raise DatasetError(f"duplicate evaluation case id {case.id!r}")
            seen_ids.add(case.id)
            cases.append(case)
        return cls(
            name=dataset_name,
            cases=tuple(cases),
            metadata=dict(raw_metadata),
            schema_version=schema_version,
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_dict())

    def source_counts(self) -> dict[str, int]:
        counts = {source: 0 for source in sorted(DATASET_SOURCES)}
        for case in self.cases:
            counts[case.source] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "source_counts": self.source_counts(),
            "cases": [case.to_dict() for case in self.cases],
        }
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value


def load_dataset(path: str | Path) -> Dataset:
    """Load one JSON dataset and retain its filename as provenance metadata."""

    dataset_path = Path(path)
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"could not load dataset {dataset_path}: {exc}") from exc
    return Dataset.from_dict(raw, name=dataset_path.stem)


def load_datasets(paths: Iterable[str | Path]) -> Dataset:
    """Load and combine one or more dataset files, rejecting duplicate case ids."""

    paths = tuple(paths)
    if not paths:
        raise DatasetError("at least one dataset path is required")
    datasets = [load_dataset(path) for path in paths]
    cases = tuple(case for dataset in datasets for case in dataset.cases)
    names = [dataset.name for dataset in datasets]
    if len(names) == 1:
        name = names[0]
        metadata: dict[str, Any] = dict(datasets[0].metadata)
    else:
        name = " + ".join(names)
        metadata = {
            "datasets": [
                {"name": dataset.name, "digest": dataset.digest}
                for dataset in datasets
            ]
        }
    return Dataset(name=name, cases=cases, metadata=metadata)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json(value: object) -> str:
    """Serialize report inputs deterministically."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def replay_digest(path: str | Path) -> str:
    """Digest replay content independently of JSON whitespace or object order."""

    replay_path = Path(path)
    try:
        raw_bytes = replay_path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"could not read replay file {replay_path}: {exc}") from exc
    try:
        parsed = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(raw_bytes).hexdigest()
    return _canonical_digest(parsed)
