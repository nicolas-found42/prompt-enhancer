"""Measure Jev option-order effects against ordinary repeat variation.

The experiment consumes a bounded set of source prompt/output pairs and their
recorded success tests. Its recordings are keyed by unique request identities so
strict replay cannot turn repeats into cache copies.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from ..gateway import Gateway
from ..jev import ChoiceDecision, JevResponseError, ScoreDecision, parse_decision

ORDER_BIAS_SCHEMA_VERSION = 1
ORDER_BIAS_POLICY_VERSION = "issue-43-order-bias-v1"
ORDER_BIAS_REPORT_KIND = "order-bias-report"
ORDER_BIAS_ARTIFACT_KIND = "order-bias-policy-artifact"
ORDER_BIAS_RECORDING_KIND = "order-bias-recording"
LEGACY_GRADING_POLICY = "legacy_min_pair"
DEFAULT_REPETITIONS = 3
MAX_CASES = 100
MAX_QUESTION_EVALUATIONS = 1000
MINIMUM_SOURCE_GROUPS = 30
EXCESS_TOLERANCE = 0.01
DEFAULT_BOOTSTRAP_SEED = 1729
DEFAULT_BOOTSTRAP_RESAMPLES = 1000


class OrderBiasError(ValueError):
    """Raised when an order-bias experiment is malformed or unsafe."""


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OrderBiasError(f"value is not finite JSON data: {exc}") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _number(value: object, *, field_name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrderBiasError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise OrderBiasError(f"{field_name} must be finite and at least {minimum}")
    return result


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrderBiasError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OrderBiasError(f"{field_name} must be an array of strings")
    result = tuple(_nonempty_string(item, field_name=field_name) for item in value)
    if len(set(result)) != len(result):
        raise OrderBiasError(f"{field_name} must not contain duplicates")
    return result


def _normalize_test(
    value: Mapping[str, Any], *, case_id: str, index: int
) -> dict[str, Any]:
    kind = str(value.get("kind", "")).strip().lower()
    if kind not in {"choice", "score", "noul"}:
        raise OrderBiasError(f"case {case_id!r} test {index} has unsupported kind")
    question = _nonempty_string(value.get("question"), field_name="test question")
    test_id = _nonempty_string(
        value.get("id", f"test-{index:03d}"), field_name="test id"
    )
    expected = _nonempty_string(value.get("expected"), field_name="test expected")
    result: dict[str, Any] = {
        "id": test_id,
        "question": question,
        "kind": kind,
        "expected": expected,
    }
    if kind == "choice":
        options = _string_list(value.get("options", ()), field_name="Choice options")
        if len(options) < 2:
            raise OrderBiasError(f"case {case_id!r} Choice test needs two options")
        if expected not in options:
            raise OrderBiasError(
                f"case {case_id!r} expected Choice option is not declared"
            )
        descriptions = value.get("option_descriptions", {})
        if not isinstance(descriptions, Mapping):
            raise OrderBiasError("option_descriptions must be an object")
        normalized_descriptions = {
            str(key): _nonempty_string(description, field_name="option description")
            for key, description in descriptions.items()
        }
        if any(option not in normalized_descriptions for option in options):
            raise OrderBiasError(f"case {case_id!r} Choice options need descriptions")
        result["options"] = list(options)
        result["option_descriptions"] = normalized_descriptions
    elif kind == "score":
        levels = _string_list(value.get("levels", ()), field_name="Score levels")
        if len(levels) < 2:
            raise OrderBiasError(f"case {case_id!r} Score test needs two levels")
        if expected not in levels:
            raise OrderBiasError(
                f"case {case_id!r} expected Score level is not declared"
            )
        result["levels"] = list(levels)
    return result


@dataclass(frozen=True, slots=True)
class OrderBiasCase:
    id: str
    source_group: str
    provenance: str
    prompt: str
    output: str
    success_tests: tuple[Mapping[str, Any], ...]
    recorded_cost_usd: float | None = None
    recorded_input_tokens: int | None = None
    recorded_output_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "source_group": self.source_group,
            "provenance": self.provenance,
            "prompt": self.prompt,
            "output": self.output,
            "success_tests": [dict(test) for test in self.success_tests],
        }
        if self.recorded_cost_usd is not None:
            result["recorded_cost_usd"] = self.recorded_cost_usd
        if self.recorded_input_tokens is not None:
            result["recorded_input_tokens"] = self.recorded_input_tokens
        if self.recorded_output_tokens is not None:
            result["recorded_output_tokens"] = self.recorded_output_tokens
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True, slots=True)
class OrderBiasManifest:
    name: str
    cases: tuple[OrderBiasCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ORDER_BIAS_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | Sequence[Mapping[str, Any]]
    ) -> OrderBiasManifest:
        if isinstance(value, Mapping):
            raw_cases = value.get("cases")
            name = _nonempty_string(
                value.get("name", "order-bias"), field_name="manifest name"
            )
            metadata = value.get("metadata", {})
            schema_version = value.get("schema_version", ORDER_BIAS_SCHEMA_VERSION)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw_cases = value
            name, metadata, schema_version = "order-bias", {}, ORDER_BIAS_SCHEMA_VERSION
        else:
            raise OrderBiasError("order-bias manifest must be an object or cases array")
        if (
            not isinstance(raw_cases, Sequence)
            or isinstance(raw_cases, (str, bytes))
            or not raw_cases
        ):
            raise OrderBiasError("order-bias manifest requires a non-empty cases array")
        if len(raw_cases) > MAX_CASES:
            raise OrderBiasError(
                f"order-bias experiments are limited to {MAX_CASES} cases"
            )
        if not isinstance(metadata, Mapping):
            raise OrderBiasError("manifest metadata must be an object")
        if schema_version != ORDER_BIAS_SCHEMA_VERSION:
            raise OrderBiasError(
                f"unsupported order-bias schema_version {schema_version!r}"
            )
        cases: list[OrderBiasCase] = []
        seen_ids: set[str] = set()
        group_sources: dict[str, tuple[str, str]] = {}
        for position, raw in enumerate(raw_cases, start=1):
            if not isinstance(raw, Mapping):
                raise OrderBiasError(f"case {position} must be an object")
            case_id = _nonempty_string(
                raw.get("id", f"case-{position:04d}"), field_name="case id"
            )
            if case_id in seen_ids:
                raise OrderBiasError(f"duplicate order-bias case id {case_id!r}")
            seen_ids.add(case_id)
            source_group = _nonempty_string(
                raw.get("source_group"), field_name="source_group"
            )
            provenance = (
                str(raw.get("provenance", "synthetic_known_answer")).strip().lower()
            )
            if provenance not in {"matched_recording", "synthetic_known_answer"}:
                raise OrderBiasError(
                    "case provenance must be matched_recording or synthetic_known_answer"
                )
            prompt = _nonempty_string(raw.get("prompt"), field_name="recorded prompt")
            output = _nonempty_string(raw.get("output"), field_name="recorded output")
            source_pair = (prompt, output)
            previous = group_sources.setdefault(source_group, source_pair)
            if previous != source_pair:
                raise OrderBiasError(
                    f"source_group {source_group!r} refers to multiple prompt/output pairs"
                )
            raw_tests = raw.get("success_tests", raw.get("tests"))
            if (
                not isinstance(raw_tests, Sequence)
                or isinstance(raw_tests, (str, bytes))
                or not raw_tests
            ):
                raise OrderBiasError(f"case {case_id!r} requires success_tests")
            tests = tuple(
                _normalize_test(test, case_id=case_id, index=index)
                for index, test in enumerate(raw_tests, start=1)
                if isinstance(test, Mapping)
            )
            if len(tests) != len(raw_tests):
                raise OrderBiasError(f"case {case_id!r} success tests must be objects")
            if len({test["id"] for test in tests}) != len(tests):
                raise OrderBiasError(
                    f"case {case_id!r} success test ids must be unique"
                )
            cost = raw.get("recorded_cost_usd")
            input_tokens = raw.get("recorded_input_tokens")
            output_tokens = raw.get("recorded_output_tokens")
            cases.append(
                OrderBiasCase(
                    id=case_id,
                    source_group=source_group,
                    provenance=provenance,
                    prompt=prompt,
                    output=output,
                    success_tests=tests,
                    recorded_cost_usd=_number(cost, field_name="recorded_cost_usd")
                    if cost is not None
                    else None,
                    recorded_input_tokens=int(input_tokens)
                    if input_tokens is not None
                    else None,
                    recorded_output_tokens=int(output_tokens)
                    if output_tokens is not None
                    else None,
                    metadata=dict(raw.get("metadata", {}))
                    if isinstance(raw.get("metadata", {}), Mapping)
                    else {},
                )
            )
        return cls(
            name=name,
            cases=tuple(cases),
            metadata=dict(metadata),
            schema_version=int(schema_version),
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema_version": self.schema_version,
                "name": self.name,
                "metadata": dict(self.metadata),
                "cases": [case.to_dict() for case in self.cases],
            }
        )


def load_order_bias_manifest(path: str | Path) -> OrderBiasManifest:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OrderBiasError(
            f"could not read order-bias manifest {source}: {exc}"
        ) from exc
    return OrderBiasManifest.from_dict(raw)


def question_schema(test: Mapping[str, Any]) -> dict[str, Any]:
    """The exact semantic Choice/Score contract used for compatibility."""
    kind = str(test.get("kind", "")).lower()
    result: dict[str, Any] = {
        "primitive": kind,
        "question": test.get("question"),
        "expected": test.get("expected"),
    }
    if kind == "choice":
        result["options"] = list(test.get("options", ()))
        result["option_descriptions"] = dict(test.get("option_descriptions", {}))
    elif kind == "score":
        result["levels"] = list(test.get("levels", ()))
    return result


def question_schema_digest(test: Mapping[str, Any]) -> str:
    return _digest(question_schema(test))


def build_order_bias_requests(
    manifest: OrderBiasManifest, *, repetitions: int = DEFAULT_REPETITIONS
) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or not 2 <= repetitions <= 3
    ):
        raise OrderBiasError("order-bias repetitions must be 2 or 3")
    requests: list[dict[str, Any]] = []
    for case in manifest.cases:
        for test in case.success_tests:
            kind = str(test["kind"])
            if kind not in {"choice", "score"}:
                continue
            base = question_schema(test)
            for order in ("direct", "reverse"):
                criteria_order = list(
                    base["options"] if kind == "choice" else base["levels"]
                )
                if order == "reverse":
                    criteria_order.reverse()
                for repeat in range(repetitions):
                    request_id = f"order-bias:{case.id}:{test['id']}:{order}:{repeat}"
                    request: dict[str, Any] = {
                        "key": request_id,
                        "case_id": case.id,
                        "model": "typesafe/jev-1.13-20260917",
                        "type": kind,
                        "question": str(test["question"]),
                        "state": {
                            "prompt": case.prompt,
                            "output": case.output,
                            "success_test": dict(test),
                        },
                        "source_group": case.source_group,
                        "test_id": test["id"],
                        "question_schema_digest": question_schema_digest(test),
                        "question_schema": base,
                        "order": order,
                        "repeat_index": repeat,
                    }
                    if kind == "choice":
                        descriptions = test["option_descriptions"]
                        request["criteria"] = {
                            option: descriptions[option] for option in criteria_order
                        }
                    else:
                        request["criteria"] = criteria_order
                    requests.append(request)
    if len(requests) > MAX_QUESTION_EVALUATIONS:
        raise OrderBiasError(
            f"order-bias experiment requires {len(requests)} questions; maximum is {MAX_QUESTION_EVALUATIONS}"
        )
    if not requests:
        raise OrderBiasError(
            "order-bias manifest contains no Choice or Score success tests"
        )
    return tuple(requests)


@dataclass(frozen=True, slots=True)
class OrderBiasArtifact:
    name: str
    dataset_digest: str
    repetitions: int
    groups: tuple[Mapping[str, Any], ...]
    policy_version: str = ORDER_BIAS_POLICY_VERSION
    schema_version: int = ORDER_BIAS_SCHEMA_VERSION
    kind: str = ORDER_BIAS_ARTIFACT_KIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "name": self.name,
            "policy_version": self.policy_version,
            "dataset_digest": self.dataset_digest,
            "repetitions": self.repetitions,
            "legacy_fallback": LEGACY_GRADING_POLICY,
            "groups": [dict(group) for group in self.groups],
        }

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


class OrderBiasPolicy:
    """Resolve one versioned experiment artifact for a runtime success test."""

    def __init__(self, artifact: Mapping[str, Any] | str | Path) -> None:
        if isinstance(artifact, (str, Path)):
            source = Path(artifact)
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise OrderBiasError(
                    f"could not load order-bias policy {source}: {exc}"
                ) from exc
        else:
            value = artifact
        if (
            not isinstance(value, Mapping)
            or value.get("kind") != ORDER_BIAS_ARTIFACT_KIND
        ):
            raise OrderBiasError("grading policy must be an order-bias policy artifact")
        if value.get("schema_version") != ORDER_BIAS_SCHEMA_VERSION:
            raise OrderBiasError("unsupported order-bias policy schema")
        if value.get("policy_version") != ORDER_BIAS_POLICY_VERSION:
            raise OrderBiasError("unsupported order-bias policy version")
        raw_groups = value.get("groups")
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
            raise OrderBiasError("order-bias policy groups must be an array")
        self.artifact = dict(value)
        self._groups: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
        for group in raw_groups:
            if not isinstance(group, Mapping):
                raise OrderBiasError("order-bias policy groups must be objects")
            key = (
                str(group.get("primitive", "")),
                str(group.get("question_schema_digest", "")),
                str(group.get("snapshot", "")),
                str(group.get("evidence_provenance", "matched_recording")),
            )
            if not all(key) or key in self._groups:
                raise OrderBiasError(
                    "order-bias policy group identities must be unique"
                )
            self._groups[key] = dict(group)

    def resolve(
        self, test: Mapping[str, Any], *, snapshot: str | None
    ) -> dict[str, Any]:
        primitive = str(test.get("kind", "")).lower()
        if primitive == "noul":
            return {"policy": "noul_direct", "reason": "outside_order_bias_experiment"}
        schema_digest = question_schema_digest(test)
        if snapshot is None:
            return {
                "policy": LEGACY_GRADING_POLICY,
                "reason": "runtime_snapshot_unavailable",
                "question_schema_digest": schema_digest,
                "snapshot": None,
            }
        group = self._groups.get(
            (primitive, schema_digest, snapshot, "matched_recording")
        )
        if group is None:
            synthetic_group = self._groups.get(
                (primitive, schema_digest, snapshot, "synthetic_known_answer")
            )
            has_schema = any(
                key[0] == primitive and key[1] == schema_digest for key in self._groups
            )
            return {
                "policy": LEGACY_GRADING_POLICY,
                "reason": (
                    "synthetic_only_evidence"
                    if synthetic_group is not None
                    else "incompatible_snapshot"
                    if has_schema
                    else "no_matching_policy_group"
                ),
                "question_schema_digest": schema_digest,
                "snapshot": snapshot,
            }
        recommendation = str(group.get("recommendation", "insufficient_evidence"))
        if not group.get("runtime_eligible"):
            return {
                "policy": LEGACY_GRADING_POLICY,
                "reason": str(group.get("activation_block", "evidence_not_eligible")),
                "question_schema_digest": schema_digest,
                "snapshot": snapshot,
                "policy_version": self.artifact["policy_version"],
                "dataset_digest": self.artifact.get("dataset_digest"),
            }
        if recommendation not in {"single", "mean_pair"}:
            return {
                "policy": LEGACY_GRADING_POLICY,
                "reason": "insufficient_evidence",
                "question_schema_digest": schema_digest,
                "snapshot": snapshot,
                "policy_version": self.artifact["policy_version"],
                "dataset_digest": self.artifact.get("dataset_digest"),
            }
        return {
            "policy": recommendation,
            "reason": "compatible_order_bias_evidence",
            "question_schema_digest": schema_digest,
            "snapshot": snapshot,
            "policy_version": self.artifact["policy_version"],
            "dataset_digest": self.artifact.get("dataset_digest"),
        }


def _distribution(
    answer: Any, request: Mapping[str, Any], *, order: str
) -> tuple[dict[str, float], str] | None:
    kind = str(request["type"])
    try:
        decision = parse_decision(answer)
    except (JevResponseError, TypeError, ValueError):
        return None
    if kind == "choice" and isinstance(decision, ChoiceDecision):
        semantic_options = tuple(
            str(option) for option in request["state"]["success_test"]["options"]
        )
        probabilities = decision.probabilities
        if set(probabilities) != set(semantic_options):
            return None
        winner = next(
            option
            for option in semantic_options
            if probabilities[option] == max(probabilities.values())
        )
        return probabilities, winner
    if kind == "score" and isinstance(decision, ScoreDecision):
        semantic_levels = tuple(
            str(level) for level in request["state"]["success_test"]["levels"]
        )
        raw = decision.probabilities
        if set(raw) != {str(index) for index in range(len(semantic_levels))}:
            return None
        probabilities = {
            semantic_levels[
                index if order == "direct" else len(semantic_levels) - 1 - index
            ]: raw[str(index)]
            for index in range(len(semantic_levels))
        }
        winner = next(
            level
            for level in semantic_levels
            if probabilities[level] == max(probabilities.values())
        )
        return probabilities, winner
    return None


def _expected_mass(test: Mapping[str, Any], distribution: Mapping[str, float]) -> float:
    expected = str(test["expected"])
    if test["kind"] == "choice":
        return distribution.get(expected, 0.0)
    levels = tuple(str(level) for level in test["levels"])
    if expected not in levels:
        return 0.0
    minimum = levels.index(expected)
    return sum(distribution.get(level, 0.0) for level in levels[minimum:])


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[right] * fraction


def _bootstrap_interval(
    values: Sequence[float], *, seed: int, resamples: int
) -> dict[str, float | None]:
    if not values:
        return {"lower": None, "upper": None}
    rng = random.Random(seed)
    size = len(values)
    means = [
        fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    ]
    return {
        "lower": _percentile(means, 0.025),
        "upper": _percentile(means, 0.975),
    }


def _event_index(events: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise OrderBiasError("recording events must be objects")
        request_id = _nonempty_string(
            event.get("request_id"), field_name="recording request_id"
        )
        if request_id in indexed:
            raise OrderBiasError(f"duplicate recording request_id {request_id!r}")
        indexed[request_id] = event
    return indexed


def _event_analysis(
    request: Mapping[str, Any],
    event: Mapping[str, Any] | None,
) -> tuple[float, str, str] | None:
    if event is None or event.get("answer") is None:
        return None
    snapshot = event.get("answered_by", event.get("snapshot"))
    if not isinstance(snapshot, str) or not snapshot.strip():
        snapshot = "unknown"
    parsed = _distribution(event.get("answer"), request, order=str(request["order"]))
    if parsed is None:
        return None
    distribution, winner = parsed
    test = request["state"]["success_test"]
    return _expected_mass(test, distribution), winner, snapshot


def analyze_order_bias(
    manifest: OrderBiasManifest,
    events: Sequence[Mapping[str, Any]],
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    request_cost_ceiling: float = 0.01,
) -> tuple[dict[str, Any], OrderBiasArtifact]:
    requests = build_order_bias_requests(manifest, repetitions=repetitions)
    cost_ceiling = _number(
        request_cost_ceiling, field_name="request_cost_ceiling", minimum=0.000001
    )
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise OrderBiasError("bootstrap_seed must be an integer")
    if (
        not isinstance(bootstrap_resamples, int)
        or isinstance(bootstrap_resamples, bool)
        or bootstrap_resamples < 1
    ):
        raise OrderBiasError("bootstrap_resamples must be a positive integer")
    event_map = _event_index(events)
    requests_by_id = {str(request["key"]): request for request in requests}
    for request_id, event in event_map.items():
        if request_id not in requests_by_id:
            raise OrderBiasError(
                f"recording contains unexpected request {request_id!r}"
            )
        recorded_request = event.get("request")
        if recorded_request != requests_by_id[request_id]:
            raise OrderBiasError(
                f"recorded request {request_id!r} does not match the manifest"
            )

    # Schema/snapshot groups are kept separate. A repeated answer served by a
    # different Jev snapshot cannot be paired with the other observations.
    observations: dict[
        tuple[str, str, str, str, str], dict[str, dict[int, tuple[float, str]]]
    ] = defaultdict(lambda: {"direct": {}, "reverse": {}})
    missing_answers = 0
    unusable_answers = 0
    live_cost = 0.0
    live_input_tokens = 0
    live_output_tokens = 0
    total_latency = 0.0
    latency_count = 0
    cases_by_id = {case.id: case for case in manifest.cases}
    schemas_by_digest = {
        question_schema_digest(test): question_schema(test)
        for case in manifest.cases
        for test in case.success_tests
        if test["kind"] in {"choice", "score"}
    }
    for request in requests:
        request_id = str(request["key"])
        event = event_map.get(request_id)
        analyzed = _event_analysis(request, event)
        if event is None or event.get("answer") is None:
            missing_answers += 1
        elif analyzed is None:
            unusable_answers += 1
        if event is not None:
            cost = event.get("cost_usd", event.get("cost"))
            if cost is not None:
                live_cost += _number(cost, field_name="recorded cost")
            usage = event.get("usage", {})
            if isinstance(usage, Mapping):
                live_input_tokens += int(
                    usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                )
                live_output_tokens += int(
                    usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                )
            latency = event.get("latency_ms")
            if latency is not None:
                total_latency += _number(latency, field_name="recorded latency")
                latency_count += 1
        case_id = str(request["case_id"])
        case = cases_by_id[case_id]
        if analyzed is None:
            continue
        mass, winner, snapshot = analyzed
        group_key = (
            str(request["type"]),
            str(request["question_schema_digest"]),
            snapshot,
            case.provenance,
            case.source_group,
        )
        observations[group_key][str(request["order"])][int(request["repeat_index"])] = (
            mass,
            winner,
        )

    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    source_pairs: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for (
        primitive,
        schema_digest,
        snapshot,
        provenance,
        source_group,
    ), ordered in observations.items():
        direct, reverse = ordered["direct"], ordered["reverse"]
        paired = [
            index
            for index in range(repetitions)
            if index in direct and index in reverse
        ]
        if paired:
            pair_diffs = [reverse[index][0] - direct[index][0] for index in paired]
            pair_abs = [abs(value) for value in pair_diffs]
            pair_flips = [direct[index][1] != reverse[index][1] for index in paired]
            cross_abs = fmean(pair_abs)
            cross_flips = fmean(float(value) for value in pair_flips)
        else:
            pair_diffs, pair_abs, pair_flips = [], [], []
            cross_abs = cross_flips = None
        repeat_abs: list[float] = []
        repeat_flips: list[float] = []
        for values in (direct, reverse):
            indexes = sorted(values)
            for left_index, left in enumerate(indexes):
                for right in indexes[left_index + 1 :]:
                    repeat_abs.append(abs(values[right][0] - values[left][0]))
                    repeat_flips.append(float(values[right][1] != values[left][1]))
        same_abs = fmean(repeat_abs) if repeat_abs else None
        same_flips = fmean(repeat_flips) if repeat_flips else None
        source_pairs[(primitive, schema_digest, snapshot, provenance, source_group)] = {
            "paired": paired,
            "direct": direct,
            "reverse": reverse,
            "signed_shifts": pair_diffs,
            "absolute_shifts": pair_abs,
            "cross_flips": pair_flips,
            "same_order_absolute_differences": repeat_abs,
            "same_order_flips": repeat_flips,
            "cross_abs": cross_abs,
            "cross_flips_rate": cross_flips,
            "same_abs": same_abs,
            "same_flip_rate": same_flips,
        }
        key = (primitive, schema_digest, snapshot, provenance)
        group = grouped.setdefault(
            key,
            {
                "primitive": primitive,
                "question_schema_digest": schema_digest,
                "question_schema": schemas_by_digest[schema_digest],
                "snapshot": snapshot,
                "evidence_provenance": provenance,
                "source_groups": {},
            },
        )
        group["source_groups"][source_group] = source_pairs[
            (primitive, schema_digest, snapshot, provenance, source_group)
        ]

    expected_request_ids = {str(request["key"]) for request in requests}
    absent = expected_request_ids - set(event_map)
    status = (
        "complete"
        if not absent and missing_answers == 0 and unusable_answers == 0
        else "partial"
    )
    experiment_complete = status == "complete"
    report_groups: list[dict[str, Any]] = []
    artifact_groups: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(grouped)):
        group = grouped[key]
        source_values = list(group["source_groups"].values())
        eligible_source_values = [
            value
            for value in source_values
            if value["cross_abs"] is not None
            and value["same_abs"] is not None
            and value["cross_flips_rate"] is not None
            and value["same_flip_rate"] is not None
        ]
        support_groups = len(eligible_source_values)
        paired_count = sum(len(value["paired"]) for value in source_values)
        direct_values = [
            mass
            for value in source_values
            for mass, _winner in value["direct"].values()
        ]
        reverse_values = [
            mass
            for value in source_values
            for mass, _winner in value["reverse"].values()
        ]
        signed = [shift for value in source_values for shift in value["signed_shifts"]]
        absolute = [
            shift for value in source_values for shift in value["absolute_shifts"]
        ]
        flip_values = [flip for value in source_values for flip in value["cross_flips"]]
        repeat_diffs = [
            diff
            for value in source_values
            for diff in value["same_order_absolute_differences"]
        ]
        group_excess_absolute = [
            float(value["cross_abs"] - value["same_abs"])
            for value in eligible_source_values
        ]
        group_excess_flips = [
            float(value["cross_flips_rate"] - value["same_flip_rate"])
            for value in eligible_source_values
        ]
        absolute_interval = _bootstrap_interval(
            group_excess_absolute,
            seed=bootstrap_seed + group_index * 2,
            resamples=bootstrap_resamples,
        )
        flip_interval = _bootstrap_interval(
            group_excess_flips,
            seed=bootstrap_seed + group_index * 2 + 1,
            resamples=bootstrap_resamples,
        )
        if support_groups < MINIMUM_SOURCE_GROUPS:
            recommendation = "insufficient_evidence"
        elif (
            absolute_interval["upper"] is not None
            and flip_interval["upper"] is not None
            and absolute_interval["upper"] <= EXCESS_TOLERANCE
            and flip_interval["upper"] <= EXCESS_TOLERANCE
        ):
            recommendation = "single"
        elif (
            absolute_interval["lower"] is not None
            and absolute_interval["lower"] > EXCESS_TOLERANCE
        ) or (
            flip_interval["lower"] is not None
            and flip_interval["lower"] > EXCESS_TOLERANCE
        ):
            recommendation = "mean_pair"
        else:
            recommendation = "insufficient_evidence"
        provenance = str(group["evidence_provenance"])
        synthetic_only = provenance != "matched_recording"
        runtime_eligible = (
            recommendation in {"single", "mean_pair"}
            and support_groups >= MINIMUM_SOURCE_GROUPS
            and not synthetic_only
            and key[2] != "unknown"
            and experiment_complete
        )
        if recommendation == "insufficient_evidence":
            activation_block = "insufficient_evidence"
        elif synthetic_only:
            activation_block = "synthetic_only_evidence"
        elif key[2] == "unknown":
            activation_block = "answering_snapshot_unavailable"
        elif not experiment_complete:
            activation_block = "incomplete_observations"
        else:
            activation_block = None
        metrics = {
            "mean_expected_mass_direct": fmean(direct_values)
            if direct_values
            else None,
            "mean_expected_mass_reversed": fmean(reverse_values)
            if reverse_values
            else None,
            "mean_signed_shift": fmean(signed) if signed else None,
            "mean_absolute_shift": fmean(absolute) if absolute else None,
            "semantic_argmax_flip_rate": fmean(float(value) for value in flip_values)
            if flip_values
            else None,
            "same_order_repeat_variation": fmean(repeat_diffs)
            if repeat_diffs
            else None,
            "excess_absolute_shift_ci95": absolute_interval,
            "excess_flip_rate_ci95": flip_interval,
        }
        report_groups.append(
            {
                "primitive": group["primitive"],
                "question_schema_digest": group["question_schema_digest"],
                "question_schema": group["question_schema"],
                "snapshot": group["snapshot"],
                "evidence_provenance": provenance,
                "recommendation": recommendation,
                "runtime_eligible": runtime_eligible,
                "activation_block": activation_block,
                "support": {
                    "distinct_source_groups": support_groups,
                    "minimum_source_groups": MINIMUM_SOURCE_GROUPS,
                    "paired_observations": paired_count,
                    "direct_answers": len(direct_values),
                    "reversed_answers": len(reverse_values),
                    "missing_answers": missing_answers,
                    "unusable_answers": unusable_answers,
                },
                "metrics": metrics,
            }
        )
        artifact_groups.append(
            {
                "primitive": group["primitive"],
                "question_schema_digest": group["question_schema_digest"],
                "question_schema": group["question_schema"],
                "snapshot": group["snapshot"],
                "evidence_provenance": provenance,
                "recommendation": recommendation,
                "runtime_eligible": runtime_eligible,
                "activation_block": activation_block,
                "support": {"distinct_source_groups": support_groups},
            }
        )

    matched_groups = {
        case.source_group
        for case in manifest.cases
        if case.provenance == "matched_recording"
    }
    synthetic_groups = {
        case.source_group
        for case in manifest.cases
        if case.provenance == "synthetic_known_answer"
    }
    source_recorded_cost = sum(case.recorded_cost_usd or 0.0 for case in manifest.cases)
    source_recorded_input_tokens = sum(
        case.recorded_input_tokens or 0 for case in manifest.cases
    )
    source_recorded_output_tokens = sum(
        case.recorded_output_tokens or 0 for case in manifest.cases
    )
    total_questions = len(requests)
    report: dict[str, Any] = {
        "schema_version": ORDER_BIAS_SCHEMA_VERSION,
        "kind": ORDER_BIAS_REPORT_KIND,
        "status": status,
        "policy_version": ORDER_BIAS_POLICY_VERSION,
        "manifest": {
            "name": manifest.name,
            "dataset_digest": manifest.digest,
            "cases": len(manifest.cases),
            "repetitions_per_order": repetitions,
            "question_evaluations": total_questions,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "order_bias_tolerance": EXCESS_TOLERANCE,
        },
        "evidence_sources": {
            "synthetic_results": {
                "source_groups": len(synthetic_groups),
                "policy_activation_allowed": False,
            },
            "matched_recordings": {
                "source_groups": len(matched_groups),
                "recorded_cost_usd": source_recorded_cost,
                "recorded_input_tokens": source_recorded_input_tokens,
                "recorded_output_tokens": source_recorded_output_tokens,
            },
        },
        "collection": {
            "recorded_answers": len(event_map),
            "missing_answers": missing_answers,
            "unusable_answers": unusable_answers,
            "live_cost_usd": live_cost,
            "live_input_tokens": live_input_tokens,
            "live_output_tokens": live_output_tokens,
            "mean_latency_ms": total_latency / latency_count if latency_count else None,
            "estimated_live_cost_ceiling_usd": total_questions * cost_ceiling,
            "estimated_live_input_tokens": None,
            "estimated_live_output_tokens": None,
            "estimate_basis": "question count multiplied by the configured per-question cost ceiling; token estimates need provider tokenization",
        },
        "groups": report_groups,
        "legacy_fallback": LEGACY_GRADING_POLICY,
        "production_default_changed": False,
        "runtime_policy_available": any(
            item["runtime_eligible"] for item in artifact_groups
        ),
        "source_limitations": (
            [
                "Synthetic known-answer observations exercise code paths and do not support a production performance claim."
            ]
            if not matched_groups
            else []
        ),
    }
    artifact = OrderBiasArtifact(
        name=manifest.name,
        dataset_digest=manifest.digest,
        repetitions=repetitions,
        groups=tuple(artifact_groups),
    )
    return report, artifact


def _role_cost(usage_report: Mapping[str, Any]) -> float:
    costs = usage_report.get("cost_by_role", {})
    if isinstance(costs, Mapping):
        value = costs.get("judge", 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0
    return 0.0


def capture_order_bias(
    manifest: OrderBiasManifest,
    gateway: Gateway,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    budget_usd: float,
    request_cost_ceiling: float = 0.01,
    run_id: str = "order-bias-experiment",
    recording_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    budget = _number(budget_usd, field_name="live budget", minimum=0.000001)
    ceiling = _number(
        request_cost_ceiling, field_name="request cost ceiling", minimum=0.000001
    )
    requests = build_order_bias_requests(manifest, repetitions=repetitions)
    destination = Path(recording_path) if recording_path is not None else None
    if destination is not None and destination.exists():
        raise OrderBiasError(f"order-bias recording {destination} already exists")
    gateway.new_run(run_id)
    recorded: list[dict[str, Any]] = []
    reserved_cost = 0.0
    for request in requests:
        reserved_cost = max(reserved_cost, _role_cost(gateway.usage_report()))
        if reserved_cost + ceiling > budget + 1e-12:
            break
        before_cost = _role_cost(gateway.usage_report())
        before_log = len(gateway.decision_log)
        started = perf_counter()
        answer = gateway.decide(request, role="judge", run_id=run_id)
        latency_ms = (perf_counter() - started) * 1000
        entries = gateway.decision_log[before_log:]
        decision = next(
            (item for item in reversed(entries) if item.get("question") == request),
            None,
        )
        if decision is None:
            raise OrderBiasError(
                "Gateway did not record the order-bias decision identity"
            )
        answered_by = decision.get("answered_by", gateway.jev_model)
        if answered_by != gateway.jev_model:
            raise OrderBiasError(
                f"order-bias response snapshot {answered_by!r} differs from configured {gateway.jev_model!r}"
            )
        usage = decision.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        event = {
            "request_id": request["key"],
            "request": request,
            "answer": answer,
            "answered_by": answered_by,
            "usage": dict(usage),
            "latency_ms": latency_ms,
            "cost_usd": max(0.0, _role_cost(gateway.usage_report()) - before_cost),
        }
        recorded.append(event)
        reserved_cost += ceiling
        if destination is not None:
            _save_recording(destination, manifest, repetitions, recorded)
    if destination is not None and not recorded:
        _save_recording(destination, manifest, repetitions, recorded)
    return tuple(recorded)


def _save_recording(
    destination: Path,
    manifest: OrderBiasManifest,
    repetitions: int,
    events: Sequence[Mapping[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": ORDER_BIAS_SCHEMA_VERSION,
                "kind": ORDER_BIAS_RECORDING_KIND,
                "dataset_digest": manifest.digest,
                "repetitions": repetitions,
                "events": list(events),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_order_bias_recording(
    path: str | Path,
    manifest: OrderBiasManifest,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
) -> tuple[Mapping[str, Any], ...]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OrderBiasError(
            f"could not load strict order-bias replay {source}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping) or raw.get("kind") != ORDER_BIAS_RECORDING_KIND:
        raise OrderBiasError("replay file is not an order-bias recording")
    if (
        raw.get("dataset_digest") != manifest.digest
        or raw.get("repetitions") != repetitions
    ):
        raise OrderBiasError(
            "order-bias replay does not match the manifest or repetition count"
        )
    events = raw.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise OrderBiasError("order-bias replay events must be an array")
    event_map = _event_index(events)
    requests = build_order_bias_requests(manifest, repetitions=repetitions)
    expected = {str(request["key"]): request for request in requests}
    missing = set(expected) - set(event_map)
    extra = set(event_map) - set(expected)
    if missing:
        raise OrderBiasError(
            f"strict replay is missing {len(missing)} recorded request(s)"
        )
    if extra:
        raise OrderBiasError(
            f"strict replay contains {len(extra)} unexpected request(s)"
        )
    for request_id, request in expected.items():
        if event_map[request_id].get("request") != request:
            raise OrderBiasError(f"strict replay request {request_id!r} does not match")
    return tuple(event_map[str(request["key"])] for request in requests)


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_REPETITIONS",
    "LEGACY_GRADING_POLICY",
    "MAX_CASES",
    "MAX_QUESTION_EVALUATIONS",
    "MINIMUM_SOURCE_GROUPS",
    "ORDER_BIAS_POLICY_VERSION",
    "OrderBiasArtifact",
    "OrderBiasCase",
    "OrderBiasError",
    "OrderBiasManifest",
    "OrderBiasPolicy",
    "analyze_order_bias",
    "build_order_bias_requests",
    "capture_order_bias",
    "load_order_bias_manifest",
    "load_order_bias_recording",
    "question_schema_digest",
]
