"""Offline failure prediction from persisted optimization evidence.

The module deliberately depends only on the Python standard library and plain
run dictionaries.  It has no gateway, provider client, optimizer, or HTTP
imports, so training cannot issue a model request or mutate runtime state.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

MODEL_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
_ALGORITHM = "gradient_boosted_decision_stumps"


class TrainingDataError(ValueError):
    """Raised when logged runs cannot produce a defensible training set."""


@dataclass(frozen=True)
class TrainingConfig:
    """Deterministic configuration for one offline training job."""

    seed: int = 1729
    holdout_fraction: float = 0.25
    failure_threshold: float = 0.5
    max_stumps: int = 30
    learning_rate: float = 0.1
    min_split_gain: float = 1e-4
    calibration_bins: int = 10
    minimum_training_runs: int = 2
    code_version: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1")
        if not 0 <= self.failure_threshold <= 1:
            raise ValueError("failure_threshold must be between 0 and 1")
        if self.max_stumps < 1:
            raise ValueError("max_stumps must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.min_split_gain < 0:
            raise ValueError("min_split_gain cannot be negative")
        if self.calibration_bins < 1:
            raise ValueError("calibration_bins must be positive")
        if self.minimum_training_runs < 2:
            raise ValueError("minimum_training_runs must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionStump:
    """One deterministic two-leaf regression stump."""

    feature: str
    threshold: float
    direction: str
    missing_left: bool
    learning_rate: float
    left_delta: float = 0.0
    right_delta: float = 0.0

    def goes_left(self, value: float | None) -> bool:
        if value is None:
            return self.missing_left
        if self.direction == "left":
            return value <= self.threshold
        return value > self.threshold

    def applies_to(self, value: float | None) -> bool:
        return self.goes_left(value)

    def delta_for(self, value: float | None) -> float:
        return self.left_delta if self.goes_left(value) else self.right_delta

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionStump:
        direction = str(value["direction"])
        if direction not in {"left", "right"}:
            raise ValueError("stump direction must be 'left' or 'right'")
        learning_rate = _finite_float(value["learning_rate"], "learning rate")
        left_delta = _finite_float(
            value.get("left_delta", learning_rate if direction == "left" else 0.0),
            "left delta",
        )
        right_delta = _finite_float(
            value.get("right_delta", learning_rate if direction == "right" else 0.0),
            "right delta",
        )
        return cls(
            feature=str(value["feature"]),
            threshold=_finite_float(value["threshold"], "stump threshold"),
            direction=direction,
            missing_left=bool(value["missing_left"]),
            learning_rate=learning_rate,
            left_delta=left_delta,
            right_delta=right_delta,
        )


@dataclass(frozen=True)
class LoggedExample:
    """Normalized model input derived from one persisted run."""

    run_id: str
    features: dict[str, float]
    weak_panel_pass_rate: float
    failure_label: int


@dataclass
class FailurePredictionModel:
    """Portable gradient-boosted stump ensemble for failure probabilities."""

    feature_names: tuple[str, ...]
    base_score: float
    stumps: tuple[DecisionStump, ...]
    training_config: dict[str, Any]
    training_run_ids: tuple[str, ...]
    label_counts: dict[str, int]

    def predict(self, features: Mapping[str, Any]) -> float:
        """Return a bounded failure probability for one normalized feature map."""

        score = self.base_score
        for stump in self.stumps:
            score += stump.delta_for(_optional_float(features.get(stump.feature)))
        return _clip(score)

    def predict_many(self, rows: Iterable[Mapping[str, Any]]) -> list[float]:
        return [self.predict(row) for row in rows]

    def contributions(
        self, features: Mapping[str, Any]
    ) -> tuple[dict[str, float], float]:
        """Return per-feature contributions, base score, and clipped output."""

        values: dict[str, float] = defaultdict(float)
        for stump in self.stumps:
            value = _optional_float(features.get(stump.feature))
            delta = stump.delta_for(value)
            if delta:
                values[stump.feature] += delta
        base = self.base_score
        return dict(values), _clip(base + sum(values.values()))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "algorithm": _ALGORITHM,
            "feature_names": list(self.feature_names),
            "base_score": self.base_score,
            "stumps": [stump.to_dict() for stump in self.stumps],
            "training_config": self.training_config,
            "training_run_ids": list(self.training_run_ids),
            "label_counts": dict(sorted(self.label_counts.items())),
        }
        payload["fingerprint"] = "sha256:" + _sha256_json(payload)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailurePredictionModel:
        if int(value.get("schema_version", 0)) != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported failure predictor schema")
        if str(value.get("algorithm")) != _ALGORITHM:
            raise ValueError("unsupported failure predictor algorithm")
        feature_names = tuple(str(name) for name in value["feature_names"])
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("model feature names must be unique")
        stumps = tuple(DecisionStump.from_dict(item) for item in value["stumps"])
        if any(stump.feature not in feature_names for stump in stumps):
            raise ValueError("stump references an unknown feature")
        return cls(
            feature_names=feature_names,
            base_score=_finite_float(value["base_score"], "base score"),
            stumps=stumps,
            training_config=dict(value["training_config"]),
            training_run_ids=tuple(str(item) for item in value["training_run_ids"]),
            label_counts={
                str(key): int(count) for key, count in value["label_counts"].items()
            },
        )


@dataclass(frozen=True)
class TrainingResult:
    """Inspectable in-memory result of one offline training run."""

    model: FailurePredictionModel
    evaluation: dict[str, Any]
    priority_queue: list[dict[str, Any]]
    run_set: dict[str, Any]
    code_version: str

    def to_report(self, artifact: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "offline": True,
            "network_calls": 0,
            "manifest": {
                "run_set": self.run_set,
                "config": self.model.training_config,
                "model": {
                    "algorithm": _ALGORITHM,
                    "feature_names": list(self.model.feature_names),
                    "label_definition": ("weak_panel_pass_rate <= failure_threshold"),
                },
                "code_version": self.code_version,
                "artifact": dict(artifact),
            },
            "evaluation": self.evaluation,
            "priority_queue": self.priority_queue,
        }


def load_logged_runs(path: str | Path) -> list[dict[str, Any]]:
    """Load persisted run dictionaries from a local JSON export.

    Both a top-level list and ``{"runs": [...]}`` are accepted.  The file is
    read only; no run repository or optimizer is opened.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(
            f"could not read logged runs from {source}: {exc}"
        ) from exc
    runs = payload.get("runs") if isinstance(payload, Mapping) else payload
    if not isinstance(runs, list):
        raise TrainingDataError(
            "logged-run JSON must be a list or contain a 'runs' list"
        )
    return [dict(run) for run in runs if isinstance(run, Mapping)]


def train_failure_predictor(
    runs: Iterable[Mapping[str, Any]], config: TrainingConfig | None = None
) -> TrainingResult:
    """Train and evaluate a predictor entirely from logged run dictionaries."""

    settings = config or TrainingConfig()
    examples = _normalize_examples(runs, settings.failure_threshold)
    if len(examples) < settings.minimum_training_runs + 1:
        raise TrainingDataError(
            "at least three uniquely identified runs are required for a held-out split"
        )

    training, held_out = _split_examples(examples, settings)
    feature_names = tuple(
        sorted({name for example in training for name in example.features})
    )
    if not feature_names:
        raise TrainingDataError("training runs contain no Jev or score features")

    model = _fit_model(training, feature_names, settings)
    baseline = fmean(example.failure_label for example in training)
    predictions = [model.predict(example.features) for example in held_out]
    outcomes = [example.failure_label for example in held_out]
    baseline_predictions = [baseline] * len(held_out)
    evaluation = {
        "held_out_runs": [example.run_id for example in held_out],
        "training_runs": [example.run_id for example in training],
        "label_counts": _label_counts(examples),
        "training_label_counts": _label_counts(training),
        "held_out_label_counts": _label_counts(held_out),
        "model": _prediction_metrics(
            outcomes,
            predictions,
            model.feature_names,
            held_out,
            settings.calibration_bins,
        ),
        "baseline": {
            "name": "training_failure_prevalence",
            "probability": baseline,
            **_prediction_metrics(
                outcomes,
                baseline_predictions,
                model.feature_names,
                held_out,
                settings.calibration_bins,
            ),
        },
        "warnings": _training_warnings(model, training, held_out),
    }
    priorities = _build_priority_queue(model, held_out, baseline)
    run_set = {
        "count": len(examples),
        "run_ids": [example.run_id for example in examples],
        "training_run_ids": [example.run_id for example in training],
        "held_out_run_ids": [example.run_id for example in held_out],
        "data_sha256": "sha256:" + _sha256_examples(examples),
    }
    code_version = settings.code_version or "sha256:" + _source_code_version()
    return TrainingResult(
        model=model,
        evaluation=evaluation,
        priority_queue=priorities,
        run_set=run_set,
        code_version=code_version,
    )


def train_and_save(
    runs: Iterable[Mapping[str, Any]],
    artifact_path: str | Path,
    report_path: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Train, write a portable model artifact, and return its manifest report."""

    result = train_failure_predictor(runs, config)
    artifact = Path(artifact_path)
    report = Path(report_path)
    artifact_payload = result.model.to_dict()
    artifact_bytes = _json_bytes(artifact_payload)
    _atomic_write(artifact, artifact_bytes)
    report_payload = result.to_report(
        {
            "path": str(artifact),
            "sha256": "sha256:" + sha256(artifact_bytes).hexdigest(),
            "schema_version": MODEL_SCHEMA_VERSION,
        }
    )
    _atomic_write(report, _json_bytes(report_payload))
    return report_payload


def _normalize_examples(
    runs: Iterable[Mapping[str, Any]], failure_threshold: float
) -> list[LoggedExample]:
    examples: list[LoggedExample] = []
    seen: set[str] = set()
    for index, raw_run in enumerate(runs):
        run = _as_mapping(raw_run)
        run_id = str(run.get("run_id") or run.get("id") or "").strip()
        if not run_id:
            raise TrainingDataError(f"run at index {index} has no run_id")
        if run_id in seen:
            raise TrainingDataError(f"duplicate run_id in training data: {run_id}")
        seen.add(run_id)
        pass_rate = _extract_weak_panel_pass_rate(run)
        if pass_rate is None:
            raise TrainingDataError(f"run {run_id} has no original weak-panel outcome")
        if not 0 <= pass_rate <= 1:
            raise TrainingDataError(
                f"run {run_id} weak-panel pass rate is outside [0, 1]"
            )
        features = extract_failure_features(run)
        if not features:
            raise TrainingDataError(
                f"run {run_id} has no Jev probabilities or score summaries"
            )
        examples.append(
            LoggedExample(
                run_id=run_id,
                features=features,
                weak_panel_pass_rate=pass_rate,
                failure_label=int(pass_rate <= failure_threshold),
            )
        )
    return sorted(examples, key=lambda item: item.run_id)


def _extract_features(run: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    probability_values: list[float] = []
    result_value = run.get("result")
    result: Mapping[str, Any] = (
        result_value if isinstance(result_value, Mapping) else {}
    )
    report_value = run.get("report")
    report: Mapping[str, Any] = (
        report_value if isinstance(report_value, Mapping) else {}
    )
    result_report_value = result.get("report")
    result_report: Mapping[str, Any] = (
        result_report_value if isinstance(result_report_value, Mapping) else {}
    )
    jev_answers = _answer_sequence(
        run.get("jev_answers")
        or run.get("answers")
        or report.get("jev_answers")
        or result.get("jev_answers")
        or result_report.get("jev_answers")
    )
    for index, raw_answer in enumerate(jev_answers):
        logged = _as_mapping(raw_answer)
        answer = (
            _as_mapping(logged.get("answer"))
            if isinstance(logged.get("answer"), Mapping)
            else logged
        )
        question = (
            _as_mapping(logged.get("question"))
            if isinstance(logged.get("question"), Mapping)
            else {}
        )
        kind = (
            str(
                answer.get("kind")
                or answer.get("answer_type")
                or answer.get("type")
                or "unknown"
            )
            .strip()
            .lower()
        )
        question_id = _identifier(
            question.get("key")
            or answer.get("question_id")
            or answer.get("question")
            or answer.get("id")
            or f"answer_{index + 1}"
        )
        prefix = f"jev/{kind}/{question_id}"
        explicit = _answer_probability(answer)
        if explicit is not None:
            features[f"{prefix}/probability"] = explicit
            probability_values.append(explicit)
        confidence = _optional_float(answer.get("confidence"))
        if confidence is not None:
            features[f"{prefix}/confidence"] = confidence
        for name, value in _probability_map(answer.get("probabilities")).items():
            features[f"{prefix}/{name}"] = value
            probability_values.append(value)
        for name, value in _probability_map(answer.get("level_probabilities")).items():
            features[f"{prefix}/level/{name}"] = value
            probability_values.append(value)

    _add_score_summary_features(
        run.get("score_summaries") or run.get("scores"), "score", features
    )
    if probability_values:
        features["jev/all/mean"] = fmean(probability_values)
        features["jev/all/spread"] = (
            pstdev(probability_values) if len(probability_values) > 1 else 0.0
        )
    if not any("/level/" in name for name in features):
        _add_score_summary_features(jev_answers, "jev/score", features)
    return dict(sorted(features.items()))


def extract_failure_features(run: Mapping[str, Any]) -> dict[str, float]:
    """Expose optional features from persisted runs without filling in unknowns.

    Historical runs did not record sentence-existence decisions. Their feature
    maps therefore omit those values; a measured low probability remains a
    numeric feature when present.
    """

    return _extract_features(_as_mapping(run))


def _answer_probability(answer: Mapping[str, Any]) -> float | None:
    for key in ("noul", "noul_probability", "probability_true", "probability", "value"):
        if key not in answer:
            continue
        value = answer[key]
        if isinstance(value, Mapping):
            value = value.get("probability", value.get("value"))
        result = _optional_float(value)
        if result is not None:
            return result
    return None


def _probability_map(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for raw_key, raw_value in value.items():
            if isinstance(raw_value, Mapping):
                raw_value = raw_value.get(
                    "probability", raw_value.get("value", raw_value.get("score"))
                )
            probability = _optional_float(raw_value)
            if probability is not None:
                result[_identifier(raw_key)] = probability
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = {}
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                name = _identifier(
                    item.get("id") or item.get("label") or item.get("name") or index
                )
                probability = _optional_float(
                    item.get("probability", item.get("value", item.get("score")))
                )
            else:
                name = str(index)
                probability = _optional_float(item)
            if probability is not None:
                result[name] = probability
        return result
    return {}


def _add_score_summary_features(
    value: Any, prefix: str, features: dict[str, float]
) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        looks_like_summary = any(
            key in value
            for key in ("mean", "spread", "std", "min", "max", "probabilities")
        )
        if looks_like_summary:
            name = _identifier(value.get("name") or value.get("id") or "summary")
            _add_one_score_summary(f"{prefix}/{name}", value, features)
        else:
            for raw_name, child in sorted(value.items(), key=lambda item: str(item[0])):
                child_prefix = f"{prefix}/{_identifier(raw_name)}"
                if isinstance(child, (int, float)) and not isinstance(child, bool):
                    features[f"{child_prefix}/mean"] = _finite_float(
                        child, f"score summary {child_prefix}"
                    )
                elif isinstance(child, Mapping):
                    _add_one_score_summary(child_prefix, child, features)
                else:
                    _add_score_summary_features(child, child_prefix, features)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            item_prefix = f"{prefix}/{index}"
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                features[f"{item_prefix}/mean"] = _finite_float(
                    item, f"score summary {item_prefix}"
                )
            else:
                _add_score_summary_features(item, item_prefix, features)


def _add_one_score_summary(
    name: str, value: Mapping[str, Any], features: dict[str, float]
) -> None:
    mean = _optional_float(value.get("mean"))
    if mean is None:
        probabilities = list(_probability_map(value.get("probabilities")).values())
        if probabilities:
            mean = fmean(probabilities)
    if mean is not None:
        features[f"{name}/mean"] = mean
    spread = _optional_float(value.get("spread", value.get("std")))
    if spread is None:
        minimum = _optional_float(value.get("min"))
        maximum = _optional_float(value.get("max"))
        if minimum is not None and maximum is not None:
            spread = maximum - minimum
    if spread is not None:
        features[f"{name}/spread"] = abs(spread)
    if spread is None and mean is not None:
        features[f"{name}/spread"] = 0.0


def _extract_weak_panel_pass_rate(run: Mapping[str, Any]) -> float | None:
    for key in (
        "original_weak_panel",
        "original_outcome",
        "original_weak_panel_outcome",
        "weak_panel_outcome",
    ):
        rate = _pass_rate_from_container(run.get(key))
        if rate is not None:
            return rate
    for key in ("original", "weak_panel", "grades", "results"):
        container = run.get(key)
        rate = _pass_rate_from_container(container)
        if rate is not None:
            return rate
        original = _find_original(container)
        if original is not None:
            rate = _pass_rate_from_container(original)
            if rate is not None:
                return rate
    for key in ("candidates", "score_summaries"):
        original = _find_original(run.get(key))
        if original is not None:
            rate = _pass_rate_from_container(original)
            if rate is not None:
                return rate
    return None


def _pass_rate_from_container(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    for key in (
        "pass_rate",
        "mean_pass_rate",
        "average_pass_rate",
        "weak_panel_pass_rate",
    ):
        rate = _optional_float(value.get(key))
        if rate is not None:
            return _clip(rate)
    for key in ("outcomes", "model_pass_rates", "per_model", "models"):
        rates = _pass_rates(value.get(key))
        if rates:
            return fmean(rates)
    return None


def _pass_rates(value: Any) -> list[float]:
    rates: list[float] = []
    if isinstance(value, Mapping):
        items = value.values()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        items = []
    for item in items:
        if isinstance(item, bool):
            rates.append(float(item))
        elif isinstance(item, Mapping):
            rate = _outcome_rate(item)
            if rate is not None:
                rates.append(rate)
        else:
            rate = _optional_float(item)
            if rate is not None:
                rates.append(rate)
    return [_clip(rate) for rate in rates]


def _outcome_rate(item: Mapping[str, Any]) -> float | None:
    if "passed" in item:
        passed = item["passed"]
        if isinstance(passed, bool):
            return float(passed)
        rate = _optional_float(passed)
        if rate is not None:
            return rate
    return _optional_float(item.get("pass_rate", item.get("score")))


def _find_original(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if value.get("original") is True or value.get("is_original") is True:
            return value
        if str(value.get("role", "")).lower() == "original":
            return value
        for key in ("original", "candidates", "results", "grades"):
            found = _find_original(value.get(key))
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found = _find_original(item)
            if found is not None:
                return found
    return None


def _split_examples(
    examples: list[LoggedExample], config: TrainingConfig
) -> tuple[list[LoggedExample], list[LoggedExample]]:
    count = len(examples)
    holdout_count = max(1, round(count * config.holdout_fraction))
    holdout_count = min(holdout_count, count - config.minimum_training_runs)
    ranked = sorted(
        examples,
        key=lambda item: sha256(f"{config.seed}:{item.run_id}".encode()).hexdigest(),
    )
    selected: set[str] = set()
    for label in (0, 1):
        if len(selected) >= holdout_count:
            break
        for example in ranked:
            if example.failure_label == label and example.run_id not in selected:
                selected.add(example.run_id)
                break
    for example in ranked:
        if len(selected) >= holdout_count:
            break
        selected.add(example.run_id)
    held_out = [item for item in examples if item.run_id in selected]
    training = [item for item in examples if item.run_id not in selected]
    if len(training) < config.minimum_training_runs or not held_out:
        raise TrainingDataError(
            "configuration does not produce a non-empty held-out split"
        )
    return training, held_out


def _fit_model(
    training: list[LoggedExample],
    feature_names: tuple[str, ...],
    config: TrainingConfig,
) -> FailurePredictionModel:
    labels = [float(example.failure_label) for example in training]
    predictions = [fmean(labels)] * len(training)
    stumps: list[DecisionStump] = []
    for _ in range(config.max_stumps):
        residuals = [
            label - prediction
            for label, prediction in zip(labels, predictions, strict=True)
        ]
        stump, gain = _best_stump(
            training, feature_names, residuals, config.learning_rate
        )
        if stump is None or gain <= config.min_split_gain:
            break
        stumps.append(stump)
        for index, example in enumerate(training):
            predictions[index] += stump.delta_for(example.features.get(stump.feature))
    counts = Counter(str(example.failure_label) for example in training)
    return FailurePredictionModel(
        feature_names=feature_names,
        base_score=fmean(labels),
        stumps=tuple(stumps),
        training_config={
            **config.to_dict(),
            "algorithm": _ALGORITHM,
            "feature_source": ["jev_probabilities", "score_summaries"],
            "label_source": "original_weak_panel_pass_rate",
        },
        training_run_ids=tuple(example.run_id for example in training),
        label_counts={label: counts.get(label, 0) for label in ("0", "1")},
    )


def _best_stump(
    training: list[LoggedExample],
    feature_names: tuple[str, ...],
    residuals: list[float],
    learning_rate: float,
) -> tuple[DecisionStump | None, float]:
    total_sse = sum(residual * residual for residual in residuals)
    best: DecisionStump | None = None
    best_sse = total_sse
    for feature in feature_names:
        values = sorted(
            {
                example.features[feature]
                for example in training
                if example.features.get(feature) is not None
            }
        )
        for lower, upper in pairwise(values):
            threshold = (lower + upper) / 2
            for direction in ("left", "right"):
                for missing_left in (False, True):
                    left_residuals: list[float] = []
                    right_residuals: list[float] = []
                    for residual, example in zip(residuals, training, strict=True):
                        value = example.features.get(feature)
                        goes_left = (
                            missing_left
                            if value is None
                            else (
                                value <= threshold
                                if direction == "left"
                                else value > threshold
                            )
                        )
                        (left_residuals if goes_left else right_residuals).append(
                            residual
                        )
                    if not left_residuals or not right_residuals:
                        continue
                    left_delta = learning_rate * fmean(left_residuals)
                    right_delta = learning_rate * fmean(right_residuals)
                    sse = sum(
                        (residual - left_delta) ** 2 for residual in left_residuals
                    ) + sum(
                        (residual - right_delta) ** 2 for residual in right_residuals
                    )
                    if sse < best_sse - 1e-15:
                        best_sse = sse
                        best = DecisionStump(
                            feature=feature,
                            threshold=threshold,
                            direction=direction,
                            missing_left=missing_left,
                            learning_rate=learning_rate,
                            left_delta=left_delta,
                            right_delta=right_delta,
                        )
    if best is None:
        return None, 0.0
    return best, total_sse - best_sse


def _prediction_metrics(
    outcomes: list[int],
    predictions: list[float],
    feature_names: tuple[str, ...],
    held_out: list[LoggedExample],
    bins: int,
) -> dict[str, Any]:
    observed = [float(value) for value in outcomes]
    positive_count = sum(1 for value in observed if value == 1)
    negative_count = len(observed) - positive_count
    missing_counts = {
        feature: sum(1 for example in held_out if feature not in example.features)
        for feature in feature_names
    }
    complete_count = sum(
        1
        for example in held_out
        if all(feature in example.features for feature in feature_names)
    )
    observed_feature_cells = sum(
        1
        for example in held_out
        for feature in feature_names
        if feature in example.features
    )
    total_feature_cells = len(held_out) * len(feature_names)
    return {
        "discrimination": {
            "roc_auc": _roc_auc(observed, predictions),
            "positive_count": positive_count,
            "negative_count": negative_count,
        },
        "calibration": {
            "brier_score": fmean(
                (probability - actual) ** 2
                for probability, actual in zip(predictions, observed, strict=True)
            ),
            "expected_calibration_error": _expected_calibration_error(
                observed, predictions, bins
            ),
            "bins": _calibration_bins(observed, predictions, bins),
        },
        "coverage": {
            "eligible_runs": len(held_out),
            "scored_runs": len(predictions),
            "sample_fraction": 1.0 if held_out else 0.0,
            "complete_feature_runs": complete_count,
            "complete_feature_fraction": (
                complete_count / len(held_out) if held_out else 0.0
            ),
            "observed_feature_fraction": (
                observed_feature_cells / total_feature_cells
                if total_feature_cells
                else 0.0
            ),
            "missing_feature_counts": dict(sorted(missing_counts.items())),
        },
        "predicted_failure_rate": fmean(predictions) if predictions else 0.0,
    }


def _roc_auc(outcomes: list[float], predictions: list[float]) -> float | None:
    if not outcomes or len(set(predictions)) <= 1:
        return None
    positives = sum(outcome == 1 for outcome in outcomes)
    negatives = len(outcomes) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(
        (
            (prediction, outcome)
            for prediction, outcome in zip(predictions, outcomes, strict=True)
        ),
        key=lambda item: (item[0], item[1]),
    )
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positive_rank_sum += average_rank * sum(item[1] for item in ranked[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def _expected_calibration_error(
    outcomes: list[float], predictions: list[float], bins: int
) -> float:
    if not outcomes:
        return 0.0
    error = 0.0
    for calibration_bin in _calibration_bins(outcomes, predictions, bins):
        if calibration_bin["count"] == 0:
            continue
        error += (
            calibration_bin["count"]
            / len(outcomes)
            * abs(
                calibration_bin["mean_prediction"]
                - calibration_bin["observed_failure_rate"]
            )
        )
    return error


def _calibration_bins(
    outcomes: list[float], predictions: list[float], bins: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for outcome, prediction in zip(outcomes, predictions, strict=True):
        bounded = _clip(prediction)
        index = min(int(bounded * bins), bins - 1)
        grouped[index].append((bounded, outcome))
    result = []
    for index in range(bins):
        values = grouped.get(index, [])
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(values),
                "mean_prediction": (
                    fmean(item[0] for item in values) if values else None
                ),
                "observed_failure_rate": (
                    fmean(item[1] for item in values) if values else None
                ),
            }
        )
    return result


def _build_priority_queue(
    model: FailurePredictionModel,
    held_out: list[LoggedExample],
    baseline: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in held_out:
        prediction = model.predict(example.features)
        contributions, clipped_prediction = model.contributions(example.features)
        if example.failure_label == 1 and prediction < 0.5:
            reason = "missed_failure"
        elif example.failure_label == 0 and prediction >= 0.5:
            reason = "false_alarm"
        elif example.failure_label == 1:
            reason = "detected_failure"
        else:
            reason = "detected_success"
        top_contributions = [
            {
                "feature": feature,
                "value": example.features.get(feature),
                "contribution": contribution,
            }
            for feature, contribution in sorted(
                contributions.items(), key=lambda item: (-item[1], item[0])
            )[:3]
        ]
        rows.append(
            {
                "run_id": example.run_id,
                "reason": reason,
                "actual_failure": example.failure_label,
                "predicted_failure": clipped_prediction,
                "baseline_failure": baseline,
                "signed_error": example.failure_label - clipped_prediction,
                "absolute_error": abs(example.failure_label - clipped_prediction),
                "top_contributions": top_contributions,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["reason"] == "missed_failure",
            row["absolute_error"],
            row["run_id"],
        ),
        reverse=True,
    )


def _training_warnings(
    model: FailurePredictionModel,
    training: list[LoggedExample],
    held_out: list[LoggedExample],
) -> list[str]:
    warnings = []
    if not model.stumps:
        warnings.append(
            "Training data did not support a useful split; model is a constant prior."
        )
    if len({example.failure_label for example in training}) < 2:
        warnings.append(
            "Training split contains only one failure label; discrimination is unavailable."
        )
    if len({example.failure_label for example in held_out}) < 2:
        warnings.append(
            "Held-out split contains only one failure label; discrimination is unavailable."
        )
    return warnings


def _label_counts(examples: Sequence[LoggedExample]) -> dict[str, int]:
    counts = Counter(example.failure_label for example in examples)
    return {"0": counts.get(0, 0), "1": counts.get(1, 0)}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        dumped = as_dict()
        if isinstance(dumped, Mapping):
            return dumped
    raise TrainingDataError(f"expected a mapping-like run, got {type(value).__name__}")


def _answer_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _identifier(value: Any) -> str:
    if isinstance(value, Mapping):
        value = (
            value.get("id")
            or value.get("name")
            or value.get("label")
            or value.get("text")
        )
    result = str(value).strip()
    return result or "unknown"


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _finite_float(value: Any, name: str) -> float:
    result = _optional_float(value)
    if result is None:
        raise ValueError(f"{name} must be a finite number")
    return result


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _sha256_examples(examples: Sequence[LoggedExample]) -> str:
    canonical = [
        {
            "run_id": example.run_id,
            "features": dict(sorted(example.features.items())),
            "weak_panel_pass_rate": example.weak_panel_pass_rate,
            "failure_label": example.failure_label,
        }
        for example in examples
    ]
    return _sha256_json(canonical)


def _source_code_version() -> str:
    source = Path(__file__).read_bytes()
    return sha256(source).hexdigest()
