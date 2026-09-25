from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from prompt_enhancer.evaluation.__main__ import main as evaluation_main
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    CalibrationError,
    CalibrationObservation,
    DecisionPolicy,
    Disposition,
    QuestionIdentity,
    VerdictPolicy,
    calibrate_question,
    compute_metrics,
    load_calibration_manifest,
    normalize_event,
    partition_groups,
)
from prompt_enhancer.jev import parse_decision


_FIXTURE = (
    Path(__file__).parent / "fixtures" / "evaluation" / "calibration_known_answer.json"
)
_SNAPSHOT = "test-jev-snapshot"


def _noul_identity() -> QuestionIdentity:
    return QuestionIdentity(
        question_id="gap:goal",
        question="Is the required piece 'goal' missing?",
        primitive="noul",
        criteria=("no", "yes"),
        event_mapping={"positive_class": "yes"},
        family="gap",
        rubric_version="default-v1",
        answering_snapshot=_SNAPSHOT,
    )


def _noul_observation(
    event_id: str,
    source_group: str,
    example_id: str,
    label: bool,
    probability: float,
    *,
    repeat_index: int = 0,
    partition: str | None = None,
    answer_id: str | None = None,
) -> CalibrationObservation:
    identity = _noul_identity()
    return CalibrationObservation(
        event_id=event_id,
        source_group=source_group,
        example_id=example_id,
        identity=identity,
        label=label,
        raw_answer={"type": "noul", "probability_true": probability},
        provenance="synthetic_known_answer",
        answering_snapshot=_SNAPSHOT,
        repeat_index=repeat_index,
        answer_id=answer_id,
        partition=partition,
    )


def test_calibration_cli_replay_writes_identical_report_and_artifact(
    tmp_path: Path,
) -> None:
    report_a = tmp_path / "first-report.json"
    artifact_a = tmp_path / "first-artifact.json"
    report_b = tmp_path / "second-report.json"
    artifact_b = tmp_path / "second-artifact.json"
    command = [
        "calibrate",
        str(_FIXTURE),
        "--bootstrap-resamples",
        "8",
    ]

    assert (
        evaluation_main(
            [*command, "--output", str(report_a), "--artifact", str(artifact_a)]
        )
        == 0
    )
    assert (
        evaluation_main(
            [*command, "--output", str(report_b), "--artifact", str(artifact_b)]
        )
        == 0
    )

    assert report_a.read_bytes() == report_b.read_bytes()
    assert artifact_a.read_bytes() == artifact_b.read_bytes()


def test_missing_answers_and_missing_choice_classes_remain_unavailable() -> None:
    manifest = load_calibration_manifest(_FIXTURE)
    noul = next(
        event for event in manifest.events if event.identity.primitive == "noul"
    )
    missing = normalize_event(
        replace(
            noul,
            event_id="missing-answer",
            raw_answer=None,
            answer_id=None,
        )
    )
    malformed = normalize_event(
        replace(
            noul,
            event_id="malformed-answer",
            raw_answer={"type": "noul", "probability_true": "not-a-probability"},
            answer_id=None,
        )
    )

    assert not missing.usable
    assert missing.unavailable_reason == "missing_answer"
    assert missing.probability is None
    assert not malformed.usable
    assert malformed.unavailable_reason == "malformed_answer"
    assert malformed.probability is None
    answer_metrics = compute_metrics([missing, malformed], threshold=0.5)
    assert answer_metrics["support"]["usable_labeled_events"] == 0
    assert answer_metrics["missing_answer_count"] == 2
    assert answer_metrics["brier"] is None

    choice = next(
        event for event in manifest.events if event.identity.primitive == "choice"
    )
    incomplete_distribution = normalize_event(
        replace(
            choice,
            event_id="choice-missing-declared-class",
            raw_answer={
                "type": "choice",
                "choice": "s0001",
                "probabilities": {"s0001": 0.7, "s0002": 0.3},
            },
            answer_id=None,
        )
    )
    assert incomplete_distribution.usable
    assert incomplete_distribution.distribution_unavailable_reason == (
        "missing_class_coverage"
    )
    distribution_metrics = compute_metrics([incomplete_distribution], threshold=0.5)
    assert distribution_metrics["distribution_brier"] is None
    assert distribution_metrics["distribution_brier_reason"] == (
        "missing_class_coverage"
    )

    missing_target = normalize_event(
        replace(
            choice,
            event_id="choice-missing-target-class",
            raw_answer={
                "type": "choice",
                "choice": "s0002",
                "probabilities": {"s0002": 0.8, "none": 0.2},
            },
            answer_id=None,
        )
    )
    assert not missing_target.usable
    assert missing_target.unavailable_reason == "missing_expected_class"
    assert missing_target.distribution_unavailable_reason == "missing_class_coverage"
    assert missing_target.probability is None


def test_cached_raw_answer_without_new_identity_is_not_an_independent_repeat() -> None:
    identity = _noul_identity()
    first = _noul_observation(
        "repeat-0", "source-1", "example-1", True, 0.9, repeat_index=0
    )
    cached_repeat = _noul_observation(
        "repeat-1", "source-1", "example-1", True, 0.9, repeat_index=1
    )

    with pytest.raises(CalibrationError, match="duplicate answer identity"):
        calibrate_question(identity, [first, cached_repeat])


def test_source_group_partitions_are_stable_and_disjoint() -> None:
    groups = [f"source-{index}" for index in range(12)]
    assignments = partition_groups([*groups, groups[0]], seed=41)
    replayed = partition_groups(list(reversed(groups)), seed=41)

    assert assignments == replayed
    assert set(assignments) == set(groups)
    assert set(assignments.values()) == {"fit", "calibration", "evaluation"}
    by_partition = {
        name: {group for group, assignment in assignments.items() if assignment == name}
        for name in ("fit", "calibration", "evaluation")
    }
    assert by_partition["fit"].isdisjoint(by_partition["calibration"])
    assert by_partition["fit"].isdisjoint(by_partition["evaluation"])
    assert by_partition["calibration"].isdisjoint(by_partition["evaluation"])
    assert (
        by_partition["fit"] | by_partition["calibration"] | by_partition["evaluation"]
    ) == set(groups)


def test_evaluation_repeat_range_straddling_cutoff_prevents_a_gate() -> None:
    identity = _noul_identity()
    observations = [
        _noul_observation(
            "fit-positive",
            "fit-positive",
            "fit-positive",
            True,
            0.95,
            partition="fit",
        ),
        _noul_observation(
            "fit-negative",
            "fit-negative",
            "fit-negative",
            False,
            0.05,
            partition="fit",
        ),
        _noul_observation(
            "calibration-positive",
            "calibration-positive",
            "calibration-positive",
            True,
            0.95,
            partition="calibration",
        ),
        _noul_observation(
            "calibration-negative",
            "calibration-negative",
            "calibration-negative",
            False,
            0.05,
            partition="calibration",
        ),
        _noul_observation(
            "evaluation-repeat-low",
            "evaluation-repeat",
            "unstable-example",
            True,
            0.94,
            repeat_index=0,
            partition="evaluation",
        ),
        _noul_observation(
            "evaluation-repeat-high",
            "evaluation-repeat",
            "unstable-example",
            True,
            0.96,
            repeat_index=1,
            partition="evaluation",
        ),
        _noul_observation(
            "evaluation-negative",
            "evaluation-negative",
            "stable-negative",
            False,
            0.05,
            partition="evaluation",
        ),
    ]
    result = calibrate_question(
        identity,
        observations,
        verdict_policy=VerdictPolicy(
            minimum_evaluation_groups=2,
            minimum_positive_examples=1,
            minimum_negative_examples=1,
            precision_floor=1.0,
            recall_floor=0.5,
            coverage_floor=1.0,
            minimum_repeat_examples=1,
            require_control=False,
            require_repeats=True,
            require_brier_better_than_control=False,
        ),
        bootstrap_resamples=8,
    )
    components = result.verdict_components["gate_components"]

    assert result.threshold == 0.95
    assert components["precision"] == 1.0
    assert components["recall"] == 0.5
    assert components["coverage"] == 1.0
    assert components["no_repeat_range_straddles_cutoff"] is False
    assert components["passes"] is False
    assert result.verdict != "gate"


def test_gate_above_confidence_predicate_can_pass_or_abstain() -> None:
    identity = _noul_identity()
    artifact = CalibrationArtifact.from_dict(
        {
            "kind": "calibration-artifact",
            "name": "confidence-policy",
            "input_digest": "known-input",
            "questions": {
                identity.question_id: {
                    "identity": identity.to_dict(),
                    "verdict": "gate-above-confidence",
                    "threshold": 0.8,
                    "predicate": {"confidence_gte": 0.75},
                }
            },
        }
    )
    policy = DecisionPolicy.from_artifact(artifact)

    def apply(confidence: float):
        raw_answer = {
            "type": "noul",
            "probability_true": 0.9,
            "confidence": confidence,
        }
        return policy.apply(
            question_id=identity.question_id,
            identity=identity,
            decision=parse_decision(raw_answer),
            raw_answer=raw_answer,
            snapshot=_SNAPSHOT,
        )

    passed = apply(0.8)
    failed = apply(0.7)

    assert passed.disposition == Disposition.GATE_ABOVE_CONFIDENCE.value
    assert passed.may_gate
    assert failed.disposition == Disposition.ABSTAIN.value
    assert not failed.may_gate
    assert failed.reason == "frozen_calibration_predicate_failed"
