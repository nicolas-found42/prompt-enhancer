from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest

from prompt_enhancer.evaluation.__main__ import main as evaluation_main
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    CalibrationBudget,
    CalibrationError,
    CalibrationObservation,
    DecisionPolicy,
    Disposition,
    QuestionIdentity,
    VerdictPolicy,
    calibrate_question,
    capture_live_manifest,
    compute_metrics,
    load_calibration_manifest,
    manifest_from_dict,
    normalize_event,
    partition_groups,
)
from prompt_enhancer.gateway import ScriptedGateway
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


def test_choice_selected_correctness_scores_confidence_in_the_selection() -> None:
    identity = QuestionIdentity(
        question_id="pointer:choice",
        question="Which option?",
        primitive="choice",
        criteria=("right", "wrong"),
        event_mapping={"selected_correctness": True},
        answering_snapshot=_SNAPSHOT,
    )
    event = normalize_event(
        CalibrationObservation(
            event_id="confidently-wrong",
            source_group="source",
            example_id="example",
            identity=identity,
            label="right",
            raw_answer={
                "type": "choice",
                "choice": "wrong",
                "probabilities": {"right": 0.1, "wrong": 0.9},
            },
            provenance="synthetic_known_answer",
            answering_snapshot=_SNAPSHOT,
        )
    )

    assert event.label is False
    assert event.probability == pytest.approx(0.9)
    assert compute_metrics([event], 0.5)["brier"] == pytest.approx(0.81)


def test_negative_polarity_noul_brier_targets_the_no_class() -> None:
    identity = replace(_noul_identity(), event_mapping={"polarity": "negative"})
    event = normalize_event(
        replace(
            _noul_observation("negative", "source", "example", True, 0.05),
            identity=identity,
        )
    )

    metrics = compute_metrics([event], 0.5)
    assert event.probability == pytest.approx(0.95)
    assert metrics["brier"] == pytest.approx(0.0025)
    assert metrics["distribution_brier"] == pytest.approx(0.005)


def test_score_distribution_uses_ordinal_cumulative_brier() -> None:
    identity = QuestionIdentity(
        question_id="fidelity:score",
        question="How faithful?",
        primitive="score",
        criteria=(1, 2, 3, 4),
        event_mapping={"boundary": 3},
        answering_snapshot=_SNAPSHOT,
    )
    event = normalize_event(
        CalibrationObservation(
            event_id="ordinal",
            source_group="source",
            example_id="example",
            identity=identity,
            label=3,
            raw_answer={
                "type": "score",
                "score": 3,
                "levels": {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4},
            },
            provenance="synthetic_known_answer",
            answering_snapshot=_SNAPSHOT,
        )
    )

    metrics = compute_metrics([event], 0.5)
    assert metrics["distribution_brier"] == pytest.approx(0.26 / 3)
    assert metrics["distribution_brier_convention"] == (
        "ranked_probability_score_mean_over_boundaries"
    )


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


def test_same_example_id_in_different_sources_is_not_repeat_evidence() -> None:
    events = [
        normalize_event(
            _noul_observation("first", "source-one", "shared-id", True, 0.1)
        ),
        normalize_event(
            _noul_observation("second", "source-two", "shared-id", True, 0.9)
        ),
    ]

    metrics = compute_metrics(events, 0.5)
    assert metrics["support"]["repeat_examples"] == 0
    assert metrics["repeat_spread"]["largest_within_example"] is None


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
                    "threshold": 0.6,
                    "predicate": {"margin_gte": 0.75},
                }
            },
        }
    )
    policy = DecisionPolicy.from_artifact(artifact)

    def apply(probability: float):
        raw_answer = {
            "type": "noul",
            "probability_true": probability,
        }
        return policy.apply(
            question_id=identity.question_id,
            identity=identity,
            decision=parse_decision(raw_answer),
            raw_answer=raw_answer,
            snapshot=_SNAPSHOT,
        )

    passed = apply(0.9)
    failed = apply(0.8)

    assert passed.disposition == Disposition.GATE_ABOVE_CONFIDENCE.value
    assert passed.may_gate
    assert failed.disposition == Disposition.ABSTAIN.value
    assert not failed.may_gate
    assert failed.reason == "frozen_calibration_predicate_failed"


def test_confidence_subset_must_pass_on_held_out_groups() -> None:
    identity = _noul_identity()
    rows = [
        ("fit-pos", "fit", True, 0.95),
        ("fit-neg", "fit", False, 0.05),
        ("cal-pos", "calibration", True, 0.95),
        ("cal-neg", "calibration", False, 0.05),
        ("eval-pos", "evaluation", True, 0.95),
        ("eval-false-positive", "evaluation", False, 0.96),
        ("eval-neg", "evaluation", False, 0.05),
    ]
    observations = [
        _noul_observation(name, name, name, label, probability, partition=partition)
        for name, partition, label, probability in rows
    ]
    result = calibrate_question(
        identity,
        observations,
        verdict_policy=VerdictPolicy(
            minimum_evaluation_groups=3,
            minimum_positive_examples=1,
            minimum_negative_examples=1,
            require_control=False,
            require_repeats=False,
            require_brier_better_than_control=False,
        ),
        bootstrap_resamples=16,
    )

    assert result.verdict not in {"gate", "gate-above-confidence"}


def test_state_blind_controls_do_not_fit_or_select_the_primary_threshold() -> None:
    identity = _noul_identity()
    primary = [
        _noul_observation(name, name, name, label, probability, partition=partition)
        for name, partition, label, probability in (
            ("fit-pos", "fit", True, 0.9),
            ("fit-neg", "fit", False, 0.1),
            ("cal-pos", "calibration", True, 0.9),
            ("cal-neg", "calibration", False, 0.1),
            ("eval-pos", "evaluation", True, 0.9),
            ("eval-neg", "evaluation", False, 0.1),
        )
    ]
    controls = [
        replace(
            event,
            event_id=f"{event.event_id}-control",
            raw_answer={
                "type": "noul",
                "probability_true": 0.1 if event.label else 0.9,
            },
            control=True,
            answer_id=f"{event.event_id}-control-answer",
        )
        for event in primary
    ]

    without_controls = calibrate_question(
        identity, primary, fit_mode="temperature", bootstrap_resamples=8
    )
    with_controls = calibrate_question(
        identity, [*primary, *controls], fit_mode="temperature", bootstrap_resamples=8
    )

    assert with_controls.fit == without_controls.fit
    assert with_controls.threshold == without_controls.threshold


def test_frozen_confidence_subset_earns_gate_only_on_held_out_support() -> None:
    identity = _noul_identity()
    observations = [
        _noul_observation("fit-pos", "fit-pos", "fit-pos", True, 0.95, partition="fit"),
        _noul_observation("fit-neg", "fit-neg", "fit-neg", False, 0.8, partition="fit"),
    ]
    for partition in ("calibration", "evaluation"):
        for index in range(30):
            probability = 0.95 if index < 2 else 0.69 if index < 5 else 0.8
            name = f"{partition}-{index}"
            observations.append(
                _noul_observation(
                    name,
                    name,
                    name,
                    index < 5,
                    probability,
                    partition=partition,
                )
            )

    result = calibrate_question(
        identity,
        observations,
        verdict_policy=VerdictPolicy(
            require_control=False,
            require_repeats=False,
            require_brier_better_than_control=False,
        ),
        bootstrap_resamples=16,
    )

    assert result.verdict == "gate-above-confidence"
    assert result.predicate["margin_gte"] >= 0.4
    assert result.verdict_components["evaluation_subset"]["coverage"] == 0.9


def test_live_capture_uses_new_requests_for_each_repeat_and_control() -> None:
    manifest = manifest_from_dict(
        {
            "name": "scripted-live",
            "events": [
                {
                    "id": "template",
                    "source_group": "source",
                    "example_id": "example",
                    "question_id": "gap:goal",
                    "question": "Is the goal missing?",
                    "family": "gap",
                    "primitive": "noul",
                    "criteria": ["no", "yes"],
                    "event": {"positive_class": "yes"},
                    "label": True,
                    "label_provenance": "synthetic_known_answer",
                    "answering_snapshot": _SNAPSHOT,
                    "state": {"prompt": "do this"},
                }
            ],
        }
    )
    gateway = ScriptedGateway(
        jev_model=_SNAPSHOT,
        decision=lambda request, **_: {
            "type": "noul",
            "probability_true": 0.5 if request["state"] == {} else 0.9,
        },
    )

    recorded, accounting = capture_live_manifest(
        manifest,
        gateway,
        budget=CalibrationBudget(
            max_source_examples=1,
            max_repeats=3,
            max_question_evaluations=6,
            budget_usd=1.0,
        ),
        runs=3,
        request_cost_ceiling=0.01,
    )

    assert len(gateway.calls) == 6
    assert [call["run_id"] for call in gateway.calls] == [
        event.request_id for event in recorded.events
    ]
    assert len({event.request_id for event in recorded.events}) == 6
    assert len({event.answer_id for event in recorded.events}) == 6
    assert {event.repeat_index for event in recorded.events} == {0, 1, 2}
    assert sum(event.control for event in recorded.events) == 3
    assert all(event.state == {} for event in recorded.events if event.control)
    assert accounting["requests"] == 6
    replayed = manifest_from_dict(recorded.to_dict())
    assert replayed.events == recorded.events


def test_live_capture_stops_before_reserved_budget_is_exhausted() -> None:
    source = load_calibration_manifest(_FIXTURE)
    template = replace(
        source.events[0], raw_answer=None, answer_id=None, request_id=None
    )
    manifest = replace(source, events=(template,))
    gateway = ScriptedGateway(
        jev_model=template.identity.answering_snapshot or _SNAPSHOT,
        decision=lambda *_args, **_kwargs: {
            "type": "noul",
            "probability_true": 0.9,
        },
    )

    recorded, accounting = capture_live_manifest(
        manifest,
        gateway,
        budget=CalibrationBudget(
            budget_usd=0.025,
            max_question_evaluations=6,
        ),
        runs=3,
        request_cost_ceiling=0.01,
    )

    assert len(recorded.events) == 2
    assert accounting["status"] == "partial"
    assert accounting["stopped_reason"] == "budget-reservation"
    assert len(gateway.calls) == 2


def test_runtime_applies_stored_temperature_before_gating() -> None:
    identity = _noul_identity()
    artifact = CalibrationArtifact.from_dict(
        {
            "kind": "calibration-artifact",
            "name": "fitted",
            "input_digest": "known-input",
            "questions": {
                identity.question_id: {
                    "identity": identity.to_dict(),
                    "verdict": "gate",
                    "threshold": 0.8,
                    "fit": {"mode": "temperature", "temperature": 0.5},
                    "predicate": {},
                }
            },
        }
    )
    raw = {"type": "noul", "probability_true": 0.7}

    decision = DecisionPolicy.from_artifact(artifact).apply(
        question_id=identity.question_id,
        identity=identity,
        decision=parse_decision(raw),
        raw_answer=raw,
        snapshot=_SNAPSHOT,
    )

    assert decision.may_gate
    assert decision.evidence["event_probability"] == pytest.approx(0.8448275862)


def test_unavailable_temperature_fit_freezes_an_explicit_no_fit_gate() -> None:
    identity = _noul_identity()
    observations = [
        replace(
            _noul_observation(
                f"fit-{index}",
                f"fit-{index}",
                f"fit-{index}",
                True,
                0.9,
                partition="fit",
            ),
            label=None,
            label_present=False,
        )
        for index in range(2)
    ]
    for partition in ("calibration", "evaluation"):
        for index in range(30):
            positive = index < 5
            name = f"{partition}-{index}"
            observations.append(
                _noul_observation(
                    name,
                    name,
                    name,
                    positive,
                    0.95 if positive else 0.05,
                    partition=partition,
                )
            )

    result = calibrate_question(
        identity,
        observations,
        fit_mode="temperature",
        verdict_policy=VerdictPolicy(
            require_control=False,
            require_repeats=False,
            require_brier_better_than_control=False,
        ),
        bootstrap_resamples=8,
    )
    raw = {"type": "noul", "probability_true": 0.95}
    runtime = DecisionPolicy.from_artifact(result.artifact()).apply(
        question_id=identity.question_id,
        identity=identity,
        decision=parse_decision(raw),
        raw_answer=raw,
        snapshot=_SNAPSHOT,
    )

    assert result.verdict == "gate"
    assert result.fit["mode"] == "none"
    assert result.fit["unavailable_reason"] == "no_usable_fit_events"
    assert runtime.may_gate


def test_runtime_finds_matching_identity_when_manifest_contains_two_versions() -> None:
    current = _noul_identity()
    stale = replace(current, question="Old wording")
    artifact = CalibrationArtifact.from_dict(
        {
            "kind": "calibration-artifact",
            "name": "two-wordings",
            "input_digest": "known-input",
            "questions": {
                current.question_id: {
                    "identity": stale.to_dict(),
                    "verdict": "unusable",
                },
                f"{current.question_id}@{current.identity_digest[:12]}": {
                    "identity": current.to_dict(),
                    "verdict": "gate",
                    "threshold": 0.8,
                },
            },
        }
    )
    raw = {"type": "noul", "probability_true": 0.9}

    decision = DecisionPolicy.from_artifact(artifact).apply(
        question_id=current.question_id,
        identity=current,
        decision=parse_decision(raw),
        raw_answer=raw,
        snapshot=_SNAPSHOT,
    )

    assert decision.may_gate


def test_live_cli_persists_raw_replay_and_partial_budget_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = load_calibration_manifest(_FIXTURE)
    template = replace(source.events[0], raw_answer=None, answer_id=None)
    input_path = tmp_path / "template.json"
    recording_path = tmp_path / "recording.json"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(replace(source, events=(template,)).to_dict()))
    gateway = ScriptedGateway(
        jev_model=template.identity.answering_snapshot or _SNAPSHOT,
        decision=lambda *_args, **_kwargs: {
            "type": "noul",
            "probability_true": 0.9,
        },
    )
    monkeypatch.setattr(
        import_module("prompt_enhancer.evaluation.__main__"),
        "HttpGateway",
        lambda **_kwargs: gateway,
    )

    assert (
        evaluation_main(
            [
                "calibrate",
                str(input_path),
                "--live",
                "--budget",
                "0.025",
                "--runs",
                "3",
                "--request-cost-ceiling",
                "0.01",
                "--record",
                str(recording_path),
                "--output",
                str(report_path),
                "--bootstrap-resamples",
                "8",
            ]
        )
        == 0
    )

    replay = load_calibration_manifest(recording_path)
    report = json.loads(report_path.read_text())
    assert len(replay.events) == 2
    assert report["status"] == "partial"
    assert report["live_capture"]["stopped_reason"] == "budget-reservation"
    assert report["questions"][template.identity.question_id]["verdict"] == (
        "too-few-examples"
    )
