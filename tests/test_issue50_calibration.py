from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.diagnosis import default_gap_question
from prompt_enhancer.evaluation.__main__ import main as evaluation_main
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    CalibrationObservation,
    DecisionPolicy,
    EventSpec,
    QuestionIdentity,
    VerdictPolicy,
    calibrate_manifest,
    calibrate_question,
    compute_metrics,
    load_calibration_manifest,
    normalize_event,
)
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def _identity(question_id: str = "gap:goal") -> QuestionIdentity:
    return QuestionIdentity(
        question_id=question_id,
        criteria=("no", "yes"),
        family="gap",
        question=default_gap_question("goal"),
        primitive="noul",
        event_mapping={"positive_class": "yes"},
        rubric_version="default-v1",
        answering_snapshot=JEV_MODEL,
    )


def _observation(
    index: int,
    label: bool,
    *,
    probability: float | None = None,
    control: bool = False,
    repeat_index: int = 0,
) -> CalibrationObservation:
    identity = _identity()
    return CalibrationObservation(
        event_id=f"event-{index}-{repeat_index}",
        source_group=f"group-{index}",
        example_id=f"example-{index}",
        identity=identity,
        label=label,
        raw_answer=(
            None
            if probability is None
            else {"type": "noul", "probability_true": probability}
        ),
        provenance="user_delegated_model",
        answering_snapshot=JEV_MODEL,
        repeat_index=repeat_index,
        control=control,
        state={} if control else {"prompt": "sample prompt"},
        answer_id=f"answer-{index}-{repeat_index}-{control}",
    )


def test_choice_and_score_adapters_preserve_declared_distributions() -> None:
    choice = normalize_event(
        CalibrationObservation(
            event_id="choice",
            source_group="source",
            example_id="choice",
            identity=QuestionIdentity(
                question_id="pointer:vagueness",
                question="Which sentence?",
                primitive="choice",
                criteria=("s1", "s2", "none"),
                event_mapping={"expected_class": "s1"},
                answering_snapshot=JEV_MODEL,
            ),
            label="s1",
            raw_answer={
                "type": "choice",
                "choice": "s1",
                "probabilities": {"s1": 0.6, "s2": 0.1, "none": 0.3},
            },
            provenance="human",
            answering_snapshot=JEV_MODEL,
        )
    )
    score = normalize_event(
        CalibrationObservation(
            event_id="score",
            source_group="source",
            example_id="score",
            identity=QuestionIdentity(
                question_id="fidelity:score",
                question="How faithful?",
                primitive="score",
                criteria=(1, 2, 3, 4),
                event_mapping={"boundary": 3, "positive_classes": [3, 4]},
                answering_snapshot=JEV_MODEL,
            ),
            label=True,
            raw_answer={
                "type": "score",
                "score": 3.2,
                "levels": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
            },
            provenance="human",
            answering_snapshot=JEV_MODEL,
        )
    )

    assert choice.usable
    assert choice.probability == pytest.approx(0.6)
    assert choice.distribution == {"s1": 0.6, "s2": 0.1, "none": 0.3}
    assert score.usable
    assert score.probability == pytest.approx(0.7)
    assert score.distribution == {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4}


def test_calibration_keeps_partitions_disjoint_and_persists_a_verdict() -> None:
    observations = []
    for index in range(40):
        label = index % 2 == 0
        for repeat_index in range(2):
            observations.append(
                _observation(
                    index,
                    label,
                    probability=0.95 if label else 0.05,
                    repeat_index=repeat_index,
                )
            )
        observations.append(
            _observation(
                index,
                label,
                probability=0.5,
                control=True,
                repeat_index=0,
            )
        )
    result = calibrate_question(
        _identity(),
        observations,
        verdict_policy=VerdictPolicy(
            minimum_evaluation_groups=3,
            minimum_positive_examples=2,
            minimum_negative_examples=2,
            minimum_control_groups=1,
            minimum_repeat_examples=1,
        ),
        bootstrap_resamples=32,
    )
    payload = result.to_dict()

    assert payload["verdict"] == "gate"
    assert payload["metrics"]["precision"] == 1.0
    assert payload["metrics"]["brier"] < payload["control_metrics"]["brier"]
    assert payload["metrics"]["reliability"]
    assert len(payload["metrics"]["reliability"]) == 10
    assert payload["uncertainty"]["bootstrap"]["resamples"] == 32
    assert set(payload["partitions"].values()) == {"fit", "calibration", "evaluation"}
    assert len(set(payload["partitions"].values())) == 3
    assert payload["evidence"]["repeat_spread"]["largest_within_example"] < 0.01
    assert result.artifact().to_dict()["questions"]["gap:goal"]["verdict"] == "gate"


def test_calibration_cli_reads_events_and_writes_machine_readable_artifact(
    tmp_path: Path,
) -> None:
    events = []
    for index in range(12):
        label = index % 2 == 0
        events.append(
            {
                "id": f"event-{index}",
                "source_group": f"group-{index}",
                "example_id": f"example-{index}",
                "question_id": "gap:goal",
                "question": default_gap_question("goal"),
                "primitive": "noul",
                "criteria": ["no", "yes"],
                "event": {"positive_class": "yes"},
                "label": label,
                "label_provenance": "user_delegated_model",
                "answering_snapshot": JEV_MODEL,
                "repeat_index": 0,
                "answer": {
                    "type": "noul",
                    "probability_true": 0.95 if label else 0.05,
                },
            }
        )
    source = tmp_path / "events.json"
    output = tmp_path / "report.json"
    artifact = tmp_path / "artifact.json"
    source.write_text(json.dumps({"name": "known", "events": events}))

    assert (
        evaluation_main(
            [
                "--calibrate",
                str(source),
                "--output",
                str(output),
                "--artifact",
                str(artifact),
                "--bootstrap-resamples",
                "8",
            ]
        )
        == 0
    )

    report = json.loads(output.read_text())
    persisted = json.loads(artifact.read_text())
    assert report["kind"] == "calibration-report"
    assert report["questions"]["gap:goal"]["verdict"] in {
        "too-few-examples",
        "gate",
        "gate-above-confidence",
        "ranker",
        "unusable",
    }
    assert report["verdict_catalog"] == [
        "too-few-examples",
        "gate",
        "gate-above-confidence",
        "ranker",
        "unusable",
    ]
    assert persisted["kind"] == "calibration-artifact"
    assert persisted["questions"]["gap:goal"]["identity"]["question_digest"]


def test_known_answer_fixture_covers_all_primitives_and_replays_deterministically() -> (
    None
):
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "evaluation"
        / "calibration_known_answer.json"
    )
    manifest = load_calibration_manifest(fixture)
    normalized = [normalize_event(event) for event in manifest.events]
    assert all(event.usable for event in normalized)

    by_primitive = {
        primitive: [
            event
            for event in normalized
            if event.observation.identity.primitive == primitive
        ]
        for primitive in ("noul", "choice", "score")
    }
    assert {key for key, events in by_primitive.items() if events} == {
        "noul",
        "choice",
        "score",
    }
    choice_metrics = compute_metrics(by_primitive["choice"], 0.5)
    score_metrics = compute_metrics(by_primitive["score"], 0.5)
    assert choice_metrics["distribution_brier"] is not None
    assert score_metrics["distribution_brier"] is not None

    artifact_a, report_a = calibrate_manifest(manifest, bootstrap_resamples=8)
    artifact_b, report_b = calibrate_manifest(manifest, bootstrap_resamples=8)
    assert json.dumps(artifact_a.to_dict(), sort_keys=True) == json.dumps(
        artifact_b.to_dict(), sort_keys=True
    )
    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)


def test_runtime_policy_gates_rankers_and_abstains_on_snapshot_mismatch() -> None:
    def artifact(verdict: str, snapshot: str = JEV_MODEL, predicate=None):
        identity = replace(_identity(), answering_snapshot=snapshot)
        return CalibrationArtifact.from_dict(
            {
                "schema_version": 1,
                "kind": "calibration-artifact",
                "name": "runtime",
                "input_digest": "digest",
                "questions": {
                    "gap:goal": {
                        "identity": identity.to_dict(),
                        "verdict": verdict,
                        "threshold": 0.8,
                        "predicate": predicate or {},
                    }
                },
            }
        )

    def gateway(probability: float) -> ScriptedGateway:
        def decide(request, **_kwargs):
            if request.get("type") == "choice":
                return {
                    "type": "choice",
                    "choice": "general",
                    "probabilities": {"general": 1.0},
                }
            return {
                "type": "noul",
                "probability_true": probability
                if request.get("key") == "gap:goal"
                else 0.01,
            }

        return ScriptedGateway(
            chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
        )

    optimizer = PromptOptimizer(
        gateway=gateway(0.9),
        store=RunStore(":memory:"),
        decision_policy=DecisionPolicy.from_artifact(artifact("gate")),
    )
    gated = optimizer.optimize("Help me plan.", {"tier": "fast"})
    assert [item["key"] for item in gated["report"]["diagnosis"]["confirmed_gaps"]] == [
        "goal"
    ]

    optimizer = PromptOptimizer(
        gateway=gateway(0.9),
        store=RunStore(":memory:"),
        decision_policy=DecisionPolicy.from_artifact(artifact("ranker")),
    )
    ranked = optimizer.optimize("Help me plan.", {"tier": "fast"})
    assert ranked["report"]["diagnosis"]["confirmed_gaps"] == []

    optimizer = PromptOptimizer(
        gateway=gateway(0.9),
        store=RunStore(":memory:"),
        decision_policy=DecisionPolicy.from_artifact(
            artifact("gate", snapshot="other-snapshot")
        ),
    )
    mismatched = optimizer.optimize("Help me plan.", {"tier": "fast"})
    assert mismatched["report"]["diagnosis"]["confirmed_gaps"] == []

    optimizer = PromptOptimizer(
        gateway=gateway(0.7),
        store=RunStore(":memory:"),
        decision_policy=DecisionPolicy.from_artifact(
            artifact(
                "gate-above-confidence",
                predicate={"probability_gte": 0.8, "margin_gte": 0.4},
            )
        ),
    )
    below = optimizer.optimize("Help me plan.", {"tier": "fast"})
    assert below["report"]["diagnosis"]["confirmed_gaps"] == []


def test_malformed_answers_are_unavailable_not_successful_zero() -> None:
    observation = replace(
        _observation(1, True, probability=None),
        raw_answer={"type": "noul", "probability_true": "bad"},
    )
    normalized = normalize_event(observation, EventSpec(primitive="noul"))
    assert not normalized.usable
    assert normalized.unavailable_reason == "malformed_answer"
    assert normalized.probability is None


def test_manifest_loader_rejects_duplicate_answer_identity(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    base = {
        "source_group": "g",
        "example_id": "e",
        "question_id": "gap:goal",
        "question": "q",
        "primitive": "noul",
        "label": True,
        "provenance": "human",
        "answering_snapshot": JEV_MODEL,
        "repeat_index": 0,
        "answer_id": "same",
        "answer": {"type": "noul", "probability_true": 0.9},
    }
    path.write_text(json.dumps({"events": [base, {**base, "id": "other"}]}))
    with pytest.raises(ValueError, match="duplicate.*answer"):
        load_calibration_manifest(path)
