from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from prompt_enhancer.diagnosis import (
    DEFAULT_RUBRIC,
    HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
    TASK_TAXONOMY_PROTOCOL_VERSION,
    ChecklistItem,
    DiagnosisRubric,
    GapImpact,
    TaskBranch,
    TaskType,
    diagnosis_from_dict,
)
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    DecisionPolicy,
    runtime_question_identity,
)
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


class TrackingScriptedGateway(ScriptedGateway):
    def __init__(self, *, decision):
        super().__init__(
            chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decision
        )
        self.decision_batches: list[list[str]] = []

    def decide_batch(self, requests, *, role="judge", run_id=None):
        self.decision_batches.append(
            [str(request.get("key", "")) for request in requests]
        )
        return super().decide_batch(requests, role=role, run_id=run_id)


def _optimizer(
    decisions: dict[str, Any] | None = None,
    *,
    rubric: DiagnosisRubric = DEFAULT_RUBRIC,
    task_taxonomy_version: int = TASK_TAXONOMY_PROTOCOL_VERSION,
) -> tuple[PromptOptimizer, TrackingScriptedGateway]:
    overrides = decisions or {}

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if key in overrides:
            return overrides[key]
        if request.get("type") == "choice":
            choice = "general" if key == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}

    gateway = TrackingScriptedGateway(decision=decide)
    return (
        PromptOptimizer(
            store=RunStore(":memory:"),
            gateway=gateway,
            diagnosis_rubric=rubric,
            task_taxonomy_version=task_taxonomy_version,
            speculative_diagnosis=False,
        ),
        gateway,
    )


def test_task_branch_options_describe_leaf_meaning_and_scope() -> None:
    optimizer, gateway = _optimizer()

    optimizer.optimize("Help me with something.", {"tier": "fast"})

    root = next(
        entry["question"]
        for entry in gateway.decision_log
        if entry["question"].get("key") == "task_type"
    )
    investigation = root["options"]["investigation"]
    assert "Analysis" in investigation
    assert "Research" in investigation
    assert "interpret information" in investigation.lower()
    assert "find or synthesize information" in investigation.lower()
    assert "intended scope" in investigation.lower()


def test_taxonomy_request_count_distinguishes_shared_prefetch() -> None:
    optimizer, _gateway = _optimizer()

    result = optimizer.optimize("Write a brief note.", {"tier": "fast"})

    evidence = result["report"]["diagnosis"]["taxonomy_evidence"]
    assert evidence["provider_requests"] == 1
    assert evidence["provider_request_delta_vs_legacy"] == 0
    assert evidence["shared_prefetch"] is False

    optimizer, _gateway = _optimizer()
    optimizer.speculative_diagnosis = True
    prefetched = optimizer.optimize("Write a brief note.", {"tier": "fast"})
    diagnosis = prefetched["report"]["diagnosis"]
    assert diagnosis["taxonomy_evidence"]["provider_requests"] == 0
    assert diagnosis["taxonomy_evidence"]["provider_request_delta_vs_legacy"] == -1
    assert diagnosis["taxonomy_evidence"]["shared_prefetch"] is True
    assert diagnosis["request_evidence"]["provider_requests"] == 1


def test_close_branch_split_batches_both_paths_and_can_choose_second_branch() -> None:
    optimizer, gateway = _optimizer(
        {
            "task_type": {
                "type": "choice",
                "choice": "communication",
                "probabilities": {
                    "communication": 0.42,
                    "investigation": 0.38,
                    "execution": 0.1,
                    "general": 0.05,
                    "unknown": 0.05,
                },
                "confidence": 0.95,
            },
            "task_type:communication": {
                "type": "choice",
                "choice": "writing",
                "probabilities": {"writing": 0.81, "chat": 0.19, "unknown": 0.0},
                "confidence": 0.9,
            },
            "task_type:investigation": {
                "type": "choice",
                "choice": "research",
                "probabilities": {"analysis": 0.01, "research": 0.99, "unknown": 0.0},
                "confidence": 0.9,
            },
        }
    )

    result = optimizer.optimize("Investigate this topic.", {"tier": "fast"})

    assert result["report"]["diagnosis"]["task_type"] == "research"
    assert any(
        set(batch) >= {"task_type:communication", "task_type:investigation"}
        for batch in gateway.decision_batches
    )
    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["task_type_path"][0]["key"] == "investigation"
    assert diagnosis["task_type_path"][1]["key"] == "research"
    assert diagnosis["taxonomy_evidence"]["protocol_version"] == 2
    assert len(diagnosis["taxonomy_evidence"]["explored_paths"]) == 2


def _execution_decisions(*, leaf_confidence: float) -> dict[str, Any]:
    return {
        "task_type": {
            "type": "choice",
            "choice": "execution",
            "probabilities": {
                "communication": 0.05,
                "investigation": 0.05,
                "execution": 0.8,
                "general": 0.05,
                "unknown": 0.05,
            },
            "confidence": 0.95,
        },
        "task_type:execution": {
            "type": "choice",
            "choice": "coding",
            "probabilities": {"coding": 0.95, "planning": 0.05, "unknown": 0.0},
            "confidence": leaf_confidence,
        },
    }


def test_supported_leaf_uses_its_own_specialized_checklist() -> None:
    optimizer, gateway = _optimizer(_execution_decisions(leaf_confidence=0.8))

    result = optimizer.optimize("Build a small parser.", {"tier": "fast"})

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["task_type"] == "coding"
    assert diagnosis["task_type_fallback_reason"] is None
    checklist_keys = {item["key"] for item in diagnosis["effective_checklist"]}
    assert {"language", "tests"} <= checklist_keys
    assert any(
        entry["question"].get("key") == "gap:language" for entry in gateway.decision_log
    )


def test_low_confidence_leaf_uses_parent_intersection_checklist() -> None:
    optimizer, gateway = _optimizer(_execution_decisions(leaf_confidence=0.79))

    result = optimizer.optimize("Build a small parser.", {"tier": "fast"})

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["task_type"] == "execution"
    assert diagnosis["task_type_fallback_reason"] == "leaf_below_confidence"
    checklist_keys = {item["key"] for item in diagnosis["effective_checklist"]}
    assert checklist_keys == {
        "goal",
        "context",
        "constraints",
        "output_format",
        "done_criteria",
        "outside_reference",
    }
    assert "gap:language" not in {
        entry["question"].get("key") for entry in gateway.decision_log
    }
    assert diagnosis["taxonomy_evidence"]["checklist_source"] == "parent_intersection"
    stored = optimizer.store.get_run(result["run_id"])
    assert stored is not None
    stored_diagnosis = stored["result"]["report"]["diagnosis"]
    assert stored_diagnosis["task_type"] == "execution"
    assert stored_diagnosis["effective_checklist"] == diagnosis["effective_checklist"]


@pytest.mark.parametrize(
    ("root_answer", "fallback_reason"),
    [
        (
            {
                "type": "choice",
                "choice": "unknown",
                "probabilities": {"unknown": 1.0},
                "confidence": 1.0,
            },
            "root_unknown",
        ),
        (
            {
                "type": "choice",
                "choice": "nonexistent",
                "probabilities": {"nonexistent": 1.0},
                "confidence": 1.0,
            },
            "root_unsupported_option",
        ),
        (None, "root_missing_or_malformed"),
    ],
)
def test_unknown_invalid_or_missing_root_uses_general_without_expansion(
    root_answer: Any, fallback_reason: str
) -> None:
    optimizer, gateway = _optimizer({"task_type": root_answer})

    result = optimizer.optimize("Help me with something.", {"tier": "fast"})

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["task_type"] == "general"
    assert diagnosis["task_type_fallback_reason"] == fallback_reason
    assert not any(
        key.startswith("task_type:")
        for batch in gateway.decision_batches
        for key in batch
    )


def test_partial_beam_answer_keeps_supported_leaf_from_other_branch() -> None:
    decisions = {
        "task_type": {
            "type": "choice",
            "choice": "communication",
            "probabilities": {
                "communication": 0.42,
                "investigation": 0.38,
                "execution": 0.1,
                "general": 0.05,
                "unknown": 0.05,
            },
            "confidence": 0.95,
        },
        "task_type:communication": None,
        "task_type:investigation": {
            "type": "choice",
            "choice": "research",
            "probabilities": {"analysis": 0.01, "research": 0.99, "unknown": 0.0},
            "confidence": 0.9,
        },
    }
    optimizer, _gateway = _optimizer(decisions)

    result = optimizer.optimize("Investigate this topic.", {"tier": "fast"})

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["task_type"] == "research"
    assert diagnosis["taxonomy_evidence"]["explored_paths"][0]["reason"] == (
        "leaf_missing_or_malformed"
    )


def test_equal_path_scores_use_taxonomy_order_for_ties() -> None:
    decisions = {
        "task_type": {
            "type": "choice",
            "choice": "communication",
            "probabilities": {
                "communication": 0.4,
                "investigation": 0.4,
                "execution": 0.1,
                "general": 0.05,
                "unknown": 0.05,
            },
            "confidence": 0.95,
        },
        "task_type:communication": {
            "type": "choice",
            "choice": "writing",
            "probabilities": {"writing": 0.9, "chat": 0.1, "unknown": 0.0},
            "confidence": 0.9,
        },
        "task_type:investigation": {
            "type": "choice",
            "choice": "analysis",
            "probabilities": {"analysis": 0.9, "research": 0.1, "unknown": 0.0},
            "confidence": 0.9,
        },
    }
    optimizer, _gateway = _optimizer(decisions)

    result = optimizer.optimize("Do a task.", {"tier": "fast"})

    assert result["report"]["diagnosis"]["task_type"] == "writing"


def test_historical_taxonomy_protocol_keeps_recorded_question_shape() -> None:
    optimizer, gateway = _optimizer(
        {
            "task_type": {
                "type": "choice",
                "choice": "investigation",
                "probabilities": {"investigation": 1.0},
                "confidence": 1.0,
            },
            "task_type:investigation": {
                "type": "choice",
                "choice": "research",
                "probabilities": {"research": 1.0},
                "confidence": 1.0,
            },
        },
        task_taxonomy_version=HISTORICAL_TASK_TAXONOMY_PROTOCOL_VERSION,
    )

    result = optimizer.optimize("Research this topic.", {"tier": "fast"})

    diagnosis = result["report"]["diagnosis"]
    root = next(
        entry["question"]
        for entry in gateway.decision_log
        if entry["question"].get("key") == "task_type"
    )
    leaf = next(
        entry["question"]
        for entry in gateway.decision_log
        if entry["question"].get("key") == "task_type:investigation"
    )
    assert diagnosis["task_type"] == "research"
    assert diagnosis["taxonomy_evidence"]["protocol_version"] == 1
    assert "question_schema" not in root
    assert root["options"]["investigation"] == "Contains analysis, research requests."
    assert "question_schema" not in leaf
    assert leaf["options"]["research"] == "Research"


def test_task_classification_path_and_checklist_round_trip_in_stored_report() -> None:
    optimizer, _gateway = _optimizer(_execution_decisions(leaf_confidence=0.9))

    result = optimizer.optimize("Build a small parser.", {"tier": "fast"})
    stored = optimizer.store.get_run(result["run_id"])

    assert stored is not None
    diagnosis = stored["result"]["report"]["diagnosis"]
    restored = diagnosis_from_dict(diagnosis).as_dict()
    assert diagnosis["task_type"] == "coding"
    assert diagnosis["effective_checklist"] == restored["effective_checklist"]
    assert diagnosis["task_type_path"][0]["evidence"]["snapshot"]
    assert diagnosis["taxonomy_evidence"]["explored_paths"][0]["accepted"] is True


def test_empty_parent_intersection_uses_general_checklist_with_reason() -> None:
    general = next(task for task in DEFAULT_RUBRIC.task_types if task.key == "general")
    rubric = DiagnosisRubric(
        task_types=(
            general,
            TaskType(
                "alpha",
                "Alpha",
                (ChecklistItem("shared", "same wording", GapImpact.HIGH),),
            ),
            TaskType(
                "beta",
                "Beta",
                (ChecklistItem("shared", "same wording", GapImpact.LOW),),
            ),
        ),
        task_branches=(
            TaskBranch(
                "parent",
                "Parent",
                "A custom parent.",
                "Alpha and beta tasks.",
                ("alpha", "beta"),
            ),
        ),
    )
    optimizer, _gateway = _optimizer(
        {
            "task_type": {
                "type": "choice",
                "choice": "parent",
                "probabilities": {"parent": 0.9, "general": 0.05, "unknown": 0.05},
                "confidence": 0.95,
            },
            "task_type:parent": {
                "type": "choice",
                "choice": "unknown",
                "probabilities": {"unknown": 1.0},
                "confidence": 1.0,
            },
        },
        rubric=rubric,
    )

    result = optimizer.optimize("Do an alpha or beta task.", {"tier": "fast"})

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["task_type"] == "parent"
    assert diagnosis["effective_checklist"] == [
        {
            "key": item.key,
            "label": item.label,
            "impact": item.impact.value,
            "question": item.question,
        }
        for item in general.checklist
    ]
    assert diagnosis["taxonomy_evidence"]["checklist_fallback_reason"] == (
        "parent_intersection_empty"
    )


def test_explicit_parent_checklist_overrides_descendant_intersection() -> None:
    general = next(task for task in DEFAULT_RUBRIC.task_types if task.key == "general")
    explicit = ChecklistItem("parent-key", "parent wording", GapImpact.HIGH)
    rubric = DiagnosisRubric(
        task_types=(
            general,
            TaskType(
                "alpha",
                "Alpha",
                (ChecklistItem("child-a", "alpha wording", GapImpact.HIGH),),
            ),
            TaskType(
                "beta",
                "Beta",
                (ChecklistItem("child-b", "beta wording", GapImpact.HIGH),),
            ),
        ),
        task_branches=(
            TaskBranch(
                "parent",
                "Parent",
                "A custom parent.",
                "Alpha and beta tasks.",
                ("alpha", "beta"),
                checklist=(explicit,),
            ),
        ),
    )
    optimizer, _gateway = _optimizer(
        {
            "task_type": {
                "type": "choice",
                "choice": "parent",
                "probabilities": {"parent": 0.9, "general": 0.05, "unknown": 0.05},
                "confidence": 0.95,
            },
            "task_type:parent": {
                "type": "choice",
                "choice": "unknown",
                "probabilities": {"unknown": 1.0},
                "confidence": 1.0,
            },
        },
        rubric=rubric,
    )

    result = optimizer.optimize("Do an alpha or beta task.", {"tier": "fast"})

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["task_type"] == "parent"
    assert [item["key"] for item in diagnosis["effective_checklist"]] == ["parent-key"]
    assert diagnosis["taxonomy_evidence"]["checklist_source"] == "explicit_parent"


def test_matching_task_type_calibration_can_override_default_confidence() -> None:
    root_answer = {
        "type": "choice",
        "choice": "execution",
        "probabilities": {
            "communication": 0.06,
            "investigation": 0.05,
            "execution": 0.79,
            "general": 0.05,
            "unknown": 0.05,
        },
        "confidence": 0.79,
    }
    probe, gateway = _optimizer(
        {
            "task_type": root_answer,
            "task_type:execution": {
                "type": "choice",
                "choice": "coding",
                "probabilities": {
                    "coding": 0.95,
                    "planning": 0.05,
                    "unknown": 0.0,
                },
                "confidence": 0.95,
            },
        }
    )
    probe.optimize("Build a parser.", {"tier": "fast"})
    root_entry = next(
        entry
        for entry in gateway.decision_log
        if entry["question"].get("key") == "task_type"
    )
    identity = runtime_question_identity(
        "task_type",
        root_entry["question"],
        family="task_type",
        rubric_version="default-v1",
        snapshot=root_entry["answered_by"],
    )
    identity = replace(identity, event_mapping={"selected_correctness": True})
    artifact = CalibrationArtifact.from_dict(
        {
            "name": "task-type-confidence",
            "questions": {
                identity.question_id: {
                    "identity": identity.to_dict(),
                    "verdict": "gate",
                    "threshold": 0.7,
                }
            },
        }
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        decision_policy=DecisionPolicy.from_artifact(artifact),
    )

    result = optimizer.optimize("Build a parser.", {"tier": "fast"})

    assert result["report"]["diagnosis"]["task_type"] == "coding"
    assert (
        result["report"]["diagnosis"]["task_type_path"][0]["evidence"]["calibration"][
            "threshold"
        ]
        == 0.7
    )
