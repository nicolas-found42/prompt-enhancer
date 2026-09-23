from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from typing import Any

from prompt_enhancer.rubric_revisions import (
    EvaluationMetrics,
    EvaluationSet,
    HarnessRevisionEvaluator,
    MaintainerDecisionKind,
    MeasuredError,
    RecommendedDecision,
    RegressionRisk,
    RevisionKind,
    RubricQuestion,
    RubricRevisionService,
    RubricVersion,
    SQLiteRubricStore,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "rubric_revisions.json").read_text()
)


class FixtureErrorSource:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def errors(self, evaluation_set: EvaluationSet) -> list[MeasuredError]:
        self.seen.append(evaluation_set.identifier)
        return [
            MeasuredError(
                error_id=item["error_id"],
                evaluation_case_ids=tuple(item["evaluation_case_ids"]),
                current_question_id=item["current_question_id"],
                proposed_question=(
                    RubricQuestion(**item["proposed_question"])
                    if item["proposed_question"] is not None
                    else None
                ),
                observation=item["observation"],
                weight=item["weight"],
                predictor_artifact=item["predictor_artifact"],
            )
            for item in FIXTURE["errors"]
        ]


class FixtureEvaluator:
    def __init__(self) -> None:
        self.revision_calls: list[tuple[str, str, str]] = []

    def evaluate_baseline(
        self,
        rubric: RubricVersion,
        evaluation_set: EvaluationSet,
    ) -> EvaluationMetrics:
        del rubric
        assert evaluation_set.replay_artifact == FIXTURE["evaluation_set"]["replay_artifact"]
        return EvaluationMetrics(**FIXTURE["baseline"])

    def evaluate_revision(
        self,
        rubric: RubricVersion,
        change: Any,
        evaluation_set: EvaluationSet,
        baseline: EvaluationMetrics,
    ) -> EvaluationMetrics:
        del rubric, evaluation_set
        self.revision_calls.append(
            (change.kind.value, change.question_id, baseline.input_digest if hasattr(baseline, "input_digest") else "fixture")
        )
        return EvaluationMetrics(**FIXTURE["candidates"][change.kind.value])


def fixture_inputs() -> tuple[RubricVersion, EvaluationSet]:
    rubric = RubricVersion(
        version_id=FIXTURE["rubric"]["version_id"],
        questions=tuple(RubricQuestion(**item) for item in FIXTURE["rubric"]["questions"]),
        parent_version_id=FIXTURE["rubric"]["parent_version_id"],
        created_at=FIXTURE["rubric"]["created_at"],
    )
    evaluation_set = EvaluationSet(**FIXTURE["evaluation_set"])
    return rubric, evaluation_set


def fixture_service(database_path: Path) -> tuple[RubricRevisionService, SQLiteRubricStore, FixtureEvaluator]:
    rubric, _ = fixture_inputs()
    store = SQLiteRubricStore(database_path)
    store.initialize(rubric)
    evaluator = FixtureEvaluator()
    identifiers = count(1)
    service = RubricRevisionService(
        store,
        FixtureErrorSource(),
        evaluator,
        clock=lambda: "2026-02-01T00:00:00Z",
        id_factory=lambda: f"offline-{next(identifiers)}",
    )
    return service, store, evaluator


def test_proposal_evaluates_each_change_and_rejection_remains_auditable(tmp_path: Path) -> None:
    service, _, evaluator = fixture_service(tmp_path / "rubrics.sqlite3")
    _, evaluation_set = fixture_inputs()

    result = service.propose(evaluation_set)

    assert [proposal.change.kind for proposal in result.workflow.proposals] == [
        RevisionKind.NEW,
        RevisionKind.REVISED,
        RevisionKind.DROPPED,
    ]
    assert all(proposal.evaluation is not None for proposal in result.workflow.proposals)
    assert all(proposal.evidence for proposal in result.workflow.proposals)
    assert len(evaluator.revision_calls) == 3
    dropped = result.workflow.proposals[2]
    assert dropped.evaluation.regression_risk is RegressionRisk.ELEVATED
    assert dropped.evaluation.additional_regressions == 1
    assert dropped.evaluation.cost_delta_usd == 0.01
    assert dropped.evaluation.recommended_decision is RecommendedDecision.HOLD

    rejected = service.reject(
        dropped.proposal_id,
        maintainer="maintainer@example.test",
        rationale="The regression evidence does not justify removing the question.",
    )

    assert rejected.decision is MaintainerDecisionKind.REJECT
    assert rejected.resulting_rubric.version_id == "rubric-v1"
    assert rejected.proposal.evidence[0].error_id == "error-contradiction-noise"
    assert rejected.proposal.evaluation is dropped.evaluation
    assert service.active_rubric().version_id == "rubric-v1"

    reopened = SQLiteRubricStore(tmp_path / "rubrics.sqlite3")
    persisted_workflow = reopened.get_workflow(result.workflow.workflow_id)
    assert persisted_workflow.evidence_artifact_ids == ("predictor/model-v3.json",)
    assert [decision.proposal_id for decision in reopened.list_decisions()] == [
        dropped.proposal_id
    ]


def test_only_explicit_adoption_changes_runtime_rubric_and_records_evidence(tmp_path: Path) -> None:
    service, _, _ = fixture_service(tmp_path / "adoption.sqlite3")
    _, evaluation_set = fixture_inputs()
    before = service.active_rubric()
    result = service.propose(evaluation_set)

    assert service.active_rubric() == before

    proposed_new = result.workflow.proposals[0]
    adopted = service.adopt(
        proposed_new.proposal_id,
        maintainer="rubric-maintainer",
        rationale="The replay shows better audience-error recall with no new regressions.",
    )

    assert adopted.decision is MaintainerDecisionKind.ADOPT
    assert adopted.proposal.evaluation is not None
    assert adopted.proposal.evidence[0].error_id == "error-audience"
    assert adopted.resulting_rubric.parent_version_id == before.version_id
    assert adopted.resulting_rubric.version_id == "rubric-from-offline-2"

    runtime_rubric = service.active_rubric()
    assert runtime_rubric.version_id == adopted.resulting_rubric.version_id
    assert runtime_rubric.question("audience-fit") is not None

    # This is the injection hook consumed by the next Diagnoser instance.
    observed_versions = []

    def diagnoser_factory(rubric: RubricVersion) -> object:
        observed_versions.append(rubric.version_id)
        return object()

    diagnoser_factory(runtime_rubric)
    assert observed_versions == [runtime_rubric.version_id]


def test_rerun_compares_output_and_current_base_with_previous_workflow(tmp_path: Path) -> None:
    service, _, _ = fixture_service(tmp_path / "rerun.sqlite3")
    _, evaluation_set = fixture_inputs()
    first = service.propose(evaluation_set)
    service.adopt(
        first.workflow.proposals[0].proposal_id,
        maintainer="rubric-maintainer",
        rationale="Adopt the measured audience question for the next comparison.",
    )

    second = service.rerun(evaluation_set, first.workflow.workflow_id)

    assert second.comparison is not None
    assert second.comparison.rubric_change.from_version_id == "rubric-v1"
    assert second.comparison.rubric_change.to_version_id == "rubric-from-offline-2"
    assert second.comparison.rubric_change.added_question_ids == ("audience-fit",)
    assert second.comparison.current_only_proposal_signatures == ()
    assert len(second.comparison.shared_proposal_signatures) == 2
    assert second.comparison.mean_candidate_f1_delta is not None
    assert second.comparison.current_mean_candidate_cost_usd is not None
    assert second.comparison.mean_candidate_regression_rate_delta is not None


def test_harness_adapter_uses_replay_and_maps_public_report_fields() -> None:
    rubric, evaluation_set = fixture_inputs()
    seen_replay_paths: list[str] = []

    class FakeEngine:
        def __init__(self, selected: RubricVersion) -> None:
            self.selected = selected

        def run(self, dataset: object, *, replay_path: str) -> dict[str, Any]:
            assert dataset == "fixed-dataset"
            seen_replay_paths.append(replay_path)
            return {
                "diagnosis": {
                    "per_gap": {
                        "audience": {"precision": 0.8, "recall": 0.6},
                        "goal": {"precision": 1.0, "recall": 0.8},
                    }
                },
                "cost": {"total": 0.125},
                "improvement": {
                    "improved": 2,
                    "unchanged": 1,
                    "regressed": 1,
                },
                "cases": [{}, {}, {}, {}],
            }

    adapter = HarnessRevisionEvaluator(
        lambda selected: FakeEngine(selected),
        "fixed-dataset",
        evaluation_set.replay_artifact,
    )
    metrics = adapter.evaluate_baseline(rubric, evaluation_set)

    assert metrics.precision == 0.9
    assert metrics.recall == 0.7
    assert metrics.total_cost_usd == 0.125
    assert metrics.regression_count == 1
    assert metrics.case_count == 4
    assert seen_replay_paths == [evaluation_set.replay_artifact]
