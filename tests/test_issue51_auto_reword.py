"""Automatic REWORD adoption uses independently labeled, sealed final rows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from prompt_enhancer.catalog import (
    DEFAULT_GO_WRITER,
    JEV_MODEL,
    ModelInfo,
    StaticModelCatalog,
)
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    DecisionPolicy,
    runtime_question_identity,
)
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import ProviderError, ReplayGateway, ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.reword_optimization import (
    RewordPolicy,
    _digest,
    _loss,
    optimize_reword,
)
from prompt_enhancer.rubric_revisions import (
    RubricQuestion,
    RubricVersion,
    SQLiteRubricStore,
)
from prompt_enhancer.store import RunStore

BASELINE = "Is the task clear?"
ALTERNATIVE = "Is the requested task clear enough to complete?"


def _dataset(*, final_groups: int = 30, provenance: str = "source") -> dict:
    rows = []
    for partition, count in (
        ("training", 10),
        ("calibration", 220),
        ("final", final_groups),
        ("regression", 2),
    ):
        for index in range(count):
            positive = index % 2 != 0
            rows.append(
                {
                    "id": f"{partition}-{index}",
                    "group_id": f"{partition}-source-{index}",
                    "partition": partition,
                    "state": {
                        "signal": "unclear" if positive else "clear",
                        "source_id": f"{partition}-{index}",
                    },
                    "label": positive,
                    "provenance": provenance,
                }
            )
    final_rows = [row for row in rows if row["partition"] == "final"]
    repeats = {
        _digest(text): {
            row["id"]: {
                "snapshot": JEV_MODEL,
                "raw_answer": {
                    "type": "noul",
                    "probability_true": (
                        0.98 if row["state"]["signal"] == "clear" else 0.02
                    )
                    if text == ALTERNATIVE
                    else (0.8 if row["state"]["signal"] == "clear" else 0.2),
                    "confidence": 1.0,
                },
            }
            for row in final_rows
        }
        for text in (BASELINE, ALTERNATIVE)
    }
    return {"rows": rows, "repeat_answers": repeats}


class RewordGateway(ScriptedGateway):
    def __init__(
        self,
        *,
        drift: bool = False,
        uncertain: bool = False,
        final_regression: bool = False,
        calibration_bad: bool = False,
    ) -> None:
        self.writer_states: list[dict] = []
        self.evaluated: list[dict] = []
        self.drift = drift
        self.uncertain = uncertain
        self.final_regression = final_regression
        self.calibration_bad = calibration_bad
        super().__init__(chat=self._chat, decision=self._decide)

    def _chat(self, _model, messages, *, role, **_kwargs):
        assert role == "writer_reword"
        self.writer_states.append(json.loads(messages[1]["content"]))
        return json.dumps({"alternatives": [ALTERNATIVE]})

    def _decide(self, request, *, role, **_kwargs):
        if role == "judge_reword_screen":
            key = request["key"]
            if key.endswith((":condition", ":options")):
                probability = 0.5 if self.uncertain else 0.05 if self.drift else 0.95
            else:
                probability = 0.05
            return {"type": "noul", "probability_true": probability, "confidence": 1.0}
        assert role == "judge_reword_eval"
        self.evaluated.append(dict(request))
        state = request["state"]
        positive = state["signal"] == "clear"
        alternative = request["question"] == ALTERNATIVE
        if (
            self.calibration_bad
            and alternative
            and state["source_id"].startswith("calibration-")
        ):
            return {"type": "noul", "probability_true": 0.5, "confidence": 1.0}
        if (
            self.final_regression
            and state["source_id"] == "regression-0"
            and alternative
        ):
            positive = False
        probability = (
            (0.98 if positive else 0.02) if alternative else (0.8 if positive else 0.2)
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}


def _store(path: Path) -> SQLiteRubricStore:
    store = SQLiteRubricStore(path)
    store.initialize(
        RubricVersion("rubric-v1", (RubricQuestion("task-clarity", BASELINE),))
    )
    return store


def test_equivalent_reword_adopts_without_human_identity_and_rolls_back(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "rubric.sqlite3")
    gateway = RewordGateway()

    result = optimize_reword(
        store, gateway, "task-clarity", _dataset(), attempt_id="good"
    )

    assert result["status"] == "adopted"
    assert all(result["gates"].values())
    question = store.active_rubric().question("task-clarity")
    assert question is not None
    assert question.text == ALTERNATIVE
    assert question.threshold != 0.5
    assert question.question_version == 2
    assert question.calibration_snapshot == gateway.jev_model
    assert question.calibration_policy_version == "issue-51-v1"
    assert question.calibration_artifact is not None
    assert (
        question.calibration_artifact["questions"]["rubric:task-clarity"]["verdict"]
        == "gate"
    )
    with pytest.raises(ValueError, match="snapshot"):
        replace(question, calibration_snapshot="typesafe/jev-moving-alias")
    rank_only = json.loads(json.dumps(question.calibration_artifact))
    rank_only["questions"]["rubric:task-clarity"]["verdict"] = "ranker"
    with pytest.raises(ValueError, match="artifact"):
        replace(question, calibration_artifact=rank_only)
    assert result["actor"] == "automatic_policy"
    decisions = store.list_decisions()
    assert len(decisions) == 1
    assert decisions[0].actor_type == "automatic"
    assert decisions[0].automatic_evidence["final_validation"]["group_count"] == 30
    assert result["final_validation"]["lower_bound_improvement_brier"] > 0.01
    assert store.get_reword_attempt("good") == result
    assert (
        optimize_reword(store, gateway, "task-clarity", _dataset(), attempt_id="good")
        == result
    )
    assert len(gateway.writer_states) == 1

    def runtime_answer(request, **_kwargs):
        if request["type"] == "choice":
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        return {"type": "noul", "probability_true": 0.1, "confidence": 1.0}

    optimized = PromptOptimizer(
        store=RunStore(":memory:"),
        rubric_store=store,
        gateway=ScriptedGateway(
            chat=lambda *_args, **_kwargs: '{"tests":[]}',
            decision=runtime_answer,
        ),
    ).optimize("Draft the note.", {"clarification_allowed": False})
    assert (
        optimized["report"]["diagnosis"]["rubric_version"]
        == result["adopted_version_id"]
    )
    assert any(
        item["question"].get("query") == ALTERNATIVE
        for item in optimized["report"]["jev_answers"]
    )

    assert store.rollback_reword(result["adopted_version_id"]).version_id == "rubric-v1"


def test_final_rows_and_labels_are_sealed_until_finalist_is_selected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "rubric.sqlite3")
    gateway = RewordGateway()
    result = optimize_reword(
        store, gateway, "task-clarity", _dataset(), attempt_id="sealed"
    )

    assert result["status"] == "adopted"
    writer_input = json.dumps(gateway.writer_states)
    assert "final-0" not in writer_input
    assert "calibration-0" not in writer_input
    assert all("label" not in request["state"] for request in gateway.evaluated)
    assert result["training"]["finalist_digest"]
    assert result["calibration"]["partitions"]
    assert result["calibration"]["verdict"] == "gate"
    assert result["partitions"]["final"]["groups"] == 30


def test_drift_uncertainty_support_regression_and_reuse_hold(tmp_path: Path) -> None:
    for name, gateway, dataset, reason in (
        ("drift", RewordGateway(drift=True), _dataset(), "semantic screening"),
        ("uncertain", RewordGateway(uncertain=True), _dataset(), "semantic screening"),
        (
            "support",
            RewordGateway(),
            _dataset(final_groups=20),
            "insufficient independent final support",
        ),
        (
            "weak-label",
            RewordGateway(),
            _dataset(provenance="weak"),
            "insufficient independent evidence",
        ),
        (
            "regression",
            RewordGateway(final_regression=True),
            _dataset(),
            "final validation gates",
        ),
        (
            "uncalibrated",
            RewordGateway(calibration_bad=True),
            _dataset(),
            "calibration did not approve a gate",
        ),
    ):
        store = _store(tmp_path / f"{name}.sqlite3")
        result = optimize_reword(
            store, gateway, "task-clarity", dataset, attempt_id=name
        )
        assert result["status"] == ("reject" if name == "drift" else "hold")
        assert reason in result["reason"]
        assert store.active_rubric().version_id == "rubric-v1"

    store = _store(tmp_path / "reused.sqlite3")
    dataset = _dataset()
    first = optimize_reword(
        store,
        RewordGateway(final_regression=True),
        "task-clarity",
        dataset,
        attempt_id="first",
    )
    assert first["status"] == "hold"
    second_gateway = RewordGateway()
    second = optimize_reword(
        store, second_gateway, "task-clarity", dataset, attempt_id="second"
    )
    assert second["reason"] == "validation budget exhausted"
    assert second_gateway.writer_states == []


def test_group_leakage_and_evaluation_cap_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path / "rubric.sqlite3")
    dataset = _dataset()
    dataset["rows"][1]["group_id"] = dataset["rows"][-1]["group_id"]
    try:
        optimize_reword(
            store, RewordGateway(), "task-clarity", dataset, attempt_id="leak"
        )
    except ValueError as exc:
        assert "crosses" in str(exc)
    else:
        raise AssertionError("a source group crossed partitions")

    capped = optimize_reword(
        store,
        RewordGateway(),
        "task-clarity",
        _dataset(),
        attempt_id="cap",
        policy=RewordPolicy(max_evaluations=1),
    )
    assert capped["status"] == "hold"
    assert "budget exhausted" in capped["reason"]
    assert store.active_rubric().version_id == "rubric-v1"


def test_choice_and_score_losses_preserve_full_label_space(tmp_path: Path) -> None:
    assert _loss(
        (0.1, 0.8, 0.1), "middle", "choice", ("low", "middle", "high")
    ) == pytest.approx((0.1**2 + 0.2**2 + 0.1**2) / 3)
    assert _loss(
        (0.1, 0.8, 0.1), 1, "score", ("low", "middle", "high")
    ) == pytest.approx((0.1**2 + 0.1**2) / 2)
    store = _store(tmp_path / "shapes.sqlite3")
    with pytest.raises(ValueError, match="Noul rewording cannot change criteria"):
        optimize_reword(
            store,
            RewordGateway(),
            "task-clarity",
            {"criteria": ["pass", "fail"], "rows": []},
            attempt_id="bad-shape",
        )


def test_exact_reword_requests_replay_without_provider_fallback(tmp_path: Path) -> None:
    catalog = StaticModelCatalog(
        (
            ModelInfo(
                DEFAULT_GO_WRITER,
                "go",
                input_cost_per_token=0.0000001,
                output_cost_per_token=0.0000002,
            ),
        ),
        (
            ModelInfo(
                JEV_MODEL,
                "openrouter",
                input_cost_per_token=0.0000001,
                output_cost_per_token=0.0000002,
            ),
        ),
    )
    scripted = RewordGateway()
    scripted.catalog = catalog
    recording_path = tmp_path / "reword-recording.json"
    first = optimize_reword(
        _store(tmp_path / "recorded.sqlite3"),
        RecordingGateway(scripted, recording_path),
        "task-clarity",
        _dataset(),
        attempt_id="strict",
    )
    bundle = json.loads(recording_path.read_text())
    replay = ReplayGateway(
        bundle["responses"],
        decision_provenance=bundle["decision_provenance"],
        jev_model=bundle["jev_model"],
        catalog=catalog,
    )
    second = optimize_reword(
        _store(tmp_path / "replayed.sqlite3"),
        replay,
        "task-clarity",
        _dataset(),
        attempt_id="strict",
    )

    assert first["status"] == second["status"] == "adopted"
    assert (
        first["final_validation"]["lower_bound_improvement_brier"]
        == second["final_validation"]["lower_bound_improvement_brier"]
    )
    assert len(replay.replayed_keys) == len(bundle["responses"])


def test_exact_training_evaluations_are_cached_across_fresh_attempts(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "cached.sqlite3")
    first = optimize_reword(
        store,
        RewordGateway(),
        "task-clarity",
        _dataset(final_groups=20),
        attempt_id="insufficient",
    )
    assert first["status"] == "hold"

    gateway = RewordGateway()
    second = optimize_reword(
        store,
        gateway,
        "task-clarity",
        _dataset(),
        attempt_id="fresh-final",
    )

    assert second["status"] == "adopted"
    assert any(item["cache_hit"] for item in second["evaluations"])
    assert all(
        item["state"]["source_id"].startswith(("final", "regression"))
        for item in gateway.evaluated
    )


def test_stale_rubric_cannot_win_atomic_adoption(tmp_path: Path) -> None:
    store = _store(tmp_path / "stale.sqlite3")

    class ConcurrentGateway(RewordGateway):
        changed = False

        def _decide(self, request, *, role, **kwargs):
            if (
                not self.changed
                and role == "judge_reword_eval"
                and request["state"]["source_id"].startswith("final-")
            ):
                self.changed = True
                store.initialize(
                    RubricVersion(
                        "rubric-v2",
                        store.active_rubric().questions,
                        parent_version_id="rubric-v1",
                    )
                )
                with sqlite3.connect(store.database_path) as connection:
                    connection.execute(
                        "UPDATE active_rubric SET version_id = 'rubric-v2' WHERE singleton = 1"
                    )
            return super()._decide(request, role=role, **kwargs)

    result = optimize_reword(
        store,
        ConcurrentGateway(),
        "task-clarity",
        _dataset(),
        attempt_id="stale",
    )

    assert result["status"] == "hold"
    assert result["reason"] == "active rubric changed after evaluation"
    assert store.active_rubric().version_id == "rubric-v2"


def test_paired_repeat_noise_floor_is_measured_in_brier_units(tmp_path: Path) -> None:
    store = _store(tmp_path / "repeat.sqlite3")
    dataset = _dataset()
    final_rows = [row for row in dataset["rows"] if row["partition"] == "final"]
    repeats = {}
    for text in (BASELINE, ALTERNATIVE):
        repeats[_digest(text)] = {
            row["id"]: {
                "snapshot": JEV_MODEL,
                "raw_answer": {
                    "type": "noul",
                    "probability_true": 0.8
                    if row["state"]["signal"] == "clear"
                    else 0.2,
                    "confidence": 1.0,
                },
            }
            for row in final_rows
        }
    dataset["repeat_answers"] = repeats

    result = optimize_reword(
        store,
        RewordGateway(),
        "task-clarity",
        dataset,
        attempt_id="repeat",
    )

    assert result["status"] == "hold"
    assert (
        result["final_validation"]["noise_floor_provenance"]
        == "paired_repeated_predictions"
    )
    assert result["final_validation"]["noise_floor_brier"] > 0.03
    assert result["gates"]["paired_brier_improvement"] is False

    without_repeats = _dataset()
    del without_repeats["repeat_answers"]
    missing = optimize_reword(
        _store(tmp_path / "missing-repeat.sqlite3"),
        RewordGateway(),
        "task-clarity",
        without_repeats,
        attempt_id="missing-repeat",
    )
    assert missing["status"] == "hold"
    assert missing["gates"]["stability_evidence"] is False


def test_rank_only_semantic_calibration_cannot_approve_reword(tmp_path: Path) -> None:
    reference_gateway = RewordGateway()
    reference = optimize_reword(
        _store(tmp_path / "reference.sqlite3"),
        reference_gateway,
        "task-clarity",
        _dataset(),
        attempt_id="reference",
    )
    assert reference["status"] == "adopted"
    request = next(
        entry["question"]
        for entry in reference_gateway.decision_log
        if entry["question"]["key"].endswith(":condition")
    )
    identity = runtime_question_identity(
        "reword_screen:condition",
        request,
        family="reword_screen",
        rubric_version="issue-51-v1",
        snapshot=JEV_MODEL,
    )
    identity = replace(identity, event_mapping={"polarity": "positive"})
    policy = DecisionPolicy.from_artifact(
        CalibrationArtifact.from_dict(
            {
                "name": "rank-only-screen",
                "input_digest": "fixture",
                "questions": {
                    "reword_screen:condition": {
                        "identity": identity.to_dict(),
                        "verdict": "ranker",
                    }
                },
            }
        )
    )
    held = optimize_reword(
        _store(tmp_path / "ranker.sqlite3"),
        RewordGateway(),
        "task-clarity",
        _dataset(),
        attempt_id="ranker",
        decision_policy=policy,
    )

    assert held["status"] == "hold"
    assert held["gates"]["semantic_equivalence"] is False


def test_missing_pricing_dollar_limit_and_provider_failure_hold(tmp_path: Path) -> None:
    unpriced = optimize_reword(
        _store(tmp_path / "unpriced.sqlite3"),
        RecordingGateway(RewordGateway(), tmp_path / "unpriced.json"),
        "task-clarity",
        _dataset(),
        attempt_id="unpriced",
    )
    assert "missing trustworthy pricing" in unpriced["reason"]

    catalog = StaticModelCatalog(
        (
            ModelInfo(
                DEFAULT_GO_WRITER,
                "go",
                input_cost_per_token=0.0000001,
                output_cost_per_token=0.0000002,
            ),
        ),
        (
            ModelInfo(
                JEV_MODEL,
                "openrouter",
                input_cost_per_token=0.0000001,
                output_cost_per_token=0.0000002,
            ),
        ),
    )
    gateway = RewordGateway()
    gateway.catalog = catalog
    capped = optimize_reword(
        _store(tmp_path / "capped.sqlite3"),
        RecordingGateway(gateway, tmp_path / "capped.json"),
        "task-clarity",
        _dataset(),
        attempt_id="capped",
        policy=RewordPolicy(max_cost_usd=0.000001),
    )
    assert "dollar budget exhausted" in capped["reason"]
    assert gateway.writer_states == []

    class FailedWriter(RewordGateway):
        def _chat(self, _model, _messages, *, role, **_kwargs):
            raise ProviderError("scripted", DEFAULT_GO_WRITER, None, role=role)

    failed = optimize_reword(
        _store(tmp_path / "failed.sqlite3"),
        FailedWriter(),
        "task-clarity",
        _dataset(),
        attempt_id="failed",
    )
    assert failed["status"] == "hold"
    assert "scripted" in failed["reason"]
