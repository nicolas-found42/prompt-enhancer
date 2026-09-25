import json
from pathlib import Path

import pytest

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.config import Settings
from prompt_enhancer.evaluation import Dataset, EvaluationHarness
from prompt_enhancer.evaluation.__main__ import main as evaluation_main
from prompt_enhancer.evaluation.harness import default_engine_factory
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import ReplayGateway, ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def test_recorded_replay_cli_completes_cases(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "evaluation"
    output = tmp_path / "report.json"

    assert (
        evaluation_main(
            [
                str(fixtures / "mixed_dataset.json"),
                "--replay",
                str(fixtures / "replay_with_latency.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text())
    assert [case["status"] for case in report["cases"]] == ["completed"] * 3
    assert report["diagnosis"]["excluded_failed_cases"] == 0


def test_replay_restores_recorded_case_cost_and_latency(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "evaluation"
    dataset = json.loads((fixtures / "mixed_dataset.json").read_text())
    first_id = dataset["cases"][0]["id"]
    replay = json.loads((fixtures / "replay_with_latency.json").read_text())
    replay["case_costs"] = {
        first_id: {"total": 0.25, "cost_by_role": {"judge": 0.1, "writer": 0.15}}
    }
    replay["case_latency_ms"] = {first_id: 4321}
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay))
    output = tmp_path / "report.json"

    assert (
        evaluation_main(
            [
                str(fixtures / "mixed_dataset.json"),
                "--replay",
                str(replay_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text())
    assert report["cases"][0]["cost"] == 0.25
    assert report["cases"][0]["cost_by_role"] == {"judge": 0.1, "writer": 0.15}
    assert report["cases"][0]["latency_ms"] == 4321


def test_replay_restores_its_question_thresholds(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "evaluation"
    replay = json.loads((fixtures / "replay_with_latency.json").read_text())
    replay["rubric_thresholds"] = {"context": 0.9}
    path = tmp_path / "historical-replay.json"
    path.write_text(json.dumps(replay))

    engine = default_engine_factory(path)

    assert engine.diagnosis_rubric.gap_threshold_for("context") == 0.9


def test_recorded_live_gateway_replays_the_same_public_run(tmp_path: Path) -> None:
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        return {"type": "noul", "probability_true": 0.1, "confidence": 0.9}

    path = tmp_path / "recorded.json"
    gateway = RecordingGateway(
        ScriptedGateway(chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide),
        path,
    )
    prompt = "Summarize the supplied article in three bullets."
    store = RunStore(":memory:")
    original = PromptOptimizer(gateway=gateway, store=store).optimize(prompt)
    responses = json.loads(path.read_text())["responses"]
    provenance = json.loads(path.read_text())["decision_provenance"]
    assert provenance
    assert {entry["answered_by"] for entry in provenance.values()} == {JEV_MODEL}
    replayed = PromptOptimizer(
        gateway=ReplayGateway(responses), store=RunStore(":memory:")
    ).optimize(prompt)

    assert responses
    assert replayed["status"] == original["status"]
    assert replayed["final_prompt"] == original["final_prompt"]
    assert original["report"]["jev_snapshot"] == [JEV_MODEL]
    assert all(
        answer["answered_by"] == JEV_MODEL
        for answer in original["report"]["jev_answers"]
    )
    assert all(
        answer["answered_by"] == JEV_MODEL
        for answer in store.get_run(original["run_id"])["jev_answers"]
    )


def test_replay_rejects_mismatched_jev_snapshot_unless_overridden(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recorded.json"
    pin = "typesafe/jev-1.13-20990101"
    original = _record_candidate_run(path, 2, pin=pin)
    with pytest.raises(Exception, match="differs from configured pin"):
        default_engine_factory(path)
    engine = default_engine_factory(path, allow_snapshot_mismatch=True)
    engine.store = RunStore(":memory:")
    replayed = engine.optimize(
        "Original request", {"tier": "fast", "clarification_allowed": False}
    )
    assert replayed["final_prompt"] == original["final_prompt"]
    assert replayed["report"]["jev_snapshot"] == [pin]


def test_replay_cli_reports_failed_cases_with_nonzero_exit(
    tmp_path: Path, capsys
) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "evaluation"
    replay = tmp_path / "empty-replay.json"
    replay.write_text('{"responses":{}}')
    output = tmp_path / "report.json"

    assert (
        evaluation_main(
            [
                str(fixtures / "mixed_dataset.json"),
                "--replay",
                str(replay),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert all(
        case["status"] == "failed" for case in json.loads(output.read_text())["cases"]
    )
    assert "3 case(s) failed" in capsys.readouterr().err


def test_harness_reads_product_selection_and_numeric_role_costs() -> None:
    class ProductResultEngine:
        def optimize(self, prompt, options):
            return {
                "status": "completed",
                "final_prompt": f"Clearer: {prompt}",
                "original_kept": False,
                "report": {
                    "diagnosis": {"confirmed_gaps": []},
                    "selection_evidence": {
                        "original_score": {"worst_model_pass_rate": 0.25},
                        "winner_score": {"worst_model_pass_rate": 0.75},
                    },
                },
                "cost": {
                    "total": 0.03,
                    "cost_by_role": {"writer": 0.02, "judge": 0.01},
                    "by_role": {"writer": [{"cost": 0.02}]},
                },
                "timing": {"total_ms": 125},
            }

    dataset = Dataset.from_dict(
        [{"id": "one", "prompt": "Write a report", "source": "real"}]
    )
    report = EvaluationHarness(ProductResultEngine()).run(dataset)

    assert report.improvement.comparable_cases == 1
    assert report.improvement.improved == 1
    assert report.cost.by_role == {"writer": 0.02, "judge": 0.01}
    assert report.cases[0].latency_ms == 125


def test_harness_resumes_only_with_source_answers_and_keeps_clarification_prediction() -> (
    None
):
    class PausingEngine:
        def __init__(self):
            self.resumed = []

        def optimize(self, prompt, options):
            return {
                "status": "needs_input",
                "run_id": "run-1",
                "report": {"diagnosis": {"confirmed_gaps": [{"key": "context"}]}},
            }

        def resume(self, run_id, answers):
            self.resumed.append((run_id, answers))
            return {
                "status": "completed",
                "final_prompt": "Answered request",
                "original_kept": False,
                "report": {
                    "diagnosis": {"confirmed_gaps": [{"key": "context"}]},
                    "selection_evidence": {"original_score": 0.25, "winner_score": 0.5},
                },
            }

    engine = PausingEngine()
    dataset = Dataset.from_dict(
        [
            {
                "id": "one",
                "prompt": "Plan a trip",
                "expected_gaps": ["clarification_need"],
                "clarification_answers": {"context": "Paris"},
                "clarification_answer_provenance": "human",
                "evaluation_gaps": ["clarification_need"],
            }
        ]
    )
    case = EvaluationHarness(engine).run(dataset).cases[0]

    assert engine.resumed == [("run-1", {"context": "Paris"})]
    assert case.status == "completed"
    assert case.predicted_gaps == ("clarification_need",)
    assert case.score_delta == 0.25


def test_harness_rejects_unattributed_clarification_answers() -> None:
    class PausingEngine:
        def optimize(self, prompt, options):
            return {"status": "needs_input", "run_id": "run-1", "report": {}}

    dataset = Dataset.from_dict(
        [
            {
                "id": "one",
                "prompt": "Plan a trip",
                "clarification_answers": {"context": "Paris"},
            }
        ]
    )
    case = EvaluationHarness(PausingEngine()).run(dataset).cases[0]

    assert case.status == "error"
    assert "require human or source provenance" in case.error


def test_dataset_round_trip_preserves_evaluation_scope_and_source_answers() -> None:
    dataset = Dataset.from_dict(
        [
            {
                "id": "one",
                "prompt": "Plan a trip",
                "expected_gaps": ["context"],
                "evaluation_gaps": ["context"],
                "clarification_answers": {"context": "Paris"},
                "clarification_answer_provenance": "human",
            }
        ]
    )

    restored = Dataset.from_dict(dataset.to_dict())

    assert restored.cases[0].metadata == dataset.cases[0].metadata
    assert restored.digest == dataset.digest


def test_harness_retains_engine_failure_reason() -> None:
    class FailedEngine:
        def optimize(self, prompt, options):
            return {
                "status": "failed",
                "final_prompt": prompt,
                "original_kept": True,
                "report": {"status": "failed", "error": "writer response was invalid"},
            }

    dataset = Dataset.from_dict(
        [
            {
                "id": "one",
                "prompt": "Write a report",
                "source": "real",
                "expected_gaps": ["context"],
            }
        ]
    )
    report = EvaluationHarness(FailedEngine()).run(dataset)

    assert report.cases[0].error == "writer response was invalid"
    assert report.diagnosis.excluded_failed_cases == 1


def test_harness_counts_false_positives_on_labeled_no_gap_cases() -> None:
    class ContextEngine:
        def optimize(self, prompt, options):
            return {
                "status": "completed",
                "final_prompt": prompt,
                "original_kept": True,
                "report": {"diagnosis": {"confirmed_gaps": [{"key": "context"}]}},
            }

    dataset = Dataset.from_dict(
        [
            {
                "id": "positive",
                "prompt": "Help with this issue",
                "source": "real",
                "expected_gaps": ["missing context"],
                "evaluation_gaps": ["context"],
            },
            {
                "id": "negative",
                "prompt": "Explain the supplied details",
                "source": "real",
                "expected_gaps": [],
                "evaluation_gaps": ["context"],
            },
        ]
    )
    report = EvaluationHarness(ContextEngine()).run(dataset)

    assert report.diagnosis.labeled_cases == 2
    assert report.diagnosis.per_gap["context"].true_positives == 1
    assert report.diagnosis.per_gap["context"].false_positives == 1
    assert report.diagnosis.per_gap["context"].precision == 0.5


def test_harness_matches_human_gap_synonyms_to_checklist_keys() -> None:
    class DiagnosingEngine:
        def optimize(self, prompt, options):
            return {
                "status": "completed",
                "final_prompt": prompt,
                "original_kept": True,
                "report": {
                    "diagnosis": {
                        "confirmed_gaps": [
                            {"key": "done_criteria"},
                            {"key": "output_format"},
                        ]
                    }
                },
            }

    dataset = Dataset.from_dict(
        [
            {
                "id": "labeled",
                "prompt": "Prepare a report",
                "source": "hand_labeled",
                "expected_gaps": ["definition of done", "missing output format"],
            }
        ]
    )
    report = EvaluationHarness(DiagnosingEngine()).run(dataset)

    assert report.diagnosis.micro.true_positives == 2
    assert report.diagnosis.micro.false_negatives == 0
    assert report.diagnosis.micro.false_positives == 0


def test_failed_cases_are_excluded_from_diagnosis_accuracy() -> None:
    class FailedEngine:
        def optimize(self, prompt, options):
            return {
                "status": "failed",
                "final_prompt": prompt,
                "report": {"diagnosis": {"confirmed_gaps": []}},
            }

    dataset = Dataset.from_dict(
        [
            {
                "id": "failed",
                "prompt": "Ambiguous request",
                "source": "real",
                "expected_gaps": ["context"],
            },
        ]
    )
    report = EvaluationHarness(FailedEngine()).run(dataset)

    assert report.diagnosis.labeled_cases == 0
    assert report.diagnosis.excluded_failed_cases == 1
    assert report.diagnosis.micro.false_negatives == 0


def _candidate_gateway() -> ScriptedGateway:
    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            return '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}],"add_missing_context":"Context rewrite","specify_output_format":"Format rewrite","add_done_criteria":"Done rewrite"}'
        return {
            "choices": [
                {
                    "message": {
                        "content": "pass"
                        if messages[0]["content"] != "Original request"
                        else "fail"
                    }
                }
            ]
        }

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "task_type":
                choice = "general"
            elif key.startswith("pointer:vagueness:"):
                choice = "s0001"
            elif key.startswith("fidelity:sentence:"):
                choice = "supported_by_original"
            else:
                choice = "none"
            probabilities = {choice: 1.0}
            if key.startswith("fidelity:sentence:"):
                probabilities = {
                    "supported_by_original": 0.99,
                    "supported_by_assumption": 0.0,
                    "new_requirement": 0.0,
                    "unknown": 0.01,
                }
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": 1.0,
            }
        if key == "fidelity:meaning":
            probability = 0.99
        elif "output" in request.get("state", {}):
            passed = request["state"]["output"] == "pass"
            probability = (
                float(not passed)
                if str(request.get("key", "")).endswith("_second")
                else float(passed)
            )
        else:
            probability = 1.0
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(chat=chat, decision=decide)


def _record_candidate_run(path: Path, version: int, *, pin: str = JEV_MODEL) -> dict:
    scripted = _candidate_gateway()
    scripted.jev_model = pin
    gateway = RecordingGateway(scripted, path)
    gateway.writer_instruction_version = version
    gateway.faithfulness_threshold = 0.9 if version == 1 else 0.8
    optimizer = PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        config=Settings(judge_model=pin),
        writer_instruction_version=version,
        faithfulness_threshold=gateway.faithfulness_threshold,
    )
    return optimizer.optimize(
        "Original request", {"tier": "fast", "clarification_allowed": False}
    )


def _replay(path: Path) -> dict:
    engine = default_engine_factory(path)
    engine.store = RunStore(":memory:")
    return engine.optimize(
        "Original request", {"tier": "fast", "clarification_allowed": False}
    )


def test_unversioned_historical_recording_replays_with_original_writer_request(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical.json"
    original = _record_candidate_run(path, 1)
    bundle = json.loads(path.read_text())
    del bundle["writer_instruction_version"]
    del bundle["faithfulness_threshold"]
    path.write_text(json.dumps(bundle))

    assert default_engine_factory(path).faithfulness_threshold == 0.9
    replayed = _replay(path)

    assert original["final_prompt"] != "Original request"
    assert replayed["final_prompt"] == original["final_prompt"]


def test_versioned_recording_selects_current_writer_request(tmp_path: Path) -> None:
    path = tmp_path / "current.json"
    original = _record_candidate_run(path, 2)
    recorded = json.loads(path.read_text())
    assert (
        recorded["writer_instruction_version"],
        recorded["faithfulness_threshold"],
    ) == (2, 0.8)
    assert default_engine_factory(path).faithfulness_threshold == 0.8

    assert _replay(path)["final_prompt"] == original["final_prompt"]

    bundle = json.loads(path.read_text())
    bundle["writer_instruction_version"] = 1
    path.write_text(json.dumps(bundle))
    assert _replay(path)["status"] == "failed"


def test_version_three_recording_replays_the_edit_permission_request(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sentence-fidelity.json"
    original = _record_candidate_run(path, 3)

    assert json.loads(path.read_text())["writer_instruction_version"] == 3
    assert original["final_prompt"] != "Original request"
    assert _replay(path)["final_prompt"] == original["final_prompt"]


def _context_impact(path: Path) -> str:
    rubric = default_engine_factory(path).diagnosis_rubric
    return next(
        item.impact.value
        for task in rubric.task_types
        for item in task.checklist
        if item.key == "context"
    )


def test_recordings_keep_the_checklist_impacts_they_were_made_with(
    tmp_path: Path,
) -> None:
    path = tmp_path / "current.json"
    _record_candidate_run(path, 2)
    bundle = json.loads(path.read_text())
    assert bundle["checklist_impacts"]["context"] == "high"
    assert _context_impact(path) == "high"

    del bundle["checklist_impacts"]
    path.write_text(json.dumps(bundle))
    # Bundles from before impacts were recorded replay with context at medium.
    assert _context_impact(path) == "medium"

    bundle["checklist_impacts"] = {"context": "urgent"}
    path.write_text(json.dumps(bundle))
    with pytest.raises(Exception, match="checklist_impacts"):
        default_engine_factory(path)
