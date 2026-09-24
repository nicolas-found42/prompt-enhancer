import json
from pathlib import Path

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

    assert evaluation_main([
        str(fixtures / "mixed_dataset.json"), "--replay",
        str(fixtures / "replay_with_latency.json"), "--output", str(output),
    ]) == 0

    report = json.loads(output.read_text())
    assert [case["status"] for case in report["cases"]] == ["completed"] * 3
    assert report["diagnosis"]["excluded_failed_cases"] == 0


def test_replay_restores_recorded_case_cost_and_latency(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "evaluation"
    dataset = json.loads((fixtures / "mixed_dataset.json").read_text())
    first_id = dataset["cases"][0]["id"]
    replay = json.loads((fixtures / "replay_with_latency.json").read_text())
    replay["case_costs"] = {first_id: {"total": 0.25, "cost_by_role": {"judge": 0.1, "writer": 0.15}}}
    replay["case_latency_ms"] = {first_id: 4321}
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay))
    output = tmp_path / "report.json"

    assert evaluation_main([str(fixtures / "mixed_dataset.json"), "--replay", str(replay_path), "--output", str(output)]) == 0

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
            return {"type": "choice", "choice": "none", "probabilities": {"none": 1.0}, "confidence": 1.0}
        return {"type": "noul", "probability_true": 0.1, "confidence": 0.9}

    path = tmp_path / "recorded.json"
    gateway = RecordingGateway(ScriptedGateway(chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide), path)
    prompt = "Summarize the supplied article in three bullets."
    original = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(prompt)
    responses = json.loads(path.read_text())["responses"]
    replayed = PromptOptimizer(gateway=ReplayGateway(responses, strict=True), store=RunStore(":memory:")).optimize(prompt)

    assert responses
    assert replayed["status"] == original["status"]
    assert replayed["final_prompt"] == original["final_prompt"]


def test_replay_cli_reports_failed_cases_with_nonzero_exit(tmp_path: Path, capsys) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "evaluation"
    replay = tmp_path / "empty-replay.json"
    replay.write_text('{"responses":{}}')
    output = tmp_path / "report.json"

    assert evaluation_main([
        str(fixtures / "mixed_dataset.json"), "--replay", str(replay),
        "--output", str(output),
    ]) == 2
    assert all(case["status"] == "failed" for case in json.loads(output.read_text())["cases"])
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

    dataset = Dataset.from_dict([{"id": "one", "prompt": "Write a report", "source": "real"}])
    report = EvaluationHarness(ProductResultEngine()).run(dataset)

    assert report.improvement.comparable_cases == 1
    assert report.improvement.improved == 1
    assert report.cost.by_role == {"writer": 0.02, "judge": 0.01}
    assert report.cases[0].latency_ms == 125


def test_harness_resumes_only_with_source_answers_and_keeps_clarification_prediction() -> None:
    class PausingEngine:
        def __init__(self):
            self.resumed = []

        def optimize(self, prompt, options):
            return {"status": "needs_input", "run_id": "run-1", "report": {"diagnosis": {"confirmed_gaps": [{"key": "context"}]}}}

        def resume(self, run_id, answers):
            self.resumed.append((run_id, answers))
            return {
                "status": "completed", "final_prompt": "Answered request", "original_kept": False,
                "report": {"diagnosis": {"confirmed_gaps": [{"key": "context"}]},
                           "selection_evidence": {"original_score": 0.25, "winner_score": 0.5}},
            }

    engine = PausingEngine()
    dataset = Dataset.from_dict([{
        "id": "one", "prompt": "Plan a trip", "expected_gaps": ["clarification_need"],
        "clarification_answers": {"context": "Paris"}, "clarification_answer_provenance": "human",
        "evaluation_gaps": ["clarification_need"],
    }])
    case = EvaluationHarness(engine).run(dataset).cases[0]

    assert engine.resumed == [("run-1", {"context": "Paris"})]
    assert case.status == "completed"
    assert case.predicted_gaps == ("clarification_need",)
    assert case.score_delta == 0.25


def test_harness_rejects_unattributed_clarification_answers() -> None:
    class PausingEngine:
        def optimize(self, prompt, options):
            return {"status": "needs_input", "run_id": "run-1", "report": {}}

    dataset = Dataset.from_dict([{"id": "one", "prompt": "Plan a trip", "clarification_answers": {"context": "Paris"}}])
    case = EvaluationHarness(PausingEngine()).run(dataset).cases[0]

    assert case.status == "error"
    assert "require human or source provenance" in case.error


def test_dataset_round_trip_preserves_evaluation_scope_and_source_answers() -> None:
    dataset = Dataset.from_dict([{
        "id": "one", "prompt": "Plan a trip", "expected_gaps": ["context"],
        "evaluation_gaps": ["context"],
        "clarification_answers": {"context": "Paris"},
        "clarification_answer_provenance": "human",
    }])

    restored = Dataset.from_dict(dataset.to_dict())

    assert restored.cases[0].metadata == dataset.cases[0].metadata
    assert restored.digest == dataset.digest


def test_harness_retains_engine_failure_reason() -> None:
    class FailedEngine:
        def optimize(self, prompt, options):
            return {"status": "failed", "final_prompt": prompt, "original_kept": True,
                    "report": {"status": "failed", "error": "writer response was invalid"}}

    dataset = Dataset.from_dict([{"id": "one", "prompt": "Write a report", "source": "real", "expected_gaps": ["context"]}])
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

    dataset = Dataset.from_dict([
        {"id": "positive", "prompt": "Help with this issue", "source": "real", "expected_gaps": ["missing context"], "evaluation_gaps": ["context"]},
        {"id": "negative", "prompt": "Explain the supplied details", "source": "real", "expected_gaps": [], "evaluation_gaps": ["context"]},
    ])
    report = EvaluationHarness(ContextEngine()).run(dataset)

    assert report.diagnosis.labeled_cases == 2
    assert report.diagnosis.per_gap["context"].true_positives == 1
    assert report.diagnosis.per_gap["context"].false_positives == 1
    assert report.diagnosis.per_gap["context"].precision == 0.5


def test_harness_matches_human_gap_synonyms_to_checklist_keys() -> None:
    class DiagnosingEngine:
        def optimize(self, prompt, options):
            return {
                "status": "completed", "final_prompt": prompt, "original_kept": True,
                "report": {"diagnosis": {"confirmed_gaps": [
                    {"key": "done_criteria"}, {"key": "output_format"},
                ]}},
            }

    dataset = Dataset.from_dict([{
        "id": "labeled", "prompt": "Prepare a report", "source": "hand_labeled",
        "expected_gaps": ["definition of done", "missing output format"],
    }])
    report = EvaluationHarness(DiagnosingEngine()).run(dataset)

    assert report.diagnosis.micro.true_positives == 2
    assert report.diagnosis.micro.false_negatives == 0
    assert report.diagnosis.micro.false_positives == 0


def test_failed_cases_are_excluded_from_diagnosis_accuracy() -> None:
    class FailedEngine:
        def optimize(self, prompt, options):
            return {"status": "failed", "final_prompt": prompt, "report": {"diagnosis": {"confirmed_gaps": []}}}

    dataset = Dataset.from_dict([
        {"id": "failed", "prompt": "Ambiguous request", "source": "real", "expected_gaps": ["context"]},
    ])
    report = EvaluationHarness(FailedEngine()).run(dataset)

    assert report.diagnosis.labeled_cases == 0
    assert report.diagnosis.excluded_failed_cases == 1
    assert report.diagnosis.micro.false_negatives == 0
