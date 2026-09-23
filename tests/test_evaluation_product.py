import json
from pathlib import Path

from prompt_enhancer.evaluation import Dataset, EvaluationHarness
from prompt_enhancer.evaluation.__main__ import main as evaluation_main


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
