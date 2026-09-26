from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_issue44_screen_and_grade import _run_screened_round

from prompt_enhancer.evaluation.grading_comparison import compare_grading_results, main


def test_paired_comparison_separates_estimates_charges_and_independent_labels() -> None:
    after, _ = _run_screened_round()
    before = deepcopy(after)
    before["report"]["grading_observation"] = {
        "gateway_batch_calls": 12,
        "serialized_input_bytes_estimate": 8000,
        "judge_cost_usd_measured": 0.01,
    }
    before["report"].pop("test_screening")
    after["report"]["grading_observation"]["judge_cost_usd_measured"] = 0.012
    after["report"]["test_screening"]["screening_observation"][
        "judge_cost_usd_measured"
    ] = 0.002
    before["cost"]["cost_by_role"] = {"judge": 0.01}
    after["cost"]["cost_by_role"] = {"judge": 0.014}

    report = compare_grading_results(
        [{"case_id": "same", "result": before}],
        [{"case_id": "same", "result": after}],
        unsafe_labels={"same": ["t0"]},
    )

    case = report["cases"][0]
    assert case["grading_requests"] == {"before": 12.0, "after": 4.0, "delta": -8.0}
    assert case["serialized_input_bytes_estimate"]["delta"] < 0
    assert case["grading_agreement"]["rate"] == 1.0
    assert case["per_role_cost_usd_measured"]["before"] == {"judge": 0.01}
    assert case["phase_judge_cost_usd_measured"]["screening_after"] == 0.002
    assert case["above_ten_percent_increase"] is True
    assert "screening cache" in case["remedy"]
    assert report["screen_false_positive_rate"] == {
        "labeled_unsafe_tests": 1,
        "false_positives": 1,
        "rate": 1.0,
    }


def test_comparison_marks_missing_paired_evidence_unavailable() -> None:
    after, _ = _run_screened_round()
    before = deepcopy(after)
    before["report"].pop("grading_observation")
    before["report"]["per_model"]["panel"]["outputs"][0]["output"] = "different"

    report = compare_grading_results({"case_id": "single", "result": before}, after)

    assert report["cases"][0]["grading_requests"]["delta"] is None
    assert report["grading_agreement"]["rate"] is None
    assert report["screen_false_positive_rate"]["rate"] is None


def test_comparison_requires_matched_case_ids() -> None:
    with pytest.raises(ValueError, match="identical case IDs"):
        compare_grading_results(
            [{"case_id": "before", "result": {}}],
            [{"case_id": "after", "result": {}}],
        )


def test_comparison_cli_writes_unavailable_evidence_honestly(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    output = tmp_path / "report.json"
    for path in (before, after):
        path.write_text(json.dumps({"status": "completed", "report": {}, "cost": {}}))

    assert main([str(before), str(after), "--output", str(output)]) == 0

    report = json.loads(output.read_text())
    assert report["matched_cases"] == 1
    assert report["cases"][0]["grading_requests"]["delta"] is None
    assert report["screen_false_positive_rate"]["rate"] is None
