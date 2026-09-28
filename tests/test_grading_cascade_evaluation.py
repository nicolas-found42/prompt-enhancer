from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt_enhancer.evaluation.grading_cascade import evaluate_cascade_cases


def test_known_answer_comparison_reports_all_three_stages_and_denominators() -> None:
    path = (
        Path(__file__).parent / "fixtures/evaluation/grading_cascade_known_answer.json"
    )
    report = evaluate_cascade_cases(json.loads(path.read_text()))

    assert report["provenance"] == "synthetic_known_answer"
    assert report["direct"]["coverage"] == {"count": 2, "denominator": 6, "rate": 2 / 6}
    assert report["confirmation"]["coverage"] == {
        "count": 4,
        "denominator": 6,
        "rate": 4 / 6,
    }
    assert report["full_cascade"]["coverage"] == {
        "count": 5,
        "denominator": 6,
        "rate": 5 / 6,
    }
    assert report["full_cascade"]["false_pass_rate"] == {
        "count": 0,
        "denominator": 3,
        "rate": 0.0,
    }
    assert report["escalation_rate"] == {"count": 2, "denominator": 6, "rate": 2 / 6}
    assert report["measured_cost_usd"] == 0.008
    assert report["production_accuracy"] == "unavailable"


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_cascade_evaluation_rejects_nonfinite_cost(cost: float | int) -> None:
    case = {
        "id": "case",
        "label_pass": True,
        "initial_pass_probability": 0.9,
        "measured_cost_usd": cost,
    }
    with pytest.raises(ValueError, match="measured_cost_usd"):
        evaluate_cascade_cases({"cases": [case]})
