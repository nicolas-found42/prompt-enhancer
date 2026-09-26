from __future__ import annotations

import json
from pathlib import Path

from prompt_enhancer.evaluation.output_screen import evaluate_output_screen_fixture


def test_known_answer_screen_fixture_reports_rates_and_provenance() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/evaluation/output_screen_known_answer.json"
        ).read_text()
    )

    report = evaluate_output_screen_fixture(fixture)

    assert report["provenance"] == "synthetic_known_answer"
    assert report["false_positive_rate"] == {
        "count": 0,
        "denominator": 4,
        "rate": 0.0,
    }
    assert report["false_negative_rate"] == {
        "count": 0,
        "denominator": 2,
        "rate": 0.0,
    }
    assert report["unresolved_rate"] == {
        "count": 1,
        "denominator": 6,
        "rate": 1 / 6,
    }
    assert (
        next(case for case in report["cases"] if case["id"] == "uncertain-quote")[
            "status"
        ]
        == "screen_unresolved"
    )
