"""Known-answer fixture check for the weak-output screen.

Scripted probabilities exercise the screening and scoring code. They do not
measure Jev's accuracy on unseen outputs or adversarial production traffic.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..gateway import ScriptedGateway
from ..grading import grade_panel_with_jev
from ..runner import PanelResult


def _rate(count: int, denominator: int) -> dict[str, float | int | None]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def evaluate_output_screen_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Run labeled synthetic cases through the same public grader as a Round."""
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise ValueError("output-screen fixture requires a cases array")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in cases:
        if not isinstance(item, Mapping):
            raise ValueError("each output-screen case must be an object")
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("each output-screen case needs a unique id")
        seen.add(case_id)
        expected = item.get("expected_hazard")
        if not isinstance(expected, bool):
            raise ValueError("expected_hazard must be boolean")
        prompt = item.get("prompt")
        output = item.get("output")
        if not isinstance(prompt, str) or not isinstance(output, str):
            raise ValueError("each case requires prompt and output text")
        hazard_answers = (
            item.get("evaluator_steering"),
            item.get("judging_override"),
        )

        def decide(
            request: Mapping[str, Any],
            hazard_answers: tuple[Any, Any] = hazard_answers,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            key = str(request["key"])
            if key.startswith("output-screen:"):
                hazard = key.rsplit(":", 1)[-1]
                answer = hazard_answers[0 if hazard == "evaluator_steering" else 1]
                return (
                    {"type": "noul", "probability_true": answer}
                    if isinstance(answer, (int, float)) and not isinstance(answer, bool)
                    else {"type": "unknown", "answer": "unavailable"}
                )
            return {"type": "noul", "probability_true": 1.0}

        gateway = ScriptedGateway(decision=decide)
        grades, evidence = grade_panel_with_jev(
            [PanelResult(case_id, "fixture-weak", 0, 7, output, prompt)],
            [
                {
                    "id": "answers",
                    "question": "Does the answer meet the request?",
                    "kind": "noul",
                    "expected": "yes",
                }
            ],
            gateway,
            judge_model=gateway.jev_model,
            run_id=f"output-screen-fixture:{case_id}",
            shared_state=True,
            output_screen=True,
        )
        screen = next(
            entry["output_screen"] for entry in evidence if "output_screen" in entry
        )
        results.append(
            {
                "id": case_id,
                "expected_hazard": expected,
                "status": screen["status"],
                "reason": screen["reason"],
                "score": grades[case_id].sample_scores[0],
            }
        )
    negatives = [case for case in results if not case["expected_hazard"]]
    positives = [case for case in results if case["expected_hazard"]]
    return {
        "fixture": str(fixture.get("name", "unnamed")),
        "provenance": str(fixture.get("provenance", "unspecified")),
        "mode": "scripted_known_answer",
        "production_robustness": "unavailable",
        "cases": results,
        "false_positive_rate": _rate(
            sum(case["status"] == "steering_detected" for case in negatives),
            len(negatives),
        ),
        "false_negative_rate": _rate(
            sum(case["status"] == "screen_clear" for case in positives),
            len(positives),
        ),
        "unresolved_rate": _rate(
            sum(case["status"] == "screen_unresolved" for case in results),
            len(results),
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate scripted output-screen cases"
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    report = evaluate_output_screen_fixture(fixture)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
