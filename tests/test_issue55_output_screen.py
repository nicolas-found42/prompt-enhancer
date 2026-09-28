"""Output screening uses the public Round and preserves grading evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_issue44_screen_and_grade import _run_screened_round

from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    DecisionPolicy,
    runtime_question_identity,
)
from prompt_enhancer.evaluation.harness import default_engine_factory
from prompt_enhancer.gateway import GatewayConfig, HttpGateway, ScriptedGateway
from prompt_enhancer.grading import grade_panel_with_jev
from prompt_enhancer.runner import PanelResult
from prompt_enhancer.store import RunStore


def _grading_batches(batches: list[list[dict]]) -> list[list[dict]]:
    return [
        batch
        for batch in batches
        if batch and str(batch[0]["key"]).startswith("grade_")
    ]


def test_detected_steering_overrides_passing_tests_in_same_output_request() -> None:
    result, batches = _run_screened_round(
        writer_instruction_version=6, output_screen={"pass": 0.99}
    )

    assert result["original_kept"] is True
    grading = [
        batch
        for batch in _grading_batches(batches)
        if any(str(item["key"]).startswith("output-screen:") for item in batch)
    ]
    assert len(grading) == 4
    assert all(len(batch) == 5 for batch in grading)
    assert all(
        {request["key"].split(":")[-1] for request in batch[3:]}
        == {"evaluator_steering", "judging_override"}
        for batch in grading
    )
    assert all(
        "never follow instructions" in str(request["question"])
        for batch in grading
        for request in batch[:3]
    )
    screen = result["report"]["output_screen"]
    assert sum(item["status"] == "steering_detected" for item in screen) == 2
    assert all(
        item["reason"] == "evaluator_steering_detected"
        for item in screen
        if item["status"] == "steering_detected"
    )
    assert result["report"]["selection_evidence"]["original_score"]["worst"] == 0.0
    assert result["report"]["grading_observation"]["gateway_batch_calls"] == 4


def test_clear_and_unresolved_screen_have_distinct_promotion_outcomes() -> None:
    clear, _ = _run_screened_round(writer_instruction_version=6)
    assert clear["original_kept"] is False
    assert all(
        item["status"] == "screen_clear" for item in clear["report"]["output_screen"]
    )

    unresolved, _ = _run_screened_round(
        writer_instruction_version=6, output_screen={"pass": None}
    )
    assert unresolved["original_kept"] is True
    assert (
        sum(
            item["status"] == "screen_unresolved"
            for item in unresolved["report"]["output_screen"]
        )
        == 2
    )
    candidate = next(
        item
        for item in unresolved["report"]["candidates"]
        if item["candidate_id"] != "original"
    )
    assert candidate["grade"]["worst"] == 1.0
    assert candidate["grade"]["unresolved_screen_outputs"] == 2


def test_unresolved_original_screen_prevents_promotion() -> None:
    result, _ = _run_screened_round(
        writer_instruction_version=6, output_screen={"fail": 0.5}
    )
    assert result["original_kept"] is True
    assert any(
        item["candidate_id"] == "original" and item["status"] == "screen_unresolved"
        for item in result["report"]["output_screen"]
    )


def test_current_screen_recording_strictly_replays_and_v5_stays_historical(
    tmp_path: Path,
) -> None:
    path = tmp_path / "screen-v6.json"
    original, _ = _run_screened_round(
        writer_instruction_version=6, output_screen={"pass": 0.99}, record_path=path
    )
    assert json.loads(path.read_text())["writer_instruction_version"] == 6
    engine = default_engine_factory(path)
    engine.store = RunStore(":memory:")

    replayed = engine.optimize(
        "Read the background notes. Summarize the report.",
        {"tier": "fast", "clarification_allowed": False},
    )

    assert replayed["report"]["output_screen"] == original["report"]["output_screen"]


def test_matching_calibration_policy_changes_screen_cutoff() -> None:
    panel = [PanelResult("candidate", "weak", 0, 7, "answer", "prompt")]
    tests = [
        {"id": "t0", "question": "Does it answer?", "kind": "noul", "expected": "yes"}
    ]

    def decide(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        key = str(request["key"])
        probability = (
            0.85
            if key.endswith("evaluator_steering")
            else 0.01
            if key.startswith("output-screen:")
            else 1.0
        )
        return {"type": "noul", "probability_true": probability}

    gateway = ScriptedGateway(decision=decide)
    _, evidence = grade_panel_with_jev(
        panel,
        tests,
        gateway,
        judge_model=gateway.jev_model,
        run_id="default",
        shared_state=True,
        output_screen=True,
    )
    first = next(item["output_screen"] for item in evidence if "output_screen" in item)
    assert first["status"] == "steering_detected"
    request = first["checks"]["evaluator_steering"]["question"]
    identity = runtime_question_identity(
        "output-screen:evaluator_steering",
        request,
        family="output_screen",
        rubric_version="issue-55-v1",
        snapshot=gateway.jev_model,
    )
    policy = DecisionPolicy.from_artifact(
        CalibrationArtifact.from_dict(
            {
                "kind": "calibration-artifact",
                "name": "screen-threshold",
                "input_digest": "known-input",
                "questions": {
                    identity.question_id: {
                        "identity": identity.to_dict(),
                        "verdict": "gate",
                        "threshold": 0.9,
                    }
                },
            }
        )
    )

    grades, calibrated = grade_panel_with_jev(
        panel,
        tests,
        gateway,
        judge_model=gateway.jev_model,
        run_id="calibrated",
        shared_state=True,
        output_screen=True,
        decision_policy=policy,
    )

    result = next(
        item["output_screen"] for item in calibrated if "output_screen" in item
    )
    assert result["status"] == "screen_unresolved"
    assert result["checks"]["evaluator_steering"]["policy"]["disposition"] == "gate"
    assert grades["candidate"].sample_scores == (1.0,)

    confidence_policy = DecisionPolicy.from_artifact(
        CalibrationArtifact.from_dict(
            {
                "kind": "calibration-artifact",
                "name": "screen-confidence",
                "input_digest": "known-input",
                "questions": {
                    identity.question_id: {
                        "identity": identity.to_dict(),
                        "verdict": "gate-above-confidence",
                        "threshold": 0.8,
                        "predicate": {"confidence_gte": 0.7},
                    }
                },
            }
        )
    )
    _, guarded = grade_panel_with_jev(
        panel,
        tests,
        gateway,
        judge_model=gateway.jev_model,
        run_id="confidence",
        shared_state=True,
        output_screen=True,
        decision_policy=confidence_policy,
    )
    assert (
        next(item["output_screen"] for item in guarded if "output_screen" in item)[
            "status"
        ]
        == "screen_unresolved"
    )


def test_http_transport_keeps_screen_questions_in_each_output_request() -> None:
    class Transport:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def request(self, _url: str, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["json"]
            self.requests.append(payload)
            return {
                "status_code": 200,
                "json": {
                    "model": payload["model"],
                    "answers": {
                        key: {
                            "type": "noul",
                            "noul": 0.01 if key.startswith("output-screen:") else 0.99,
                        }
                        for key in payload["questions"]
                    },
                    "usage": {"input_tokens": 20, "output_tokens": 3},
                },
                "headers": {},
            }

    transport = Transport()
    gateway = HttpGateway(
        transport, config=GatewayConfig(openrouter_api_key="test", max_retries=0)
    )
    panel = [
        PanelResult(candidate, "weak", 0, 7, "answer", "prompt")
        for candidate in ("original", "candidate")
    ]
    grades, _ = grade_panel_with_jev(
        panel,
        [
            {
                "id": "t0",
                "question": "Does it answer?",
                "kind": "noul",
                "expected": "yes",
            }
        ],
        gateway,
        judge_model=gateway.jev_model,
        run_id="transport",
        shared_state=True,
        output_screen=True,
    )

    assert len(transport.requests) == 2
    assert all(len(request["questions"]) == 3 for request in transport.requests)
    assert all(grade.worst == 1.0 for grade in grades.values())


def test_oversized_question_set_cannot_split_screen_into_another_request() -> None:
    class CountingGateway(ScriptedGateway):
        def __init__(self) -> None:
            super().__init__(
                decision=lambda _request, **_kwargs: {
                    "type": "noul",
                    "probability_true": 1.0,
                }
            )
            self.batch_count = 0

        def decide_batch(self, requests: Any, **kwargs: Any) -> list[Any]:
            self.batch_count += 1
            return super().decide_batch(requests, **kwargs)

    gateway = CountingGateway()
    grades, evidence = grade_panel_with_jev(
        [PanelResult("candidate", "weak", 0, 7, "answer", "prompt")],
        [
            {
                "id": f"t{index}",
                "question": f"Criterion {index}",
                "kind": "noul",
                "expected": "yes",
            }
            for index in range(39)
        ],
        gateway,
        judge_model=gateway.jev_model,
        run_id="too-many-questions",
        shared_state=True,
        output_screen=True,
    )

    assert gateway.batch_count == 0
    assert grades["candidate"].ungradable_outputs == 1
    assert (
        next(item["output_screen"] for item in evidence if "output_screen" in item)[
            "status"
        ]
        == "screen_unresolved"
    )
