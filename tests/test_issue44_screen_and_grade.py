"""Public new-protocol screening and one-output grading regressions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prompt_enhancer.catalog import JEV_MODEL, ModelInfo, StaticModelCatalog
from prompt_enhancer.config import Settings
from prompt_enhancer.evaluation.harness import default_engine_factory
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    ProviderError,
    ScriptedGateway,
)
from prompt_enhancer.grading import grade_panel_with_jev
from prompt_enhancer.jev import batch_decision_payload
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.runner import PanelResult
from prompt_enhancer.store import RunStore


def _run_screened_round(
    *,
    evaluator_hazard: bool = False,
    incomplete_screen: bool = False,
    partial_grading: bool = False,
    oversized_output: bool = False,
    screen_provider_error: bool = False,
    record_path: Path | None = None,
    writer_instruction_version: int = 5,
    output_screen: Mapping[str, float | None] | None = None,
    tier: str = "fast",
    grade_pass_probability: float = 1.0,
    confirmation_answers: tuple[float, float, float] | None = None,
    generated_test_count: int = 3,
    strong_evidence: Mapping[str, Any] | None = None,
    verification_answers: tuple[float, float, float, float] | None = None,
    decision_policy: Any = None,
    priced_catalog: bool = False,
    criterion_text: str | None = None,
    settings: Settings | None = None,
    confirmation_provider_error: bool = False,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    batches: list[list[dict[str, Any]]] = []
    prompt = "Read the background notes. Summarize the report."

    def chat(
        _model: str, messages: Sequence[Mapping[str, str]], *, role: str, **_kwargs: Any
    ) -> str:
        if role == "judge_escalation":
            return json.dumps(strong_evidence or {})
        if role == "writer":
            state = json.loads(messages[1]["content"])
            assert "strategies" not in state
            return json.dumps(
                {
                    "tests": [
                        {
                            "question": criterion_text
                            or f"Does the answer satisfy criterion {index}?",
                            "kind": "noul",
                            "expected": "yes",
                        }
                        for index in range(generated_test_count)
                    ]
                }
            )
        output = (
            "pass"
            if role == "strong_check" or "### Task" in messages[0]["content"]
            else "fail"
        )
        return output + ("x" * 65_000 if oversized_output and role == "weak" else "")

    def decide(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "strategy_choice":
                choice = "restructure_lossless"
            elif key.startswith("restructure_lossless:role:"):
                choice = (
                    "context"
                    if "background" in request["state"]["target_unit_text"]
                    else "task"
                )
            else:
                choice = "general" if key == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        if key.startswith("success-test-screen:"):
            if incomplete_screen and key.endswith("assessability"):
                return {"type": "unknown", "answer": "yes"}
            probability = (
                (0.99 if evaluator_hazard else 0.01)
                if key.endswith("evaluator_instructions")
                else 0.99
            )
        elif key.startswith("grade_"):
            probability = (
                grade_pass_probability if request["state"]["output"] == "pass" else 0.0
            )
        elif key.startswith("grade-confirm:"):
            hazard = key.rsplit(":", 1)[-1]
            probabilities = dict(
                zip(
                    ("sufficient", "meets", "violation"),
                    confirmation_answers or (0.5, 0.5, 0.5),
                    strict=True,
                )
            )
            probability = probabilities[hazard]
        elif key.startswith("grade-verify:"):
            check = key.rsplit(":", 1)[-1]
            probabilities = dict(
                zip(
                    ("sufficient", "meets", "violation", "support"),
                    verification_answers or (0.5, 0.5, 0.5, 0.5),
                    strict=True,
                )
            )
            probability = probabilities[check]
        elif key.startswith("output-screen:"):
            hazard = (output_screen or {}).get(str(request["state"]["output"]), 0.01)
            if hazard is None:
                return {"type": "unknown", "answer": "yes"}
            probability = hazard
        elif key.startswith(
            ("gap:goal", "strategy_recheck:restructure_lossless", "fidelity:meaning")
        ):
            probability = 1.0
        else:
            probability = 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    class CountingGateway(ScriptedGateway):
        def decide_batch(
            self,
            requests: Sequence[Mapping[str, Any]],
            *,
            role: str = "judge",
            run_id: str | None = None,
        ) -> list[Any]:
            batches.append([dict(request) for request in requests])
            if (
                screen_provider_error
                and requests
                and str(requests[0]["key"]).startswith("success-test-screen:")
            ):
                raise ProviderError("scripted", self.jev_model, None, role=role)
            if (
                partial_grading
                and requests
                and str(requests[0]["key"]).startswith("grade_")
            ):
                return super().decide_batch(requests[:-1], role=role, run_id=run_id)
            if (
                confirmation_provider_error
                and requests
                and str(requests[0]["key"]).startswith("grade-confirm:")
            ):
                raise ProviderError("scripted", self.jev_model, None, role=role)
            return super().decide_batch(requests, role=role, run_id=run_id)

    catalog = (
        StaticModelCatalog(
            [
                ModelInfo(
                    id="glm-5.3-flash",
                    provider="go",
                    input_cost_per_token=0.0000001,
                    output_cost_per_token=0.0000002,
                )
            ],
            [
                ModelInfo(
                    id=JEV_MODEL,
                    provider="openrouter",
                    input_cost_per_token=0.0000001,
                    output_cost_per_token=0.0000002,
                )
            ],
        )
        if priced_catalog
        else None
    )
    gateway = CountingGateway(chat=chat, decision=decide, catalog=catalog)
    recording = RecordingGateway(gateway, record_path) if record_path else None
    if recording is not None:
        recording.writer_instruction_version = writer_instruction_version
        recording.faithfulness_threshold = 0.8
    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=recording or gateway,
        config=settings or Settings(),
        writer_instruction_version=writer_instruction_version,
        decision_policy=decision_policy,
    ).optimize(prompt, {"tier": tier, "clarification_allowed": False})
    return result, batches


def test_new_optimizer_screens_three_tests_and_groups_four_outputs() -> None:
    result, batches = _run_screened_round()

    assert result["status"] == "completed"
    assert result["original_kept"] is False
    screening = result["report"]["test_screening"]
    assert len(screening["screening_checks"]) == 3
    assert all(check["accepted"] for check in screening["screening_checks"])
    fidelity_index = next(
        index
        for index, batch in enumerate(batches)
        if batch and str(batch[0]["key"]).startswith("fidelity:")
    )
    grade_batches = [
        batch
        for batch in batches[:fidelity_index]
        if batch and str(batch[0]["key"]).startswith("grade_")
    ]
    assert len(grade_batches) == 4
    assert all(len(batch) == 3 for batch in grade_batches)
    for batch in grade_batches:
        _, envelope = batch_decision_payload(batch, model=gateway_model(batch))
        assert "items" not in envelope["state"]
        assert list(envelope["state"]) == ["prompt", "output", "success_tests"]
        assert len(envelope["state"]["success_tests"]) == 3
        assert all(question["state"] == envelope["state"] for question in batch)
        assert all(
            "Does the answer satisfy" not in json.dumps(question["question"])
            for question in batch
        )


def gateway_model(batch: Sequence[Mapping[str, Any]]) -> str:
    return str(batch[0]["model"])


def test_screened_out_or_unknown_tests_cannot_claim_an_improvement() -> None:
    for options, expected_reason in (
        ({"evaluator_hazard": True}, "evaluator_instructions"),
        ({"incomplete_screen": True}, "incomplete_screening"),
        ({"screen_provider_error": True}, "screen_provider_error"),
    ):
        result, batches = _run_screened_round(**options)
        assert result["original_kept"] is True
        assert result["report"]["tests"] == []
        assert all(
            check["reason"] == expected_reason
            for check in result["report"]["test_screening"]["screening_checks"]
        )
        assert not any(
            batch and str(batch[0]["key"]).startswith("grade_") for batch in batches
        )


def test_partial_or_oversized_grading_keeps_original_without_human_prompt() -> None:
    for options in ({"partial_grading": True}, {"oversized_output": True}):
        result, batches = _run_screened_round(**options)
        assert result["status"] == "completed"
        assert result["original_kept"] is True
        rejected = result["report"]["selection_evidence"]["rejected_candidates"]
        assert any(
            "weak-panel grading was incomplete or oversized"
            in item["rejection_reasons"]
            for item in rejected
        )
        if options.get("oversized_output"):
            assert not any(
                batch
                and str(batch[0]["key"]).startswith("grade_")
                and len(batch[0]["state"]["output"]) > 64_000
                for batch in batches
            )


def test_shared_state_preserves_noul_choice_score_and_approved_descriptions() -> None:
    batches: list[list[dict[str, Any]]] = []
    tests = [
        {"id": "t0", "question": "Does it answer?", "kind": "noul", "expected": "yes"},
        {
            "id": "t1",
            "question": "Which response?",
            "kind": "choice",
            "expected": "pass",
            "options": ["pass", "fail", "unknown"],
            "option_descriptions": {
                "pass": "Answers the user's request.",
                "fail": "Does not answer the request.",
                "unknown": "The answer is insufficient to choose.",
            },
        },
        {
            "id": "t2",
            "question": "How complete?",
            "kind": "score",
            "expected": "good",
            "levels": ["bad", "good"],
        },
    ]

    def decide(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        key = str(request["key"])
        if request["type"] == "noul":
            return {"type": "noul", "probability_true": 0.9}
        if request["type"] == "choice":
            return {
                "type": "choice",
                "choice": "pass",
                "probabilities": {"pass": 0.8, "fail": 0.1, "unknown": 0.1},
                "confidence": 0.9,
            }
        assert request["type"] == "score"
        return {
            "type": "score",
            "score": 0.7,
            "probabilities": {"0": 0.7, "1": 0.3}
            if key.endswith("_second")
            else {"0": 0.3, "1": 0.7},
            "legend": {
                str(index): level for index, level in enumerate(request["criteria"])
            },
        }

    class CapturingGateway(ScriptedGateway):
        def decide_batch(
            self,
            requests: Sequence[Mapping[str, Any]],
            *,
            role: str = "judge",
            run_id: str | None = None,
        ) -> list[Any]:
            batches.append([dict(request) for request in requests])
            return super().decide_batch(requests, role=role, run_id=run_id)

    gateway = CapturingGateway(decision=decide)
    grades, _ = grade_panel_with_jev(
        [PanelResult("candidate", "weak", 0, 7, "An answer.", "A prompt.")],
        tests,
        gateway,
        judge_model=gateway.jev_model,
        run_id="one-output",
        shared_state=True,
    )

    assert grades["candidate"].sample_scores == (0.7,)
    assert len(batches) == 1
    assert len(batches[0]) == 5
    _, envelope = batch_decision_payload(batches[0], model=gateway.jev_model)
    assert "items" not in envelope["state"]
    assert envelope["state"]["success_tests"]["t1"]["criterion"] == "Which response?"
    assert all(
        "Which response?" not in json.dumps(request["question"])
        for request in batches[0]
    )
    choices = [request for request in batches[0] if request["type"] == "choice"]
    assert choices[0]["criteria"]["pass"] == "Answers the user's request."
    scores = [request for request in batches[0] if request["type"] == "score"]
    assert [request["criteria"] for request in scores] == [
        ["bad", "good"],
        ["good", "bad"],
    ]


def test_current_screening_and_grading_protocol_strictly_replays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "screened-replay.json"
    original, _ = _run_screened_round(record_path=path)
    recorded = json.loads(path.read_text())
    assert recorded["writer_instruction_version"] == 5
    engine = default_engine_factory(path)
    engine.store = RunStore(":memory:")

    replayed = engine.optimize(
        "Read the background notes. Summarize the report.",
        {"tier": "fast", "clarification_allowed": False},
    )

    assert replayed["status"] == "completed"
    assert replayed["final_prompt"] == original["final_prompt"]
    assert (
        replayed["report"]["test_screening"]["screening_version"]
        == original["report"]["test_screening"]["screening_version"]
    )


def test_http_transport_sends_four_single_output_grading_requests() -> None:
    class GradingTransport:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def request(self, url: str, **kwargs: Any) -> dict[str, Any]:
            self.requests.append({"url": url, **kwargs})
            payload = kwargs["json"]
            probability = 0.9 if payload["state"]["output"] == "pass" else 0.1
            return {
                "status_code": 200,
                "json": {
                    "model": JEV_MODEL,
                    "answers": {
                        key: {"type": "noul", "noul": probability}
                        for key in payload["questions"]
                    },
                    "usage": {"input_tokens": 20, "output_tokens": 3},
                },
                "headers": {},
            }

    transport = GradingTransport()
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(openrouter_api_key="test", max_retries=0),
    )
    tests = [
        {
            "id": f"t{index}",
            "question": f"Criterion {index}",
            "kind": "noul",
            "expected": "yes",
        }
        for index in range(3)
    ]
    panel = [
        PanelResult(candidate, model, 0, 7, output, f"Prompt {candidate}")
        for candidate, output in (("original", "fail"), ("candidate", "pass"))
        for model in ("weak-a", "weak-b")
    ]
    grades, _ = grade_panel_with_jev(
        panel,
        tests,
        gateway,
        judge_model=JEV_MODEL,
        run_id="recorded",
        shared_state=True,
    )

    assert len(transport.requests) == 4
    for request in transport.requests:
        payload = request["json"]
        assert "items" not in payload["state"]
        assert set(payload["state"]) == {"prompt", "output", "success_tests"}
        assert len(payload["questions"]) == 3
    assert grades["candidate"].worst == 1.0
    assert grades["original"].worst == 0.0
