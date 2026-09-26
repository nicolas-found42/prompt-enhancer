from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest

from prompt_enhancer.diagnosis import (
    ChecklistItem,
    DiagnosisRubric,
    GapImpact,
    TaskType,
)
from prompt_enhancer.evaluation.__main__ import main as evaluation_main
from prompt_enhancer.evaluation.order_bias import (
    OrderBiasManifest,
    analyze_order_bias,
    build_order_bias_requests,
    capture_order_bias,
    question_schema_digest,
)
from prompt_enhancer.gateway import GatewayConfig, HttpGateway, HttpTransport
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore

PromptOptimizer = partial(PromptOptimizer, writer_instruction_version=4)


def _manifest(*, kind: str = "choice", provenance: str = "synthetic_known_answer"):
    cases = []
    for index in range(30):
        test = {
            "id": "success",
            "question": "Does the answer satisfy the requested outcome?",
            "kind": kind,
            "expected": "pass" if kind == "choice" else "acceptable",
        }
        if kind == "choice":
            test.update(
                options=["pass", "fail", "unknown"],
                option_descriptions={
                    "pass": "The requested outcome is satisfied.",
                    "fail": "The requested outcome is not satisfied.",
                    "unknown": "The evidence does not decide.",
                },
            )
        else:
            test.update(levels=["poor", "acceptable", "excellent"])
        cases.append(
            {
                "id": f"case-{index:02d}",
                "source_group": f"source-{index:02d}",
                "provenance": provenance,
                "prompt": f"Prompt {index}",
                "output": f"Output {index}",
                "success_tests": [test],
            }
        )
    return OrderBiasManifest.from_dict({"name": "order-bias-test", "cases": cases})


def _events_for(manifest, answer_for):
    events = []
    for request in build_order_bias_requests(manifest, repetitions=3):
        events.append(
            {
                "request_id": request["key"],
                "request": request,
                "answer": answer_for(request),
                "answered_by": "jev-test-snapshot",
                "usage": {"input_tokens": 12, "output_tokens": 4},
                "cost_usd": 0.0001,
                "latency_ms": 18.0,
            }
        )
    return events


def test_score_order_comparison_preserves_semantic_levels_and_stable_ties() -> None:
    manifest = _manifest(kind="score")

    def answer(request):
        reversed_order = ":reverse:" in request["key"]
        return {
            "type": "score",
            "score": 1.5,
            "probabilities": (
                {"0": 0.3, "1": 0.6, "2": 0.1}
                if reversed_order
                else {"0": 0.1, "1": 0.6, "2": 0.3}
            ),
        }

    report, _artifact = analyze_order_bias(manifest, _events_for(manifest, answer))

    group = report["groups"][0]
    assert group["metrics"]["mean_expected_mass_direct"] == pytest.approx(0.9)
    assert group["metrics"]["mean_expected_mass_reversed"] == pytest.approx(0.9)
    assert group["metrics"]["mean_signed_shift"] == pytest.approx(0.0)
    assert group["metrics"]["semantic_argmax_flip_rate"] == 0.0


def test_order_bias_replay_is_repeatable_and_synthetic_results_do_not_activate() -> (
    None
):
    manifest = _manifest()

    def answer(request):
        if ":reverse:" in request["key"]:
            probabilities = {"pass": 0.8, "fail": 0.1, "unknown": 0.1}
        else:
            probabilities = {"pass": 0.8, "fail": 0.1, "unknown": 0.1}
        return {"type": "choice", "choice": "pass", "probabilities": probabilities}

    events = _events_for(manifest, answer)
    report, artifact = analyze_order_bias(manifest, events)

    group = report["groups"][0]
    assert group["recommendation"] == "single"
    assert group["runtime_eligible"] is False
    assert artifact.to_dict()["groups"][0]["runtime_eligible"] is False
    report_again, artifact_again = analyze_order_bias(manifest, events)
    assert json.dumps(report_again, sort_keys=True) == json.dumps(
        report, sort_keys=True
    )
    assert json.dumps(artifact_again.to_dict(), sort_keys=True) == json.dumps(
        artifact.to_dict(), sort_keys=True
    )


def test_synthetic_and_matched_evidence_have_separate_policy_groups() -> None:
    synthetic = _manifest()
    matched_cases = [
        {
            **case.to_dict(),
            "id": f"matched-{case.id}",
            "source_group": f"matched-{case.source_group}",
            "provenance": "matched_recording",
        }
        for case in synthetic.cases
    ]
    manifest = OrderBiasManifest.from_dict(
        {
            "name": "mixed-evidence",
            "cases": [*(case.to_dict() for case in synthetic.cases), *matched_cases],
        }
    )
    events = _events_for(
        manifest,
        lambda _request: {
            "type": "choice",
            "choice": "pass",
            "probabilities": {"pass": 0.9, "fail": 0.05, "unknown": 0.05},
        },
    )

    report, artifact = analyze_order_bias(manifest, events)

    assert len(report["groups"]) == 2
    by_source = {group["evidence_provenance"]: group for group in report["groups"]}
    assert by_source["synthetic_known_answer"]["runtime_eligible"] is False
    assert by_source["matched_recording"]["runtime_eligible"] is True
    assert {group["evidence_provenance"] for group in artifact.to_dict()["groups"]} == {
        "synthetic_known_answer",
        "matched_recording",
    }


def test_score_ties_choose_the_same_semantic_level_in_both_orders() -> None:
    manifest = _manifest(kind="score")

    def answer(request):
        reverse = ":reverse:" in request["key"]
        return {
            "type": "score",
            "score": 0.5,
            "probabilities": (
                {"0": 0.0, "1": 0.5, "2": 0.5}
                if reverse
                else {"0": 0.5, "1": 0.5, "2": 0.0}
            ),
        }

    report, _artifact = analyze_order_bias(manifest, _events_for(manifest, answer))

    assert report["groups"][0]["metrics"]["semantic_argmax_flip_rate"] == 0.0


def test_matched_order_bias_recommends_mean_pair_when_shift_exceeds_repeat_noise() -> (
    None
):
    manifest = _manifest(provenance="matched_recording")

    def answer(request):
        reverse = ":reverse:" in request["key"]
        probabilities = (
            {"pass": 0.4, "fail": 0.5, "unknown": 0.1}
            if reverse
            else {"pass": 0.9, "fail": 0.05, "unknown": 0.05}
        )
        return {"type": "choice", "choice": "pass", "probabilities": probabilities}

    report, artifact = analyze_order_bias(manifest, _events_for(manifest, answer))

    group = report["groups"][0]
    assert group["recommendation"] == "mean_pair"
    assert group["runtime_eligible"] is True
    assert artifact.to_dict()["groups"][0]["recommendation"] == "mean_pair"


def test_order_bias_with_mixed_group_effects_is_inconclusive() -> None:
    manifest = _manifest(provenance="matched_recording")

    def answer(request):
        shifted_source = request["source_group"] == "source-00"
        shifted = ":reverse:" in request["key"] and shifted_source
        probability = 0.5 if shifted else 0.9
        return {
            "type": "choice",
            "choice": "pass",
            "probabilities": {
                "pass": probability,
                "fail": 1.0 - probability,
                "unknown": 0.0,
            },
        }

    report, artifact = analyze_order_bias(manifest, _events_for(manifest, answer))

    assert report["groups"][0]["recommendation"] == "insufficient_evidence"
    assert report["groups"][0]["runtime_eligible"] is False
    assert (
        artifact.to_dict()["groups"][0]["activation_block"] == "insufficient_evidence"
    )


def test_order_bias_missing_repeat_is_reported_and_cannot_activate_policy() -> None:
    manifest = _manifest(provenance="matched_recording")
    events = _events_for(
        manifest,
        lambda _request: {
            "type": "choice",
            "choice": "pass",
            "probabilities": {"pass": 0.9, "fail": 0.05, "unknown": 0.05},
        },
    )
    events[0]["answer"] = None

    report, artifact = analyze_order_bias(manifest, events)

    assert report["status"] == "partial"
    assert report["collection"]["missing_answers"] == 1
    assert artifact.to_dict()["groups"][0]["runtime_eligible"] is False


def test_order_bias_cli_strict_replay_writes_report_and_policy_artifact(
    tmp_path: Path,
) -> None:
    manifest = _manifest(provenance="matched_recording")
    input_path = tmp_path / "manifest.json"
    recording_path = tmp_path / "recording.json"
    report_path = tmp_path / "report.json"
    artifact_path = tmp_path / "policy.json"
    input_path.write_text(
        json.dumps(
            {
                "name": manifest.name,
                "cases": [case.to_dict() for case in manifest.cases],
            }
        ),
        encoding="utf-8",
    )
    requests = build_order_bias_requests(manifest, repetitions=3)
    events = _events_for(
        manifest,
        lambda _request: {
            "type": "choice",
            "choice": "pass",
            "probabilities": {"pass": 0.9, "fail": 0.05, "unknown": 0.05},
        },
    )
    # A strict recording binds every answer to the exact request, including its unique id.
    assert [item["key"] for item in requests] == [
        event["request_id"] for event in events
    ]
    recording_path.write_text(
        json.dumps(
            {
                "kind": "order-bias-recording",
                "dataset_digest": manifest.digest,
                "repetitions": 3,
                "events": events,
            }
        ),
        encoding="utf-8",
    )

    code = evaluation_main(
        [
            "order-bias",
            str(input_path),
            "--replay",
            str(recording_path),
            "--output",
            str(report_path),
            "--artifact",
            str(artifact_path),
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["evidence_sources"]["matched_recordings"]["source_groups"] == 30
    assert artifact["groups"][0]["recommendation"] == "single"
    assert artifact["groups"][0]["runtime_eligible"] is True


def test_live_collection_sends_three_independent_asks_in_each_option_order() -> None:
    manifest = OrderBiasManifest.from_dict(
        {
            "cases": [
                {
                    "id": "one",
                    "source_group": "source-one",
                    "provenance": "matched_recording",
                    "prompt": "Use the short format.",
                    "output": "A short answer.",
                    "success_tests": [
                        {
                            "id": "format",
                            "question": "Does the answer use the requested format?",
                            "kind": "choice",
                            "expected": "pass",
                            "options": ["pass", "fail", "unknown"],
                            "option_descriptions": {
                                "pass": "The answer uses the requested format.",
                                "fail": "The answer ignores the requested format.",
                                "unknown": "There is not enough evidence.",
                            },
                        }
                    ],
                }
            ]
        }
    )

    class InspectingTransport:
        def __init__(self) -> None:
            self.requests = []

        def request(self, _url, *, json, **_kwargs):
            self.requests.append(json)
            key = next(iter(json["questions"]))
            return {
                "status_code": 200,
                "json": {
                    "model": "typesafe/jev-test-snapshot",
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                    "answers": {
                        key: {
                            "type": "choice",
                            "choice": "pass",
                            "probabilities": {"pass": 0.8, "fail": 0.1, "unknown": 0.1},
                        }
                    },
                },
            }

    transport = InspectingTransport()
    gateway = HttpGateway(
        transport,
        config=GatewayConfig(
            openrouter_api_key="test-key",
            jev_model="typesafe/jev-test-snapshot",
            max_retries=0,
        ),
    )
    events = capture_order_bias(
        manifest,
        gateway,
        repetitions=3,
        budget_usd=0.20,
        request_cost_ceiling=0.01,
    )

    assert len(transport.requests) == 6
    assert len({next(iter(item["questions"])) for item in transport.requests}) == 6
    criteria_order = [
        list(next(iter(item["questions"].values()))["criteria"])
        for item in transport.requests
    ]
    assert criteria_order == [
        ["pass", "fail", "unknown"],
        ["pass", "fail", "unknown"],
        ["pass", "fail", "unknown"],
        ["unknown", "fail", "pass"],
        ["unknown", "fail", "pass"],
        ["unknown", "fail", "pass"],
    ]
    assert len(events) == 6


def test_live_collection_reserves_budget_before_each_independent_ask() -> None:
    from prompt_enhancer.gateway import ScriptedGateway

    manifest = OrderBiasManifest.from_dict(
        {
            "cases": [
                {
                    "id": "one",
                    "source_group": "source-one",
                    "prompt": "Use the short format.",
                    "output": "A short answer.",
                    "success_tests": [
                        {
                            "question": "Does it use the format?",
                            "kind": "choice",
                            "expected": "pass",
                            "options": ["pass", "fail"],
                            "option_descriptions": {
                                "pass": "Uses the format.",
                                "fail": "Does not use the format.",
                            },
                        }
                    ],
                }
            ]
        }
    )
    gateway = ScriptedGateway(
        decision=lambda *_args, **_kwargs: {
            "type": "choice",
            "choice": "pass",
            "probabilities": {"pass": 1.0, "fail": 0.0},
        }
    )

    events = capture_order_bias(
        manifest,
        gateway,
        repetitions=3,
        budget_usd=0.01,
        request_cost_ceiling=0.01,
    )

    assert len(events) == 1


def test_http_transport_serialization_preserves_choice_order() -> None:
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    serialized = []

    def opener(request, *, timeout):
        del timeout
        serialized.append(json.loads(request.data))
        return Response()

    transport = HttpTransport(opener=opener)
    for options in (["pass", "fail", "unknown"], ["unknown", "fail", "pass"]):
        transport.request(
            "https://example.test",
            method="POST",
            json={
                "questions": {"q": {"criteria": {option: option for option in options}}}
            },
        )

    assert [list(item["questions"]["q"]["criteria"]) for item in serialized] == [
        ["pass", "fail", "unknown"],
        ["unknown", "fail", "pass"],
    ]


def _runtime_policy(recommendation: str, *, snapshot: str = "jev-test-snapshot"):
    test = {
        "question": "Does the answer satisfy the requested outcome?",
        "kind": "choice",
        "expected": "pass",
        "options": ["pass", "fail", "unknown"],
        "option_descriptions": {
            "pass": "The requested outcome is satisfied.",
            "fail": "The requested outcome is not satisfied.",
            "unknown": "The evidence does not decide.",
        },
    }
    return {
        "schema_version": 1,
        "kind": "order-bias-policy-artifact",
        "name": "matched-order-test",
        "policy_version": "issue-43-order-bias-v1",
        "dataset_digest": "known-dataset-digest",
        "repetitions": 3,
        "groups": [
            {
                "primitive": "choice",
                "question_schema_digest": question_schema_digest(test),
                "question_schema": test,
                "snapshot": snapshot,
                "recommendation": recommendation,
                "runtime_eligible": True,
                "activation_block": None,
                "support": {"distinct_source_groups": 30},
            }
        ],
    }


def _runtime_score_policy(recommendation: str, *, snapshot: str = "jev-test-snapshot"):
    test = {
        "question": "How good is the answer?",
        "kind": "score",
        "expected": "acceptable",
        "levels": ["poor", "acceptable", "excellent"],
    }
    return {
        "schema_version": 1,
        "kind": "order-bias-policy-artifact",
        "name": "matched-score-order-test",
        "policy_version": "issue-43-order-bias-v1",
        "dataset_digest": "known-score-dataset-digest",
        "repetitions": 3,
        "groups": [
            {
                "primitive": "score",
                "question_schema_digest": question_schema_digest(test),
                "question_schema": test,
                "snapshot": snapshot,
                "evidence_provenance": "matched_recording",
                "recommendation": recommendation,
                "runtime_eligible": True,
                "activation_block": None,
                "support": {"distinct_source_groups": 30},
            }
        ],
    }


def _synthetic_runtime_policy():
    manifest = _manifest()
    events = _events_for(
        manifest,
        lambda _request: {
            "type": "choice",
            "choice": "pass",
            "probabilities": {"pass": 0.8, "fail": 0.1, "unknown": 0.1},
        },
    )
    return analyze_order_bias(manifest, events)[1].to_dict()


def _optimizer_gateway(
    snapshot: str = "jev-test-snapshot", *, success_test_kind: str = "choice"
):
    from prompt_enhancer.gateway import ScriptedGateway

    test = {
        "question": "Does the answer satisfy the requested outcome?",
        "kind": success_test_kind,
        "expected": (
            "yes"
            if success_test_kind == "noul"
            else "acceptable"
            if success_test_kind == "score"
            else "pass"
        ),
    }
    if success_test_kind == "choice":
        test["options"] = [
            {"value": "pass", "description": "The requested outcome is satisfied."},
            {"value": "fail", "description": "The requested outcome is not satisfied."},
            {"value": "unknown", "description": "The evidence does not decide."},
        ]
    elif success_test_kind == "score":
        test["question"] = "How good is the answer?"
        test["levels"] = ["poor", "acceptable", "excellent"]
    success_test = {"tests": [test]}

    def chat(_model, messages, *, role, **_kwargs):
        if role in {"weak", "strong_check"}:
            return {"choices": [{"message": {"content": "pass"}}]}
        state = json.loads(messages[1]["content"])
        if "strategies" in state:
            return json.dumps(
                {
                    item["name"]: f"Rewrite {item['name']}"
                    for item in state["strategies"]
                }
            )
        if "gaps" in state:
            return '{"gaps":{}}'
        return json.dumps(success_test)

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "task_type":
                selected = "general"
                probabilities = {selected: 1.0}
            elif key == "strategy_choice":
                selected = "specify_output_format"
                probabilities = {selected: 1.0}
            elif key.startswith("grade_"):
                criteria = request["criteria"]
                reversed_order = list(criteria)[0] == "unknown"
                pass_mass = 0.4 if reversed_order else 0.9
                probabilities = {
                    "pass": pass_mass,
                    "fail": 1.0 - pass_mass - 0.05,
                    "unknown": 0.05,
                }
                selected = "pass"
            else:
                criteria = request.get("criteria", request.get("options", {}))
                selected = next(iter(criteria))
                probabilities = {selected: 1.0}
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": probabilities,
                "confidence": 1.0,
            }
        if request.get("type") == "score" and key.startswith("grade_"):
            reversed_order = request["criteria"][0] == "excellent"
            return {
                "type": "score",
                "score": 1.5,
                "probabilities": (
                    {"0": 0.1, "1": 0.4, "2": 0.5}
                    if reversed_order
                    else {"0": 0.1, "1": 0.6, "2": 0.3}
                ),
            }
        return {"type": "noul", "probability_true": 0.99}

    return ScriptedGateway(
        jev_model=snapshot,
        chat=chat,
        decision=decide,
    )


@pytest.mark.parametrize(
    ("recommendation", "expected_mass", "orders"),
    [("single", 0.9, 1), ("mean_pair", 0.65, 2)],
)
def test_optimizer_applies_compatible_order_policy(
    recommendation: str, expected_mass: float, orders: int
) -> None:
    gateway = _optimizer_gateway()
    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (ChecklistItem("goal", "goal", GapImpact.HIGH),),
            ),
        )
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        diagnosis_rubric=rubric,
        grading_policy=_runtime_policy(recommendation),
    )

    result = optimizer.optimize(
        "Summarize the supplied report.",
        {"tier": "fast", "clarification_allowed": False},
    )

    assert result["status"] == "completed", (result["report"], gateway.decision_log)
    grading_requests = [
        event["question"]
        for event in gateway.decision_log
        if str(event["question"].get("key", "")).startswith("grade_")
    ]
    assert len(grading_requests) >= 2
    assert sum(request["key"].endswith("_first") for request in grading_requests) >= 1
    assert sum(request["key"].endswith("_second") for request in grading_requests) == (
        (len(grading_requests) // orders) * (orders - 1)
    )
    first_choice_grade = next(
        candidate["grade"]
        for candidate in result["report"]["candidates"]
        if candidate["candidate_id"] != "original"
    )
    assert first_choice_grade["per_model_samples"][
        "meta-llama/llama-3.1-8b-instruct"
    ] == [expected_mass]
    applied = result["report"]["grading_policy"]
    assert applied[0]["policy"] == recommendation
    assert applied[0]["reason"] == "compatible_order_bias_evidence"


def test_optimizer_aligns_reversed_score_levels_before_averaging() -> None:
    gateway = _optimizer_gateway(success_test_kind="score")
    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (ChecklistItem("goal", "goal", GapImpact.HIGH),),
            ),
        )
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        diagnosis_rubric=rubric,
        grading_policy=_runtime_score_policy("mean_pair"),
    )

    result = optimizer.optimize(
        "Summarize the supplied report.",
        {"tier": "fast", "clarification_allowed": False},
    )

    assert result["status"] == "completed"
    candidate = next(
        item
        for item in result["report"]["candidates"]
        if item["candidate_id"] != "original"
    )
    assert candidate["grade"]["per_model_samples"][
        "meta-llama/llama-3.1-8b-instruct"
    ] == [0.7]
    grading_requests = [
        event["question"]
        for event in gateway.decision_log
        if str(event["question"].get("key", "")).startswith("grade_")
    ]
    assert any(request["key"].endswith("_second") for request in grading_requests)


def test_optimizer_loads_saved_policy_from_server_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    policy_path = tmp_path / "order-bias-policy.json"
    policy_path.write_text(json.dumps(_runtime_policy("single")), encoding="utf-8")
    monkeypatch.setenv("PROMPT_ENHANCER_ORDER_BIAS_POLICY", str(policy_path))
    gateway = _optimizer_gateway()
    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (ChecklistItem("goal", "goal", GapImpact.HIGH),),
            ),
        )
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, diagnosis_rubric=rubric
    )

    result = optimizer.optimize(
        "Summarize the supplied report.",
        {"tier": "fast", "clarification_allowed": False},
    )

    assert result["report"]["grading_policy"][0]["policy"] == "single"


@pytest.mark.parametrize(
    ("artifact", "reason"),
    [
        (None, "no_order_bias_artifact"),
        (
            _runtime_policy("single", snapshot="another-jev-snapshot"),
            "incompatible_snapshot",
        ),
        (_synthetic_runtime_policy(), "synthetic_only_evidence"),
        (
            {
                **_runtime_policy("insufficient_evidence"),
                "groups": [
                    {
                        **_runtime_policy("insufficient_evidence")["groups"][0],
                        "runtime_eligible": False,
                        "activation_block": "insufficient_evidence",
                    }
                ],
            },
            "insufficient_evidence",
        ),
    ],
)
def test_optimizer_keeps_legacy_min_pair_without_compatible_evidence(
    artifact, reason: str
) -> None:
    gateway = _optimizer_gateway()
    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (ChecklistItem("goal", "goal", GapImpact.HIGH),),
            ),
        )
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        diagnosis_rubric=rubric,
        grading_policy=artifact,
    )

    result = optimizer.optimize(
        "Summarize the supplied report.",
        {"tier": "fast", "clarification_allowed": False},
    )

    requests = [
        event["question"]
        for event in gateway.decision_log
        if str(event["question"].get("key", "")).startswith("grade_")
    ]
    assert requests
    assert sum(request["key"].endswith("_second") for request in requests) > 0
    policy = result["report"]["grading_policy"][0]
    assert policy["policy"] == "legacy_min_pair"
    assert policy["reason"] == reason
    candidate = next(
        item
        for item in result["report"]["candidates"]
        if item["candidate_id"] != "original"
    )
    assert candidate["grade"]["per_model_samples"][
        "meta-llama/llama-3.1-8b-instruct"
    ] == [0.4]


def test_optimizer_keeps_noul_grading_as_one_direct_question() -> None:
    gateway = _optimizer_gateway(success_test_kind="noul")
    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (ChecklistItem("goal", "goal", GapImpact.HIGH),),
            ),
        )
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        diagnosis_rubric=rubric,
        grading_policy=_runtime_policy("single"),
    )

    result = optimizer.optimize(
        "Summarize the supplied report.",
        {"tier": "fast", "clarification_allowed": False},
    )

    requests = [
        event["question"]
        for event in gateway.decision_log
        if str(event["question"].get("key", "")).startswith("grade_")
    ]
    assert requests
    assert all(request["key"].endswith("_first") for request in requests)
    assert result["report"]["grading_policy"][0]["policy"] == "noul_direct"
