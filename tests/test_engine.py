import json
from functools import partial

import pytest

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.config import Settings
from prompt_enhancer.diagnosis import (
    ChecklistItem,
    DiagnosisRubric,
    GapImpact,
    TaskType,
)
from prompt_enhancer.gateway import ProviderError, ScriptedGateway

# These regressions pin the pre-screening request protocol; current screening
# and shared-state grading are exercised in test_issue44_screen_and_grade.py.
PromptOptimizer = partial(PromptOptimizer, writer_instruction_version=4)


def test_model_settings_do_not_expose_server_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_KEY", "go-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_no_test_gateway())

    assert "key" not in str(optimizer.get_model_settings())


def test_environment_cannot_change_fixed_jev_model(monkeypatch) -> None:
    monkeypatch.setenv("PROMPT_ENHANCER_JUDGE_MODEL", "another-judge")

    assert Settings.from_env().judge_model == "typesafe/jev-1.13-20260917"


def _no_test_gateway(*, gaps: tuple[str, ...] = ()) -> ScriptedGateway:
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = (
            0.99 if request.get("key") in {f"gap:{gap}" for gap in gaps} else 0.01
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
    )


def test_clear_prompt_is_returned_unchanged_and_persisted() -> None:
    prompt = "Summarize this article in three concise bullets for a busy reader."
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(store=store, gateway=_no_test_gateway())

    result = optimizer.optimize(prompt, {"tier": "Fast"})

    assert result["status"] == "completed"
    assert result["final_prompt"] == prompt
    assert result["original_kept"] is True
    assert result["report"]["status"] == "unverified"
    assert "no faithful success tests" in result["report"]["summary"].casefold()
    assert result["run_id"]

    record = store.get_run(result["run_id"])
    assert record is not None
    assert record["prompt"] == prompt
    assert record["options"]["tier"] == "fast"
    assert record["result"] == result
    assert record["cost"] == result["cost"]
    assert record["timing"] == result["timing"]


def test_unsupported_added_sentence_is_rejected_and_persisted() -> None:
    prompt = "Summarize the report in English."
    candidate = f"{prompt} Respond in French."
    store = RunStore(":memory:")
    requests = []

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                return json.dumps(
                    {
                        item["name"]: (
                            candidate
                            if item["name"] == "specify_output_format"
                            else "Translate the report into English."
                        )
                        for item in state["strategies"]
                    }
                )
            if "gaps" in state:
                return '{"gaps":{}}'
            return (
                '{"tests":[{"question":"Does the answer summarize the report?",'
                '"kind":"noul","expected":"yes"}]}'
            )
        output = (
            "pass" if role != "weak" or messages[0]["content"] != prompt else "fail"
        )
        return {"choices": [{"message": {"content": output}}]}

    def decide(request, **_kwargs):
        requests.append(request)
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "strategy_choice":
                selected = "specify_output_format"
            elif key.startswith("fidelity:sentence:"):
                selected = "new_requirement"
                return {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": {
                        "supported_by_original": 0.01,
                        "supported_by_assumption": 0.0,
                        "new_requirement": 0.99,
                        "unknown": 0.0,
                    },
                    "confidence": 0.99,
                }
            else:
                selected = "none"
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": {selected: 1.0},
                "confidence": 1.0,
            }
        if key.startswith("faithful:"):
            probability = 0.99
        elif key == "gap:output_format":
            probability = 0.99
        elif key.startswith("strategy_recheck:"):
            probability = float(
                key.endswith("specify_output_format")
                or key.endswith("add_done_criteria")
            )
        elif key.startswith("grade_"):
            probability = float(request["state"]["output"] == "pass")
        elif key == "fidelity:meaning":
            probability = 0.99
        else:
            probability = 0.01
        return {
            "type": "noul",
            "probability_true": probability,
            "confidence": 1.0,
        }

    class BatchRecordingGateway(ScriptedGateway):
        def __init__(self):
            super().__init__(chat=chat, decision=decide)
            self.batches = []

        def decide_batch(self, questions, **kwargs):
            self.batches.append(list(questions))
            return super().decide_batch(questions, **kwargs)

    gateway = BatchRecordingGateway()
    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (ChecklistItem("output_format", "output format", GapImpact.LOW),),
            ),
        )
    )
    optimizer = PromptOptimizer(store=store, gateway=gateway, diagnosis_rubric=rubric)

    result = optimizer.optimize(prompt, {"tier": "fast"})

    rejected = result["report"]["selection_evidence"]["rejected_candidates"]
    assert result["final_prompt"] == prompt
    rejected_by_strategy = {item["strategy"]: item for item in rejected}
    assert set(rejected_by_strategy) == {"specify_output_format", "add_done_criteria"}
    assert rejected_by_strategy["specify_output_format"]["eligible"] is False
    assert any(
        "Respond in French." in reason
        for reason in rejected_by_strategy["specify_output_format"]["rejection_reasons"]
    )
    fidelity = rejected_by_strategy["specify_output_format"]["metadata"]["fidelity"]
    support = fidelity["evidence"]["sentence_support"][0]
    assert support["sentence"] == "Respond in French."
    assert support["selected"] == "new_requirement"
    assert support["probability"] == 0.99
    assert support["source_gap"] == 1
    assert any(
        "Translate the report into English." in reason
        for reason in rejected_by_strategy["add_done_criteria"]["rejection_reasons"]
    )
    assert (
        rejected_by_strategy["specify_output_format"]["grade"]["mean"]
        > result["report"]["selection_evidence"]["original_score"]["mean"]
    )
    fidelity_requests = [
        request
        for request in requests
        if str(request.get("key", "")).startswith("fidelity:")
    ]
    assert len(fidelity_requests) == 2
    assert {request["type"] for request in fidelity_requests} == {"choice", "noul"}
    assert all("diagnosis" not in request["state"] for request in fidelity_requests)
    assert all(
        "Respond in French." in request["state"]["candidate_prompt"]
        for request in fidelity_requests
    )
    assert not any(
        "Translate the report into English." in request["state"]["candidate_prompt"]
        for request in fidelity_requests
    )
    fidelity_batches = [
        batch
        for batch in gateway.batches
        if batch and all(str(item["key"]).startswith("fidelity:") for item in batch)
    ]
    assert len(fidelity_batches) == 1
    assert [item["key"] for item in fidelity_batches[0]] == [
        "fidelity:sentence:change-0001:candidate-s0002",
        "fidelity:meaning",
    ]
    assert all(
        item["state"] == fidelity_batches[0][0]["state"] for item in fidelity_batches[0]
    )
    persisted = store.get_run(result["run_id"])
    assert persisted is not None
    persisted_rejected = persisted["result"]["report"]["selection_evidence"][
        "rejected_candidates"
    ]
    persisted_by_id = {item["candidate_id"]: item for item in persisted_rejected}
    assert all(
        persisted_by_id[item["candidate_id"]]["metadata"]["fidelity"]
        == item["metadata"]["fidelity"]
        for item in rejected
    )


def test_confirmed_answer_supports_an_authorized_gap_fill() -> None:
    prompt = "Summarize the report."
    candidate_sentence = "Respond in French."

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "gaps" in state:
                return json.dumps(
                    {
                        "gaps": {
                            "language": {
                                "options": [{"value": "French", "label": "French"}]
                            }
                        }
                    }
                )
            if "strategies" in state:
                return json.dumps(
                    {
                        item["name"]: f"{state['prompt']}\n\n{candidate_sentence}"
                        for item in state["strategies"]
                    }
                )
            return (
                '{"tests":[{"question":"Does the answer summarize the report?",'
                '"kind":"noul","expected":"yes"}]}'
            )
        return {"choices": [{"message": {"content": "pass"}}]}

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "task_type":
                selected = "general"
            elif key == "strategy_choice":
                selected = "add_missing_context"
            elif key == "infer:language":
                selected = "unknown"
            elif key.startswith("fidelity:sentence:"):
                selected = "supported_by_assumption"
                return {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": {
                        "supported_by_original": 0.0,
                        "supported_by_assumption": 0.99,
                        "new_requirement": 0.0,
                        "unknown": 0.01,
                    },
                    "confidence": 0.99,
                }
            else:
                selected = "none"
            options = request.get("criteria", {})
            probabilities = {option: 0.0 for option in options}
            probabilities[selected] = 1.0
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": probabilities,
                "confidence": 1.0,
            }
        if key.startswith("faithful:") or key == "fidelity:meaning":
            probability = 0.99
        elif key == "gap:language":
            probability = 0.99
        elif key.startswith("strategy_recheck:"):
            probability = float(key.endswith("add_missing_context"))
        else:
            probability = 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (ChecklistItem("language", "response language", GapImpact.HIGH),),
            ),
        )
    )
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(
        store=store,
        gateway=ScriptedGateway(chat=chat, decision=decide),
        diagnosis_rubric=rubric,
    )

    paused = optimizer.optimize(prompt, {"tier": "fast"})
    result = optimizer.resume(paused["run_id"], {"language": "French"})

    assert paused["status"] == "needs_input"
    assert result["report"]["assumptions"][0]["source"] == "answer"
    selected_round = result["report"]["selection_evidence"]["ranking"]
    candidate = next(
        item for item in selected_round if item["strategy"] == "add_missing_context"
    )
    assert candidate["eligible"] is True
    fidelity = candidate["metadata"]["fidelity"]
    assert fidelity["no_invention"] is True
    assert fidelity["evidence"]["confirmed_assumptions"] == [
        {
            "key": "language",
            "value": "French",
            "source": "answer",
            "label": "response language",
        }
    ]
    persisted = store.get_run(result["run_id"])
    assert persisted is not None
    persisted_candidate = next(
        item
        for item in persisted["result"]["report"]["selection_evidence"]["ranking"]
        if item["strategy"] == "add_missing_context"
    )
    assert persisted_candidate["metadata"]["fidelity"] == fidelity


def test_structured_jev_question_keeps_user_prompt_in_state() -> None:
    prompt = "Summarize this article in three concise bullets."
    gateway = _no_test_gateway()
    rubric = DiagnosisRubric(
        task_types=(
            TaskType(
                "general",
                "General",
                (
                    ChecklistItem(
                        "goal",
                        "goal",
                        GapImpact.HIGH,
                        question={
                            "question": "Is the goal missing from `prompt`?",
                            "focus": ["goal"],
                        },
                    ),
                ),
            ),
        )
    )

    PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        diagnosis_rubric=rubric,
    ).optimize(prompt)

    gap_request = next(
        entry["question"]
        for entry in gateway.decision_log
        if entry["question"].get("key") == "gap:goal"
    )
    assert gap_request["state"] == {"prompt": prompt}
    assert gap_request["query"] == {
        "question": "Is the goal missing from `prompt`?",
        "focus": ["goal"],
    }
    assert prompt not in json.dumps(gap_request["query"])


def test_context_gap_uses_its_calibrated_threshold_without_changing_goal_threshold() -> (
    None
):
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "unknown"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = (
            0.88 if request.get("key") in {"gap:context", "gap:goal"} else 0.01
        )
        return {"type": "noul", "probability_true": probability}

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Help me plan.", {"tier": "fast"}
    )

    diagnosis = result.get("diagnosis") or result["report"]["diagnosis"]
    assert [gap["key"] for gap in diagnosis["confirmed_gaps"]] == ["context"]
    assert diagnosis["confirmed_gaps"][0]["threshold"] == 0.83


@pytest.mark.parametrize(("probability", "hinted"), [(0.78, True), (0.7, False)])
def test_near_miss_outside_reference_is_hinted_without_confirming_a_gap(
    probability: float, hinted: bool
) -> None:
    prompt = "Tell the warehouse team about the new rules for the vans. Keep it short."

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if key == "task_type":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        if key.startswith("pointer:vagueness"):
            return {
                "type": "choice",
                "choice": "s0001",
                "probabilities": {"s0001": 0.9},
                "confidence": 0.9,
            }
        if key.startswith("existence:"):
            return {
                "type": "noul",
                "probability_true": 0.95,
                "confidence": 1.0,
            }
        if request.get("type") == "choice":
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        return {
            "type": "noul",
            "probability_true": probability if key == "gap:outside_reference" else 0.01,
        }

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        prompt, {"tier": "fast"}
    )

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["confirmed_gaps"] == []
    assert result["status"] == "completed"
    if hinted:
        assert diagnosis["possible_gaps"] == [
            {
                "key": "outside_reference",
                "label": "details only you know",
                "missing_probability": probability,
                "threshold": 0.8,
                "sentence": "Tell the warehouse team about the new rules for the vans.",
            }
        ]
    else:
        assert diagnosis["possible_gaps"] == []


@pytest.mark.parametrize(
    ("key", "probability", "confirmed"),
    [
        ("context", 0.83, True),
        ("context", 0.82, False),
        ("outside_reference", 0.8, True),
        ("outside_reference", 0.79, False),
    ],
)
def test_recalibrated_gap_cutoffs_confirm_at_their_boundary(
    key: str, probability: float, confirmed: bool
) -> None:
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        return {
            "type": "noul",
            "probability_true": probability
            if request.get("key") == f"gap:{key}"
            else 0.01,
        }

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Tell the team about the new rules.",
        {"tier": "fast", "clarification_allowed": False},
    )

    keys = [gap["key"] for gap in result["report"]["diagnosis"]["confirmed_gaps"]]
    assert (key in keys) is confirmed


def test_possible_gaps_never_reach_model_requests() -> None:
    sent: list[str] = []

    def chat(_model, messages, *, role, **_kwargs):
        sent.append(json.dumps(messages, sort_keys=True))
        if role == "writer":
            return '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}],"add_missing_context":"Rewrite"}'
        return "An answer."

    def decide(request, **_kwargs):
        sent.append(json.dumps(request, sort_keys=True))
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            choice = "general" if key == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = {"gap:goal": 0.99, "gap:outside_reference": 0.78}.get(
            key, 1.0 if key.startswith("faithful:") else 0.01
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    gateway = ScriptedGateway(chat=chat, decision=decide)
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Tell the team about the new rules.",
        {"tier": "fast", "clarification_allowed": False},
    )

    assert [gap["key"] for gap in result["report"]["diagnosis"]["possible_gaps"]] == [
        "outside_reference"
    ]
    assert any('"strategy_choice"' in request for request in sent)
    assert not [request for request in sent if "possible_gaps" in request]


def test_clear_prompt_with_success_tests_is_never_rewritten() -> None:
    def chat(_model, _messages, *, role, **_kwargs):
        if role == "writer":
            return '{"tests":[{"question":"Does the response satisfy the request?","kind":"noul","expected":"yes"}],"add_missing_context":"Unneeded rewrite"}'
        raise AssertionError("A clear prompt should not generate or run candidates")

    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = (
            1.0 if str(request.get("key", "")).startswith("faithful:") else 0.01
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    prompt = "Summarize the supplied article in three concise bullets."
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=ScriptedGateway(chat=chat, decision=decide)
    ).optimize(prompt)

    assert result["status"] == "completed"
    assert result["original_kept"] is True
    assert result["final_prompt"] == prompt
    assert result["report"]["status"] == "no_change"
    assert len(result["report"]["tests"]) == 1


def test_writer_choice_test_without_unknown_does_not_fail_run() -> None:
    def chat(_model, _messages, *, role, **_kwargs):
        if role == "writer":
            return (
                '{"tests":[{"question":"Which response helps?","kind":"choice",'
                '"expected":"Asks for context","options":['
                '{"value":"Asks for context","description":"Requests missing context."},'
                '{"value":"Guesses","description":"Invents missing context."}]}]}'
            )
        raise AssertionError("The clear prompt should not need candidate outputs")

    gateway = ScriptedGateway(
        chat=chat,
        decision=lambda request, **_kwargs: (
            {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
            if request.get("type") == "choice"
            else {
                "type": "noul",
                "probability_true": 1.0
                if str(request.get("key", "")).startswith("faithful:")
                else 0.01,
                "confidence": 1.0,
            }
        ),
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Summarize the supplied article in three bullets."
    )

    assert result["status"] == "completed"
    assert "unknown" in result["report"]["tests"][0]["options"]


def test_writer_missing_final_json_delimiters_does_not_fail_run() -> None:
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        probability = (
            1.0 if str(request.get("key", "")).startswith("faithful:") else 0.01
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: (
            '{"tests":[{"question":"Does it summarize the article?","kind":"noul","expected":"yes"}'
        ),
        decision=decide,
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Summarize the supplied article."
    )

    assert result["status"] == "completed"
    assert len(result["report"]["tests"]) == 1


def test_writer_invalid_score_test_is_discarded_without_losing_valid_test() -> None:
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        probability = (
            1.0 if str(request.get("key", "")).startswith("faithful:") else 0.01
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: (
            '{"tests":[{"question":"Does it summarize the article?","kind":"noul","expected":"yes"},{"question":"How well?","kind":"score","levels":[]}]}'
        ),
        decision=decide,
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Summarize the supplied article."
    )

    assert result["status"] == "completed"
    assert len(result["report"]["tests"]) == 1
    assert result["report"]["tests"][0]["kind"] == "noul"


def test_optimize_grades_noul_from_direct_answer_only() -> None:
    grading_requests = []

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            if "state.strategies" in messages[0]["content"]:
                return json.dumps(
                    {
                        strategy: "Answer clearly"
                        for strategy in (
                            "add_missing_context",
                            "specify_output_format",
                            "add_done_criteria",
                        )
                    }
                )
            return '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}],"add_missing_context":"Answer clearly"}'
        return "A useful answer."

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if key.startswith("grade_"):
            grading_requests.append(request)
            return {
                "type": "noul",
                "probability_true": 0.3 if key.endswith("_second") else 0.9,
            }
        if request.get("type") == "choice":
            choice = (
                "general"
                if key == "task_type"
                else "add_missing_context"
                if key == "strategy_choice"
                else "none"
            )
            return {"type": "choice", "choice": choice, "probabilities": {choice: 1.0}}
        probability = (
            1.0
            if key.startswith(("gap:goal", "faithful:", "strategy_recheck:"))
            else 0.01
        )
        return {"type": "noul", "probability_true": probability}

    store = RunStore(":memory:")
    result = PromptOptimizer(
        store=store, gateway=ScriptedGateway(chat=chat, decision=decide)
    ).optimize("Answer my question.", {"tier": "fast", "clarification_allowed": False})

    assert result["status"] == "completed", result["report"]
    record = store.get_run(result["run_id"])
    assert record is not None
    assert record["original_weak_panel"]["mean_pass_rate"] == 1.0
    assert all(
        score == 0.9
        for scores in record["original_weak_panel"]["per_model_samples"].values()
        for score in scores
    )
    assert all(not request["key"].endswith("_second") for request in grading_requests)
    assert all(request["question"].endswith("yes?") for request in grading_requests)


@pytest.mark.parametrize("second_is_good", [True, False])
def test_optimize_grades_score_test_from_probability_mass_and_sends_plain_levels(
    second_is_good: bool,
) -> None:
    levels = ["Neither", "Partial", "Good", "Excellent"]
    seen_criteria = []

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            if "state.strategies" in messages[0]["content"]:
                return json.dumps(
                    {
                        strategy: "Answer my question clearly."
                        for strategy in (
                            "add_missing_context",
                            "specify_output_format",
                            "add_done_criteria",
                        )
                    }
                )
            return json.dumps(
                {
                    "tests": [
                        {
                            "question": "How well does the output answer?",
                            "kind": "score",
                            "expected": "2: Good",
                            "levels": [
                                f"{index}: {level}"
                                for index, level in enumerate(levels)
                            ],
                        }
                    ],
                    "add_missing_context": "Answer clearly",
                }
            )
        return "A good answer."

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "score":
            criteria = request["criteria"]
            seen_criteria.append(criteria)
            assert criteria in (levels, list(reversed(levels)))
            reverse = key.endswith("_second")
            return {
                "type": "score",
                "score": (0.5 if second_is_good else 2.6) if reverse else 2.45,
                "probabilities": (
                    (
                        {"0": 0.6, "1": 0.3, "2": 0.1, "3": 0.0}
                        if second_is_good
                        else {"0": 0.0, "1": 0.0, "2": 0.4, "3": 0.6}
                    )
                    if reverse
                    else {"0": 0, "1": 0.1, "2": 0.45, "3": 0.45}
                ),
                "legend": {str(index): level for index, level in enumerate(criteria)},
                "confidence": 0.9,
            }
        if request.get("type") == "choice":
            choice = (
                "general"
                if key == "task_type"
                else "add_missing_context"
                if key == "strategy_choice"
                else "none"
            )
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = (
            1.0
            if key.startswith(("gap:goal", "faithful:", "strategy_recheck:"))
            else 0.01
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    store = RunStore(":memory:")
    result = PromptOptimizer(
        store=store, gateway=ScriptedGateway(chat=chat, decision=decide)
    ).optimize("Answer my question.", {"tier": "fast", "clarification_allowed": False})

    assert result["status"] == "completed"
    assert seen_criteria
    record = store.get_run(result["run_id"])
    assert record is not None
    assert (record["original_weak_panel"]["mean_pass_rate"] > 0) is second_is_good
    assert result["report"]["tests"][0]["levels"] == tuple(levels)


def test_task_taxonomy_descends_to_a_research_leaf() -> None:
    def decide(request, **_kwargs):
        if request.get("key") == "task_type":
            return {
                "type": "choice",
                "choice": "investigation",
                "probabilities": {"investigation": 1.0},
                "confidence": 1.0,
            }
        if request.get("key") == "task_type:investigation":
            return {
                "type": "choice",
                "choice": "research",
                "probabilities": {"research": 1.0},
                "confidence": 1.0,
            }
        if request.get("type") == "choice":
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Research the history of this topic."
    )

    assert result["report"]["diagnosis"]["task_type"] == "research"


def test_empty_prompt_is_rejected() -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_no_test_gateway())

    try:
        optimizer.optimize("   ")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty prompt should be rejected")


def test_update_assumption_replaces_recorded_value_and_refreshes_report() -> None:
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(store=store, gateway=_clarification_gateway(0.95))
    original = optimizer.optimize("Write a concise report.")
    completed = optimizer.resume(original["run_id"], {"goal": "summarize"})
    assert "goal: Summarize" in completed["final_prompt"]
    updated = optimizer.update_assumption(
        original["run_id"], {"key": "goal", "value": "analyze"}
    )

    assert "goal: analyze" in updated["final_prompt"]
    assert updated["report"]["assumption_check"] == {
        "passed": True,
        "checked": True,
        "score": 0.95,
    }
    assert updated["report"]["diff"]
    assert updated["report"]["status"] == "edited"
    assert updated["cost"]["total"] == completed["cost"]["total"]
    assert (
        store.get_run(original["run_id"])["result"]["final_prompt"]
        == updated["final_prompt"]
    )


def test_update_assumption_rejects_negative_jev_meaning_check() -> None:
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=_clarification_gateway(0.0)
    )
    original = optimizer.optimize("Write a concise report.")
    completed = optimizer.resume(original["run_id"], {"goal": "summarize"})
    run_id = original["run_id"]

    try:
        optimizer.update_assumption(run_id, {"key": "goal", "value": "analyze"})
    except ValueError as exc:
        assert "preserve" in str(exc)
    else:
        raise AssertionError("negative meaning check should reject the edit")

    assert (
        optimizer.store.get_run(run_id)["result"]["final_prompt"]
        == completed["final_prompt"]
    )


def test_update_assumption_requires_recorded_assumption_and_jev_answer() -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"))
    result = optimizer.optimize("Write a concise report.")
    try:
        optimizer.update_assumption(
            result["run_id"], {"key": "audience", "value": "novices"}
        )
    except ValueError as exc:
        assert "assumption" in str(exc)
    else:
        raise AssertionError("unrecorded assumption should not be editable")

    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=_clarification_gateway(None)
    )
    pending = optimizer.optimize("Write a concise report.")
    completed = optimizer.resume(pending["run_id"], {"goal": "summarize"})
    try:
        optimizer.update_assumption(
            completed["run_id"], {"key": "goal", "value": "analyze"}
        )
    except RuntimeError as exc:
        assert "meaning check" in str(exc)
    else:
        raise AssertionError("unavailable Jev check should block the edit")


def _clarification_gateway(
    meaning_score: float | None, initial_cost: float = 0.0
) -> ScriptedGateway:
    def chat(_model, _messages, **_kwargs):
        return '{"gaps":{"goal":{"question":"What should the assistant do?","options":[{"value":"summarize","label":"Summarize"},{"value":"analyze","label":"Analyze"}]}},"tests":[]}'

    def decide(request, **_kwargs):
        key = request.get("key")
        if key == "gap:goal" and initial_cost:
            gateway.usage.record(
                role="judge",
                provider="openrouter",
                model="typesafe/jev-1.13",
                cost=initial_cost,
            )
        if key == "task_type":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        if key == "infer:goal":
            return {
                "type": "choice",
                "choice": "unknown",
                "probabilities": {"unknown": 1.0},
                "confidence": 1.0,
            }
        if request.get("type") == "choice":
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        if "updated_prompt" in request.get("state", {}):
            if meaning_score is None:
                raise RuntimeError("judge unavailable")
            probability = meaning_score
        else:
            probability = 0.99 if key == "gap:goal" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    gateway = ScriptedGateway(chat=chat, decision=decide)
    return gateway


def test_run_model_overrides_are_reported_without_changing_defaults() -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_no_test_gateway())

    result = optimizer.optimize(
        "Write a concise report.",
        {
            "tier": "fast",
            "model_overrides": {
                "writer": "one-run-writer",
                "strong": "one-run-strong",
                "weak": ["weak-a", "weak-b"],
            },
        },
    )

    assert result["report"]["models"] == {
        "judge": "typesafe/jev-1.13-20260917",
        "writer": "one-run-writer",
        "strong": "one-run-strong",
        "weak": ["weak-a", "weak-b"],
    }
    next_result = optimizer.optimize("Summarize this report.", {"tier": "fast"})
    assert next_result["report"]["models"]["writer"] == "space-bunny-free"


def test_engine_model_defaults_persist_and_apply_after_restart(tmp_path) -> None:
    database = tmp_path / "runs.sqlite3"
    first = PromptOptimizer(store=RunStore(database), gateway=_no_test_gateway())

    saved = first.update_model_settings({"writer_model": "writer-v2"})
    assert saved["writer_model"] == "writer-v2"
    assert (
        first.optimize("Write a report.")["report"]["models"]["writer"] == "writer-v2"
    )
    first.store.close()

    reopened = PromptOptimizer(store=RunStore(database), gateway=_no_test_gateway())
    assert reopened.get_model_settings()["writer_model"] == "writer-v2"
    assert (
        reopened.optimize("Write another report.")["report"]["models"]["writer"]
        == "writer-v2"
    )


@pytest.mark.parametrize(
    "tier,weak",
    [
        ("fast", ["one"]),
        ("standard", ["one", "two"]),
        ("deep", ["one", "two", "three", "four"]),
        ("fast", ["one", "one"]),
    ],
)
def test_run_rejects_weak_panels_below_tier_budget(tier: str, weak: list[str]) -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_no_test_gateway())

    with pytest.raises(ValueError, match="weak panel"):
        optimizer.optimize(
            "Write a release note.", {"tier": tier, "model_overrides": {"weak": weak}}
        )


@pytest.mark.parametrize("failing_gate", ["no_invention", "meaning"])
def test_engine_rejects_unfaithful_candidate_even_when_weak_models_prefer_it(
    failing_gate: str,
) -> None:
    rejected_prompt = (
        "Invented rewrite"
        if failing_gate == "no_invention"
        else "Meaning-changing rewrite"
    )

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            return json.dumps(
                {
                    "tests": [
                        {
                            "question": "Does the output answer?",
                            "kind": "noul",
                            "expected": "yes",
                        }
                    ],
                    "add_missing_context": rejected_prompt,
                    "specify_output_format": "Safe rewrite",
                    "add_done_criteria": "Other rewrite",
                }
            )
        prompt = messages[0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": "pass" if prompt != "Original request." else "fail"
                    }
                }
            ]
        }

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        state = request.get("state", {})
        if request.get("type") == "choice":
            if key == "task_type":
                choice = "general"
            elif key.startswith("pointer:vagueness:"):
                choice = "s0001"
            elif key.startswith("fidelity:sentence:"):
                sentence = next(iter(state["changed_sentences"].values()))[
                    "candidate_sentences"
                ][0]["text"]
                choice = (
                    "new_requirement"
                    if failing_gate == "no_invention" and sentence == rejected_prompt
                    else "supported_by_original"
                )
            else:
                choice = "none"
            probabilities = {
                choice: 0.99 if key.startswith("fidelity:sentence:") else 1.0
            }
            if key.startswith("fidelity:sentence:"):
                probabilities.update(
                    {
                        "supported_by_original": 0.99
                        if choice == "supported_by_original"
                        else 0.0,
                        "supported_by_assumption": 0.0,
                        "new_requirement": 0.99 if choice == "new_requirement" else 0.0,
                        "unknown": 0.0,
                    }
                )
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": 1.0,
            }
        if key.startswith(("gap:goal", "faithful:")):
            probability = 1.0
        elif key == "fidelity:meaning":
            probability = (
                0.0
                if failing_gate == "meaning"
                and state["candidate_prompt"] == rejected_prompt
                else 1.0
            )
        elif "output" in state:
            passed = state["output"] == "pass"
            probability = (
                float(not passed) if key.endswith("_second") else float(passed)
            )
        else:
            probability = 1.0
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=ScriptedGateway(chat=chat, decision=decide)
    )
    result = optimizer.optimize(
        "Original request.", {"tier": "fast", "clarification_allowed": False}
    )

    assert result["final_prompt"] == "Safe rewrite"
    rejection_reasons = result["report"]["selection_evidence"]["rejection_reasons"][
        "candidate-1-add_missing_context"
    ]
    assert rejection_reasons
    if failing_gate == "meaning":
        assert any("whole-prompt meaning" in reason for reason in rejection_reasons)
    else:
        assert any(rejected_prompt in reason for reason in rejection_reasons)
    rejected = next(
        item
        for item in result["report"]["selection_evidence"]["rejected_candidates"]
        if item["candidate_id"] == "candidate-1-add_missing_context"
    )
    assert (
        rejected["grade"]["mean"]
        > result["report"]["selection_evidence"]["original_score"]["mean"]
    )
    record = optimizer.store.get_run(result["run_id"])
    assert record["jev_answers"]
    assert record["original_weak_panel"]["mean_pass_rate"] == 0.0
    assert record["score_summaries"]["original"]["mean_pass_rate"] == 0.0


def test_run_without_confirmed_gaps_does_not_offer_deep() -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_no_test_gateway())
    result = optimizer.optimize("Write a clear report.", {"tier": "fast"})

    assert result["report"]["diagnosis"]["confirmed_gaps"] == []
    assert result["report"]["offer_deep"] is None


def test_deep_pass_executes_again_under_same_run_id() -> None:
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=_no_test_gateway(gaps=("goal",))
    )
    first = optimizer.optimize(
        "Write a clear report.", {"tier": "fast", "clarification_allowed": False}
    )

    assert first["report"]["offer_deep"]["to_tier"] == "deep"
    deep = optimizer.start_deep_pass(first["run_id"])

    assert deep["run_id"] == first["run_id"]
    assert deep["report"]["escalation"]["status"] == "completed"
    assert [entry["tier"] for entry in deep["report"]["history"]] == ["fast", "deep"]
    assert optimizer.store.get_run(first["run_id"])["tier"] == "deep"
    assert optimizer.history.list_runs(None)[0]["escalated_from"] == "fast"


def test_deep_pass_expands_a_fast_run_weak_panel_to_deep_budget() -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_no_test_gateway())
    first = optimizer.optimize(
        "Write a clear report.",
        {
            "tier": "fast",
            "model_overrides": {"weak": ["chosen-one", "chosen-two"]},
        },
    )

    deep = optimizer.start_deep_pass(first["run_id"])

    assert deep["run_id"] == first["run_id"]
    assert deep["report"]["models"]["weak"][:2] == ["chosen-one", "chosen-two"]
    assert len(deep["report"]["models"]["weak"]) == 5
    assert "muse-spark-1.3-contributor" in deep["report"]["models"]["weak"]
    assert "qwen3.8-flash" not in deep["report"]["models"]["weak"]


def test_provider_failure_preserves_original_prompt_and_run_record() -> None:
    def chat(_model, _messages, *, role, **_kwargs):
        if role == "writer":
            return '{"tests":[{"question":"Does the answer address the request?","kind":"noul","expected":"yes"}],"add_missing_context":"A clearer request","specify_output_format":"A concise request","add_done_criteria":"An explicit request"}'
        raise ProviderError("openrouter", "weak-model", 503)

    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = (
            0.99 if str(request.get("key", "")).startswith("gap:goal") else 1.0
        )
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    store = RunStore(":memory:")
    optimizer = PromptOptimizer(
        store=store, gateway=ScriptedGateway(chat=chat, decision=decide)
    )
    result = optimizer.optimize(
        "Explain recursion.", {"tier": "fast", "clarification_allowed": False}
    )

    assert result["status"] == "failed"
    assert result["final_prompt"] == "Explain recursion."
    assert store.get_run(result["run_id"])["prompt"] == "Explain recursion."


def test_failed_strong_reference_never_approves_a_candidate() -> None:
    def chat(_model, _messages, *, role, **_kwargs):
        if role == "writer":
            return '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}],"add_missing_context":"Rewrite A","specify_output_format":"Rewrite B","add_done_criteria":"Rewrite C"}'
        if role == "strong_check":
            raise ProviderError("opencode", "strong-reference", 503)
        return "pass"

    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = (
            0.99 if str(request.get("key", "")).startswith("gap:goal") else 1.0
        )
        if str(request.get("key", "")).endswith("_second"):
            probability = 0.0
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    store = RunStore(":memory:")
    result = PromptOptimizer(
        store=store, gateway=ScriptedGateway(chat=chat, decision=decide)
    ).optimize("Original request", {"tier": "fast", "clarification_allowed": False})

    assert result["status"] == "failed"
    assert result["final_prompt"] == "Original request"
    assert store.get_run(result["run_id"])["result"]["status"] == "failed"


def test_resume_after_restart_retains_cost_spent_before_clarification(tmp_path) -> None:
    database = tmp_path / "runs.sqlite3"
    first = PromptOptimizer(
        store=RunStore(database),
        gateway=_clarification_gateway(0.95, initial_cost=0.02),
    )
    pending = first.optimize("Write a concise report.")
    assert pending["status"] == "needs_input"
    assert pending["cost"]["total"] == 0.02
    first.store.close()

    resumed = PromptOptimizer(
        store=RunStore(database), gateway=_clarification_gateway(0.95)
    ).resume(pending["run_id"], {"goal": "summarize"})

    assert resumed["cost"]["total"] == 0.02
    assert resumed["cost"]["cost_by_role"] == {"judge": 0.02}


def test_unknown_high_impact_gap_offers_writer_choices_and_preserves_diagnosis() -> (
    None
):
    def chat(_model, _messages, **_kwargs):
        return '{"gaps":{"goal":{"question":"What is the goal?","options":[{"value":"summarize","label":"Summarize"},{"value":"analyze","label":"Analyze"}]}}}'

    def decide(request, **_kwargs):
        if request.get("key") == "task_type":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        if request.get("key") == "infer:goal":
            return {
                "type": "choice",
                "choice": "unknown",
                "probabilities": {"summarize": 0.6, "analyze": 0.2, "unknown": 0.2},
                "confidence": 0.6,
            }
        if request.get("type") == "choice":
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        probability = 0.99 if request.get("key") == "gap:goal" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=ScriptedGateway(chat=chat, decision=decide)
    )
    result = optimizer.optimize("Help me with this.")

    assert result["status"] == "needs_input"
    assert result["report"]["diagnosis"]["confirmed_gaps"][0]["key"] == "goal"
    question = result["questions"][0]
    assert question["prompt"] == "What is the goal?"
    assert [option["value"] for option in question["options"]] == [
        "summarize",
        "analyze",
        "other",
    ]
    assert question["default"] == "summarize"
