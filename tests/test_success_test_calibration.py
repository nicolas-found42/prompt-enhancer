import json

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.success_tests import SuccessTestCompiler


def test_calibrated_faithfulness_gate_accepts_boundary_probability() -> None:
    tests = {
        "tests": [
            {
                "id": "accepted",
                "question": "Does the response give the requested summary?",
                "kind": "noul",
                "expected": "yes",
            },
            {
                "id": "rejected",
                "question": "Does the response add a chart?",
                "kind": "noul",
                "expected": "yes",
            },
        ]
    }
    faithfulness = {"faithful:accepted": 0.8, "faithful:rejected": 0.79}
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": json.dumps(tests)}}]
        },
        decision=lambda request, **_kwargs: {
            "type": "noul",
            "noul": faithfulness[request["key"]],
        },
    )

    compiled = SuccessTestCompiler(gateway).compile("Summarize this article.")

    assert [test.id for test in compiled.tests] == ["accepted"]
    assert [item.test.id for item in compiled.rejected] == ["rejected"]
    assert all(check.threshold == 0.8 for check in compiled.faithfulness_checks)


def test_choice_descriptions_are_repaired_then_checked_by_jev() -> None:
    writer_replies = iter(
        [
            {
                "tests": [
                    {
                        "id": "helpful",
                        "question": "Which response helps?",
                        "kind": "choice",
                        "expected": "asks",
                        "options": ["asks", "guesses"],
                    }
                ]
            },
            {
                "descriptions": {
                    "helpful": {
                        "asks": "Requests missing context before answering.",
                        "guesses": "Invents the missing context.",
                    }
                }
            },
        ]
    )

    def chat(*_args, **_kwargs):
        return {"choices": [{"message": {"content": json.dumps(next(writer_replies))}}]}

    def decide(request, **_kwargs):
        if request["key"] == "task_type":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        probability = 0.01
        if str(request["key"]).startswith("faithful:"):
            descriptions = request["state"]["proposed_test"]["option_descriptions"]
            probability = (
                0.95
                if descriptions
                == {
                    "asks": "Requests missing context before answering.",
                    "guesses": "Invents the missing context.",
                    "unknown": "The output does not give enough evidence to choose another option.",
                }
                else 0.01
            )
        return {"type": "noul", "noul": probability, "confidence": 0.9}

    gateway = ScriptedGateway(chat=chat, decision=decide)
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Ask for context before answering."
    )

    assert result["report"]["status"] == "no_change"
    assert len(result["report"]["tests"]) == 1
    assert result["report"]["tests"][0]["option_descriptions"] == {
        "asks": "Requests missing context before answering.",
        "guesses": "Invents the missing context.",
        "unknown": "The output does not give enough evidence to choose another option.",
    }


def test_choice_with_missing_description_after_repair_is_unverified() -> None:
    writer_replies = iter(
        [
            '{"tests":[{"id":"helpful","question":"Which helps?","kind":"choice",'
            '"expected":"asks","options":["asks","guesses"]}]}',
            '{"descriptions":{"helpful":{"asks":"Requests context."}}}',
        ]
    )

    def decide(request, **_kwargs):
        if request["key"] == "task_type":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        if str(request["key"]).startswith("faithful:"):
            raise AssertionError("Incomplete tests must not reach Jev")
        return {"type": "noul", "noul": 0.01, "confidence": 1.0}

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: next(writer_replies),
        decision=decide,
    )

    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Ask for context first."
    )

    assert result["original_kept"] is True
    assert result["report"]["status"] == "unverified"
    assert len(result["report"]["tests"]) == 0
