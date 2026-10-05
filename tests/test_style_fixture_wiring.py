"""Reviewed style examples traverse the real optimizer and judgment seams."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore
from prompt_enhancer.styles import IMPROVEMENT_STYLES, style_authorization_for

CORPUS = json.loads(
    (Path(__file__).parent / "fixtures/improvement_styles.json").read_text()
)
NAMED = {item["style"]: item["candidate_prompt"] for item in CORPUS["named_styles"]}
CASES = [(style, style, 1.0) for style in NAMED] + [
    ("auto", "creative", 1.0),
    ("auto", "clearer", 0.1),
]


@pytest.mark.parametrize("requested,applied,confidence", CASES)
def test_style_example_reaches_writer_fidelity_evaluate_and_report(
    requested, applied, confidence
):
    original = CORPUS["shared_input"]
    candidate = NAMED[applied]
    writer_states = []

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            writer_states.append(state)
            return json.dumps(
                {strategy["name"]: candidate for strategy in state["strategies"]}
            )
        if role == "writer":
            return '{"tests":[]}'
        return "pass"

    def decide(request, **_kwargs):
        key = request["key"]
        state = request.get("state", {})
        if request["type"] == "choice":
            criteria = request.get("criteria", {})
            if key == "understand:classify:style":
                choice = "creative"
            elif key == "task_type":
                choice = "general"
            elif key.startswith("fidelity:sentence:"):
                choice = "authorized_style_presentation"
            elif key.endswith(":verbosity_direction"):
                choice = "same"
            else:
                choice = "none" if "none" in criteria else next(iter(criteria), "none")
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": confidence if key == "understand:classify:style" else 1.0,
            }
        if key.startswith("score:"):
            probability = 0.1 if state["candidate_prompt"] == original else 0.99
        elif key.startswith(("fidelity:", "strategy_recheck:", "evaluate:")):
            probability = 0.99
        else:
            probability = 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    gateway = ScriptedGateway(chat=chat, decision=decide)
    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        config=Settings(),
        writer_instruction_version=4,
    ).optimize(
        original, {"improvement_style": requested, "clarification_allowed": False}
    )

    assert result["status"] != "failed", result["report"]
    assert result["final_prompt"] == candidate, result["report"]
    assert result["report"]["applied_style"] == applied
    assert result["report"]["improvement_style"] == requested
    assert result["report"]["convergence"]["status"] == "converged"
    assert result["report"].get("outcome") == "converged"
    assert result["report"].get("outcome_reason")
    permission = style_authorization_for(applied)
    assert writer_states
    assert all(
        state["applied_style"] == applied and state["style_authorization"] == permission
        for state in writer_states
    )
    questions = [entry["question"] for entry in gateway.decision_log]
    for prefix in ("fidelity:sentence:", "evaluate:compare:", "evaluate:accept:"):
        observed = [
            question for question in questions if question["key"].startswith(prefix)
        ]
        assert observed, prefix
        assert all(
            question["state"]["applied_style"] == applied
            and question["state"]["style_authorization"] == permission
            for question in observed
        )
    assert set(NAMED) == set(IMPROVEMENT_STYLES) - {"auto"}
    assert len(set(NAMED.values())) == 21
    assert NAMED["proofread_only"] == original.replace("volunter", "volunteer")
