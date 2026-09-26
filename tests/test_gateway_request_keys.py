"""Pin the request keys that reach the gateway.

Strict replay looks up each recorded answer by a hash of the exact request, so
any change to how the engine builds a request silently orphans the recordings
in ``.local/evaluation``. These scenarios cover every stage that calls the
gateway; if a key changes on purpose, regenerate the fixture with
``UPDATE_GATEWAY_KEYS=1 uv run pytest tests/test_gateway_request_keys.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import ScriptedGateway

FIXTURE = Path(__file__).parent / "fixtures" / "gateway_request_keys.json"


def _pipeline_gateway() -> ScriptedGateway:
    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            return '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}],"add_missing_context":"Invented rewrite","specify_output_format":"Safe rewrite","add_done_criteria":"Other rewrite"}'
        prompt = messages[0]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": "pass" if prompt != "Original request" else "fail"
                    }
                }
            ]
        }

    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        if str(request.get("key", "")).startswith(("gap:goal", "faithful:")):
            probability = 1.0
        elif "candidate_prompt" in request.get("state", {}):
            probability = (
                0.0
                if request["state"]["candidate_prompt"] == "Invented rewrite"
                else 1.0
            )
        elif "output" in request.get("state", {}):
            probability = float(request["state"]["output"] == "pass")
        else:
            probability = 1.0
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(chat=chat, decision=decide)


def _deep_gateway() -> ScriptedGateway:
    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        probability = 0.99 if request.get("key") == "gap:goal" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
    )


def _clarification_gateway() -> ScriptedGateway:
    def chat(_model, _messages, **_kwargs):
        return '{"gaps":{"goal":{"question":"What should the assistant do?","options":[{"value":"summarize","label":"Summarize"},{"value":"analyze","label":"Analyze"}]}},"tests":[]}'

    def decide(request, **_kwargs):
        key = request.get("key")
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
            return {"type": "noul", "probability_true": 0.95, "confidence": 1.0}
        probability = 0.99 if key == "gap:goal" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return ScriptedGateway(chat=chat, decision=decide)


def _no_candidate_beats_gateway() -> ScriptedGateway:
    base = _pipeline_gateway().decision_handler

    def decide(request, **kwargs):
        return base(request, **kwargs)

    def only_original_passes(_model, messages, *, role, **_kwargs):
        if role == "writer":
            return '{"tests":[{"question":"Does the output answer?","kind":"noul","expected":"yes"}],"add_missing_context":"Rewrite one","specify_output_format":"Rewrite two","add_done_criteria":"Rewrite three","remove_contradictions":"Rewrite four","add_example":"Rewrite five","split_into_steps":"Rewrite six"}'
        return "pass" if messages[0]["content"] == "Original request" else "fail"

    return ScriptedGateway(chat=only_original_passes, decision=decide)


def _no_strategy_gateway() -> ScriptedGateway:
    gateway = _pipeline_gateway()
    decide = gateway.decision_handler

    def reject_every_strategy(request, **kwargs):
        if str(request.get("key", "")).startswith("strategy_recheck:"):
            return {"type": "noul", "probability_true": 0.0, "confidence": 1.0}
        return decide(request, **kwargs)

    return ScriptedGateway(chat=gateway.chat_handler, decision=reject_every_strategy)


def _kept_over_two_rounds(optimizer: PromptOptimizer) -> list[Any]:
    return [
        optimizer.optimize(
            "Original request", {"tier": "standard", "clarification_allowed": False}
        )
    ]


def _deep_pass_after_kept(optimizer: PromptOptimizer) -> list[Any]:
    # The Deep pass rebuilds its first round's failures from the stored run.
    kept = optimizer.optimize(
        "Original request", {"tier": "standard", "clarification_allowed": False}
    )
    return [kept, optimizer.start_deep_pass(kept["run_id"])]


def _no_strategy(optimizer: PromptOptimizer) -> list[Any]:
    return [
        optimizer.optimize(
            "Original request", {"tier": "fast", "clarification_allowed": False}
        )
    ]


def _full_pipeline(optimizer: PromptOptimizer) -> list[Any]:
    return [
        optimizer.optimize(
            "Original request", {"tier": "fast", "clarification_allowed": False}
        )
    ]


def _deep_pass(optimizer: PromptOptimizer) -> list[Any]:
    first = optimizer.optimize(
        "Write a clear report.", {"tier": "fast", "clarification_allowed": False}
    )
    return [first, optimizer.start_deep_pass(first["run_id"])]


def _clarify_resume_edit(optimizer: PromptOptimizer) -> list[Any]:
    pending = optimizer.optimize("Write a concise report.")
    resumed = optimizer.resume(pending["run_id"], {"goal": "summarize"})
    edited = optimizer.update_assumption(
        pending["run_id"], {"key": "goal", "value": "analyze"}
    )
    return [pending, resumed, edited]


# Each scenario returns every public result it produced, in order.
SCENARIOS: dict[
    str, tuple[Callable[[], ScriptedGateway], Callable[[PromptOptimizer], list[Any]]]
] = {
    "full_pipeline": (_pipeline_gateway, _full_pipeline),
    "deep_pass": (_deep_gateway, _deep_pass),
    "clarify_resume_edit": (_clarification_gateway, _clarify_resume_edit),
    "kept_over_two_rounds": (_no_candidate_beats_gateway, _kept_over_two_rounds),
    "deep_pass_after_kept": (_no_candidate_beats_gateway, _deep_pass_after_kept),
    "no_strategy": (_no_strategy_gateway, _no_strategy),
}


def _recorded_keys(tmp_path: Path, name: str) -> list[str]:
    make_gateway, run = SCENARIOS[name]
    recording = RecordingGateway(make_gateway(), tmp_path / f"{name}.json")
    run(
        PromptOptimizer(
            gateway=recording,
            store=RunStore(":memory:"),
            writer_instruction_version=4,
            speculative_diagnosis=False,
        )
    )
    return sorted(recording.responses)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gateway_request_keys_are_unchanged(tmp_path: Path, name: str) -> None:
    keys = _recorded_keys(tmp_path, name)
    assert keys, "scenario made no gateway requests"

    if os.environ.get("UPDATE_GATEWAY_KEYS"):
        pinned = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {}
        pinned[name] = keys
        FIXTURE.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")

    pinned = json.loads(FIXTURE.read_text())
    assert keys == pinned[name]
