from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from prompt_enhancer.rewrite import CandidateWriter, VerifiedRewrite


class ScriptedRewriteGateway:
    """A no-network gateway with responses selected by model role/state."""

    def __init__(self, *, candidate: str, fidelity: bool = True) -> None:
        self.candidate = candidate
        self.fidelity = fidelity
        self.chat_calls: list[dict[str, Any]] = []
        self.decision_calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, str]:
        self.chat_calls.append({"model": model, "messages": list(messages), **kwargs})
        if kwargs.get("role") == "writer":
            return {"content": self.candidate}
        prompt = str(messages[-1]["content"])
        return {"content": "specific answer" if "specific" in prompt else "vague answer"}

    def decide(self, payload: Mapping[str, Any], **kwargs: Any) -> dict[str, float]:
        self.decision_calls.append({"payload": dict(payload), **kwargs})
        state = payload["state"]
        if "test" in state:
            return {"score": 1.0 if "specific" in state["output"] else 0.0}
        return {"score": 1.0 if self.fidelity else 0.0}


def _run(gateway: ScriptedRewriteGateway, candidate: str) -> Any:
    writer = CandidateWriter(gateway)
    proposal = writer.write(
        "Write a summary.",
        [{"flagged_sentences": ["summary"]}],
        [{"id": "useful", "question": "Is the answer useful?"}],
    )
    evaluator = VerifiedRewrite(gateway)
    return evaluator.evaluate(
        "Write a summary.",
        proposal,
        diagnosis=[{"flagged_sentences": ["summary"]}],
        tests=[{"id": "useful", "question": "Is the answer useful?"}],
    )


def test_candidate_writer_preserves_language_and_keeps_input_in_state() -> None:
    gateway = ScriptedRewriteGateway(candidate="Escribe un resumen específico.")
    candidate = CandidateWriter(gateway).write(
        "Escribe un resumen.",
        [{"flagged_sentences": ["resumen"]}],
        [{"id": "clear", "question": "¿Es claro?"}],
    )

    assert candidate.text == "Escribe un resumen específico."
    assert gateway.chat_calls[0]["model"] == "deepseek-v4.1-flash"
    messages = gateway.chat_calls[0]["messages"]
    assert "Escribe un resumen." not in messages[0]["content"]
    assert "Escribe un resumen." in messages[1]["content"]


def test_verified_candidate_is_selected_with_clean_prompt_and_evidence() -> None:
    gateway = ScriptedRewriteGateway(candidate="Write a specific summary.")
    result = _run(gateway, "Write a specific summary.")

    assert result.final_prompt == "Write a specific summary."
    assert result.original_kept is False
    assert result.report()["status"] == "improved"
    assert "specific summary" in result.diff
    assert result.selection_evidence["reason"] == "verified_improvement"
    assert result.per_model["meta-llama/llama-3.1-8b-instruct"]["beats_original"] is True


@pytest.mark.parametrize("failure", ["meaning", "invention", "confined"])
def test_candidate_is_retained_when_fidelity_check_fails(failure: str) -> None:
    gateway = ScriptedRewriteGateway(candidate="Write a specific summary.", fidelity=False)
    writer_candidate = CandidateWriter(gateway).write(
        "Write a summary.",
        [{"flagged_sentences": ["summary"]}],
        [{"id": "useful", "question": "Is the answer useful?"}],
    )
    fidelity: dict[str, bool] = {"meaning_preserved": True, "no_invention": True, "edits_confined": True}
    fidelity[failure] = False
    result = VerifiedRewrite(
        gateway,
        check_fidelity=lambda **_: fidelity,
    ).evaluate(
        "Write a summary.",
        writer_candidate,
        diagnosis=[{"flagged_sentences": ["summary"]}],
        tests=[{"id": "useful", "question": "Is the answer useful?"}],
    )

    assert result.final_prompt == "Write a summary."
    assert result.original_kept is True
    assert result.report()["status"] == "no_change"
    assert result.selection_evidence["reason"] == "failed_fidelity"
    assert result.report()["fidelity"][failure] is False


def test_candidate_that_does_not_beat_original_is_retained() -> None:
    gateway = ScriptedRewriteGateway(candidate="Write a summary.")
    result = _run(gateway, "Write a summary.")

    assert result.final_prompt == "Write a summary."
    assert result.original_kept is True
    assert result.selection_evidence["reason"] == "did_not_beat_original"
    assert result.report()["diff"] == ""
