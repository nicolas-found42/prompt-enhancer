from __future__ import annotations

import json

import pytest

from prompt_enhancer.fidelity import check_candidate_fidelity
from prompt_enhancer.gateway import ProviderError, ScriptedGateway
from prompt_enhancer.rewrite import CandidateWriter
from prompt_enhancer.strategies import search_strategies


def test_candidate_writer_keeps_user_text_in_state_and_preserves_language() -> None:
    calls = []

    def chat(model, messages, **kwargs):
        calls.append({"model": model, "messages": messages, **kwargs})
        strategies = json.loads(messages[1]["content"])["strategies"]
        return json.dumps(
            {
                strategy["name"]: "Escribe un resumen específico."
                for strategy in strategies
            }
        )

    result = search_strategies(
        "Escribe un resumen.",
        tier="fast",
        writer=CandidateWriter(ScriptedGateway(chat=chat)),
    )

    assert {candidate.text for candidate in result.candidates} == {
        "Escribe un resumen específico."
    }
    assert calls[0]["model"] == "space-bunny-free"
    assert calls[0]["role"] == "writer"
    system, user = calls[0]["messages"]
    assert "Escribe un resumen." not in system["content"]
    assert json.loads(user["content"])["prompt"] == "Escribe un resumen."


def _fidelity(decide):
    return check_candidate_fidelity(
        ScriptedGateway(decision=decide),
        "Write a summary.",
        "Write a specific summary.",
        {"confirmed_gaps": []},
        "specify_output_format",
        run_id="run",
        judge_model="typesafe/jev-1.13",
    )


@pytest.mark.parametrize(
    "failing", ["meaning_preserved", "no_invention", "edits_confined"]
)
def test_each_fidelity_check_can_reject_a_candidate(failing: str) -> None:
    result = _fidelity(
        lambda request, **_: {
            "type": "noul",
            "noul": 0.1 if request["key"] == failing else 0.99,
        }
    )

    assert result.passed is False
    assert result.to_dict()[failing] is False


def test_fidelity_fails_closed_when_jev_is_unavailable() -> None:
    def decide(_request, **_kwargs):
        raise ProviderError("openrouter", "typesafe/jev-1.13", 503)

    assert _fidelity(decide).passed is False


def test_fidelity_does_not_hide_a_bug_as_a_failed_check() -> None:
    def decide(_request, **_kwargs):
        raise TypeError("bug in the caller")

    with pytest.raises(TypeError):
        _fidelity(decide)
