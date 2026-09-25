from __future__ import annotations

import json

import pytest

from prompt_enhancer.fidelity import check_candidate_fidelity, sentence_edit_script
from prompt_enhancer.gateway import ProviderError, ScriptedGateway
from prompt_enhancer.rewrite import CandidateWriter
from prompt_enhancer.strategies import (
    CandidateBatchRequest,
    RewriteStrategy,
    search_strategies,
)


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


def test_current_candidate_writer_exposes_edit_permissions_and_unverified_failures() -> (
    None
):
    calls = []
    strategy = RewriteStrategy(
        "add_missing_context",
        "safe",
        "Add missing context.",
        gap_fill_keys=("context",),
    )
    gateway = ScriptedGateway(
        chat=lambda _model, messages, **kwargs: (
            calls.append((messages, kwargs)) or '{"add_missing_context":"Prompt."}'
        )
    )

    CandidateWriter(gateway).generate_candidates(
        CandidateBatchRequest(
            "Prompt.",
            (strategy,),
            previous_failures=("prior fidelity evidence did not verify a sentence",),
        )
    )

    system, user = calls[0][0]
    state = json.loads(user["content"])
    assert state["strategies"][0]["gap_fill_keys"] == ["context"]
    assert state["strategies"][0]["restructures"] is False
    assert state["previous_failures"] == [
        "prior fidelity evidence did not verify a sentence"
    ]
    assert "not as facts about the user's intent" in system["content"]


def test_version_two_writer_keeps_its_recorded_strategy_request_shape() -> None:
    calls = []
    strategy = RewriteStrategy(
        "add_missing_context",
        "safe",
        "Add missing context.",
        gap_fill_keys=("context",),
    )
    gateway = ScriptedGateway(
        chat=lambda _model, messages, **kwargs: (
            calls.append((messages, kwargs)) or '{"add_missing_context":"Prompt."}'
        )
    )

    CandidateWriter(gateway, instruction_version=2).generate_candidates(
        CandidateBatchRequest("Prompt.", (strategy,))
    )

    system, user = calls[0][0]
    state = json.loads(user["content"])
    assert "gap_fill_keys" not in state["strategies"][0]
    assert "restructures" not in state["strategies"][0]
    assert "not as facts about the user's intent" not in system["content"]


@pytest.mark.parametrize(
    ("original", "candidate", "operation", "source_ids", "candidate_ids", "gap"),
    [
        (
            "Keep it short. Keep it short. Then summarize.",
            "Keep it short. Then summarize.",
            "delete",
            ["source-s0002"],
            [],
            None,
        ),
        (
            "A. C.",
            "A. B. C.",
            "insert",
            [],
            ["candidate-s0002"],
            1,
        ),
        ("A. B.", "A.", "delete", ["source-s0002"], [], None),
    ],
)
def test_sentence_edit_script_maps_duplicates_insertions_and_deletions(
    original, candidate, operation, source_ids, candidate_ids, gap
) -> None:
    script = sentence_edit_script(original, candidate)

    assert script["correspondence_errors"] == []
    assert len(script["edits"]) == 1
    edit = script["edits"][0]
    assert edit["operation"] == operation
    assert edit["source_sentence_ids"] == source_ids
    assert edit["candidate_sentence_ids"] == candidate_ids
    assert edit["source_gap"] == gap


def test_sentence_edit_script_rejects_ambiguous_sentence_correspondence() -> None:
    script = sentence_edit_script("A. B.", "A single combined request.")

    assert script["correspondence_errors"] == [
        "sentence correspondence could not be established for A single combined request."
    ]


def test_sentence_edit_script_records_exact_unchanged_anchors() -> None:
    script = sentence_edit_script(
        "Rewrite this. Keep this constraint.", "Revise this. Keep this constraint."
    )

    assert script["unchanged_anchors"] == [
        {
            "source_sentence_id": "source-s0002",
            "candidate_sentence_id": "candidate-s0002",
            "text": "Keep this constraint.",
        }
    ]


def _fidelity(decide):
    original = "Write a summary."
    return check_candidate_fidelity(
        ScriptedGateway(decision=decide),
        original,
        "Write a clearer summary.",
        {
            "confirmed_gaps": [],
            "problem_sentences": [
                {
                    "sentence_id": "s0001",
                    "sentence": {
                        "id": "s0001",
                        "text": original,
                        "start": 0,
                        "end": len(original),
                    },
                }
            ],
        },
        {
            "name": "specify_output_format",
            "restructures": False,
            "gap_fill_keys": [],
        },
        run_id="run",
        judge_model="typesafe/jev-1.13",
    )


@pytest.mark.parametrize(
    "failing", ["meaning_preserved", "unknown_support", "missing_support"]
)
def test_each_fidelity_check_can_reject_a_candidate(failing: str) -> None:
    def decide(request, **_):
        if request["type"] == "choice":
            if failing == "missing_support":
                return None
            selected = (
                "unknown" if failing == "unknown_support" else "supported_by_original"
            )
            probabilities = {
                "supported_by_original": 0.99,
                "supported_by_assumption": 0.0,
                "new_requirement": 0.0,
                "unknown": 0.0,
            }
            probabilities[selected] = 0.99
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": probabilities,
                "confidence": 0.99,
            }
        return {
            "type": "noul",
            "noul": 0.1 if failing == "meaning_preserved" else 0.99,
        }

    result = _fidelity(decide)

    assert result.passed is False
    assert result.to_dict()["meaning_preserved"] is (failing != "meaning_preserved")
    assert result.to_dict()["no_invention"] is (
        failing not in {"unknown_support", "missing_support"}
    )


def test_fidelity_fails_closed_when_jev_is_unavailable() -> None:
    def decide(_request, **_kwargs):
        raise ProviderError("openrouter", "typesafe/jev-1.13", 503)

    assert _fidelity(decide).passed is False


def test_fidelity_fails_closed_for_malformed_gateway_batch() -> None:
    class MalformedBatchGateway(ScriptedGateway):
        def decide_batch(self, *_args, **_kwargs):
            return None

    result = check_candidate_fidelity(
        MalformedBatchGateway(),
        "Write a summary.",
        "Write a clearer summary.",
        {
            "confirmed_gaps": [],
            "problem_sentences": [
                {
                    "sentence_id": "s0001",
                    "sentence": {
                        "id": "s0001",
                        "text": "Write a summary.",
                        "start": 0,
                        "end": len("Write a summary."),
                    },
                }
            ],
        },
        {"name": "rewrite", "gap_fill_keys": [], "restructures": False},
        run_id="run",
        judge_model="typesafe/jev-1.13",
    )

    assert result.passed is False
    assert result.evidence["reasons"] == [
        "fidelity checks unavailable or incomplete (JevResponseError)"
    ]


def test_fidelity_does_not_hide_a_bug_as_a_failed_check() -> None:
    def decide(_request, **_kwargs):
        raise TypeError("bug in the caller")

    with pytest.raises(TypeError):
        _fidelity(decide)


def test_unconfined_edit_is_rejected_without_a_jev_request() -> None:
    gateway = ScriptedGateway(
        decision=lambda *_args, **_kwargs: pytest.fail("fidelity must not call Jev")
    )

    result = check_candidate_fidelity(
        gateway,
        "Write a summary. Keep it brief.",
        "Write a detailed summary. Keep it brief.",
        {"confirmed_gaps": [], "problem_sentences": []},
        {"name": "specify_output_format", "gap_fill_keys": [], "restructures": False},
        run_id="run",
        judge_model="typesafe/jev-1.13",
    )

    assert result.edits_confined is False
    assert result.passed is False
    assert gateway.decision_log == []


def test_unmatched_gap_cannot_authorize_an_arbitrary_insertion() -> None:
    gateway = ScriptedGateway(
        decision=lambda *_args, **_kwargs: pytest.fail("fidelity must not call Jev")
    )

    result = check_candidate_fidelity(
        gateway,
        "Summarize the report.",
        "Summarize the report. Use only two words.",
        {
            "confirmed_gaps": [{"key": "output_format"}],
            "problem_sentences": [],
        },
        {
            "name": "add_done_criteria",
            "gap_fill_keys": ["done_criteria"],
            "restructures": False,
        },
        run_id="run",
        judge_model="typesafe/jev-1.13",
    )

    assert result.edits_confined is False
    assert result.evidence["confinement"]["authorized_gap_keys"] == []
    assert gateway.decision_log == []


def test_deletion_uses_whole_prompt_meaning_check_without_sentence_support() -> None:
    requests = []

    def decide(request, **_kwargs):
        requests.append(request)
        return {"type": "noul", "probability_true": 0.1, "confidence": 1.0}

    original = "Summarize the report. Keep the warning."
    gateway = ScriptedGateway(decision=decide)
    result = check_candidate_fidelity(
        gateway,
        original,
        "Summarize the report.",
        {
            "confirmed_gaps": [],
            "problem_sentences": [
                {
                    "sentence_id": "s0002",
                    "sentence": {
                        "id": "s0002",
                        "text": "Keep the warning.",
                        "start": len("Summarize the report. "),
                        "end": len(original),
                    },
                }
            ],
        },
        {"name": "remove_contradictions", "gap_fill_keys": [], "restructures": False},
        run_id="run",
        judge_model="typesafe/jev-1.13",
    )

    assert result.edits_confined is True
    assert result.meaning_preserved is False
    assert [request["key"] for request in requests] == ["fidelity:meaning"]


def test_jev_cannot_claim_an_assumption_when_no_user_answer_was_confirmed() -> None:
    result = _fidelity(
        lambda request, **_kwargs: (
            {
                "type": "choice",
                "choice": "supported_by_assumption",
                "probabilities": {
                    "supported_by_original": 0.0,
                    "supported_by_assumption": 0.99,
                    "new_requirement": 0.0,
                    "unknown": 0.01,
                },
                "confidence": 0.99,
            }
            if request["type"] == "choice"
            else {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        )
    )

    assert result.no_invention is False
    assert (
        "no confirmed user answer was available"
        in result.evidence["sentence_support"][0]["reason"]
    )


def test_explicit_restructuring_allows_reordering_but_keeps_fidelity_checks() -> None:
    requests = []

    def decide(request, **_kwargs):
        requests.append(request)
        if request["type"] == "choice":
            return {
                "type": "choice",
                "choice": "supported_by_original",
                "probabilities": {
                    "supported_by_original": 0.99,
                    "supported_by_assumption": 0.0,
                    "new_requirement": 0.0,
                    "unknown": 0.01,
                },
                "confidence": 0.99,
            }
        return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}

    gateway = ScriptedGateway(decision=decide)
    original = "Write a summary. Keep it brief."
    candidate = "Keep it brief. Write a summary."
    result = check_candidate_fidelity(
        gateway,
        original,
        candidate,
        {"confirmed_gaps": [], "problem_sentences": []},
        {"name": "restructure", "gap_fill_keys": [], "restructures": True},
        run_id="run",
        judge_model="typesafe/jev-1.13",
    )

    assert result.passed is True
    assert result.edits_confined is True
    assert result.evidence["confinement"]["restructuring_authorized"] is True
    assert {request["type"] for request in requests} == {"choice", "noul"}
    assert all("diagnosis" not in request["state"] for request in requests)
    assert all(
        request["state"]["candidate_prompt"] == candidate for request in requests
    )
