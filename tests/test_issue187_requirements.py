"""Public-flow controls for the next requirement-ledger slice of #187."""

import pytest
from active_clock import TickingClock, advancing_chat
from test_always_attempt import _gateway

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.clarification import InvalidAnswerError


def test_bare_exact_output_literal_blocks_rewrite_even_when_semantic_judge_passes():
    original = "Reply with exactly PING and nothing else."
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Respond with exactly PONG and nothing else.",
        weak_output="PONG",
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)

    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original
    assert result["report"]["control_state"] == "deadline_reached"


def test_literal_as_a_substring_of_a_changed_token_is_not_preservation():
    original = "Reply with exactly PING and nothing else."
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Respond with exactly PINGPONG and nothing else.",
        weak_output="PING",
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)

    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original


def test_bare_json_format_is_not_promoted_to_the_exact_literal_json():
    gateway = _gateway(
        candidate_text="Return valid JSON without any surrounding text.",
        weak_output='"hello"',
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Return only JSON."
    )

    assert result["report"]["outcome"] == "converged"
    assert result["report"]["understand"]["exact_output"] is False
    assert result["report"]["understand"]["hard_constraints"] == []


def test_faithful_surrounding_edit_qualifies_with_its_own_literal_findings():
    original = "Reply with exactly PING and nothing else."
    rewrite = "Respond with exactly PING and nothing else."
    gateway = _gateway(candidate_text=rewrite, weak_output="PING")
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        original
    )

    assert result["report"]["outcome"] == "converged"
    assert result["final_prompt"] == rewrite
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    findings = [
        item
        for item in selected["metadata"]["requirement_findings"]
        if item["scope"] == "whole_output"
    ]
    assert len(findings) == 15
    assert all(item["status"] == "tested" for item in findings)
    assert all(item["candidate_id"] == selected["candidate_id"] for item in findings)
    source = result["report"]["understand"]["provenance"]["requirements"]
    assert source["requirements"][0]["source"] == original
    assert source["coverage"] == "partial"
    assert result["report"]["convergence"]["verification"] == "unverified"


def test_exact_output_failure_blocks_qualification_without_accepted_success_tests():
    original = "Reply with exactly PING and nothing else."
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Respond with exactly PING and nothing else.",
        weak_output="PING\nPING",
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)

    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original
    assert result["report"]["control_state"] == "deadline_reached"


def test_punctuation_only_output_cannot_change_case_even_if_jev_approves():
    original = (
        "Improve punctuation only, preserving every word and its order: "
        '"maybe tomorrow we can ship if tests pass"'
    )
    rewrite = (
        'Correct punctuation only in "maybe tomorrow we can ship if tests pass", '
        "preserving the words and their order."
    )
    clock = TickingClock()
    gateway = _gateway(
        candidate_text=rewrite,
        weak_output="Maybe tomorrow we can ship if tests pass.",
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)

    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original
    assert result["report"]["control_state"] == "deadline_reached"


def test_punctuation_only_positive_control_keeps_words_case_and_order():
    original = (
        "Improve punctuation only, preserving every word and its order: "
        '"maybe tomorrow we can ship if tests pass"'
    )
    rewrite = (
        'Correct punctuation only in "maybe tomorrow we can ship if tests pass", '
        "preserving the words and their order."
    )
    gateway = _gateway(
        candidate_text=rewrite,
        weak_output="maybe, tomorrow, we can ship if tests pass.",
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        original
    )

    assert result["final_prompt"] == rewrite
    assert result["report"]["outcome"] == "converged"
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    findings = selected["metadata"]["requirement_findings"]
    assert all(item["status"] == "tested" for item in findings)
    assert {item["check"] for item in findings} == {
        "punctuation_only",
        "protected_value",
    }


def test_reusable_placeholder_cannot_be_filled_with_an_invented_fact():
    original = "Summarize {{article}} in plain language."
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Summarize the 2026 product launch announcement in plain language.",
        weak_output="A short summary.",
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)

    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original


def test_explicit_word_count_survives_rejected_success_test_proposals():
    original = "Write exactly two words."
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Use exactly two words in the reply.",
        weak_output="one two three",
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)

    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original
    assert result["report"]["control_state"] == "deadline_reached"
    assert (
        result["report"]["requirements"]["requirements"][0]["oracle"]["expected"] == "2"
    )


def test_word_count_positive_control_does_not_protect_the_spelling_of_two():
    gateway = _gateway(
        candidate_text="Use exactly 2 words in the reply.", weak_output="one two"
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Write exactly two words."
    )

    assert result["report"]["outcome"] == "converged"
    assert result["final_prompt"] == "Use exactly 2 words in the reply."
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    findings = selected["metadata"]["requirement_findings"]
    assert len(findings) == 15
    assert all(item["status"] == "tested" for item in findings)
    assert {item["check"] for item in findings} == {"word_count"}


def test_ambiguous_word_boundaries_remain_untestable_instead_of_a_false_failure():
    gateway = _gateway(
        candidate_text="Use exactly 2 words in the reply.", weak_output="don't ship"
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        "Write exactly two words."
    )

    assert result["report"]["outcome"] == "converged"
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    findings = selected["metadata"]["requirement_findings"]
    assert len(findings) == 15
    assert all(item["status"] == "untestable" for item in findings)
    assert all("word boundaries" in item["reason"] for item in findings)


def test_incompatible_sentence_counts_pause_for_the_smallest_explicit_choice():
    original = "Write exactly two sentences and exactly three sentences."
    store = RunStore(":memory:")
    gateway = _gateway(candidate_text="Write exactly two sentences.")
    optimizer = PromptOptimizer(store=store, gateway=gateway)

    result = optimizer.optimize(original)

    assert result["status"] == "needs_input"
    assert result["final_prompt"] == original
    assert result["original_kept"] is True
    assert len(result["questions"]) == 1
    question = result["questions"][0]
    assert question["required_answer"] is True
    assert question["default_answer"] == ""
    assert {item["label"] for item in question["options"]} == {
        "Use exactly 2 sentences.",
        "Use exactly 3 sentences.",
        "Other",
    }
    ledger = result["report"]["requirements"]
    assert len(ledger["requirements"]) == 2
    assert ledger["contradictions"][0]["status"] == "unresolved"
    assert gateway.calls == []

    with pytest.raises(InvalidAnswerError, match="explicit answer"):
        optimizer.skip_clarification(result["run_id"])
    assert store.get_run(result["run_id"])["result"]["status"] == "needs_input"


def test_conflict_resume_preserves_sources_and_records_only_the_users_choice():
    original = "Write exactly two sentences and exactly three sentences."
    rewrite = "Respond using exactly 3 sentences."
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(
        store=store,
        gateway=_gateway(candidate_text=rewrite, weak_output="One. Two. Three."),
    )
    paused = optimizer.optimize(original)
    result = optimizer.resume(paused["run_id"], {"conflict:sentence_count": "3"})

    assert result["run_id"] == paused["run_id"]
    assert result["final_prompt"] == rewrite
    assert result["report"]["outcome"] == "converged"
    ledger = result["report"]["requirements"]
    assert [item["oracle"]["expected"] for item in ledger["requirements"]] == ["2", "3"]
    assert ledger["contradictions"][0]["status"] == "resolved_by_user"
    assert ledger["contradictions"][0]["selected_count"] == "3"
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    assert all(
        item["expected"] == "3" for item in selected["metadata"]["requirement_findings"]
    )
    assert all(
        item["status"] == "untestable"
        for item in selected["metadata"]["requirement_findings"]
    )


def test_a_custom_count_is_user_evidence_and_unresolved_text_cannot_resolve_it():
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=_gateway(candidate_text="Respond using exactly 4 sentences."),
    )
    paused = optimizer.optimize(
        "Write exactly two sentences and exactly three sentences."
    )
    with pytest.raises(InvalidAnswerError, match="count"):
        optimizer.resume(
            paused["run_id"],
            {"conflict:sentence_count": {"value": "other", "text": "Whatever"}},
        )
    result = optimizer.resume(
        paused["run_id"], {"conflict:sentence_count": {"value": "other", "text": "4"}}
    )

    assert result["report"]["outcome"] == "converged"
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    assert all(
        item["expected"] == "4" for item in selected["metadata"]["requirement_findings"]
    )
    ledger = result["report"]["requirements"]
    assert len(ledger["requirements"]) == 3
    assert ledger["requirements"][-1]["source_kind"] == "user_answer"
    assert ledger["requirements"][-1]["oracle"]["expected"] == "4"


@pytest.mark.parametrize(
    ("unit", "output", "status"),
    [
        ("lines", "one\ntwo", "tested"),
        ("lines", "one\ntwo\nthree", "failed"),
        ("bullets", "- one\n- two", "tested"),
        ("bullets", "- one\n- two\n- three", "failed"),
        ("bullets", "- one\n  - nested\n- two", "untestable"),
    ],
)
def test_whole_answer_count_checks_retain_failures_and_scope_uncertainty(
    unit, output, status
):
    original = f"Write exactly two {unit}."
    rewrite = f"Respond using exactly 2 {unit}."
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite, weak_output=output)
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)

    assert result["report"]["outcome"] == (None if status == "failed" else "converged")
    evidence = result["report"]["history"][0]["evidence"]["selection_evidence"]
    findings = evidence["ranking"][0]["metadata"]["requirement_findings"]
    assert len(findings) == 15
    assert all(item["status"] == status for item in findings)
