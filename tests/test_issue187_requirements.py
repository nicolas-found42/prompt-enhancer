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


@pytest.mark.parametrize(
    "output", ["not JSON", "NaN", '{"value": Infinity}', "```json\n{}\n```"]
)
def test_json_format_failures_cannot_qualify_when_writer_tests_are_rejected(output):
    original = "Return only JSON."
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Return valid JSON without any surrounding text.",
        weak_output=output,
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original
    evidence = result["report"]["history"][0]["evidence"]["selection_evidence"]
    findings = evidence["ranking"][0]["metadata"]["requirement_findings"]
    assert findings and all(item["status"] == "failed" for item in findings)
    assert all(item["check"] == "json_format" for item in findings)


@pytest.mark.parametrize(
    ("output", "status"),
    [
        ('{"name":"A","count":2}', "tested"),
        ('{"name":"B","count":2.0}', "tested"),
        ('{"name":"A","count":true}', "failed"),
        ('{"name":2,"count":2}', "failed"),
        ('{"name":"A"}', "failed"),
        ('{"name":"A","count":2,"extra":0}', "failed"),
        ('{"name":"A","count":1,"count":"two"}', "untestable"),
    ],
)
def test_json_keys_and_types_use_only_the_explicit_whole_answer_shape(output, status):
    original = 'Return a JSON object with exactly these keys and types: {"name":"string","count":"integer"}.'
    rewrite = 'Respond with a JSON object with exactly these keys and types: {"name":"string","count":"integer"}.'
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite, weak_output=output)
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] == (None if status == "failed" else "converged")
    evidence = result["report"]["history"][0]["evidence"]["selection_evidence"]
    findings = [
        item
        for item in evidence["ranking"][0]["metadata"]["requirement_findings"]
        if item["scope"] == "whole_output"
    ]
    assert len(findings) == 15
    assert all(
        item["check"] == "json_schema" and item["status"] == status for item in findings
    )
    assert result["report"]["understand"]["exact_output"] is False


@pytest.mark.parametrize(
    "shape",
    [
        '{"name":"string","count":"string"}',
        '{"name":"integer","count":"string"}',
    ],
)
def test_a_rewrite_cannot_change_json_type_bindings_despite_correct_sampled_answers(
    shape,
):
    original = 'Return a JSON object with exactly these keys and types: {"name":"string","count":"integer"}.'
    clock = TickingClock()
    gateway = _gateway(
        candidate_text=f"Respond with a JSON object with exactly these keys and types: {shape}.",
        weak_output='{"name":"A","count":2}',
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original
    ranking = result["report"]["history"][0]["evidence"]["selection_evidence"][
        "ranking"
    ]
    assert any(
        item["check"] == "json_type_bindings" and item["status"] == "failed"
        for item in ranking[0]["metadata"]["requirement_findings"]
    )


@pytest.mark.parametrize(
    ("output", "status"),
    [
        ("name,count\n", "tested"),
        ('name,count\n"A,B",2\n', "tested"),
        ("count,name\n2,A\n", "failed"),
        ("name,count\nA,2,3\n", "failed"),
        ("name;count\nA;2\n", "failed"),
    ],
)
def test_csv_shape_checks_keep_header_order_and_allow_unconstrained_rows(
    output, status
):
    original = "Return comma-delimited CSV with exactly this header and matching record width: name,count."
    rewrite = "Respond with comma-delimited CSV with exactly this header and matching record width: name,count."
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite, weak_output=output)
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] == (None if status == "failed" else "converged")
    evidence = result["report"]["history"][0]["evidence"]["selection_evidence"]
    findings = [
        item
        for item in evidence["ranking"][0]["metadata"]["requirement_findings"]
        if item["scope"] == "whole_output"
    ]
    assert len(findings) == 15
    assert all(
        item["check"] == "csv_shape" and item["status"] == status for item in findings
    )
    assert result["report"]["understand"]["exact_output"] is False


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


def test_json_types_do_not_add_rules_to_unconstrained_nested_values():
    shape = '{"payload":"object","values":"array","ok":"boolean","missing":"null","ratio":"number"}'
    original = f"Return a JSON object with exactly these keys and types: {shape}."
    gateway = _gateway(
        candidate_text=f"Respond with a JSON object with exactly these keys and types: {shape}.",
        weak_output='{"payload":{"x":1,"x":2},"values":[1,2],"ok":true,"missing":null,"ratio":1e999}',
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        original
    )
    assert result["report"]["outcome"] == "converged"
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    findings = [
        item
        for item in selected["metadata"]["requirement_findings"]
        if item["scope"] == "whole_output"
    ]
    assert findings and all(item["status"] == "tested" for item in findings)


def test_csv_column_list_does_not_invent_a_required_header_row():
    original = (
        "Return comma-delimited CSV with exactly these columns in order: name,count."
    )
    gateway = _gateway(
        candidate_text="Provide comma-delimited CSV with exactly these columns in order: name,count.",
        weak_output="A,2\n",
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        original
    )
    assert result["report"]["outcome"] == "converged"
    assert result["report"]["requirements"]["coverage"] == "partial"
    assert result["report"]["requirements"]["requirements"] == []
