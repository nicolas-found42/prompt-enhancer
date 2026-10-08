"""Compound obligations through the public optimize/history seam (#189–#195)."""

import json

import pytest
from test_always_attempt import _gateway

from prompt_enhancer import PromptOptimizer, RunStore


def compound_gateway(prompt, obligations, *, output="4", candidate=None, audit=0.99):
    gateway = _gateway(
        candidate_text=candidate or prompt + " Please.", weak_output=output
    )
    chat = gateway.chat_handler
    decide = gateway.decision_handler

    def scripted_chat(model, messages, **params):
        if "source-requirements-extraction" in messages[0]["content"]:
            assert model == "qwen3.8-flash"
            assert params["role"] == "writer"
            return json.dumps({"obligations": obligations})
        return chat(model, messages, **params)

    def scripted_decision(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:conflict:"):
            return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}
        if str(request.get("key", "")).startswith("requirements:audit:"):
            return {"type": "noul", "probability_true": audit, "confidence": 1.0}
        return decide(request, **params)

    gateway.chat_handler = scripted_chat
    gateway.decision_handler = scripted_decision
    return gateway


def obligation(prompt, source, **extra):
    start = prompt.index(source)
    return {
        "source": source,
        "start": start,
        "end": start + len(source),
        "kind": "semantic",
        "scope": "whole_output",
        "expected": source,
        **extra,
    }


def test_compound_ledger_has_source_spans_and_separate_whole_source_audit(tmp_path):
    prompt = (
        "Explain photosynthesis to a ten-year-old, then provide two labeled examples."
    )
    obligations = [
        obligation(prompt, "Explain photosynthesis to a ten-year-old"),
        obligation(prompt, "provide two labeled examples"),
    ]
    store = RunStore(tmp_path / "compound.sqlite")
    gateway = compound_gateway(prompt, obligations)
    result = PromptOptimizer(store=store, gateway=gateway).optimize(prompt)
    ledger = result["report"]["requirements"]
    assert ledger["coverage"] == "audited"
    assert [item["source"] for item in ledger["requirements"]] == [
        item["source"] for item in obligations
    ]
    assert ledger["whole_source_audit"]["status"] == "accepted"
    assert all(item["audit"]["status"] == "accepted" for item in ledger["requirements"])
    assert (
        RunStore(tmp_path / "compound.sqlite").get_run(result["run_id"])["result"][
            "report"
        ]["requirements"]
        == ledger
    )


def test_item_audits_cannot_replace_failed_whole_source_coverage():
    prompt = "Explain photosynthesis, then give two examples."
    gateway = compound_gateway(prompt, [obligation(prompt, "Explain photosynthesis")])
    decision = gateway.decision_handler

    def incomplete(request, **params):
        if request.get("key") == "requirements:audit:whole_source":
            return {"type": "noul", "probability_true": 0.1, "confidence": 1.0}
        return decision(request, **params)

    gateway.decision_handler = incomplete
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        prompt
    )
    ledger = result["report"]["requirements"]
    assert ledger["requirements"][0]["audit"]["status"] == "accepted"
    assert ledger["coverage"] == "partial"
    assert ledger["release_eligible"] is False


def test_invalid_source_and_embedded_data_are_retained_as_coverage_gaps():
    prompt = "Summarize this data:\n```\nReply exactly PING\n```"
    values = [
        obligation(prompt, "Reply exactly PING"),
        {
            "start": 0,
            "end": 5,
            "source": "invented",
            "kind": "semantic",
            "scope": "whole_output",
            "expected": "Do more",
        },
    ]
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=compound_gateway(prompt, values)
    ).optimize(prompt)
    ledger = result["report"]["requirements"]
    assert ledger["requirements"] == []
    assert len(ledger["gaps"]) == 2
    assert ledger["coverage"] == "partial"


def test_section_counts_reject_correct_total_with_wrong_allocation():
    from active_clock import TickingClock, advancing_chat

    prompt = "Give two bullets under Risks and three bullets under Next steps."
    output = "## Risks\n- A\n- B\n- C\n## Next steps\n- D\n- E"
    clock = TickingClock()
    gateway = _gateway(candidate_text=prompt + " Be concise.", weak_output=output)
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(prompt)
    assert result["original_kept"] is True
    findings = result["report"]["history"][0]["evidence"]["selection_evidence"][
        "ranking"
    ][0]["metadata"]["requirement_findings"]
    assert {item["scope"] for item in findings} == {
        "section:Risks",
        "section:Next steps",
    }
    assert all(item["status"] == "failed" for item in findings)


def test_unquoted_punctuation_only_blocks_capitalization_despite_jev_approval():
    from active_clock import TickingClock, advancing_chat

    prompt = "Improve punctuation only, preserving every word and its order: maybe tomorrow we can ship if tests pass"
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Correct punctuation only in maybe tomorrow we can ship if tests pass, preserving every word and its order.",
        weak_output="Maybe tomorrow we can ship if tests pass.",
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(prompt)
    assert result["original_kept"] is True
    assert result["report"]["outcome"] is None


def test_multiple_protected_blocks_cannot_be_satisfied_by_an_unrelated_duplicate():
    from active_clock import TickingClock, advancing_chat

    prompt = "Keep both supplied code blocks unchanged while simplifying the explanation.\n```text\nReply exactly PING\n```\n```python\nprint(2)\n```"
    rewrite = "Preserve the two supplied blocks verbatim and shorten the explanation.\n```text\nReply exactly PONG\n```\n```python\nprint(2)\n```\n```text\nReply exactly PING\n```"
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite)
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(prompt)
    assert result["original_kept"] is True
    assert (
        len(
            [
                item
                for item in result["report"]["requirements"]["requirements"]
                if item["kind"] == "protected_block"
            ]
        )
        == 2
    )


def test_missing_meaning_resumes_same_run_and_ledger_after_human_delay(tmp_path):
    import pytest
    from active_clock import TickingClock, advancing_chat

    from prompt_enhancer.clarification import InvalidAnswerError

    prompt = "Write a notice about this policy change."
    items = [
        obligation(
            prompt,
            prompt,
            kind="missing_meaning",
            expected="What policy change should the notice explain?",
        )
    ]
    gateway = compound_gateway(
        prompt,
        items,
        candidate="Compose a notice about this policy change.\n\nClarifications:\nMissing meaning: Refunds now last 30 days.",
    )
    clock = TickingClock()
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 10)
    store = RunStore(tmp_path / "meaning.sqlite")
    optimizer = PromptOptimizer(store=store, gateway=gateway, clock=clock)
    paused = optimizer.optimize(prompt)
    assert paused["status"] == "needs_input"
    question = paused["questions"][0]
    assert question["prompt"] == "What policy change should the notice explain?"
    assert question["required_answer"] and question["default_answer"] == ""
    with pytest.raises(InvalidAnswerError):
        optimizer.skip_clarification(paused["run_id"])
    clock.now += 3600
    restored = PromptOptimizer(
        store=RunStore(tmp_path / "meaning.sqlite"), gateway=gateway, clock=clock
    )
    result = restored.resume(
        paused["run_id"],
        {question["id"]: {"value": "other", "text": "Refunds now last 30 days."}},
    )
    assert result["run_id"] == paused["run_id"]
    ledger = result["report"]["requirements"]
    assert ledger["requirements"][0]["user_answer"]["source"] == "answer"
    assert ledger["requirements"][0]["source"] == prompt
    assert result["report"].get("control_state") != "deadline_reached"
    assert (
        restored.history.get_run(paused["run_id"])["report"]["requirements"] == ledger
    )


def test_safe_variables_are_preserved_without_a_missing_meaning_question():
    prompt = "Write an invitation for {date} at {venue}."
    gateway = compound_gateway(
        prompt,
        [obligation(prompt, prompt)],
        candidate="Compose an invitation for {date} at {venue}.",
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        prompt
    )
    assert result["status"] == "completed"
    assert "{date}" in result["final_prompt"] and "{venue}" in result["final_prompt"]
    assert {
        value
        for item in result["report"]["requirements"]["requirements"]
        for value in item["protected_values"]
    } == {"{date}", "{venue}"}


def test_scoped_conflict_requires_a_choice_and_keeps_other_obligations(tmp_path):
    import pytest

    from prompt_enhancer.clarification import InvalidAnswerError

    prompt = "Give Section A exactly two bullets and exactly three bullets. Preserve {topic}."
    gateway = compound_gateway(
        prompt,
        [],
        output="## Section A\n- One\n- Two",
        candidate="Provide Section A exactly two bullets. Preserve {topic}.\n\nClarifications:\nResolve conflicting requirements: two bullets",
    )
    optimizer = PromptOptimizer(
        store=RunStore(tmp_path / "conflict.sqlite"), gateway=gateway
    )
    paused = optimizer.optimize(prompt)
    assert paused["status"] == "needs_input"
    question = paused["questions"][0]
    assert question["default_answer"] == ""
    assert not any(option["preselected"] for option in question["options"])
    with pytest.raises(InvalidAnswerError):
        optimizer.resume(paused["run_id"], {question["id"]: "invalid"})
    restored = PromptOptimizer(
        store=RunStore(tmp_path / "conflict.sqlite"), gateway=gateway
    )
    result = restored.resume(
        paused["run_id"], {question["id"]: question["options"][0]["value"]}
    )
    ledger = result["report"]["requirements"]
    assert ledger["contradictions"][0]["status"] == "resolved_by_user"
    assert (
        len([item for item in ledger["requirements"] if item.get("superseded_by")]) == 1
    )
    assert any(
        item["protected_values"] == ["{topic}"] for item in ledger["requirements"]
    )
    assert result["report"]["outcome"] == "converged"


def test_non_count_conflict_choice_never_supersedes_an_unrelated_request():
    prompt = "Use only JSON. Use only plain text. Explain photosynthesis."
    values = [
        obligation(prompt, source)
        for source in [
            "Use only JSON.",
            "Use only plain text.",
            "Explain photosynthesis.",
        ]
    ]
    gateway = compound_gateway(prompt, values)
    decide = gateway.decision_handler

    def conflict(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:conflict:"):
            sources = [item["source"] for item in request["state"]["requirements"]]
            return {
                "type": "noul",
                "probability_true": 0.99
                if sources == ["Use only JSON.", "Use only plain text."]
                else 0.01,
                "confidence": 1.0,
            }
        return decide(request, **params)

    gateway.decision_handler = conflict
    from active_clock import TickingClock, advancing_chat

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    )
    paused = optimizer.optimize(prompt)
    question = paused["questions"][0]
    assert len(question["options"]) == 2
    result = optimizer.resume(
        paused["run_id"], {question["id"]: question["options"][0]["value"]}
    )
    ledger = result["report"]["requirements"]
    assert not next(
        item
        for item in ledger["requirements"]
        if item["source"] == "Explain photosynthesis."
    ).get("superseded_by")
    assert (
        next(
            item
            for item in ledger["requirements"]
            if item["source"] == "Use only plain text."
        )["superseded_by"]["source"]
        == "answer"
    )


def test_different_section_counts_have_no_conflict_and_pass_their_own_checks():
    prompt = "Give two bullets under Risks and three bullets under Next steps."
    gateway = compound_gateway(
        prompt,
        [],
        output="## Risks\n- A\n- B\n## Next steps\n- C\n- D\n- E",
        candidate="Provide two bullets under Risks and three bullets under Next steps.",
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        prompt
    )
    assert result["status"] == "completed"
    assert result["report"]["requirements"]["contradictions"] == []
    findings = result["report"]["selection_evidence"]["selected_candidate"]["metadata"][
        "requirement_findings"
    ]
    assert {item["scope"] for item in findings} == {
        "section:Risks",
        "section:Next steps",
    }
    assert all(item["status"] == "tested" for item in findings)


@pytest.mark.parametrize(
    "operation,preservation,source,output,status",
    [
        (
            "capitalization",
            "every word and its order",
            "maybe tomorrow",
            "Maybe Tomorrow",
            "tested",
        ),
        (
            "capitalization",
            "every word and its order",
            "maybe tomorrow",
            "Tomorrow Maybe",
            "failed",
        ),
        (
            "word order",
            "every word and its case",
            "maybe tomorrow",
            "tomorrow maybe",
            "tested",
        ),
        (
            "word order",
            "every word and its case",
            "maybe tomorrow",
            "Tomorrow maybe",
            "failed",
        ),
        (
            "punctuation, capitalization and word order",
            "every word",
            "maybe tomorrow",
            "Tomorrow, maybe.",
            "tested",
        ),
        (
            "punctuation and capitalization",
            "every word and its order",
            "maybe tomorrow",
            "Maybe, tomorrow.",
            "tested",
        ),
        (
            "punctuation",
            "every word and its order",
            "maybe tomorrow",
            "maybe, soon.",
            "failed",
        ),
        (
            "punctuation",
            "every word and its order",
            "don't ship",
            "don't ship.",
            "untestable",
        ),
    ],
)
def test_source_edit_permissions_govern_despite_passing_semantic_judgment(
    operation, preservation, source, output, status
):
    from active_clock import TickingClock, advancing_chat

    prompt = f"Improve {operation} only, preserving {preservation}: {source}"
    candidate = f"Correct {operation} only in {source}, preserving {preservation}."
    clock = TickingClock()
    gateway = _gateway(candidate_text=candidate, weak_output=output)
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(prompt)
    findings = result["report"]["history"][0]["evidence"]["selection_evidence"][
        "ranking"
    ][0]["metadata"]["requirement_findings"]
    own = [item for item in findings if item["check"] == "edit_restriction"]
    assert own and {item["status"] for item in own} == {status}
    if status == "failed":
        assert result["original_kept"] is True
    else:
        assert result["report"]["outcome"] == "converged"


@pytest.mark.parametrize(
    "output,status",
    [
        ("## Risks\n- A\n  - Nested\n- B", "untestable"),
        ("## Risks\n- A\n## Risks\n- B", "untestable"),
        ("Risks - A - B", "untestable"),
        ("## Benefits\n- A\n- B", "failed"),
    ],
)
def test_section_boundary_uncertainty_and_missing_sections_remain_distinct(
    output, status
):
    from active_clock import TickingClock, advancing_chat

    prompt = "Give two bullets under Risks."
    clock = TickingClock()
    gateway = _gateway(
        candidate_text="Provide two bullets under Risks.", weak_output=output
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=gateway, clock=clock
    ).optimize(prompt)
    own = result["report"]["history"][0]["evidence"]["selection_evidence"]["ranking"][
        0
    ]["metadata"]["requirement_findings"]
    assert own and {item["status"] for item in own} == {status}
    assert all(item["scope"] == "section:Risks" and item["source_span"] for item in own)


def test_http_jobs_preserve_section_evidence_and_coverage_in_saved_history(tmp_path):
    from fastapi.testclient import TestClient

    from prompt_enhancer.api import create_app
    from prompt_enhancer.jobs import RunJobs

    prompt = "Give two bullets under Risks and three bullets under Next steps."
    gateway = compound_gateway(
        prompt,
        [],
        output="## Risks\n- A\n- B\n## Next steps\n- C\n- D\n- E",
        candidate="Provide two bullets under Risks and three bullets under Next steps.",
    )
    store = RunStore(tmp_path / "http.sqlite")
    optimizer = PromptOptimizer(store=store, gateway=gateway)
    app = create_app(optimizer=optimizer)
    with TestClient(app) as client:
        submitted = client.post("/api/jobs/optimize", json={"prompt": prompt})
        assert submitted.status_code == 202
        run_id = submitted.json()["run_id"]
        app.state.jobs.wait(run_id)
        result = client.get(f"/api/jobs/{run_id}").json()
        assert result["result"]["report"]["requirements"]["coverage"] == "audited"
        events = [event for event in result["events"] if event["kind"] == "checks"]
        assert events and {item["scope"] for item in events[0]["checks"]} == {
            "section:Risks",
            "section:Next steps",
        }
        detail = client.get(f"/api/runs/{run_id}").json()
        assert (
            detail["report"]["requirements"]
            == result["result"]["report"]["requirements"]
        )
        assert (
            RunJobs(store=RunStore(tmp_path / "http.sqlite")).get(run_id)["events"]
            == result["events"]
        )


def test_unquoted_delegated_count_instructions_remain_source_data():
    prompt = "Summarize this instruction: Give Section A exactly two bullets and exactly three bullets."
    gateway = compound_gateway(
        prompt,
        [],
        candidate="Describe this instruction: Give Section A exactly two bullets and exactly three bullets.",
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        prompt
    )
    assert not any(
        item["kind"].endswith("_count")
        for item in result["report"]["requirements"]["requirements"]
    )


def test_extractor_cannot_promote_blockquoted_or_delegated_data():
    for prompt, data in [
        ("Summarize this instruction: Use only JSON.", "Use only JSON."),
        ("Summarize the supplied text:\n> Use only JSON.", "Use only JSON."),
    ]:
        gateway = compound_gateway(prompt, [obligation(prompt, data)])
        result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
            prompt
        )
        ledger = result["report"]["requirements"]
        assert ledger["requirements"] == []
        assert ledger["gaps"] and ledger["coverage"] == "partial"


def test_same_scope_conflicts_are_audited_across_mechanical_and_semantic_kinds():
    prompt = "Give two bullets under Risks. Write no bullets under Risks."
    values = [
        obligation(prompt, "Write no bullets under Risks.", scope="section:Risks")
    ]
    gateway = compound_gateway(prompt, values)
    decide = gateway.decision_handler

    def conflict(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:conflict:"):
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        return decide(request, **params)

    gateway.decision_handler = conflict
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        prompt
    )
    assert result["status"] == "needs_input"
    assert len(result["questions"][0]["options"]) == 2


@pytest.mark.parametrize("failed_stage", ["extraction", "item", "whole"])
def test_provider_failure_preserves_partial_ledger_in_reloaded_history(
    tmp_path, failed_stage
):
    from active_clock import TickingClock, advancing_chat

    from prompt_enhancer.gateway import ProviderError

    prompt = "Reply with exactly two bullets."
    gateway = compound_gateway(
        prompt, [obligation(prompt, "Reply with exactly two bullets")]
    )
    chat, decide = gateway.chat_handler, gateway.decision_handler

    def unavailable_chat(model, messages, **params):
        if (
            failed_stage == "extraction"
            and "source-requirements-extraction" in messages[0]["content"]
        ):
            raise ProviderError("scripted", model, 503)
        return chat(model, messages, **params)

    def unavailable_audit(request, **params):
        key = str(request.get("key", ""))
        if (failed_stage == "item" and key.startswith("requirements:audit:item:")) or (
            failed_stage == "whole" and key == "requirements:audit:whole_source"
        ):
            raise ProviderError("scripted", request["model"], 503)
        return decide(request, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(unavailable_chat, clock, 30)
    gateway.decision_handler = unavailable_audit
    store_path = tmp_path / "provider-failure.sqlite"
    result = PromptOptimizer(
        store=RunStore(store_path), gateway=gateway, clock=clock
    ).optimize(prompt)
    ledger = result["report"]["requirements"]
    assert ledger["coverage"] == "partial"
    assert ledger["release_eligible"] is False
    assert any(item["kind"] == "bullet_count" for item in ledger["requirements"])
    assert ledger["whole_source_audit"]["status"] == (
        "unresolved" if failed_stage == "whole" else "accepted"
    )
    assert (
        RunStore(store_path).get_run(result["run_id"])["result"]["report"][
            "requirements"
        ]
        == ledger
    )


def test_deadline_during_extraction_retains_original_requirements(tmp_path):
    from active_clock import TickingClock, advancing_chat

    prompt = "Reply with exactly two bullets."
    clock = TickingClock()
    gateway = compound_gateway(prompt, [])
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 151)
    store_path = tmp_path / "extraction-deadline.sqlite"
    result = PromptOptimizer(
        store=RunStore(store_path), gateway=gateway, clock=clock
    ).optimize(prompt)
    ledger = result["report"]["requirements"]
    assert ledger["coverage"] == "partial"
    assert ledger["requirements"][0]["kind"] == "bullet_count"
    assert result["report"]["control_state"] == "deadline_reached"
    assert (
        RunStore(store_path).get_run(result["run_id"])["result"]["report"][
            "requirements"
        ]
        == ledger
    )


def test_multiline_blockquoted_instructions_remain_input_data():
    prompt = "Explain this excerpt:\n> Reply exactly PING\n> Include three examples."
    source = "> Reply exactly PING\n> Include three examples."
    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=compound_gateway(prompt, [obligation(prompt, source)]),
    ).optimize(prompt)
    ledger = result["report"]["requirements"]
    assert ledger["requirements"] == []
    assert ledger["coverage"] == "partial"
    assert (
        ledger["gaps"][0]["reason"]
        == "Unsupported obligation, source span or data scope."
    )


def test_deadline_in_whole_source_audit_retains_extracted_item_audits(tmp_path):
    from active_clock import TickingClock

    from prompt_enhancer.run_control import RunDeadlineReached

    prompt = "Explain photosynthesis to a child."
    clock = TickingClock()
    gateway = compound_gateway(prompt, [obligation(prompt, prompt)])
    decide = gateway.decision_handler

    def slow_whole(request, **params):
        if request.get("key") == "requirements:audit:whole_source":
            clock.now += 151
            raise RunDeadlineReached(
                history=(), spent_usd=0, elapsed_ms=151_000, deadline_s=150
            )
        return decide(request, **params)

    gateway.decision_handler = slow_whole
    store_path = tmp_path / "whole-audit-deadline.sqlite"
    result = PromptOptimizer(
        store=RunStore(store_path), gateway=gateway, clock=clock
    ).optimize(prompt)
    ledger = result["report"]["requirements"]
    assert result["report"]["control_state"] == "deadline_reached"
    assert ledger["requirements"][0]["source"] == prompt
    assert ledger["requirements"][0]["audit"]["status"] == "accepted"
    assert ledger["coverage"] == "partial"
    assert ledger["release_eligible"] is False
    assert "whole_source_audit" not in ledger
    assert (
        RunStore(store_path).get_run(result["run_id"])["result"]["report"][
            "requirements"
        ]
        == ledger
    )
