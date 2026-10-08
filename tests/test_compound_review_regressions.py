"""Review regressions through optimizer persistence and requirement oracles."""

import json

import pytest
from test_compound_requirements import compound_gateway, obligation

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.requirement_scopes import scoped_counts, section_text
from prompt_enhancer.requirements import prompt_findings


@pytest.mark.parametrize("tail", ["in Spanish", "in a friendly tone"])
def test_language_and_tone_are_not_section_headings(tail):
    assert scoped_counts(f"Give two bullets {tail}.") == ()


@pytest.mark.parametrize(
    "data",
    [
        "> Hello. Give Section A exactly two bullets.",
        "Hello. Give Section A exactly two bullets.",
    ],
)
def test_delegated_scoped_counts_are_data(data):
    prompt = "Summarize the supplied text:\n" + data
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=compound_gateway(prompt, [])
    ).optimize(prompt)
    assert not any(
        item["scope"].startswith("section:")
        for item in result["report"]["requirements"]["requirements"]
    )


def test_instruction_after_delegated_excerpt_is_audited():
    prompt = "Summarize this text: hello.\nThen give the summary a title."
    sources = [
        obligation(prompt, "hello."),
        obligation(prompt, "Then give the summary a title."),
    ]
    result = PromptOptimizer(
        store=RunStore(":memory:"), gateway=compound_gateway(prompt, sources)
    ).optimize(prompt)
    ledger = result["report"]["requirements"]
    assert [item["source"] for item in ledger["requirements"]] == [
        "Then give the summary a title."
    ]
    assert ledger["requirements"][0]["audit"]["status"] == "accepted"


@pytest.mark.parametrize("heading", ["## Next Steps", "Next Steps:"])
def test_section_identity_ignores_heading_case(heading):
    body, uncertainty = section_text(heading + "\n- One\n- Two", "section:Next steps")
    assert uncertainty is None
    assert body.splitlines() == ["- One", "- Two"]


def test_section_duplicate_case_and_blank_lines_are_preserved():
    body, uncertainty = section_text("## A\ntext\n\n## B\nother", "section:A")
    assert uncertainty is None
    assert body.splitlines() == ["text", ""]
    assert section_text("## A\none\n## a\ntwo", "section:A")[1] is not None


@pytest.mark.parametrize(
    "choice,expected",
    [
        (0, "Give Section A exactly two bullets."),
        (1, "Give Section A exactly three bullets."),
    ],
)
def test_scoped_choice_preserves_grammatical_working_instruction(choice, expected):
    prompt = "Give Section A exactly two bullets and exactly three bullets. Preserve {topic}."
    gateway = compound_gateway(prompt, [])
    seen = []
    chat = gateway.chat_handler

    def capture(model, messages, **params):
        if "state.strategies" in messages[0]["content"]:
            seen.append(json.loads(messages[1]["content"])["prompt"])
        return chat(model, messages, **params)

    gateway.chat_handler = capture
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway)
    paused = optimizer.optimize(prompt)
    q = paused["questions"][0]
    optimizer.resume(paused["run_id"], {q["id"]: q["options"][choice]["value"]})
    assert seen and any(expected in text for text in seen)
    assert all("Preserve {topic}." in text for text in seen)


def test_legacy_count_choice_gets_source_audits_before_rewriting(tmp_path):
    prompt = "Reply with exactly two bullets and exactly three bullets."
    path = tmp_path / "legacy.sqlite"
    gateway = compound_gateway(prompt, [])
    paused = PromptOptimizer(store=RunStore(path), gateway=gateway).optimize(prompt)
    assert paused["status"] == "needs_input"
    q = paused["questions"][0]
    result = PromptOptimizer(store=RunStore(path), gateway=gateway).resume(
        paused["run_id"], {q["id"]: q["options"][0]["value"]}
    )
    ledger = result["report"]["requirements"]
    assert ledger["whole_source_audit"]["status"] == "accepted"
    assert all(
        item.get("audit", {}).get("status") == "accepted"
        for item in ledger["requirements"]
        if item["source_kind"] == "original_prompt"
    )


def test_two_conflicts_resume_after_each_database_reload_and_reuse_audits(tmp_path):
    prompt = "Use only JSON. Use only plain text. Use only XML. Explain photosynthesis."
    values = [
        obligation(prompt, source)
        for source in [
            "Use only JSON.",
            "Use only plain text.",
            "Use only XML.",
            "Explain photosynthesis.",
        ]
    ]
    gateway = compound_gateway(prompt, values)
    decide = gateway.decision_handler
    calls = []
    batches = []
    batch = gateway.decide_batch

    def capture_batch(requests, **params):
        batches.append(requests)
        return batch(requests, **params)

    gateway.decide_batch = capture_batch

    def conflicts(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:conflict:"):
            calls.append(request["key"])
            sources = [item["source"] for item in request["state"]["requirements"]]
            return {
                "type": "noul",
                "probability_true": 0.99
                if all(source.startswith("Use only") for source in sources)
                else 0.01,
                "confidence": 1.0,
            }
        return decide(request, **params)

    gateway.decision_handler = conflicts
    path = tmp_path / "sequential.sqlite"
    result = PromptOptimizer(store=RunStore(path), gateway=gateway).optimize(prompt)
    for _ in range(2):
        assert result["status"] == "needs_input"
        q = result["questions"][0]
        result = PromptOptimizer(store=RunStore(path), gateway=gateway).resume(
            result["run_id"], {q["id"]: q["options"][0]["value"]}
        )
    assert result["status"] == "completed"
    assert len(calls) == len(set(calls))
    assert any(
        len(
            [
                request
                for request in batch
                if str(request.get("key", "")).startswith(
                    "requirements:audit:conflict:"
                )
            ]
        )
        > 1
        for batch in batches
    )
    assert not next(
        item
        for item in result["report"]["requirements"]["requirements"]
        if item["source"] == "Explain photosynthesis."
    ).get("superseded_by")
    assert (
        RunStore(path).get_run(result["run_id"])["result"]["report"]["requirements"]
        == result["report"]["requirements"]
    )


def test_extracted_literal_is_bound_to_its_original_fenced_region():
    prompt = "Preserve the first supplied block unchanged.\n```\nTOKEN\n```\nSecond example:\n```\nTOKEN\n```"
    source = prompt[: prompt.index("Second example:")].rstrip()
    candidate = prompt.replace("TOKEN", "CHANGED", 1)
    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=compound_gateway(
            prompt,
            [obligation(prompt, source, protected_values=["TOKEN"])],
            candidate=candidate,
        ),
    ).optimize(prompt)
    ledger = result["report"]["requirements"]
    from prompt_enhancer.compound_requirements import ledger_requirements

    findings = prompt_findings(
        ledger_requirements(ledger, prompt, []), "rewrite", candidate
    )
    assert any(
        item["check"] == "protected_value"
        and item["status"] == "failed"
        and item["region_index"] == 0
        for item in findings
    )


@pytest.mark.parametrize(
    "candidate,status",
    [
        ("```\nTOKEN\n```\n```\nCHANGED\n```", "tested"),
        ("```\nCHANGED\n```\n```\nTOKEN\n```", "failed"),
    ],
)
def test_protected_literal_region_accepts_original_and_rejects_decoy(candidate, status):
    from prompt_enhancer.compound_requirements import audit_ledger, ledger_requirements

    prompt = "Preserve this literal in the first block:\n```\nTOKEN\n```\nExample:\n```\nTOKEN\n```"
    source = prompt[: prompt.index("Example:")].rstrip()
    ledger = audit_ledger(
        compound_gateway(
            prompt, [obligation(prompt, source, protected_values=["TOKEN"])]
        ),
        prompt,
        judge_model="typesafe/jev-1.13",
        run_id="literal-region",
    )
    findings = prompt_findings(
        ledger_requirements(ledger, prompt, []), "candidate", candidate
    )
    assert [item["status"] for item in findings] == [status]


def test_ambiguous_extracted_literal_region_requires_scope_decision():
    from prompt_enhancer.compound_requirements import audit_ledger, ledger_requirements

    prompt = "Preserve TOKEN in the supplied block.\n```\nTOKEN\n```\n```\nTOKEN\n```"
    ledger = audit_ledger(
        compound_gateway(
            prompt, [obligation(prompt, prompt, protected_values=["TOKEN"])]
        ),
        prompt,
        judge_model="typesafe/jev-1.13",
        run_id="ambiguous-region",
    )
    findings = prompt_findings(
        ledger_requirements(ledger, prompt, []), "candidate", prompt
    )
    assert [item["status"] for item in findings] == ["untestable"]


def test_explicit_in_section_count_still_has_supported_scope():
    assert [
        (item.scope, item.expected)
        for item in scoped_counts("Give two bullets in section Risks.")
    ] == [("section:Risks", "2")]


def test_under_pressure_is_not_a_named_section():
    assert scoped_counts("Write three sentences under pressure.") == ()


def test_extracted_unchanged_block_rejects_changes_beyond_declared_token():
    from prompt_enhancer.compound_requirements import audit_ledger, ledger_requirements

    prompt = "Preserve the first block unchanged.\n```\nTOKEN\nOther line\n```\nExample:\n```\nTOKEN\nOther line\n```"
    source = prompt[: prompt.index("Example:")].rstrip()
    gateway = compound_gateway(
        prompt, [obligation(prompt, source, protected_values=["TOKEN"])]
    )
    ledger = audit_ledger(
        gateway, prompt, judge_model="typesafe/jev-1.13", run_id="entire-block"
    )
    requirements = ledger_requirements(ledger, prompt, [])
    assert [
        item["status"] for item in prompt_findings(requirements, "original", prompt)
    ] == ["tested"]
    candidate = prompt.replace("Other line", "Changed line", 1)
    assert [
        item["status"] for item in prompt_findings(requirements, "changed", candidate)
    ] == ["failed"]
