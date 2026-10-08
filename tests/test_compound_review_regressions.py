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


def test_deleted_first_protected_block_cannot_be_satisfied_by_shifted_duplicate():
    from prompt_enhancer.compound_requirements import audit_ledger, ledger_requirements

    prompt = "Preserve the first block unchanged.\n```\nTOKEN\n```\nExample:\n```\nTOKEN\n```"
    source = prompt[: prompt.index("Example:")].rstrip()
    ledger = audit_ledger(
        compound_gateway(
            prompt, [obligation(prompt, source, protected_values=["TOKEN"])]
        ),
        prompt,
        judge_model="typesafe/jev-1.13",
        run_id="deleted-region",
    )
    candidate = "Preserve the first block unchanged.\nExample:\n```\nTOKEN\n```"
    findings = prompt_findings(
        ledger_requirements(ledger, prompt, []), "deleted", candidate
    )
    assert all(item["status"] != "tested" for item in findings)


def test_section_suffix_names_the_heading_without_its_description():
    assert [
        (item.scope, item.expected)
        for item in scoped_counts("Give two bullets under the Risks section.")
    ] == [("section:Risks", "2")]


def test_then_recipe_step_remains_delegated_data():
    prompt = "Summarize this text: Preheat the oven.\nThen add the flour and stir."
    source = "Then add the flour and stir."
    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=compound_gateway(prompt, [obligation(prompt, source)]),
    ).optimize(prompt)
    assert result["report"]["requirements"]["requirements"] == []
    assert result["report"]["requirements"]["coverage"] == "partial"


@pytest.mark.parametrize("selected", [0, 1])
def test_count_choice_also_supersedes_extracted_copy_of_rejected_clause(
    tmp_path, selected
):
    prompt = "Reply with exactly two bullets and exactly three bullets."
    values = [
        obligation(prompt, source)
        for source in ["exactly two bullets", "exactly three bullets"]
    ]
    gateway = compound_gateway(prompt, values)
    decide = gateway.decision_handler

    def contradict(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:conflict:"):
            items = request["state"]["requirements"]
            counts = [
                item["oracle"]["expected"]
                if item["kind"] == "bullet_count"
                else ("2" if "two" in item["source"] else "3")
                for item in items
            ]
            return {
                "type": "noul",
                "probability_true": 0.99 if len(set(counts)) > 1 else 0.01,
                "confidence": 1.0,
            }
        return decide(request, **params)

    gateway.decision_handler = contradict
    path = tmp_path / "count-copies.sqlite"
    paused = PromptOptimizer(store=RunStore(path), gateway=gateway).optimize(prompt)
    q = paused["questions"][0]
    result = PromptOptimizer(store=RunStore(path), gateway=gateway).resume(
        paused["run_id"], {q["id"]: q["options"][selected]["value"]}
    )
    assert result["status"] == "completed"
    superseded = [
        item
        for item in result["report"]["requirements"]["requirements"]
        if item.get("superseded_by")
    ]
    assert len(superseded) == 2


def test_failed_conflict_audit_recovers_after_choice_without_losing_attempt(tmp_path):
    from prompt_enhancer.gateway import ProviderError

    prompt = "Explain photosynthesis. Give examples. Write the invitation."
    values = [
        obligation(prompt, source)
        for source in ["Explain photosynthesis.", "Give examples."]
    ]
    values.append(
        obligation(
            prompt,
            "Write the invitation.",
            kind="missing_meaning",
            expected="Who is the invitation for?",
        )
    )
    gateway = compound_gateway(prompt, values)
    decide = gateway.decision_handler
    attempts = {}

    def recover(request, **params):
        key = str(request.get("key", ""))
        if key.startswith("requirements:audit:conflict:"):
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] == 1:
                raise ProviderError("scripted", request["model"], 503)
            return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}
        return decide(request, **params)

    gateway.decision_handler = recover
    path = tmp_path / "conflict-recovery.sqlite"
    paused = PromptOptimizer(store=RunStore(path), gateway=gateway).optimize(prompt)
    q = paused["questions"][0]
    result = PromptOptimizer(store=RunStore(path), gateway=gateway).resume(
        paused["run_id"],
        {q["id"]: {"value": "other", "text": "For the science class."}},
    )
    ledger = result["report"]["requirements"]
    assert ledger["coverage"] == "audited"
    assert ledger["release_eligible"] is True
    assert any(
        item["failure_kind"] == "provider_unavailable"
        for item in ledger["conflict_audits"]
    )
    assert any(count > 1 for count in attempts.values())


def test_accumulated_custom_count_answers_have_one_stable_requirement_id():
    from prompt_enhancer.compound_requirements import apply_ledger_answers, audit_ledger

    prompt = "Reply with exactly two bullets and exactly three bullets."
    ledger = audit_ledger(
        compound_gateway(prompt, []),
        prompt,
        judge_model="typesafe/jev-1.13",
        run_id="custom-count",
    )
    answers = [
        {
            "key": "conflict:bullet_count",
            "value": "Use exactly 7 bullets.",
            "source": "answer",
        }
    ]
    once = apply_ledger_answers(ledger, answers, prompt)
    twice = apply_ledger_answers(once, answers, prompt)
    ids = [item["id"] for item in twice["requirements"]]
    assert len(ids) == len(set(ids))


def test_broad_extracted_span_retains_unrelated_meaning_after_count_choice(tmp_path):
    prompt = "Give Section A exactly two bullets and exactly three bullets. Explain photosynthesis."
    gateway = compound_gateway(
        prompt, [obligation(prompt, prompt, scope="section:Section A")]
    )
    decide = gateway.decision_handler
    states = []
    chat = gateway.chat_handler

    def capture(model, messages, **params):
        if "state.strategies" in messages[0]["content"]:
            states.append(json.loads(messages[1]["content"]))
        return chat(model, messages, **params)

    def conflicts(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:conflict:"):
            expected = [
                item["oracle"]["expected"] for item in request["state"]["requirements"]
            ]
            return {
                "type": "noul",
                "probability_true": 0.99
                if any("three bullets" in value for value in expected)
                else 0.01,
                "confidence": 1.0,
            }
        return decide(request, **params)

    gateway.chat_handler, gateway.decision_handler = capture, conflicts
    path = tmp_path / "broad-count.sqlite"
    paused = PromptOptimizer(store=RunStore(path), gateway=gateway).optimize(prompt)
    q = paused["questions"][0]
    result = PromptOptimizer(store=RunStore(path), gateway=gateway).resume(
        paused["run_id"], {q["id"]: q["options"][0]["value"]}
    )
    assert result["status"] == "completed"
    broad = next(
        item
        for item in result["report"]["requirements"]["requirements"]
        if item["kind"] == "semantic"
    )
    assert broad["source"] == prompt and not broad.get("superseded_by")
    effective = broad["effective_interpretation"]
    assert effective["audit"]["status"] == "accepted"
    assert (
        effective["source"]
        == "Give Section A exactly two bullets. Explain photosynthesis."
    )
    assert (
        "".join(
            prompt[part["start"] : part["end"]]
            for part in effective["source_fragments"]
        )
        == effective["source"]
    )
    assert effective["overrides"][0]["provenance"]["source"] == "answer"
    assert states and "Explain photosynthesis." in states[0]["prompt"]
    assert "three bullets" not in states[0]["prompt"]
    constraints = result["report"]["understand"]["hard_constraints"]
    assert any("Explain photosynthesis." in value for value in constraints)
    assert not any("three bullets" in value for value in constraints)


def test_failed_effective_interpretation_retains_original_and_partial_coverage():
    prompt = "Give Section A exactly two bullets and exactly three bullets. Explain photosynthesis."
    gateway = compound_gateway(
        prompt, [obligation(prompt, prompt, scope="section:Section A")]
    )
    decide = gateway.decision_handler

    def uncertain(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:effective:"):
            return {"type": "noul", "probability_true": 0.5, "confidence": 1.0}
        return decide(request, **params)

    gateway.decision_handler = uncertain
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway)
    paused = optimizer.optimize(prompt)
    q = paused["questions"][0]
    result = optimizer.resume(paused["run_id"], {q["id"]: q["options"][0]["value"]})
    ledger = result["report"]["requirements"]
    assert ledger["coverage"] == "partial" and ledger["release_eligible"] is False
    assert any(
        item["source"] == prompt and not item.get("superseded_by")
        for item in ledger["requirements"]
    )


def test_deadline_during_residual_audit_retains_choice_and_original_source(tmp_path):
    from active_clock import TickingClock

    from prompt_enhancer.run_control import RunDeadlineReached

    prompt = "Give Section A exactly two bullets and exactly three bullets. Explain photosynthesis."
    gateway = compound_gateway(
        prompt, [obligation(prompt, prompt, scope="section:Section A")]
    )
    clock = TickingClock()
    decide = gateway.decision_handler

    def deadline(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:effective:"):
            clock.now += 151
            raise RunDeadlineReached(
                history=(), spent_usd=0, elapsed_ms=151_000, deadline_s=150
            )
        return decide(request, **params)

    gateway.decision_handler = deadline
    path = tmp_path / "residual-deadline.sqlite"
    paused = PromptOptimizer(
        store=RunStore(path), gateway=gateway, clock=clock
    ).optimize(prompt)
    q = paused["questions"][0]
    result = PromptOptimizer(store=RunStore(path), gateway=gateway, clock=clock).resume(
        paused["run_id"], {q["id"]: q["options"][0]["value"]}
    )
    ledger = result["report"]["requirements"]
    assert result["report"]["control_state"] == "deadline_reached"
    assert ledger["coverage"] == "partial" and ledger["release_eligible"] is False
    broad = next(item for item in ledger["requirements"] if item["kind"] == "semantic")
    assert broad["source"] == prompt
    assert (
        broad["effective_interpretation"]["overrides"][0]["provenance"]["source"]
        == "answer"
    )
    assert result["report"]["assumptions"][0]["source"] == "answer"
    assert (
        RunStore(path).get_run(result["run_id"])["result"]["report"]["requirements"]
        == ledger
    )


@pytest.mark.parametrize(
    "name",
    ["each", "every", "all", "any", "this", "that", "following", "both", "other"],
)
def test_generic_section_references_are_not_named_sections(name):
    assert scoped_counts(f"Give two bullets under the {name} section.") == ()


def test_selected_count_and_independent_title_survive_rejected_broad_span(tmp_path):
    prompt = "Give Section A exactly two bullets. For Section A, give three bullets and add a title."
    broad_source = "two bullets. For Section A, give three bullets and add a title."
    gateway = compound_gateway(
        prompt,
        [
            obligation(prompt, broad_source, scope="section:Section A"),
            obligation(prompt, "add a title.", scope="section:Section A"),
        ],
    )
    decide = gateway.decision_handler
    chat = gateway.chat_handler
    states = []

    def conflict(request, **params):
        if str(request.get("key", "")).startswith("requirements:audit:conflict:"):
            items = request["state"]["requirements"]
            return {
                "type": "noul",
                "probability_true": 0.99
                if any(item["source"] == broad_source for item in items)
                and any(item["kind"] == "bullet_count" for item in items)
                else 0.01,
                "confidence": 1.0,
            }
        return decide(request, **params)

    def capture(model, messages, **params):
        if "state.strategies" in messages[0]["content"]:
            states.append(json.loads(messages[1]["content"]))
        return chat(model, messages, **params)

    gateway.decision_handler, gateway.chat_handler = conflict, capture
    path = tmp_path / "contained-title.sqlite"
    paused = PromptOptimizer(store=RunStore(path), gateway=gateway).optimize(prompt)
    q = paused["questions"][0]
    selected = next(
        option["value"] for option in q["options"] if option["label"] == "two bullets"
    )
    result = PromptOptimizer(store=RunStore(path), gateway=gateway).resume(
        paused["run_id"], {q["id"]: selected}
    )
    assert result["status"] == "completed"
    ledger = result["report"]["requirements"]
    active = [item for item in ledger["requirements"] if not item.get("superseded_by")]
    assert any(item["id"] == selected for item in active)
    assert any(
        item["source"] == "add a title." and item["audit"]["status"] == "accepted"
        for item in active
    )
    assert states and all(
        "two bullets" in state["prompt"]
        and "add a title." in state["prompt"]
        and "three bullets" not in state["prompt"]
        for state in states
    )
    assert any(
        "add a title." in value
        for value in result["report"]["understand"]["hard_constraints"]
    )


@pytest.mark.parametrize("name", ["All", "Other Recommendations"])
def test_explicit_capitalized_section_name_can_start_with_determiner(name):
    assert [
        (item.scope, item.expected)
        for item in scoped_counts(f"Give two bullets under the {name} section.")
    ] == [("section:" + name, "2")]
