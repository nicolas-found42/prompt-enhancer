"""Source contracts at the public optimize/history seam (#196, #197)."""

import pytest
from active_clock import TickingClock, advancing_chat
from test_compound_requirements import compound_gateway

from prompt_enhancer import PromptOptimizer, RunStore


def run_contract(tmp_path, prompt, output, *, candidate=None):
    clock = TickingClock()
    gateway = compound_gateway(
        prompt,
        [],
        output=output,
        candidate=candidate or prompt.replace("Return", "Output"),
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 10)
    store = RunStore(tmp_path / "contracts.sqlite")
    result = PromptOptimizer(store=store, gateway=gateway, clock=clock).optimize(
        prompt, {"time_limit_s": 0}
    )
    assert store.get_run(result["run_id"])["result"]["report"] == result["report"]
    evidence = result["report"]["history"][0]["evidence"]["selection_evidence"]
    return result, evidence["ranking"][0]["metadata"].get("requirement_findings", [])


@pytest.mark.parametrize(
    ("output", "status"),
    [
        ('{"title":"A","items":["one"],"extra":true}', "tested"),
        ('{"title":"B","items":[]}', "tested"),
        ('{"title":2,"items":[]}', "failed"),
        ('{"title":"A","items":[2]}', "failed"),
        ('{"items":[]}', "failed"),
        ('{"title":"A","title":2,"items":[]}', "failed"),
        ("not json", "failed"),
    ],
)
def test_compound_json_checks_only_explicit_keys_and_types(tmp_path, output, status):
    prompt = "Summarize the article, then Return JSON containing title as a string and items as an array of strings."
    result, findings = run_contract(tmp_path, prompt, output)
    checks = [
        item
        for item in findings
        if item["check"] == "json_contract" and item["scope"] == "whole_output"
    ]
    assert checks and all(item["status"] == status for item in checks)
    assert result["original_kept"] is (status == "failed")


def test_swapped_compound_json_declaration_cannot_qualify(tmp_path):
    prompt = "Summarize the article, then Return JSON containing title as a string and items as an array of strings."
    candidate = prompt.replace(
        "title as a string and items as an array of strings",
        "title as an array of strings and items as a string",
    )
    result, findings = run_contract(
        tmp_path, prompt, '{"title":"A","items":[]}', candidate=candidate
    )
    assert result["original_kept"]
    assert any(
        item["check"] == "json_type_bindings" and item["status"] == "failed"
        for item in findings
    )


@pytest.mark.parametrize(
    ("output", "status"),
    [
        ('{"user":{"name":"A","extra":1},"items":[{"label":"a"}],"extra":2}', "tested"),
        ('{"user":{"name":1},"items":[]}', "failed"),
        ('{"user":{"name":"A"},"items":[{"label":false}]}', "failed"),
        ('{"user":{"name":"A","name":"B"},"items":[]}', "failed"),
    ],
)
def test_nested_json_declarations_retain_only_source_structure(
    tmp_path, output, status
):
    prompt = 'Read the article, then Return JSON with keys and types: {"user":{"name":"string"},"items":[{"label":"string"}]}.'
    result, findings = run_contract(tmp_path, prompt, output)
    checks = [item for item in findings if item["check"] == "json_contract"]
    assert any(item["status"] == status for item in checks)
    assert result["original_kept"] is (status == "failed")


@pytest.mark.parametrize(
    ("output", "status"),
    [
        ('name,status\n"A,B",ready\nC,done', "tested"),
        ("name,status\nA,ready", "failed"),
        ("name,status\n", "failed"),
        ("name,status\nA,ready,extra\nB,done", "failed"),
        ('name,status\n"unclosed,ready\nB,done', "failed"),
        ("name,status\n\nA,ready\nB,done", "untestable"),
    ],
)
def test_compound_csv_counts_data_records_and_honors_quoted_commas(
    tmp_path, output, status
):
    prompt = "Read the data, then Return CSV with columns name,status, with exactly two data rows; quoted commas inside name are valid."
    result, findings = run_contract(tmp_path, prompt, output)
    checks = [
        item
        for item in findings
        if item["check"] == "csv_contract" and item["scope"] == "whole_output"
    ]
    assert checks and all(item["status"] == status for item in checks)
    assert result["original_kept"] is (status == "failed")


@pytest.mark.parametrize(
    ("prompt", "output"),
    [
        ("Read the data, then Return CSV with columns name,status.", "name,status\n"),
        (
            "Read the data, then Return semicolon-delimited CSV with columns name;status, with exactly one data row.",
            'name;status\n"A;B";ready',
        ),
    ],
)
def test_header_only_and_supported_delimiter_have_no_invented_values(
    tmp_path, prompt, output
):
    result, findings = run_contract(tmp_path, prompt, output)
    assert not result["original_kept"]
    assert all(item["status"] == "tested" for item in findings)


def test_unsupported_csv_dialect_remains_untestable(tmp_path):
    prompt = "Read the data, then Return CSV with columns name,status, using backslash escapes."
    _, findings = run_contract(tmp_path, prompt, "name,status\nA,ready")
    assert any(
        item["check"] == "csv_contract" and item["status"] == "untestable"
        for item in findings
    )


def test_empty_csv_cannot_qualify_a_changed_prompt(tmp_path):
    clock = TickingClock()
    prompt = "Read the data, then Return CSV with columns name,status."
    gateway = compound_gateway(
        prompt, [], output="", candidate=prompt.replace("Return", "Output")
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 10)
    result = PromptOptimizer(
        store=RunStore(tmp_path / "empty-csv.sqlite"), gateway=gateway, clock=clock
    ).optimize(prompt, {"time_limit_s": 0})
    assert result["original_kept"]
    assert result["report"]["outcome"] not in {
        "converged",
        "improved_tested",
        "improved_unverified",
    }


def test_csv_rewrite_cannot_swap_headers_despite_compliant_sampled_answers(tmp_path):
    prompt = "Read the data, then Return CSV with columns name,status, with exactly two data rows."
    result, findings = run_contract(
        tmp_path,
        prompt,
        "name,status\nA,ready\nB,done",
        candidate=prompt.replace("name,status", "status,name"),
    )
    assert result["original_kept"]
    assert any(
        item["check"] == "csv_declaration_bindings" and item["status"] == "failed"
        for item in findings
    )


def test_array_only_json_type_does_not_require_the_singular_spelling_in_a_draft(
    tmp_path,
):
    prompt = (
        "Read the article, then Return JSON containing items as an array of strings."
    )
    result, findings = run_contract(tmp_path, prompt, '{"items":["a"]}')
    assert not result["original_kept"]
    assert all(item["status"] == "tested" for item in findings)


def test_duplicate_undeclared_json_keys_do_not_become_a_stronger_key_constraint(
    tmp_path,
):
    prompt = "Read the article, then Return JSON containing title as a string."
    _, findings = run_contract(tmp_path, prompt, '{"title":"A","extra":1,"extra":2}')
    checks = [item for item in findings if item["check"] == "json_contract"]
    assert checks and all(item["status"] == "untestable" for item in checks)


def test_json_declaration_before_another_sentence_still_checks_required_type(tmp_path):
    prompt = "Read the article, then Return JSON containing title as a string. Preserve the supplied facts."
    result, findings = run_contract(tmp_path, prompt, '{"title":2}')
    assert result["original_kept"]
    assert any(
        item["check"] == "json_contract" and item["status"] == "failed"
        for item in findings
    )


def test_named_json_section_checks_its_own_content_and_allows_other_sections(tmp_path):
    prompt = "Explain the outcome. In section Data, Return JSON containing title as a string."
    result, findings = run_contract(
        tmp_path, prompt, '## Summary\nAll good.\n## Data\n{"title":"A"}'
    )
    assert not result["original_kept"]
    assert any(
        item["scope"] == "section:Data" and item["status"] == "tested"
        for item in findings
    )


def test_unsupported_json_schema_is_disclosed_without_requiring_unstated_values(
    tmp_path,
):
    prompt = "Read the article, then Return JSON matching a union of string or number."
    _, findings = run_contract(tmp_path, prompt, '{"value":1}')
    assert any(
        item["check"] == "json_contract" and item["status"] == "untestable"
        for item in findings
    )


@pytest.mark.parametrize("change", ["delete", "contradict"])
def test_candidate_cannot_delete_or_countermand_a_source_contract(tmp_path, change):
    prompt = "Summarize the article, then Return JSON containing title as a string."
    candidate = "Summarize the article using the title."
    if change == "contradict":
        candidate = prompt + " Return JSON containing title as an integer."
    result, findings = run_contract(
        tmp_path, prompt, '{"title":"A"}', candidate=candidate
    )
    assert result["original_kept"]
    assert any(
        item["check"] == "json_type_bindings" and item["status"] == "failed"
        for item in findings
    )


def test_nested_duplicate_source_schema_is_ambiguous(tmp_path):
    prompt = 'Summarize, then Return JSON with keys and types: {"user":{"name":"string","name":"integer"}}.'
    _, findings = run_contract(tmp_path, prompt, '{"user":{"name":1}}')
    assert any(
        item["check"] == "json_contract" and item["status"] == "untestable"
        for item in findings
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain how to return JSON from a Flask endpoint.",
        "Describe how a Flask endpoint should return JSON.",
        "Explain how you return JSON from Flask.",
    ],
)
def test_describing_json_in_a_tutorial_does_not_require_json_output(tmp_path, prompt):
    result, findings = run_contract(
        tmp_path,
        prompt,
        "Use jsonify to serialize a response.",
        candidate=prompt + " Include a clear explanation.",
    )
    assert not result["original_kept"]
    assert not any(item["check"] == "json_contract" for item in findings)


def test_json_directive_after_a_tutorial_clause_remains_binding(tmp_path):
    prompt = (
        "Explain how the endpoint works, then Return JSON containing title as a string."
    )
    result, findings = run_contract(tmp_path, prompt, '{"title":2}')
    assert result["original_kept"]
    assert any(
        item["check"] == "json_contract" and item["status"] == "failed"
        for item in findings
    )


def test_uncertain_rewrite_cannot_silently_drop_known_type_bindings(tmp_path):
    prompt = "Summarize, then Return JSON containing title as a string."
    result, findings = run_contract(
        tmp_path,
        prompt,
        '{"title":"A"}',
        candidate="Summarize, then Return JSON with title typed as text.",
    )
    assert result["original_kept"]
    assert any(
        item["check"] == "json_type_bindings" and item["status"] == "unresolved"
        for item in findings
    )


@pytest.mark.parametrize(
    ("output", "status"), [("not json", "failed"), ("{}", "untestable")]
)
def test_unsupported_json_schema_still_has_a_known_syntax_requirement(
    tmp_path, output, status
):
    prompt = "Read the article, then Return JSON matching a union of string or number."
    result, findings = run_contract(tmp_path, prompt, output)
    assert any(
        item["check"] == "json_contract" and item["status"] == status
        for item in findings
    )
    assert result["original_kept"] is (status == "failed")
