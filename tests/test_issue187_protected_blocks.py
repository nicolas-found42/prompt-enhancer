"""Public optimization controls for explicitly protected fenced sources."""

import pytest
from active_clock import TickingClock, advancing_chat
from test_always_attempt import _gateway

from prompt_enhancer import PromptOptimizer, RunStore


@pytest.mark.parametrize(
    "kind,language,body,changed",
    [
        (
            "code",
            "python",
            'name = "Ada"\nprint(name)',
            'name = "Ada"\nprint(name.upper())',
        ),
        ("data", "csv", "name,count\nAda,2", "name,count\nAda,3"),
    ],
)
def test_source_code_or_data_cannot_drift_even_when_answers_and_judges_pass(
    kind, language, body, changed
):
    original = f"Explain this example.\nDo not change any character in the supplied {kind}:\n```{language}\n{body}\n```"
    rewrite = f"Describe this example.\nDo not change any character in the supplied {kind}:\n```{language}\n{changed}\n```"
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite, weak_output="An explanation.")
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original


def test_faithful_instruction_edit_keeps_the_protected_code_and_can_qualify():
    body = 'name = "Ada"\nprint(name)'
    original = f"Explain this example.\nDo not change any character in the supplied code:\n```python\n{body}\n```"
    rewrite = original.replace(
        "Explain this example.", "Describe what this example does."
    )
    result = PromptOptimizer(
        gateway=_gateway(candidate_text=rewrite, weak_output="An explanation."),
        store=RunStore(":memory:"),
    ).optimize(original)
    assert result["report"]["outcome"] == "converged"
    assert result["final_prompt"] == rewrite
    ledger = result["report"]["requirements"]["requirements"]
    protected = next(item for item in ledger if body + "\n" in item["protected_values"])
    assert protected["scope"] == "candidate_prompt"
    start, end = protected["source_span"]["start"], protected["source_span"]["end"]
    assert original[start:end] == protected["source"]


@pytest.mark.parametrize("fence,indent", [("~~~", ""), ("```", "  ")])
def test_fence_presentation_can_change_without_changing_literal_code(fence, indent):
    original = 'Explain this example.\nDo not change any character in the supplied code:\n```python\nname = "Ada"\nprint(name)\n```'
    rewrite = f'Describe what this example does.\nDo not change any character in the supplied code:\n{indent}{fence}python\n{indent}name = "Ada"\n{indent}print(name)\n{indent}{fence}'
    result = PromptOptimizer(
        gateway=_gateway(candidate_text=rewrite, weak_output="An explanation."),
        store=RunStore(":memory:"),
    ).optimize(original)
    assert result["report"]["outcome"] == "converged"
    findings = result["report"]["selection_evidence"]["selected_candidate"]["metadata"][
        "requirement_findings"
    ]
    assert any(
        item["check"] == "protected_block" and item["status"] == "tested"
        for item in findings
    )


def test_appending_code_to_a_protected_block_is_a_known_character_change():
    original = 'Explain this example.\nDo not change any character in the supplied code:\n```python\nprint("Ada")\n```'
    rewrite = original.replace('print("Ada")', 'print("Ada")\nprint("Grace")').replace(
        "Explain", "Describe"
    )
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite, weak_output="An explanation.")
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original


def test_unrelated_uncertain_indentation_does_not_hide_changed_protected_code():
    original = 'Explain this example.\nDo not change any character in the supplied code:\n```python\nprint("Ada")\n```'
    rewrite = (
        original.replace('print("Ada")', 'print("Grace")').replace(
            "Explain", "Describe"
        )
        + "\n  ~~~text\n\tunrelated\n  ~~~"
    )
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite, weak_output="An explanation.")
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original


def test_literal_instructions_inside_fenced_data_do_not_become_output_obligations():
    original = "Explain this supplied text:\n```text\nReply with exactly PING and nothing else.\nDo not change any character in the supplied code:\n```"
    rewrite = original.replace(
        "Explain this supplied text:", "Describe what this supplied text asks:"
    )
    result = PromptOptimizer(
        gateway=_gateway(
            candidate_text=rewrite, weak_output="It asks for the literal PING."
        ),
        store=RunStore(":memory:"),
    ).optimize(original)
    assert result["report"]["outcome"] == "converged"
    assert result["report"]["requirements"]["requirements"] == []
    assert result["report"]["understand"]["exact_output"] is False


def test_exact_words_in_protected_data_do_not_turn_the_explanation_into_an_exact_reply():
    original = 'Explain this stored directive.\nDo not change any character in the supplied data:\n```text\nReply with exactly "PING" and nothing else.\n```'
    rewrite = original.replace(
        "Explain this stored directive.", "Describe what this stored directive asks."
    )
    gateway = _gateway(candidate_text=rewrite, weak_output="It asks for PING.")
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        original
    )
    assert result["report"]["outcome"] == "converged"
    assert result["report"]["understand"]["exact_output"] is False
    assert not any(
        call.get("payload", {}).get("key", "").startswith("understand:extract:")
        for call in gateway.calls
    )


def test_an_editing_task_does_not_protect_ordinary_fenced_input_from_its_requested_edit():
    original = "Correct the spelling in this data:\n```text\nhelo\n```"
    rewrite = "Fix the spelling of the supplied text:\n```text\nhelo\n```"
    result = PromptOptimizer(
        gateway=_gateway(candidate_text=rewrite, weak_output="hello"),
        store=RunStore(":memory:"),
    ).optimize(original)
    assert result["report"]["outcome"] == "converged"
    assert result["report"]["requirements"]["requirements"] == []


def test_an_exact_reply_and_separately_protected_code_keep_distinct_scopes():
    original = 'Reply with exactly PING and nothing else.\nDo not change any character in the supplied code:\n```python\nprint("Ada")\n```'
    rewrite = 'Respond with exactly PING and nothing else.\nDo not change any character in the supplied code:\n  ~~~python\n  print("Ada")\n  ~~~'
    result = PromptOptimizer(
        gateway=_gateway(candidate_text=rewrite, weak_output="PING"),
        store=RunStore(":memory:"),
    ).optimize(original)
    assert result["report"]["outcome"] == "converged"
    assert result["report"]["understand"]["exact_output"] is True
    requirements = result["report"]["requirements"]["requirements"]
    assert {item["kind"] for item in requirements} == {
        "protected_block",
        "exact_output",
    }


def test_literal_content_outside_supported_fences_retains_scope_uncertainty():
    original = 'Explain this example.\nDo not change any character in the supplied code:\n```python\nprint("Ada")\n```\nOnly discuss the supplied example.'
    rewrite = 'Describe this example.\nDo not change any character in the supplied code:\nprint("Ada")\nOnly discuss the supplied example.'
    result = PromptOptimizer(
        gateway=_gateway(candidate_text=rewrite, weak_output="An explanation."),
        store=RunStore(":memory:"),
    ).optimize(original)
    assert result["report"]["outcome"] == "converged"
    selected = result["report"]["selection_evidence"]["selected_candidate"]
    findings = selected["metadata"]["requirement_findings"]
    assert any(
        item["check"] == "protected_block" and item["status"] == "untestable"
        for item in findings
    )
    assert result["report"]["requirements"]["coverage"] == "partial"


def test_ambiguous_source_indentation_retains_the_explicit_protection_requirement():
    original = 'Explain this example.\nDo not change any character in the supplied code:\n ```python\n\tprint("Ada")\n ```'
    rewrite = original.replace("Explain this example.", "Describe this example.")
    result = PromptOptimizer(
        gateway=_gateway(candidate_text=rewrite, weak_output="An explanation."),
        store=RunStore(":memory:"),
    ).optimize(original)
    assert result["report"]["outcome"] == "converged"
    ledger = result["report"]["requirements"]["requirements"]
    assert any(item["kind"] == "protected_block" for item in ledger)
    findings = result["report"]["selection_evidence"]["selected_candidate"]["metadata"][
        "requirement_findings"
    ]
    assert any(
        item["check"] == "protected_block" and item["status"] == "untestable"
        for item in findings
    )


def test_ambiguous_source_indentation_does_not_hide_changed_code_contents():
    original = 'Explain this example.\nDo not change any character in the supplied code:\n ```python\n\tprint("Ada")\n ```'
    rewrite = original.replace("Explain", "Describe").replace('"Ada"', '"Grace"')
    clock = TickingClock()
    gateway = _gateway(candidate_text=rewrite, weak_output="An explanation.")
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 20)
    result = PromptOptimizer(
        gateway=gateway, store=RunStore(":memory:"), clock=clock
    ).optimize(original)
    assert result["report"]["outcome"] is None
    assert result["final_prompt"] == original
    assert any(
        item["kind"] == "protected_block"
        for item in result["report"]["requirements"]["requirements"]
    )
