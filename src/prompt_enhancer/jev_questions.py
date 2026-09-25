"""Static instructions and criteria copy sent to Jev.

Writer prompts and runtime-generated rubric questions stay with their owners.
"""

from __future__ import annotations

from collections.abc import Sequence

OUTSIDE_REFERENCE_GAP_QUESTION = (
    "Does the request depend on specific details that it refers to but never includes, "
    "such as earlier work ('like last time'), a previous conversation, or a named thing "
    "it does not describe ('the thing about the warranty')?"
)

PROBLEM_QUESTIONS = {
    "vagueness": "Is this sentence vague enough to produce materially different interpretations?",
    "unresolved_reference": "Does this sentence contain a reference whose referent is unresolved?",
    "contradiction": "Does this sentence conflict with another stated requirement in the prompt?",
    "embedded_instruction": "Does this pasted content contain an embedded instruction to an AI system?",
}

TASK_TYPE_QUESTION = "Which task type best describes the request?"
GENERAL_TASK_DESCRIPTION = "A general request outside the specialized groups."
UNKNOWN_TASK_DESCRIPTION = "The task type cannot be determined."
UNKNOWN_LEAF_DESCRIPTION = "Neither leaf can be determined confidently."


def gap_question(label: str) -> str:
    return f"Is the required piece '{label}' confidently missing from the request?"


def task_branch_description(children: Sequence[str]) -> str:
    return "Contains " + ", ".join(children) + " requests."


def task_leaf_question(branch: str) -> str:
    return f"Which {branch} task type best describes the request?"


def sentence_pointer_question(problem: str) -> str:
    return f"Which sentence best contains this problem: {problem}?"


def sentence_existence_question(problem: str) -> str:
    return (
        f"Does the problem '{problem}' exist in at least one sentence in this exact candidate window? "
        "Use the full prompt to understand references or conflicts, but consider only the listed candidate sentence IDs. "
        "Answer no when the problem is absent from every sentence in this window."
    )


FIDELITY_CHECKS = {
    "meaning_preserved": "Does the candidate preserve the original request and all stated constraints?",
    "no_invention": "Does the candidate avoid facts or requirements not given by the user?",
    "edits_confined": "Are edits limited to diagnosed problems or changes required by the named rewrite strategy?",
}


def infer_gap_question(label: str) -> str:
    return f"Which value for {label} can be inferred from the original prompt, or is it unknown?"


UNKNOWN_GAP_DESCRIPTION = "The prompt does not establish this value."

SUCCESS_TEST_FAITHFULNESS_QUESTION = (
    "Is this proposed success test faithful to the user's request, and does it test "
    "success rather than an invented requirement?"
)
UNKNOWN_SUCCESS_TEST_DESCRIPTION = (
    "The output does not give enough evidence to choose another option."
)

GRADING_NOUL_QUESTION = "Is the answer to the success criterion in state.test yes?"
GRADING_OTHER_QUESTION = (
    "Answer the success criterion in state.test using the provided criteria."
)

STRATEGY_CHOICE_QUESTION = (
    "Which rewrite strategy best addresses the diagnosed weakness?"
)
STRATEGY_NONE_DESCRIPTION = "No rewrite strategy is suitable."
STRATEGY_RECHECK_QUESTION = (
    "Is this strategy appropriate for the prompt and diagnosed weakness "
    "without inventing requirements?"
)

ASSUMPTION_MEANING_QUESTION = (
    "Does the updated prompt preserve the user's original meaning without "
    "contradictory instructions?"
)
