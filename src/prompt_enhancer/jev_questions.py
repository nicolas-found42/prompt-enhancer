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


def task_branch_description(
    *, label: str, description: str, scope: str, children: Sequence[str]
) -> str:
    descendants = "; ".join(children)
    return (
        f"{label}: {description} Intended scope: {scope} "
        f"This subtree contains: {descendants}."
    )


def task_leaf_question(branch: str) -> str:
    return f"Which {branch} task type best describes the request?"


def sentence_pointer_question(problem: str) -> str:
    return f"Which sentence best contains this problem: {problem}?"


FIDELITY_MEANING_QUESTION = "Does the candidate preserve the original prompt's meaning and all stated constraints?"
FIDELITY_SUPPORT_OPTIONS = {
    "supported_by_original": "The original prompt states or clearly entails this sentence.",
    "supported_by_assumption": "A confirmed user answer in state.confirmed_assumptions supports this sentence.",
    "new_requirement": "This sentence adds a fact or requirement not supported by the prompt or a confirmed answer.",
    "unknown": "The available prompt and confirmed answers do not establish whether this sentence is supported.",
}


def sentence_existence_question(problem: str) -> str:
    return (
        f"Does the problem '{problem}' exist in at least one sentence in this exact candidate window? "
        "Use the full prompt to understand references or conflicts, but consider only the listed candidate sentence IDs. "
        "Answer no when the problem is absent from every sentence in this window."
    )


def fidelity_sentence_support_question(change_id: str) -> str:
    return (
        f"Does the candidate sentence recorded at state.changed_sentences[{change_id!r}] "
        "follow from state.original_prompt or a confirmed user answer?"
    )


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

RESTRUCTURE_ROLE_QUESTION = (
    "Which role best describes state.target_unit_id in the user's complete prompt? "
    "Choose only from the listed roles. Do not rewrite the source unit."
)
RESTRUCTURE_ROLE_DESCRIPTIONS = {
    "context": "Background, facts, audience, or setting that frames the request.",
    "task": "The action or deliverable the user asks for.",
    "constraint": "A requirement, prohibition, boundary, or success criterion.",
    "output_format": "The requested shape, structure, or ordering of the response.",
    "example": "An example, sample, or pattern supplied to guide the response.",
    "other": "Original content that does not clearly fit another role.",
    "unknown": "The role cannot be determined from the available context.",
}

ASSUMPTION_MEANING_QUESTION = (
    "Does the updated prompt preserve the user's original meaning without "
    "contradictory instructions?"
)
