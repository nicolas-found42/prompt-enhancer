"""The candidate writer: one structured writer call per round.

The writer receives the original prompt, diagnosis, strategies and previous
failures as state data. The instructions contain no user text, so a prompt
cannot steer the request or leak into the wrong field.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .catalog import DEFAULT_GO_WRITER
from .gateway import Gateway, completion_text, writer_messages

# Version 1 is the historical request without diagnosis; version 2 adds it.
WRITER_INSTRUCTION_VERSIONS = (1, 2)
CURRENT_WRITER_INSTRUCTION_VERSION = 2


class CandidateWriter:
    """Write a rewrite for every selected strategy in one writer call."""

    def __init__(
        self,
        gateway: Gateway,
        *,
        writer_model: str = DEFAULT_GO_WRITER,
        instruction_version: int = CURRENT_WRITER_INSTRUCTION_VERSION,
    ) -> None:
        if instruction_version not in WRITER_INSTRUCTION_VERSIONS:
            raise ValueError("unknown candidate writer instruction version")
        self.gateway = gateway
        self.writer_model = writer_model
        self.instruction_version = instruction_version

    def generate_candidates(self, request: Any) -> Mapping[str, str]:
        """Write every selected strategy in one structured model call."""
        state = request.to_dict()
        original_instructions = (
            "Return JSON only: an object mapping each strategy name in state.strategies "
            "to one complete rewritten prompt. Use a distinct strategy for each. "
            "Preserve the user's language, intent, and unflagged wording. "
            "Use state.previous_failures to address prior round failures."
        )
        current_instructions = (
            "Return JSON only: an object mapping each strategy name in state.strategies "
            "to one complete rewritten prompt. Use a distinct strategy for each. "
            "Use state.diagnosis.confirmed_gaps and problem_sentences to find the diagnosed weaknesses. "
            "Preserve the user's language, intent, and every requirement. Keep unaffected text verbatim "
            "unless the strategy explicitly restructures it. Do not invent facts, requirements, examples, "
            "roles, output formats, or constraints. If essential information is missing, ask for that exact "
            "information instead of supplying a value or adding optional details. If a strategy has no safe "
            "edit, return the original prompt for that strategy. Use state.previous_failures to address "
            "prior round failures."
        )
        if self.instruction_version == 1:
            state.pop("diagnosis", None)
        instructions = original_instructions if self.instruction_version == 1 else current_instructions
        response = self.gateway.chat(self.writer_model, writer_messages(instructions, state), role="writer")
        payload = json.loads(completion_text(response))
        if not isinstance(payload, Mapping):
            raise TypeError("candidate writer must return a JSON object")
        if any(not isinstance(payload.get(strategy.name), str) or not payload[strategy.name].strip() for strategy in request.strategies):
            raise ValueError("candidate writer omitted a selected strategy")
        return {strategy.name: payload[strategy.name].strip() for strategy in request.strategies}


__all__ = [
    "CURRENT_WRITER_INSTRUCTION_VERSION",
    "WRITER_INSTRUCTION_VERSIONS",
    "CandidateWriter",
]
