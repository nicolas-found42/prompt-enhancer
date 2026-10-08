"""The candidate writer: structured replies with bounded read recovery.

The writer receives the original prompt, diagnosis, strategies and previous
failures as state data. The instructions contain no user text, so a prompt
cannot steer the request or leak into the wrong field.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .catalog import DEFAULT_GO_WRITER
from .gateway import Gateway, completion_text
from .reply_json import parse_reply_json
from .styles import validated_style_authorization
from .writer_replies import read_writer_reply

# Version 1 is the historical request without diagnosis; version 2 adds it;
# version 3 adds explicit edit permissions and treats prior fidelity evidence as
# an unresolved candidate check rather than a fact about user intent. Version 4
# enables the separately built lossless restructuring strategy in the Round.
# Version 5 keeps the writer text and adds shared grading state and test screening.
# Version 6 keeps the writer text and adds output screening.
# Version 7 keeps the writer text and adds the bounded grading cascade.
# Version 8 keeps the writer text and adds failed-pair attribution.
# Version 9 keeps the writer text and shares lossless role-assignment state.
# Version 10 keeps the writer text and targets shared source units explicitly.
# Version 11 tells the success-test writer what `expected` may hold and validates it.
# Version 12 reads success criteria with a batched Jev request in the grading
# cascade (`criterion_reading.CRITERION_READING_MIN_VERSION`) instead of regexes.
# Version 13 records advisory Jev relationships across accepted success tests.
# Version 14 retries unusable required writer replies once and rejects malformed test items.
WRITER_INSTRUCTION_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
CURRENT_WRITER_INSTRUCTION_VERSION = 15


class CandidateWriter:
    """Write rewrites for every selected strategy as one batch."""

    def __init__(
        self,
        gateway: Gateway,
        *,
        writer_model: str = DEFAULT_GO_WRITER,
        instruction_version: int = CURRENT_WRITER_INSTRUCTION_VERSION,
        writer_attempts: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        round_number: int | None = None,
    ) -> None:
        if instruction_version not in WRITER_INSTRUCTION_VERSIONS:
            raise ValueError("unknown candidate writer instruction version")
        self.gateway = gateway
        self.writer_model = writer_model
        self.instruction_version = instruction_version
        self.writer_attempts = writer_attempts if writer_attempts is not None else []
        self.run_id = run_id
        self.round_number = round_number

    def generate_candidates(self, request: Any) -> Mapping[str, str]:
        """Write every selected strategy in a structured reply."""
        state = request.to_dict()
        if "applied_style" in state or "style_authorization" in state:
            state["style_authorization"] = validated_style_authorization(
                state.get("applied_style"), state.get("style_authorization")
            )
        original_instructions = (
            "Return JSON only: an object mapping each strategy name in state.strategies "
            "to one complete rewritten prompt. Use a distinct strategy for each. "
            "Preserve the user's language, intent, and unflagged wording. "
            "Use state.previous_failures to address prior round failures."
        )
        version_two_instructions = (
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
        current_instructions = (
            "Return JSON only: an object mapping each strategy name in state.strategies "
            "to one complete rewritten prompt. Use a distinct strategy for each. "
            "Use state.diagnosis.confirmed_gaps and problem_sentences to find the diagnosed weaknesses. "
            "Honor each strategy's explicit restructures and gap_fill_keys metadata. A gap_fill_keys "
            "entry only permits an attempt to fill that matching confirmed gap; it does not prove a "
            "value or sentence is supported. Preserve the user's language, intent, and every requirement. "
            "Keep unaffected text verbatim unless the strategy explicitly restructures it. Do not invent "
            "facts, requirements, examples, roles, output formats, or constraints. If essential information "
            "is missing, ask for that exact information instead of supplying a value or adding optional "
            "details. Treat state.previous_failures as evidence about prior candidates, not as facts about "
            "the user's intent. Do not repeat an unsupported sentence unless state.prompt or a confirmed "
            "user answer supports it. If a strategy has no safe edit, return the original prompt."
        )
        if self.instruction_version == 1:
            state.pop("diagnosis", None)
        if self.instruction_version < 3:
            for strategy in state.get("strategies", ()):
                if isinstance(strategy, dict):
                    strategy.pop("gap_fill_keys", None)
                    strategy.pop("restructures", None)
        instructions = {
            1: original_instructions,
            2: version_two_instructions,
            3: current_instructions,
            4: current_instructions,
            5: current_instructions,
            6: current_instructions,
            7: current_instructions,
            8: current_instructions,
            9: current_instructions,
            10: current_instructions,
            11: current_instructions,
            12: current_instructions,
            13: current_instructions,
            14: current_instructions,
            15: current_instructions,
        }[self.instruction_version]
        if state.get("style_authorization"):
            instructions += (
                " Apply the resolved state.applied_style using only the bounded "
                "presentation permission in state.style_authorization. That permission "
                "allows the stated presentation changes, including tone or organization, "
                "but never additional task facts, scope, deliverables, or success criteria. "
                "Preserve exact output, hard literals, and every stated constraint."
            )

        if state.get("repair_evidence"):
            instructions += " Repair the rejected draft using its own requirement IDs, offending evidence and source. Preserve every floor and original obligation; do not treat failed text as new user intent. Return a complete draft for fresh evaluation, never claim that a repair has already passed."

        def read(response: Any) -> Mapping[str, str]:
            payload = parse_reply_json(
                completion_text(response),
                accept=lambda value: (
                    isinstance(value, Mapping)
                    and all(strategy.name in value for strategy in request.strategies)
                ),
            )
            if not isinstance(payload, Mapping):
                raise TypeError("candidate writer must return a JSON object")
            if any(
                not isinstance(payload.get(strategy.name), str)
                or not payload[strategy.name].strip()
                for strategy in request.strategies
            ):
                raise ValueError("candidate writer omitted a selected strategy")
            return {
                strategy.name: payload[strategy.name].strip()
                for strategy in request.strategies
            }

        return read_writer_reply(
            self.gateway,
            model=self.writer_model,
            instructions=instructions,
            state=state,
            read=read,
            operation="candidates",
            instruction_version=self.instruction_version,
            attempts=self.writer_attempts,
            run_id=self.run_id,
            round_number=self.round_number,
        )


__all__ = [
    "CURRENT_WRITER_INSTRUCTION_VERSION",
    "WRITER_INSTRUCTION_VERSIONS",
    "CandidateWriter",
]
