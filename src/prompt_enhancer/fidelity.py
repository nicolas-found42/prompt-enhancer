"""Code determines rewrite boundaries; Jev checks changed sentence support."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from . import jev_questions
from .diagnosis import split_sentences
from .gateway import Gateway, ProviderError
from .jev import ChoiceDecision, JevResponseError, NoulDecision, parse_decision
from .lossless_restructuring import verify_lossless_proof

FIDELITY_THRESHOLD = 0.8
DECISION_BATCH_LIMIT = 40


@dataclass(frozen=True)
class FidelityResult:
    """Aggregate fidelity gates and their sentence-level audit evidence."""

    meaning_preserved: bool
    no_invention: bool
    edits_confined: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.meaning_preserved and self.no_invention and self.edits_confined

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.evidence.get("reasons", ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "meaning_preserved": self.meaning_preserved,
            "no_invention": self.no_invention,
            "edits_confined": self.edits_confined,
            "meaning": self.meaning_preserved,
            "invention": self.no_invention,
            "confined": self.edits_confined,
            "passed": self.passed,
            "evidence": dict(self.evidence),
        }


def sentence_edit_script(original_prompt: str, candidate_prompt: str) -> dict[str, Any]:
    """Return a stable sentence edit script with unique source occurrence IDs.

    Equal sentences are exact anchors. Duplicate sentences receive distinct IDs
    in source order, so removing one occurrence does not erase its provenance.
    Unequal replacement groups are reported as correspondence errors instead
    of guessing which source sentence supports each candidate sentence.
    """
    source = _sentence_entries(original_prompt, "source")
    candidate = _sentence_entries(candidate_prompt, "candidate")
    matcher = SequenceMatcher(
        a=[(item["text"], item["occurrence"]) for item in source],
        b=[(item["text"], item["occurrence"]) for item in candidate],
        autojunk=False,
    )
    edits: list[dict[str, Any]] = []
    errors: list[str] = []
    anchors: list[dict[str, Any]] = []

    def append_edit(
        operation: str,
        source_sentences: Sequence[Mapping[str, Any]],
        candidate_sentences: Sequence[Mapping[str, Any]],
        *,
        source_gap: int | None = None,
    ) -> None:
        change_id = f"change-{len(edits) + 1:04d}"
        source_values = [dict(item) for item in source_sentences]
        candidate_values = [dict(item) for item in candidate_sentences]
        edits.append(
            {
                "change_id": change_id,
                "operation": operation,
                "source_sentence_ids": [item["id"] for item in source_values],
                "source_sentences": source_values,
                "candidate_sentence_ids": [item["id"] for item in candidate_values],
                "candidate_sentences": candidate_values,
                "source_gap": source_gap,
            }
        )

    for (
        tag,
        source_start,
        source_end,
        candidate_start,
        candidate_end,
    ) in matcher.get_opcodes():
        if tag == "equal":
            anchors.extend(
                {
                    "source_sentence_id": source[source_index]["id"],
                    "candidate_sentence_id": candidate[candidate_index]["id"],
                    "text": source[source_index]["text"],
                }
                for source_index, candidate_index in zip(
                    range(source_start, source_end),
                    range(candidate_start, candidate_end),
                    strict=True,
                )
            )
            continue
        removed = source[source_start:source_end]
        added = candidate[candidate_start:candidate_end]
        if tag == "replace" and removed and added:
            if len(removed) != len(added):
                append_edit("replace", removed, added)
                errors.append(
                    "sentence correspondence could not be established for "
                    + "; ".join(str(item["text"]) for item in added)
                )
            else:
                for old, new in zip(removed, added, strict=True):
                    append_edit("replace", (old,), (new,))
        elif tag == "delete":
            for old in removed:
                append_edit("delete", (old,), ())
        elif tag == "insert":
            for new in added:
                append_edit("insert", (), (new,), source_gap=source_start)

    return {
        "edits": edits,
        "unchanged_anchors": anchors,
        "correspondence_errors": errors,
    }


def _sentence_entries(prompt: str, source: str) -> list[dict[str, Any]]:
    occurrences: Counter[str] = Counter()
    values: list[dict[str, Any]] = []
    for sentence in split_sentences(prompt):
        occurrences[sentence.text] += 1
        values.append(
            {
                "id": f"{source}-{sentence.id}",
                "diagnosis_id": sentence.id,
                "occurrence": occurrences[sentence.text],
                "text": sentence.text,
                "start": sentence.start,
                "end": sentence.end,
            }
        )
    return values


def _diagnosed_source_ids(
    diagnosis: Mapping[str, Any], source: Sequence[Mapping[str, Any]]
) -> set[str]:
    diagnosed: set[str] = set()
    for problem in diagnosis.get("problem_sentences", ()):
        if not isinstance(problem, Mapping):
            continue
        sentence = problem.get("sentence")
        sentence = sentence if isinstance(sentence, Mapping) else {}
        sentence_id = problem.get("sentence_id", sentence.get("id"))
        if sentence_id is not None:
            matched = [
                item for item in source if item["diagnosis_id"] == str(sentence_id)
            ]
            if len(matched) == 1:
                diagnosed.add(str(matched[0]["id"]))
                continue
        start, end = sentence.get("start"), sentence.get("end")
        if isinstance(start, int) and isinstance(end, int):
            matched = [
                item for item in source if item["start"] == start and item["end"] == end
            ]
            if len(matched) == 1:
                diagnosed.add(str(matched[0]["id"]))
                continue
        text = sentence.get("text")
        if isinstance(text, str):
            matched = [item for item in source if item["text"] == text]
            if len(matched) == 1:
                diagnosed.add(str(matched[0]["id"]))
    return diagnosed


def _strategy_metadata(strategy: Any) -> dict[str, Any]:
    if isinstance(strategy, Mapping):
        return dict(strategy)
    to_dict = getattr(strategy, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    return {"name": str(strategy), "gap_fill_keys": (), "restructures": False}


def _confirmed_answers(assumptions: Sequence[Any]) -> list[dict[str, Any]]:
    answers = []
    for assumption in assumptions:
        if not isinstance(assumption, Mapping) or assumption.get("source") != "answer":
            continue
        if assumption.get("key") is None or assumption.get("value") is None:
            continue
        answer = {
            "key": str(assumption["key"]),
            "value": str(assumption["value"]),
            "source": "answer",
        }
        if assumption.get("label"):
            answer["label"] = str(assumption["label"])
        answers.append(answer)
    return answers


def _confinement(
    original_prompt: str,
    diagnosis: Mapping[str, Any],
    strategy: Mapping[str, Any],
    edits: Sequence[Mapping[str, Any]],
    correspondence_errors: Sequence[str],
) -> dict[str, Any]:
    source = _sentence_entries(original_prompt, "source")
    diagnosed_ids = _diagnosed_source_ids(diagnosis, source)
    confirmed_gap_keys = {
        str(gap["key"])
        for gap in diagnosis.get("confirmed_gaps", ())
        if isinstance(gap, Mapping) and gap.get("key") is not None
    }
    authorized_gap_keys = confirmed_gap_keys & {
        str(key) for key in strategy.get("gap_fill_keys", ())
    }
    restructuring = strategy.get("restructures") is True
    failures: list[dict[str, Any]] = []

    for edit in edits:
        operation = str(edit.get("operation", ""))
        source_ids = {str(item) for item in edit.get("source_sentence_ids", ())}
        if operation in {"replace", "delete"} and not restructuring:
            if not source_ids or not source_ids.issubset(diagnosed_ids):
                for sentence in edit.get("candidate_sentences", ()) or edit.get(
                    "source_sentences", ()
                ):
                    failures.append(
                        {
                            "change_id": edit.get("change_id"),
                            "reason": "edit outside a diagnosed problem span",
                            "sentence": sentence.get("text", ""),
                            "source_sentence_ids": list(
                                edit.get("source_sentence_ids", ())
                            ),
                            "source_gap": edit.get("source_gap"),
                        }
                    )
        if operation == "insert" and not restructuring:
            gap = edit.get("source_gap")
            boundary_is_diagnosed = isinstance(gap, int) and any(
                item["id"] in diagnosed_ids
                for index, item in enumerate(source)
                if gap in (index, index + 1)
            )
            gap_fill_authorized = gap == len(source) and bool(authorized_gap_keys)
            if not boundary_is_diagnosed and not gap_fill_authorized:
                for sentence in edit.get("candidate_sentences", ()):
                    failures.append(
                        {
                            "change_id": edit.get("change_id"),
                            "reason": "insertion outside a diagnosed boundary or authorized gap slot",
                            "sentence": sentence.get("text", ""),
                            "source_sentence_ids": [],
                            "source_gap": gap,
                        }
                    )

    for error in correspondence_errors:
        failures.append(
            {
                "change_id": None,
                "reason": error,
                "sentence": error.removeprefix(
                    "sentence correspondence could not be established for "
                ),
                "source_sentence_ids": [],
                "source_gap": None,
            }
        )

    return {
        "passed": not failures,
        "diagnosed_source_sentence_ids": sorted(diagnosed_ids),
        "confirmed_gap_keys": sorted(confirmed_gap_keys),
        "authorized_gap_keys": sorted(authorized_gap_keys),
        "restructuring_authorized": restructuring,
        "failures": failures,
    }


def _confinement_reasons(failures: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"fidelity confinement rejected {str(item.get('sentence') or 'the edit')!r}: "
        f"{item.get('reason')} (source mapping: "
        f"{', '.join(str(value) for value in item.get('source_sentence_ids', ())) or 'gap ' + str(item.get('source_gap'))})"
        for item in failures
    ]


def check_candidate_fidelity(
    gateway: Gateway,
    original_prompt: str,
    candidate_prompt: str,
    diagnosis: Mapping[str, Any],
    strategy: Any,
    *,
    run_id: str,
    judge_model: str,
    assumptions: Sequence[Any] = (),
    support_prompt: str | None = None,
    preservation_proof: Mapping[str, Any] | None = None,
) -> FidelityResult:
    """Check deterministic edit confinement, then semantic support and meaning.

    A confinement failure is returned without a Jev call. Otherwise one shared
    candidate state is sent with one Choice per inserted or changed candidate
    sentence and one whole-prompt meaning decision. Responses follow ADR-0001:
    the Gateway returns raw values and this caller parses them fail-closed.
    """
    script = sentence_edit_script(original_prompt, candidate_prompt)
    edits = script["edits"]
    strategy_value = _strategy_metadata(strategy)
    is_lossless = strategy_value.get("name") == "restructure_lossless"
    confinement: dict[str, Any]
    if is_lossless:
        proof_check = verify_lossless_proof(
            original_prompt, candidate_prompt, preservation_proof
        )
        proof_passed = proof_check.passed and strategy_value.get("restructures") is True
        confinement = {
            "passed": proof_passed,
            "restructuring_authorized": strategy_value.get("restructures") is True,
            "source_preservation": {
                "status": "passed" if proof_passed else "failed",
                "proof_kind": preservation_proof.get("kind")
                if isinstance(preservation_proof, Mapping)
                else None,
                "reasons": list(proof_check.reasons),
            },
            "failures": []
            if proof_passed
            else [
                {
                    "reason": reason,
                    "sentence": "the restructured candidate",
                    "source_sentence_ids": [],
                    "source_gap": None,
                }
                for reason in (
                    proof_check.reasons
                    if proof_check.reasons
                    else ("lossless strategy was not authorized",)
                )
            ],
        }
    else:
        confinement = _confinement(
            original_prompt,
            diagnosis,
            strategy_value,
            edits,
            script["correspondence_errors"],
        )
    confirmed_assumptions = _confirmed_answers(assumptions)
    evidence: dict[str, Any] = {
        "edit_script": edits,
        "unchanged_anchors": script["unchanged_anchors"],
        "confinement": confinement,
        "confirmed_assumptions": confirmed_assumptions,
        "sentence_support": [],
        "meaning_preservation": None,
        "reasons": [],
    }
    if is_lossless:
        evidence["source_preservation"] = confinement["source_preservation"]

    if not confinement["passed"]:
        evidence["reasons"] = _confinement_reasons(confinement["failures"])
        return FidelityResult(False, False, False, evidence)

    if not edits and not is_lossless:
        evidence["meaning_preservation"] = {"required": False, "accepted": True}
        return FidelityResult(True, True, True, evidence)

    changed_sentences = {str(edit["change_id"]): dict(edit) for edit in edits}
    state = {
        "original_prompt": support_prompt or original_prompt,
        "candidate_prompt": candidate_prompt,
        "confirmed_assumptions": confirmed_assumptions,
        "changed_sentences": {
            change_id: {
                key: value for key, value in edit.items() if key != "source_sentences"
            }
            for change_id, edit in changed_sentences.items()
        },
    }
    requests: list[dict[str, Any]] = []
    support_edit_by_key: dict[str, Mapping[str, Any]] = {}
    for edit in () if is_lossless else edits:
        for sentence in edit.get("candidate_sentences", ()):
            change_id = str(edit["change_id"])
            sentence_id = str(sentence["id"])
            key = f"fidelity:sentence:{change_id}:{sentence_id}"
            requests.append(
                {
                    "model": judge_model,
                    "key": key,
                    "type": "choice",
                    "query": jev_questions.fidelity_sentence_support_question(
                        change_id
                    ),
                    "criteria": jev_questions.FIDELITY_SUPPORT_OPTIONS,
                    "state": state,
                }
            )
            support_edit_by_key[key] = edit
    meaning_key = "fidelity:meaning"
    requests.append(
        {
            "model": judge_model,
            "key": meaning_key,
            "type": "noul",
            "query": jev_questions.FIDELITY_MEANING_QUESTION,
            "state": state,
        }
    )

    raw_answers: list[Any] = []
    try:
        for offset in range(0, len(requests), DECISION_BATCH_LIMIT):
            batch_answers = gateway.decide_batch(
                requests[offset : offset + DECISION_BATCH_LIMIT],
                role="judge",
                run_id=run_id,
            )
            if not isinstance(batch_answers, list):
                raise JevResponseError("malformed fidelity response batch")
            raw_answers.extend(batch_answers)
        if len(raw_answers) != len(requests):
            raise JevResponseError("incomplete fidelity response")
    except (ProviderError, JevResponseError) as exc:
        reason = f"fidelity checks unavailable or incomplete ({type(exc).__name__})"
        evidence["reasons"] = [reason]
        evidence["request_count"] = len(requests)
        evidence["response_count"] = len(raw_answers)
        return FidelityResult(False, False, True, evidence)

    by_key = dict(
        zip((str(request["key"]) for request in requests), raw_answers, strict=True)
    )
    no_invention = True
    for key, edit in support_edit_by_key.items():
        sentence = next(
            item
            for item in edit["candidate_sentences"]
            if key.endswith(str(item["id"]))
        )
        selected = None
        probability = 0.0
        probabilities: dict[str, float] = {}
        confidence = 0.0
        try:
            decision = parse_decision(by_key[key])
            if isinstance(decision, ChoiceDecision):
                selected = decision.selected
                confidence = decision.confidence
                probabilities = decision.probabilities
                probability = probabilities.get(selected, 0.0)
        except JevResponseError:
            decision = None
        accepted = bool(
            selected in {"supported_by_original", "supported_by_assumption"}
            and probability >= FIDELITY_THRESHOLD
            and not (
                selected == "supported_by_assumption" and not confirmed_assumptions
            )
        )
        support = {
            "change_id": edit["change_id"],
            "candidate_sentence_id": sentence["id"],
            "sentence": sentence["text"],
            "source_sentence_ids": list(edit["source_sentence_ids"]),
            "source_gap": edit["source_gap"],
            "selected": selected,
            "probability": probability,
            "confidence": confidence,
            "probabilities": probabilities,
            "accepted": accepted,
        }
        if selected == "supported_by_assumption" and not confirmed_assumptions:
            support["reason"] = "no confirmed user answer was available"
        elif selected == "new_requirement":
            support["reason"] = "new requirement"
        elif selected == "unknown":
            support["reason"] = "support unknown"
        elif selected not in {"supported_by_original", "supported_by_assumption"}:
            support["reason"] = "malformed or missing support answer"
        elif probability < FIDELITY_THRESHOLD:
            support["reason"] = "supported option probability below 0.80"
        evidence["sentence_support"].append(support)
        if not accepted:
            no_invention = False
            reason = str(support.get("reason", "sentence support failed"))
            mapping = (
                ", ".join(support["source_sentence_ids"])
                or f"gap {support['source_gap']}"
            )
            evidence["reasons"].append(
                f"fidelity rejected {sentence['text']!r}: {reason} "
                f"(source mapping: {mapping}; probability={probability:.2f})"
            )

    meaning_probability = 0.0
    meaning_error = None
    try:
        meaning_decision = parse_decision(by_key[meaning_key])
        if isinstance(meaning_decision, NoulDecision):
            meaning_probability = meaning_decision.probability
        else:
            meaning_error = "wrong Jev decision type"
    except JevResponseError as exc:
        meaning_error = type(exc).__name__
    meaning_preserved = meaning_probability >= FIDELITY_THRESHOLD
    evidence["meaning_preservation"] = {
        "probability": meaning_probability,
        "threshold": FIDELITY_THRESHOLD,
        "accepted": meaning_preserved,
        **({"reason": meaning_error} if meaning_error else {}),
    }
    if not meaning_preserved:
        evidence["reasons"].append(
            "whole-prompt meaning preservation did not reach 0.80 "
            f"(probability={meaning_probability:.2f}"
            f"{'; ' + meaning_error if meaning_error else ''})"
        )

    return FidelityResult(
        meaning_preserved,
        no_invention,
        True,
        evidence,
    )
