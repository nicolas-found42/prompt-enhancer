"""Candidate-level Evaluate and Accept judgments (#171).

The selector owns deterministic eligibility. Jev orders only survivors, then
evaluates each package independently and applies an evidence-backed accept gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from .gateway import Gateway
from .jev import ChoiceDecision, JevDecision, NoulDecision, parse_decision
from .selector import RankingCandidate
from .styles import validated_style_authorization

EVALUATION_ASPECTS = (
    "task_preserved",
    "no_invented_detail",
    "structure_added",
    "verbosity_direction",
)
ACCEPT_THRESHOLD = 0.8
CAPABILITIES = (
    "verify",
    "screen",
    "noul",
    "find",
    "rerank",
    "classify",
    "decide",
    "compare",
    "extract",
    "audit",
    "review",
    "gate",
)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evidence for each evaluated package and its answered judgments."""

    candidates: Mapping[str, Mapping[str, Any]]
    order: Mapping[str, float]
    provenance: tuple[Mapping[str, Any], ...]


def _token(candidate_id: str) -> str:
    return quote(candidate_id, safe="-_.")


def _request(
    *,
    key: str,
    model: str,
    query: str,
    state: Mapping[str, Any],
    kind: str = "noul",
    criteria: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request = {
        "model": model,
        "key": key,
        "type": kind,
        "query": query,
        "state": dict(state),
    }
    if criteria is not None:
        request["criteria"] = dict(criteria)
    return request


def _parse(raw: Any) -> JevDecision | None:
    try:
        decision = parse_decision(raw)
    except (TypeError, ValueError):
        return None
    return decision


def _matches_request_shape(
    request: Mapping[str, Any], decision: JevDecision | None
) -> bool:
    request_type = request.get("type")
    if request_type == "noul":
        return isinstance(decision, NoulDecision)
    if request_type == "choice":
        if not isinstance(decision, ChoiceDecision):
            return False
        criteria = request.get("criteria")
        return not isinstance(criteria, Mapping) or decision.selected in criteria
    return decision is not None


def _answer_evidence(request: Mapping[str, Any], raw: Any) -> dict[str, Any]:
    decision = _parse(raw)
    usable = _matches_request_shape(request, decision)
    answer: dict[str, Any] = {"raw_answer": raw, "usable": usable}
    if not usable:
        return answer
    if isinstance(decision, NoulDecision):
        answer["probability"] = decision.probability
    elif isinstance(decision, ChoiceDecision):
        answer["selected"] = decision.selected
        answer["probability"] = decision.probabilities.get(decision.selected)
    return answer


def _answered_record(
    request: Mapping[str, Any],
    raw: Any,
    entry: Mapping[str, Any],
    *,
    capability: str,
    stage: str,
    candidate_id: str,
    round_number: int,
) -> dict[str, Any]:
    decision = _parse(raw)
    usable = _matches_request_shape(request, decision)
    normalized: dict[str, Any] = {}
    if usable and isinstance(decision, NoulDecision):
        normalized["probability"] = decision.probability
    elif usable and isinstance(decision, ChoiceDecision):
        normalized["selected"] = decision.selected
        normalized["probability"] = decision.probabilities.get(decision.selected)
    return {
        "capability": capability,
        "stage": stage,
        "candidate_id": candidate_id,
        "round_number": round_number,
        "source_round": round_number,
        "question_key": str(request.get("key") or ""),
        "model": entry.get("answered_by"),
        "raw_answer": raw,
        "usable": usable,
        **normalized,
    }


def _log_entries(gateway: Gateway, before: int) -> dict[str, Mapping[str, Any]]:
    return {
        str(question.get("key") or ""): entry
        for entry in gateway.decision_log[before:]
        if isinstance(entry, Mapping)
        and isinstance((question := entry.get("question")), Mapping)
    }


def _capability_for(question: Mapping[str, Any]) -> str | None:
    key = str(question.get("key") or "")
    if key == "task_type" or key.startswith("task_type:"):
        return "classify"
    if key == "strategy_choice":
        return "decide"
    if key.startswith("strategy_recheck:"):
        return "noul"
    if key.startswith("pointer:"):
        return "find"
    if key.startswith(("existence:", "gap:", "problem:")):
        return "noul"
    if key.startswith("infer:"):
        return "extract"
    if key.startswith("faithful:"):
        return "verify"
    if key == "assumption_meaning":
        return "verify"
    if key.startswith("restructure_lossless:role:"):
        return "classify"
    if key.startswith("understand:screen"):
        return "screen"
    if key.startswith("understand:probe:"):
        return "noul"
    if key.startswith("understand:classify"):
        return "classify"
    if key.startswith("understand:extract"):
        return "extract"
    if key.startswith("understand:audit"):
        return "audit"
    if key.startswith(("success-test-screen:", "output-screen:")):
        return "screen"
    if key.startswith("route:find"):
        return "find"
    if key.startswith("route:decide"):
        return "decide"
    if key.startswith("evaluate:compare"):
        return "compare"
    if key.startswith("evaluate:verify"):
        return "verify"
    if key.startswith("evaluate:audit"):
        return "audit"
    if key.startswith("evaluate:rerank"):
        return "rerank"
    if key.startswith("evaluate:review"):
        return "review"
    if key.startswith("evaluate:accept"):
        return "gate"
    if key.startswith(("fidelity:", "grade-verify:", "grade-confirm:", "faithful:")):
        return "verify"
    if key.startswith("failure-attribution:pointer"):
        return "find"
    if key.startswith("failure-attribution:kind"):
        return "classify"
    if key.startswith("failure-attribution:attributable"):
        return "noul"
    if key.startswith("score:") and question.get("type") == "noul":
        return "noul"
    # Primitive response shape does not identify the semantic capability.
    # Unknown callers remain visible in provenance without being miscounted.
    if key.startswith("grade_"):
        if question.get("type") == "noul":
            return "noul"
        if question.get("type") == "choice":
            return "classify"
    return None


def stage_for(question: Mapping[str, Any]) -> str:
    """Identify the pipeline stage from the request's stable semantic key."""
    key = str(question.get("key") or "")
    for prefix, stage in (
        ("task_type", "diagnosis"),
        ("gap:", "diagnosis"),
        ("pointer:", "diagnosis"),
        ("existence:", "diagnosis"),
        ("problem:", "diagnosis"),
        ("rubric:", "diagnosis"),
        ("infer:", "clarification_inference"),
        ("faithful:", "success_test_validation"),
        ("assumption_meaning", "clarification_assumption"),
        ("strategy_choice", "strategy_selection"),
        ("strategy_recheck:", "strategy_selection"),
        ("restructure_lossless:", "lossless_restructuring"),
        ("understand:", "understand"),
        ("route:", "route"),
        ("evaluate:accept:", "accept"),
        ("evaluate:", "evaluate"),
        ("grade_", "grading"),
        ("grade-verify:", "grading_verification"),
        ("grade-confirm:", "grading_confirmation"),
        ("fidelity:", "fidelity"),
        ("score:", "score_vector"),
        ("strong:", "strong_check"),
        ("output-screen:", "output_screen"),
        ("success-test-screen:", "success_test_screen"),
        ("failure-attribution:", "failure_attribution"),
    ):
        if key.startswith(prefix):
            return stage
    return "unknown"


def round_judgment_provenance(
    gateway: Gateway,
    start_index: int,
    candidate_prompts: Mapping[str, str],
    round_number: int | None,
) -> tuple[Mapping[str, Any], ...]:
    """Capture this round's raw answered decisions with explicit local context."""
    prompts_to_ids: dict[str, list[str]] = {}
    for candidate_id, prompt in candidate_prompts.items():
        prompts_to_ids.setdefault(prompt, []).append(candidate_id)
    records: list[Mapping[str, Any]] = []
    for entry in gateway.decision_log[start_index:]:
        if not isinstance(entry, Mapping):
            continue
        question = entry.get("question")
        if not isinstance(question, Mapping):
            continue
        raw = entry.get("answer")
        parsed = _parse(raw)
        state = question.get("state")
        state = state if isinstance(state, Mapping) else {}
        key = str(question.get("key") or "")
        candidate_id = state.get("candidate_id")
        if not isinstance(candidate_id, str):
            candidate_keys = (
                "evaluate:",
                "grade_",
                "score:",
                "fidelity:",
                "strong:",
                "output-screen:",
            )
            if key.startswith(candidate_keys):
                candidate_text = state.get("candidate_prompt", state.get("prompt"))
                matching_ids = prompts_to_ids.get(candidate_text, ())
                candidate_id = matching_ids[0] if len(matching_ids) == 1 else None
        request_round = state.get("round_number")
        record_round = (
            int(request_round)
            if isinstance(request_round, int) and not isinstance(request_round, bool)
            else round_number
        )
        capability = _capability_for(question)
        stage = stage_for(question)
        usable = _matches_request_shape(question, parsed)
        normalized: dict[str, Any] = {}
        if usable and isinstance(parsed, NoulDecision):
            normalized["probability"] = parsed.probability
        elif usable and isinstance(parsed, ChoiceDecision):
            normalized["selected"] = parsed.selected
            normalized["probability"] = parsed.probabilities.get(parsed.selected)
        records.append(
            {
                "capability": capability,
                "stage": stage,
                "candidate_id": candidate_id,
                "round_number": record_round,
                "source_round": record_round if candidate_id else None,
                "question_key": key,
                "model": entry.get("answered_by"),
                "raw_answer": raw,
                "usable": usable,
                **normalized,
            }
        )
    return tuple(records)


def summarize_capabilities(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Count every Jev capability, retaining explicit zeroes and stage counts."""
    summary: dict[str, Any] = {
        capability: {"count": 0, "ran": False, "stages": {}}
        for capability in CAPABILITIES
    }
    for record in records:
        capability = record.get("capability")
        if capability not in summary:
            continue
        entry = summary[str(capability)]
        entry["count"] += 1
        entry["ran"] = True
        stage = str(record.get("stage") or "unknown")
        stages = entry["stages"]
        stages[stage] = stages.get(stage, 0) + 1
    return summary


def evaluate_candidate_packages(
    gateway: Gateway,
    original_prompt: str,
    candidates: Sequence[RankingCandidate],
    *,
    constraints: Sequence[str],
    improvement_style: str,
    style_bundle: Sequence[str],
    success_tests: Sequence[Mapping[str, Any]],
    candidate_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    strong_evidence: Mapping[str, Mapping[str, Any]],
    judge_model: str,
    run_id: str,
    round_number: int,
    on_stage: Callable[[str], None] | None = None,
    style_authorization: Mapping[str, Any] | None = None,
    legacy_evidence: bool = False,
) -> CandidateEvaluation:
    """Rerank and fully evaluate already-eligible candidate packages.

    The first batch independently compares, verifies, audits, reranks, and
    reviews each candidate. Accept is a second batch because its state depends
    on the complete first-batch evidence. Invalid judgments fail closed for
    acceptance; incomplete reranking falls back to deterministic selector order.
    """
    bundle = ", ".join(style_bundle) if style_bundle else "no named strategies"
    authorization = validated_style_authorization(
        improvement_style, style_authorization
    )
    requests: list[dict[str, Any]] = []
    request_info: list[tuple[str, str, str]] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        token = _token(candidate_id)
        common_state = {
            "candidate_id": candidate_id,
            "round_number": round_number,
            "original_prompt": original_prompt,
            "candidate_prompt": candidate.text,
            **(
                {
                    "applied_style": improvement_style,
                    "style_authorization": authorization,
                }
                if authorization
                else {}
            ),
        }
        for aspect in EVALUATION_ASPECTS:
            if aspect == "task_preserved":
                query = (
                    "Does the candidate prompt preserve the task that the user "
                    "requested in the original prompt? Answer yes only when the "
                    "task and intended outcome remain intact."
                )
            elif aspect == "no_invented_detail":
                query = (
                    "Does the candidate avoid adding facts or requirements not "
                    "supported by the original prompt? A nonempty canonical "
                    "state.style_authorization permits only its expressly bounded "
                    "presentation changes. It never supports new task facts, scope, "
                    "deliverables, or success criteria."
                )
            elif aspect == "structure_added":
                query = (
                    "Does the candidate add headings, grouping, or other "
                    "organization that was absent from the original prompt? "
                    "This is descriptive evidence; either answer may be valid."
                )
            else:
                query = (
                    "Compared with the original prompt, is this candidate "
                    "shorter, about the same length, or longer?"
                )
            requests.append(
                _request(
                    key=f"evaluate:compare:{round_number}:{token}:{aspect}",
                    model=judge_model,
                    query=query,
                    state={**common_state, "aspect": aspect},
                    kind="choice" if aspect == "verbosity_direction" else "noul",
                    criteria={
                        "shorter": "The candidate is materially shorter than the original.",
                        "same": "The candidate has about the same level of detail and length.",
                        "longer": "The candidate is materially longer than the original.",
                    }
                    if aspect == "verbosity_direction"
                    else None,
                )
            )
            request_info.append((candidate_id, "compare", aspect))
        for index, requirement in enumerate(constraints):
            requests.append(
                _request(
                    key=f"evaluate:verify:{round_number}:{token}:{index}",
                    model=judge_model,
                    query=(
                        "Does the candidate preserve this extracted requirement "
                        "from the original prompt? Judge the candidate against "
                        "the exact requirement and source prompt."
                    ),
                    state={**common_state, "requirement": requirement},
                )
            )
            request_info.append((candidate_id, "verify", str(index)))
            requests.append(
                _request(
                    key=f"evaluate:audit:{round_number}:{token}:{index}",
                    model=judge_model,
                    query=(
                        "Audit this extracted requirement against the original "
                        "prompt and candidate. Answer yes only if the original "
                        "supports the requirement and the candidate preserves "
                        "its intended meaning. Exact-output literals are checked "
                        "separately and must remain verbatim."
                    ),
                    state={
                        **common_state,
                        "source": original_prompt,
                        "value": requirement,
                    },
                )
            )
            request_info.append((candidate_id, "audit", str(index)))
        requests.append(
            _request(
                key=f"evaluate:rerank:{round_number}:{token}",
                model=judge_model,
                query=(
                    "How relevant is this candidate prompt to the user's original "
                    "task, considering the requested style and all stated "
                    "requirements? Answer yes when it is a strong fit."
                ),
                state={
                    **common_state,
                    "improvement_style": improvement_style,
                    "style_bundle": list(style_bundle),
                },
            )
        )
        request_info.append((candidate_id, "rerank", "relevance"))
        requests.append(
            _request(
                key=f"evaluate:review:{round_number}:{token}",
                model=judge_model,
                query=(
                    "Review this candidate package against the requested style. "
                    "Does its wording and organization fit the style while "
                    "preserving the user's original task?"
                ),
                state={
                    **common_state,
                    "improvement_style": improvement_style,
                    "style_bundle": bundle,
                    "score_vector": candidate.metadata.get("score_vector"),
                    "success_test_grade": (
                        candidate.grade.to_dict()
                        if (success_tests or legacy_evidence)
                        and candidate.grade is not None
                        else None
                    ),
                },
            )
        )
        request_info.append((candidate_id, "review", "style_package"))

    pre_answers: dict[tuple[str, str, str], dict[str, Any]] = {}
    pre_records: list[dict[str, Any]] = []
    if requests:
        if on_stage is not None:
            on_stage("evaluating_candidates")
        before = len(gateway.decision_log)
        raw_answers = gateway.decide_batch(requests, role="judge", run_id=run_id)
        entries = _log_entries(gateway, before)
        for request, info, raw in zip(requests, request_info, raw_answers, strict=True):
            candidate_id, capability, item = info
            entry = entries.get(str(request["key"]), {})
            answer = _answer_evidence(request, raw)
            pre_answers[info] = answer
            pre_records.append(
                _answered_record(
                    request,
                    raw,
                    entry,
                    capability=capability,
                    stage="evaluate",
                    candidate_id=candidate_id,
                    round_number=round_number,
                )
            )

    packages: dict[str, dict[str, Any]] = {}
    order: dict[str, float] = {}
    gate_requests: list[dict[str, Any]] = []
    gate_candidates: list[RankingCandidate] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        evidence: dict[str, Any] = {
            "candidate_id": candidate_id,
            "round_number": round_number,
            "comparison": {
                aspect: pre_answers.get((candidate_id, "compare", aspect), {})
                for aspect in EVALUATION_ASPECTS
            },
            "verification": {
                str(index): pre_answers.get((candidate_id, "verify", str(index)), {})
                for index, _ in enumerate(constraints)
            },
            "audit": {
                str(index): pre_answers.get((candidate_id, "audit", str(index)), {})
                for index, _ in enumerate(constraints)
            },
            "rerank": pre_answers.get((candidate_id, "rerank", "relevance"), {}),
            "review": pre_answers.get((candidate_id, "review", "style_package"), {}),
            "score_vector": candidate.metadata.get("score_vector"),
            "fidelity": candidate.metadata.get("fidelity"),
            "strong_check": strong_evidence.get(candidate_id),
            "downstream_verification": ("verified" if success_tests else "unverified"),
            "success_tests": [dict(test) for test in success_tests],
            "success_test_outputs": [
                dict(item) for item in candidate_outputs.get(candidate_id, ())
            ],
            "success_test_grade": (
                candidate.grade.to_dict()
                if (success_tests or legacy_evidence) and candidate.grade is not None
                else None
            ),
        }
        rerank = evidence["rerank"]
        if rerank.get("usable"):
            order[candidate_id] = float(rerank["probability"])
        accept_evidence = deepcopy(evidence)
        if not success_tests and not legacy_evidence:
            # Keep raw answers in diagnostic evidence, but do not introduce
            # an implicit downstream grader without accepted success tests.
            accept_evidence["success_test_outputs"] = []
        gate_requests.append(
            _request(
                key=f"evaluate:accept:{round_number}:{_token(candidate_id)}",
                model=judge_model,
                query=(
                    "Accept this candidate only when the complete evidence bundle "
                    "supports the original task, all stated constraints, the "
                    "requested style, its fidelity and score-vector gates, and "
                    "the independent comparison, verification, audit, and review "
                    "judgments. Use the candidate's weak-panel outputs and "
                    "success-test grade when tests exist. If no success tests "
                    "exist, preserve downstream_verification as unverified; do "
                    "not invent a pass or reject solely for missing tests. "
                    "Otherwise reject it."
                    if legacy_evidence
                    else (
                        "Should this prompt be accepted for the original task? Check "
                        "all stated constraints, requested presentation style, fidelity "
                        "and score-vector floors. Use the supplied success tests, grades, "
                        "and weak outputs when tests exist. Use comparison and review "
                        "judgments, and verification/audit judgments only when those "
                        "checks are applicable. Empty verification/audit mappings mean "
                        "no applicable checks, not failures. An unchanged original may "
                        "be accepted without a rewrite, strategy application, added "
                        "structure, or measured improvement. Strategy names are "
                        "suggestions, not acceptance criteria. No-tests downstream "
                        "verification must stay unverified and is neutral. Reject "
                        "substantive failures or insufficient applicable evidence."
                    )
                ),
                state={
                    "candidate_id": candidate_id,
                    "round_number": round_number,
                    "original_prompt": original_prompt,
                    "candidate_prompt": candidate.text,
                    "improvement_style": improvement_style,
                    **(
                        {
                            "applied_style": improvement_style,
                            "style_authorization": authorization,
                        }
                        if authorization
                        else {}
                    ),
                    "style_bundle": list(style_bundle),
                    "complete_candidate_evidence": accept_evidence,
                },
            )
        )
        gate_candidates.append(candidate)
        packages[candidate_id] = evidence

    gate_records: list[dict[str, Any]] = []
    if gate_requests:
        if on_stage is not None:
            on_stage("accepting_candidates")
        before = len(gateway.decision_log)
        raw_gates = gateway.decide_batch(gate_requests, role="judge", run_id=run_id)
        entries = _log_entries(gateway, before)
        for request, candidate, raw in zip(
            gate_requests, gate_candidates, raw_gates, strict=True
        ):
            parsed = _parse(raw)
            accepted = (
                isinstance(parsed, NoulDecision)
                and parsed.probability >= ACCEPT_THRESHOLD
            )
            candidate_evidence = packages[candidate.candidate_id]
            candidate_evidence["accept"] = {
                "raw_answer": raw,
                "usable": isinstance(parsed, NoulDecision),
                "probability": (
                    parsed.probability if isinstance(parsed, NoulDecision) else None
                ),
                "threshold": ACCEPT_THRESHOLD,
                "accepted": accepted,
            }
            failures: list[str] = []
            for aspect in ("task_preserved", "no_invented_detail"):
                answer = candidate_evidence["comparison"][aspect]
                if (
                    not answer.get("usable")
                    or answer.get("probability", 0.0) < ACCEPT_THRESHOLD
                ):
                    failures.append(f"comparison failed or unusable: {aspect}")
            for aspect in ("structure_added", "verbosity_direction"):
                if not candidate_evidence["comparison"][aspect].get("usable"):
                    failures.append(f"comparison incomplete or unusable: {aspect}")
            for index, answer in candidate_evidence["verification"].items():
                if (
                    not answer.get("usable")
                    or answer.get("probability", 0.0) < ACCEPT_THRESHOLD
                ):
                    failures.append(
                        f"requirement verification failed or unusable: {index}"
                    )
            for index, answer in candidate_evidence["audit"].items():
                if (
                    not answer.get("usable")
                    or answer.get("probability", 0.0) < ACCEPT_THRESHOLD
                ):
                    failures.append(
                        f"candidate-time requirement audit failed or unusable: {index}"
                    )
            review = candidate_evidence["review"]
            if (
                not review.get("usable")
                or review.get("probability", 0.0) < ACCEPT_THRESHOLD
            ):
                failures.append("style package review failed or unusable")
            if not accepted:
                failures.append("final acceptance gate rejected or unusable")
            candidate_evidence["eligible"] = not failures
            candidate_evidence["rejection_reasons"] = failures
            gate_records.append(
                _answered_record(
                    request,
                    raw,
                    entries.get(str(request["key"]), {}),
                    capability="gate",
                    stage="accept",
                    candidate_id=candidate.candidate_id,
                    round_number=round_number,
                )
            )

    return CandidateEvaluation(
        candidates=packages,
        order=order,
        provenance=tuple(pre_records + gate_records),
    )
