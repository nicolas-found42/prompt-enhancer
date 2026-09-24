"""Jev checks that a rewrite remains faithful to the user's request."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .gateway import ProviderError
from .jev import JevResponseError, NoulDecision, parse_decision
from .rewrite import FidelityResult

_CHECKS = {
    "meaning_preserved": "Does the candidate preserve the original request and all stated constraints?",
    "no_invention": "Does the candidate avoid facts or requirements not given by the user?",
    "edits_confined": "Are edits limited to diagnosed problems or changes required by the named rewrite strategy?",
}


def check_candidate_fidelity(
    gateway: Any,
    original_prompt: str,
    candidate_prompt: str,
    diagnosis: Mapping[str, Any],
    strategy: str,
    *,
    run_id: str,
    judge_model: str,
) -> FidelityResult:
    """Fail closed when any required Jev answer is absent or uncertain."""
    state = {
        "original_prompt": original_prompt,
        "candidate_prompt": candidate_prompt,
        "diagnosis": dict(diagnosis),
        "strategy": strategy,
    }
    requests = [
        {
            "model": judge_model,
            "key": name,
            "type": "noul",
            "query": question,
            "state": state,
        }
        for name, question in _CHECKS.items()
    ]
    try:
        answers = gateway.jev_batch(requests, role="judge", run_id=run_id)
        decisions = [parse_decision(answer) for answer in answers]
        if len(decisions) != len(_CHECKS) or any(not isinstance(answer, NoulDecision) for answer in decisions):
            raise ValueError("incomplete fidelity response")
    except (ProviderError, JevResponseError, ValueError, TypeError, KeyError) as exc:
        return FidelityResult(False, False, False, {"error": type(exc).__name__})
    probabilities = {
        name: decision.probability
        for name, decision in zip(_CHECKS, cast(list[NoulDecision], decisions), strict=True)
    }
    return FidelityResult(
        meaning_preserved=probabilities["meaning_preserved"] >= 0.8,
        no_invention=probabilities["no_invention"] >= 0.8,
        edits_confined=probabilities["edits_confined"] >= 0.8,
        evidence=probabilities,
    )
