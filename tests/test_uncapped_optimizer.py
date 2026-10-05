"""Public-API regression for score-vector retries beyond the former round cap."""

from __future__ import annotations

import json

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.gateway import ScriptedGateway


def test_run_retries_until_a_changed_candidate_passes_then_plateaus() -> None:
    prompt = "Summarize the report."
    early_candidates = (
        "Provide a summary of the report.",
        "Give an overview of the report's contents.",
        "Write a summary of the report's key points.",
        "Summarize the report's main points.",
    )
    winner = "Summarize the report clearly."
    writer_rounds = 0
    candidate_score_vectors: list[tuple[str, dict[str, float]]] = []
    baseline_score_vector: dict[str, float] = {}
    verbosity_requests: list[str] = []

    def chat(_model, messages, *, role, **_kwargs):
        nonlocal writer_rounds
        if role == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            writer_rounds += 1
            candidate = (
                early_candidates[writer_rounds - 1]
                if writer_rounds <= len(early_candidates)
                else winner
            )
            return json.dumps(
                {strategy["name"]: candidate for strategy in state["strategies"]}
            )
        if role == "writer":
            return '{"tests": []}'
        return "pass"

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            criteria = request.get("criteria", {})
            if key == "task_type":
                selected = "general"
            elif key == "strategy_choice":
                selected = next(option for option in criteria if option != "none")
            elif key.endswith(":verbosity_direction"):
                selected = "same"
                verbosity_requests.append(key)
            elif key.startswith("fidelity:sentence:"):
                selected = "supported_by_original"
            elif key == "route:find" and "none" in criteria:
                selected = "none"
            else:
                selected = "general" if "general" in criteria else "none"
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": {selected: 1.0},
                "confidence": 1.0,
            }

        if key == "fidelity:meaning" or key.startswith("faithful:"):
            probability = 1.0
        elif key.startswith("strategy_recheck:"):
            probability = 1.0
        elif key.startswith("score:"):
            candidate = str(request["state"]["candidate_prompt"])
            if candidate == prompt:
                probability = 0.1
            elif candidate in early_candidates and key == "score:clarity":
                probability = 0.1
            else:
                probability = 0.99
            if candidate == prompt:
                baseline_score_vector[key.removeprefix("score:")] = probability
            else:
                vector = next(
                    (
                        scores
                        for text, scores in candidate_score_vectors
                        if text == candidate
                    ),
                    None,
                )
                if vector is None:
                    vector = {}
                    candidate_score_vectors.append((candidate, vector))
                vector[key.removeprefix("score:")] = probability
        else:
            probability = 1.0
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=ScriptedGateway(chat=chat, decision=decide),
        writer_instruction_version=4,
    ).optimize(
        prompt,
        {"clarification_allowed": False},
    )

    report = result["report"]
    history = report["history"]
    assert len(history) == 6
    assert writer_rounds == 6
    assert len(history) > 3
    assert report["convergence"]["status"] == "converged"
    assert result["final_prompt"] == winner
    assert result["final_prompt"] != prompt
    assert report["convergence"]["status"] == "converged"
    assert report["convergence"]["gain"] <= report["convergence"]["epsilon"]
    assert report["convergence"]["selected_candidate_id"]
    assert verbosity_requests
    assert all(key.startswith("evaluate:compare:") for key in verbosity_requests)

    for round_result, candidate_text in zip(history[:4], early_candidates, strict=True):
        selection = round_result["evidence"]["selection_evidence"]
        rejected = next(
            item for item in selection["ranking"] if item["text"] == candidate_text
        )
        assert rejected["selected"] is False
        assert rejected["metadata"]["score_vector"]["passed"] is False
        assert any(
            reason.startswith("score floor breached: clarity")
            for reason in rejected["rejection_reasons"]
        )

    winner_evidence = next(
        item["evidence"]
        for item in history
        if item["evidence"]["selection_evidence"]["selected_candidate"]
        and item["evidence"]["selection_evidence"]["selected_candidate"]["text"]
        == winner
    )
    selected = winner_evidence["selection_evidence"]["selected_candidate"]
    vector = selected["metadata"]["score_vector"]
    assert selected["candidate_id"] == report["convergence"]["selected_candidate_id"]
    assert vector["passed"] is True
    assert all(
        score >= vector["floors"][dimension]
        for dimension, score in vector["scores"].items()
    )
    assert baseline_score_vector["clarity"] < vector["floors"]["clarity"]
    early_vectors = {
        text: scores
        for text, scores in candidate_score_vectors
        if text in early_candidates
    }
    assert set(early_vectors) == set(early_candidates)
    assert all(
        scores["clarity"] < vector["floors"]["clarity"]
        for scores in early_vectors.values()
    )
