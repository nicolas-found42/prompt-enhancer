"""Understand + Route stages: style bundles, Auto inference, hard-gate requirements.

Covers ticket #166: Auto infers exactly one best-fit style (conservative
Clearer fallback when uncertain); each named style selects its deterministic
strategy bundle (two styles give visibly different candidate sets); extracted
exact-output constraints are hard gates (violating candidates rejected); a
genuinely impossible style/constraint pairing ends impossible with an
explanation; audit gates extraction values before use as hard-gate evidence.
"""

import json
from functools import partial

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.styles import IMPROVEMENT_STYLES

PromptOptimizer = partial(PromptOptimizer, writer_instruction_version=4)


def _gateway(
    *,
    classify_choice: str = "shorter",
    classify_confidence: float = 0.9,
    audit_probability: float = 0.99,
    screen_probability: float = 0.01,
    probe_probability: float = 0.01,
    find_choice: str = "none",
    decide_probability: float = 0.99,
    writer_texts: dict[str, str] | None = None,
    default_candidate: str | None = None,
    recheck_probability: float = 0.99,
    meaning_probability: float = 0.99,
    support: str = "supported_by_original",
):
    """Scripted gateway covering the v4 keys plus the Understand/Route keys."""

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            prompt = state["prompt"]
            payload = {}
            for item in state["strategies"]:
                name = item["name"]
                if writer_texts is not None and name in writer_texts:
                    payload[name] = writer_texts[name]
                elif default_candidate is not None:
                    payload[name] = default_candidate
                else:
                    payload[name] = f"{prompt} [{name}]"
            return json.dumps(payload)
        if role == "writer":
            return '{"tests":[]}'
        return "4"

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key == "task_type":
                selected, probability = "general", 1.0
            elif key == "understand:classify:style":
                selected, probability = classify_choice, classify_confidence
                options = {selected: probability}
                if selected != "none":
                    options = {
                        **{style: 0.0 for style in (selected,)},
                        selected: probability,
                    }
                return {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": options,
                    "confidence": classify_confidence,
                }
            elif key == "route:find":
                selected = find_choice
                return {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": {selected: 1.0},
                    "confidence": 1.0,
                }
            elif key.startswith("fidelity:sentence:"):
                return {
                    "type": "choice",
                    "choice": support,
                    "probabilities": {support: 1.0},
                    "confidence": 0.99,
                }
            else:
                selected, probability = "none", 1.0
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": {selected: probability},
                "confidence": 1.0,
            }
        if key == "fidelity:meaning":
            probability = meaning_probability
        elif key.startswith("strategy_recheck:"):
            probability = recheck_probability
        elif key.startswith("understand:audit"):
            probability = audit_probability
        elif key == "understand:screen":
            probability = screen_probability
        elif key.startswith("understand:probe:"):
            probability = probe_probability
        elif key == "route:decide":
            probability = decide_probability
        elif key.startswith("faithful:"):
            probability = 0.01
        elif key.startswith("grade_"):
            probability = float(request["state"]["output"] == "pass")
        else:
            probability = 0.01
        return {
            "type": "noul",
            "probability_true": probability,
            "confidence": 1.0,
        }

    return ScriptedGateway(chat=chat, decision=decide)


def test_auto_infers_confident_style_choice() -> None:
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=_gateway(classify_choice="shorter", classify_confidence=0.9),
    )

    result = optimizer.optimize("whats 2 plus 2", {"improvement_style": "auto"})

    report = result["report"]
    assert report["improvement_style"] == "auto"
    assert report["applied_style"] == "shorter"
    assert report["understand"]["inferred_style"] == "shorter"
    assert report["understand"]["provenance"]["classify"]["fired"] is True


def test_auto_falls_back_to_clearer_when_uncertain() -> None:
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=_gateway(classify_choice="surprise_me", classify_confidence=0.2),
    )

    result = optimizer.optimize("whats 2 plus 2", {"improvement_style": "auto"})

    report = result["report"]
    assert report["improvement_style"] == "auto"
    assert report["applied_style"] == "clearer"
    assert report["understand"]["inferred_style"] == "clearer"
    assert report["understand"]["provenance"]["classify"]["fallback"] == "clearer"


def test_named_styles_produce_different_candidate_sets() -> None:
    prompt = "Explain photosynthesis for a middle-school student."
    clearer = PromptOptimizer(store=RunStore(":memory:"), gateway=_gateway()).optimize(
        prompt, {"improvement_style": "clearer"}
    )
    shorter = PromptOptimizer(store=RunStore(":memory:"), gateway=_gateway()).optimize(
        prompt, {"improvement_style": "shorter"}
    )

    assert clearer["report"]["applied_style"] == "clearer"
    assert shorter["report"]["applied_style"] == "shorter"

    def strategies(result) -> set[str]:
        return {
            item["strategy"]
            for item in result["report"]["selection_evidence"]["ranking"]
        }

    assert clearer["report"]["route"]["bundle"] != shorter["report"]["route"]["bundle"]
    assert strategies(clearer) != strategies(shorter)


def test_hard_gate_rejects_candidate_dropping_exact_literal() -> None:
    prompt = 'Reply with exactly: "OK"'
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=_gateway(
            writer_texts={
                "specify_output_format": prompt + " [specify_output_format] OK",
                "remove_contradictions": "Reply with something else entirely",
            }
        ),
    )

    result = optimizer.optimize(prompt, {"improvement_style": "exact_format"})

    report = result["report"]
    assert report["applied_style"] == "exact_format"
    assert "OK" in report["understand"]["hard_constraints"]
    assert result["original_kept"] is False
    selected = report["selection_evidence"]["selected_candidate"]
    assert selected is not None and "OK" in selected["text"]
    rejected = {
        item["strategy"]: item
        for item in report["selection_evidence"]["rejected_candidates"]
    }
    assert "remove_contradictions" in rejected
    assert any(
        "hard requirement" in reason
        for reason in rejected["remove_contradictions"]["rejection_reasons"]
    )


def test_impossible_style_constraint_pair_ends_impossible() -> None:
    prompt = 'Reply with exactly: "OK"'
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_gateway())

    result = optimizer.optimize(prompt, {"improvement_style": "creative"})

    assert result["status"] == "failed"
    assert result["final_prompt"] == prompt
    assert result["original_kept"] is True
    report = result["report"]
    assert report["status"] == "impossible"
    assert report["applied_style"] == "creative"
    assert "creative" in report["summary"].lower()
    assert "OK" in report["summary"]
    assert report["selection_evidence"]["selected_candidate_id"] is None


def test_audit_gates_extraction_before_hard_gate_use() -> None:
    prompt = 'Say "hello" to greet.'
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=_gateway(
            audit_probability=0.01,
            writer_texts={
                "specify_output_format": 'Say "hi" to greet, please.',
                "remove_contradictions": 'Say "hi" to greet kindly.',
            },
        ),
    )

    result = optimizer.optimize(prompt, {"improvement_style": "shorter"})

    report = result["report"]
    assert report["understand"]["provenance"]["audit"]["fired"] is True
    assert report["understand"]["hard_constraints"] == []
    assert result["original_kept"] is False
    assert "hello" not in result["final_prompt"]


def test_every_named_style_maps_to_a_bundle() -> None:
    from prompt_enhancer.strategies import STYLE_STRATEGY_BUNDLES

    named = [style for style in IMPROVEMENT_STYLES if style != "auto"]
    assert set(STYLE_STRATEGY_BUNDLES) == set(named)
    for style, bundle in STYLE_STRATEGY_BUNDLES.items():
        assert len(bundle) >= 1, style
