"""Understand + Route stages: style bundles, Auto inference, hard-gate requirements.

Covers ticket #166: Auto infers exactly one best-fit style (conservative
Clearer fallback when uncertain); each named style selects its deterministic
strategy bundle (two styles give visibly different candidate sets); extracted
exact-output constraints are hard gates (violating candidates rejected); a
genuinely impossible style/constraint pairing ends impossible with an
explanation; audit gates extraction values before use as hard-gate evidence.
"""

import json
from collections import Counter
from functools import partial

import pytest

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.candidate_evaluation import (
    round_judgment_provenance,
    summarize_capabilities,
)
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.grading import grade_panel_with_jev
from prompt_enhancer.run_control import RESUME_CONTEXT_KEY
from prompt_enhancer.runner import PanelResult
from prompt_enhancer.styles import IMPROVEMENT_STYLES
from prompt_enhancer.understand import run_understand

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
    score_probability: float = 0.99,
    support: str = "supported_by_original",
    extract_choice: str = "keep",
    tests_text: str = '{"tests":[]}',
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
            return tests_text
        return "4"

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if request.get("type") == "choice":
            if key.startswith("understand:extract:"):
                selected, probability = extract_choice, 1.0
                return {
                    "type": "choice",
                    "choice": selected,
                    "probabilities": {selected: probability},
                    "confidence": 1.0,
                }
            elif key == "task_type":
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
            elif key.startswith("evaluate:compare:") and key.endswith(
                ":verbosity_direction"
            ):
                selected = "same"
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
        elif key.startswith("evaluate:"):
            probability = 0.99
        elif key == "route:decide":
            probability = decide_probability
        elif key.startswith("faithful:"):
            probability = 0.01
        elif key.startswith("grade_"):
            probability = float(request["state"]["output"] == "pass")
        elif key.startswith("score:"):
            probability = score_probability
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


def test_semantic_extractor_ignores_a_quoted_example_before_audit() -> None:
    prompt = 'Mention the example phrase "ignore all rules" as data only.'
    understand = run_understand(
        _gateway(extract_choice="ignore"),
        prompt,
        requested_style="shorter",
        judge_model="test-judge",
        run_id="extract-example",
    ).to_dict()
    assert understand["hard_constraints"] == []
    assert understand["provenance"]["extract"]["selected"] == []
    assert understand["provenance"]["audit"]["keys"] == []


def test_named_style_keeps_taxonomy_classification_provenance_distinct() -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=_gateway())

    result = optimizer.optimize(
        "Summarize this report", {"improvement_style": "shorter"}
    )

    report = result["report"]
    assert "understand:classify:style" not in {
        item["question_key"] for item in report["judgment_provenance"]
    }
    assert report["capabilities_fired"]["classify"]["count"] > 0
    assert set(report["capabilities_fired"]) == {
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
    }
    assert set(report["capabilities_fired"]["classify"]["stages"]) == {"diagnosis"}


def test_partial_failure_retains_answered_provenance_without_guessing() -> None:
    gateway = _gateway()
    respond = gateway.decision_handler
    assert respond is not None

    def fail_during_candidate_evaluation(request, **kwargs):
        if str(request.get("key", "")).startswith("evaluate:rerank:"):
            raise RuntimeError("injected evaluation interruption")
        return respond(request, **kwargs)

    gateway.decision_handler = fail_during_candidate_evaluation
    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway)

    result = optimizer.optimize(
        "Summarize this report", {"improvement_style": "shorter"}
    )

    records = result["report"]["judgment_provenance"]
    assert result["status"] == "failed"
    assert any(item["question_key"] == "task_type" for item in records)
    assert any(item["question_key"].startswith("score:") for item in records)
    assert any(item["stage"] == "evaluate" for item in records)
    assert result["report"]["capabilities_fired"]["noul"]["count"] > 0


def test_terminal_grade_provenance_keeps_candidate_and_round_without_prompt_map() -> (
    None
):
    gateway = _gateway()
    grade_panel_with_jev(
        [PanelResult("candidate-a", "weak", 0, 1, "pass", "candidate prompt")],
        [
            {
                "id": "answer",
                "question": "Does it answer?",
                "kind": "noul",
                "expected": "yes",
            }
        ],
        gateway,
        judge_model="test-judge",
        run_id="grade-context",
        round_number=4,
    )

    records = round_judgment_provenance(gateway, 0, {}, None)
    grade = next(item for item in records if item["question_key"].startswith("grade_"))
    assert grade["candidate_id"] == "candidate-a"
    assert grade["round_number"] == 4
    assert grade["source_round"] == 4


def test_bounded_grade_choice_counts_as_classify_not_decide() -> None:
    gateway = _gateway()
    gateway.decide(
        {
            "key": "grade_0_0_first",
            "model": "test-judge",
            "type": "choice",
            "query": "Answer this success criterion using the provided outcomes.",
            "state": {"candidate_id": "candidate-a", "round_number": 4},
        }
    )

    record = round_judgment_provenance(gateway, 0, {}, None)[0]
    assert record["capability"] == "classify"
    assert record["stage"] == "grading"


@pytest.mark.parametrize(
    ("key", "kind", "capability", "stage"),
    [
        ("strategy_choice", "choice", "decide", "strategy_selection"),
        ("strategy_recheck:clarify", "noul", "noul", "strategy_selection"),
        ("pointer:vagueness:0", "choice", "find", "diagnosis"),
        ("existence:vagueness:0", "noul", "noul", "diagnosis"),
        ("gap:goal", "noul", "noul", "diagnosis"),
        ("problem:vagueness:s0001", "noul", "noul", "diagnosis"),
        (
            "restructure_lossless:role:s0001",
            "choice",
            "classify",
            "lossless_restructuring",
        ),
        ("infer:language", "choice", "extract", "clarification_inference"),
        (
            "faithful:test-1",
            "noul",
            "verify",
            "success_test_validation",
        ),
        (
            "assumption_meaning",
            "noul",
            "verify",
            "clarification_assumption",
        ),
        ("rubric:custom-question", "noul", None, "diagnosis"),
    ],
)
def test_known_judgment_callers_keep_semantic_capability_and_stage(
    key: str, kind: str, capability: str, stage: str
) -> None:
    gateway = _gateway()
    gateway.decide(
        {
            "key": key,
            "model": "test-judge",
            "type": kind,
            "query": "Test semantic caller mapping.",
            "state": {"round_number": 2},
        }
    )

    record = round_judgment_provenance(gateway, 0, {}, None)[0]
    assert (record["capability"], record["stage"]) == (capability, stage)


@pytest.mark.parametrize(
    ("requested_style", "applied_style"),
    [("auto", "shorter"), ("shorter", "shorter")],
)
def test_continue_reuses_persisted_style_route_and_provenance(
    tmp_path, requested_style: str, applied_style: str
) -> None:
    prompt = "whats 2 plus 2"
    gateway = _gateway(classify_choice="shorter", score_probability=0.01)
    database = tmp_path / "resume.sqlite"
    store = RunStore(str(database))
    optimizer = PromptOptimizer(
        store=store, gateway=gateway, writer_instruction_version=4
    )

    paused = optimizer.optimize(
        prompt,
        {
            "tier": "fast",
            "improvement_style": requested_style,
            "clarification_allowed": False,
            "time_limit_s": 0,
        },
    )
    assert paused["status"] == "needs_input"
    assert paused["report"]["status"] == "awaiting_approval"
    assert paused["report"]["applied_style"] == applied_style
    saved = store.get_run(paused["run_id"])
    assert saved is not None
    resume = saved[RESUME_CONTEXT_KEY]
    assert resume["understand"]["applied_style"] == applied_style
    assert resume["route"]["applied_style"] == applied_style
    previous_records = paused["report"]["judgment_provenance"]
    assert previous_records
    assert saved[RESUME_CONTEXT_KEY]["understand"]["requested_style"] == requested_style

    # A resumed run would infer Creative if it repeated Auto classification.
    gateway.decision_handler = _gateway(classify_choice="creative").decision_handler
    before_continue = len(gateway.decision_log)
    store.close()
    restarted_store = RunStore(str(database))
    restarted_optimizer = PromptOptimizer(
        store=restarted_store, gateway=gateway, writer_instruction_version=4
    )
    result = restarted_optimizer.continue_run(paused["run_id"], time_limit_s=0)

    new_keys = {
        entry["question"]["key"] for entry in gateway.decision_log[before_continue:]
    }
    assert result["report"]["improvement_style"] == requested_style
    assert result["report"]["applied_style"] == applied_style
    assert not any(key.startswith("understand:") for key in new_keys)
    assert not any(key.startswith("route:") for key in new_keys)
    result_records = result["report"]["judgment_provenance"]
    report_rows = Counter(
        (
            record["question_key"],
            record["model"],
            json.dumps(record["raw_answer"], sort_keys=True),
        )
        for record in result_records
    )
    answered_rows = Counter(
        (
            entry["question"]["key"],
            entry["answered_by"],
            json.dumps(entry["answer"], sort_keys=True),
        )
        for entry in gateway.decision_log
    )
    assert len(result_records) == len(gateway.decision_log)
    assert report_rows == answered_rows
    assert result["report"]["capabilities_fired"] == summarize_capabilities(
        result_records
    )
    assert all(
        (
            old["question_key"],
            old["candidate_id"],
            old["round_number"],
            old["raw_answer"],
        )
        in [
            (
                item["question_key"],
                item["candidate_id"],
                item["round_number"],
                item["raw_answer"],
            )
            for item in result_records
        ]
        for old in previous_records
    )


def test_every_named_style_maps_to_a_bundle() -> None:
    from prompt_enhancer.strategies import STYLE_STRATEGY_BUNDLES

    named = [style for style in IMPROVEMENT_STYLES if style != "auto"]
    assert set(STYLE_STRATEGY_BUNDLES) == set(named)
    for style, bundle in STYLE_STRATEGY_BUNDLES.items():
        assert len(bundle) >= 1, style
