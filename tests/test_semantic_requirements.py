"""Semantic findings belong to their own draft and model/sample evidence."""

import pytest
from active_clock import TickingClock, advancing_chat
from test_compound_requirements import compound_gateway, obligation

from prompt_enhancer import PromptOptimizer, RunStore


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        ({"type": "noul", "probability_true": 0.99, "confidence": 1.0}, "tested"),
        ({"type": "noul", "probability_true": 0.01, "confidence": 1.0}, "failed"),
        ({"type": "noul", "probability_true": 0.5, "confidence": 1.0}, "unresolved"),
        ({"type": "noul", "probability_true": 0.99, "confidence": 0.1}, "unresolved"),
        ({"type": "noul", "probability_true": "bad"}, "unresolved"),
        ({"type": "noul", "probability": 0.99, "certainty": 0.99}, "tested"),
        ({"data": {"noul": {"probability_true": 0.99}}}, "tested"),
    ],
)
def test_semantic_checks_retain_raw_decisions_at_public_history_seam(
    tmp_path, answer, status
):
    prompt = "Explain gravity for a ten-year-old without changing the supplied facts."
    gateway = compound_gateway(
        prompt,
        [obligation(prompt, prompt)],
        output="Things fall because Earth attracts them.",
    )
    decide = gateway.decision_handler
    requests = []

    def semantic(request, **params):
        if str(request.get("key", "")).startswith("requirement:semantic:"):
            requests.append(request)
            return answer
        return decide(request, **params)

    gateway.decision_handler = semantic
    clock = TickingClock()
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 10)
    store = RunStore(tmp_path / "semantic.sqlite")
    result = PromptOptimizer(store=store, gateway=gateway, clock=clock).optimize(
        prompt, {"time_limit_s": 0}
    )
    assert requests
    ranking = result["report"]["history"][0]["evidence"]["selection_evidence"][
        "ranking"
    ]
    findings = ranking[0]["metadata"]["requirement_findings"]
    semantic_findings = [
        item for item in findings if item["check"] == "semantic_obligation"
    ]
    assert semantic_findings and all(
        item["status"] == status for item in semantic_findings
    )
    assert all(
        item["candidate_id"] == ranking[0]["candidate_id"] for item in semantic_findings
    )
    assert all(item["raw_decision"] == answer for item in semantic_findings)
    assert any(
        item.get("model") and item.get("sample") is not None
        for item in semantic_findings
    )
    assert result["original_kept"] is (status != "tested")
    assert (
        RunStore(tmp_path / "semantic.sqlite").get_run(result["run_id"])["result"][
            "report"
        ]
        == result["report"]
    )


def test_one_failed_sample_blocks_a_draft_despite_other_semantic_passes(tmp_path):
    prompt = "Explain gravity without changing the supplied facts."
    gateway = compound_gateway(prompt, [obligation(prompt, prompt)])
    decide = gateway.decision_handler

    def disagree(request, **params):
        if str(request.get("key", "")).startswith("requirement:semantic:"):
            return {
                "type": "noul",
                "probability_true": 0.01
                if request["state"].get("sample") == 1
                else 0.99,
                "confidence": 1.0,
            }
        return decide(request, **params)

    gateway.decision_handler = disagree
    result = PromptOptimizer(
        store=RunStore(tmp_path / "disagreement.sqlite"), gateway=gateway
    ).optimize(prompt, {"time_limit_s": 0})
    ranking = result["report"]["history"][0]["evidence"]["selection_evidence"][
        "ranking"
    ]
    findings = ranking[0]["metadata"]["requirement_findings"]
    assert {"tested", "failed"} <= {
        item["status"] for item in findings if item["check"] == "semantic_obligation"
    }
    assert result["original_kept"]


def test_known_mechanical_failure_remains_authoritative_over_semantic_passes():
    prompt = "Reply with exactly PING and nothing else."
    gateway = compound_gateway(
        prompt,
        [obligation(prompt, prompt)],
        output="PONG",
        candidate="Respond with exactly PING and nothing else.",
    )
    result = PromptOptimizer(store=RunStore(":memory:"), gateway=gateway).optimize(
        prompt, {"time_limit_s": 0}
    )
    findings = result["report"]["history"][0]["evidence"]["selection_evidence"][
        "ranking"
    ][0]["metadata"]["requirement_findings"]
    assert any(
        item["status"] == "failed" and item["check"] == "exact_output"
        for item in findings
    )
    assert any(
        item["status"] == "tested" and item["check"] == "semantic_obligation"
        for item in findings
    )
    assert result["original_kept"]


def test_typed_requirement_decision_omits_private_explanations():
    from prompt_enhancer.requirement_decisions import typed_evidence

    assert typed_evidence(
        {
            "data": {
                "noul": {
                    "probability": 0.99,
                    "certainty": 0.95,
                    "reasoning": "private",
                    "message": "private",
                }
            },
            "diagnostics": "private",
        }
    ) == {"data": {"noul": {"probability": 0.99, "certainty": 0.95}}}
