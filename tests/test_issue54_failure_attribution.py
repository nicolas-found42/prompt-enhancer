"""Weak-output failure hypotheses reach later Rounds only with source support."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prompt_enhancer.catalog import JEV_MODEL, ModelInfo, StaticModelCatalog
from prompt_enhancer.config import Settings
from prompt_enhancer.evaluation import Dataset, EvaluationHarness
from prompt_enhancer.evaluation.harness import default_engine_factory
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import ProviderError, ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore

PROMPT = "Summarize the report. Keep it concise."
CANDIDATE = "Summarize the report.\n\nKeep it concise."


class AttributionGateway(ScriptedGateway):
    def __init__(
        self,
        *,
        pointer: str = "s0002",
        kind: str = "ignored_constraint",
        attributable: float = 0.95,
        confidence: float = 0.95,
        priced: bool = True,
        fail_attribution: bool = False,
        malformed_attribution: bool = False,
    ) -> None:
        self.pointer = pointer
        self.kind = kind
        self.attributable = attributable
        self.confidence = confidence
        self.fail_attribution = fail_attribution
        self.malformed_attribution = malformed_attribution
        self.writer_states: list[dict[str, Any]] = []
        self.attribution_batches: list[list[dict[str, Any]]] = []
        catalog = (
            StaticModelCatalog(
                (),
                (
                    ModelInfo(
                        id=JEV_MODEL,
                        provider="openrouter",
                        input_cost_per_token=0.0000001,
                        output_cost_per_token=0.0000002,
                    ),
                ),
            )
            if priced
            else None
        )
        super().__init__(chat=self._chat, decision=self._decide, catalog=catalog)

    def decide_batch(self, requests, *, role="judge", run_id=None):
        if role == "judge_attribution":
            self.attribution_batches.append([dict(item) for item in requests])
            if self.fail_attribution:
                raise ProviderError("scripted", JEV_MODEL, None, role=role)
        return super().decide_batch(requests, role=role, run_id=run_id)

    def _chat(self, _model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if "strategies" in state:
                self.writer_states.append(state)
                return json.dumps(
                    {item["name"]: CANDIDATE for item in state["strategies"]}
                )
            return json.dumps(
                {
                    "tests": [
                        {
                            "question": "Does the answer summarize the report concisely?",
                            "kind": "noul",
                            "expected": "yes",
                        }
                    ]
                }
            )
        if role == "weak":
            return "fail" if "\n\n" in messages[0]["content"] else "pass"
        return "pass"

    def _decide(self, request: Mapping[str, Any], **_kwargs):
        key = str(request.get("key", ""))
        if self.malformed_attribution and key.startswith("failure-attribution:"):
            return {"unexpected": "answer"}
        if request.get("type") == "choice":
            if key == "task_type":
                choice = "general"
            elif key == "strategy_choice":
                choice = "specify_output_format"
            elif key.startswith("fidelity:sentence:"):
                choice = "supported_by_original"
            elif key.startswith("failure-attribution:"):
                choice = self.pointer if key.endswith(":pointer") else self.kind
            else:
                choice = "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: self.confidence, "none": 1 - self.confidence}
                if choice != "none"
                else {"none": 1.0},
                "confidence": self.confidence,
            }
        if key.startswith("failure-attribution:"):
            probability = self.attributable
        elif key.startswith("grade_"):
            probability = float(request["state"]["output"] == "pass")
        elif key.startswith("output-screen:"):
            probability = 0.01
        elif key.startswith("success-test-screen:"):
            probability = (
                0.99
                if key.endswith((":faithfulness", ":no_invention", ":assessability"))
                else 0.01
            )
        elif key == "gap:output_format":
            probability = 0.99
        elif key.startswith(("strategy_recheck:", "fidelity:")):
            probability = 0.99
        else:
            probability = 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}


def _optimize(gateway: AttributionGateway, *, tier: str = "standard", settings=None):
    return PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        config=settings or Settings(),
        writer_instruction_version=8,
    ).optimize(PROMPT, {"tier": tier, "clarification_allowed": False})


def test_supported_source_attribution_reaches_next_round_writer() -> None:
    gateway = AttributionGateway()

    result = _optimize(gateway)

    assert result["original_kept"] is True
    assert len(gateway.writer_states) == 2
    previous = gateway.writer_states[1]["previous_failures"]
    assert any(
        "s0002" in item
        and "Keep it concise." in item
        and "ignored_constraint" in item
        and "candidate-" in item
        for item in previous
    )
    first_round = result["report"]["history"][0]
    records = first_round["candidate_failures"][0]["attributions"]
    assert len(records) >= 2
    assert all(record["prompt_digest"] for record in records)
    assert {record["model"] for record in records} >= {
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-nemo",
    }
    assert first_round["evidence"]["failure_attribution"]["attributed_count"] > 0


def test_low_confidence_none_unknown_and_provider_failure_stay_auditable() -> None:
    for options in (
        {"confidence": 0.79},
        {"pointer": "none"},
        {"kind": "unknown"},
        {"fail_attribution": True},
        {"malformed_attribution": True},
    ):
        gateway = AttributionGateway(**options)
        result = _optimize(gateway)

        assert len(gateway.writer_states) == 2
        assert not any(
            "attribution hypothesis" in item
            for item in gateway.writer_states[1]["previous_failures"]
        )
        records = result["report"]["history"][0]["candidate_failures"][0][
            "attributions"
        ]
        assert records
        assert all(record["status"] == "unresolved" for record in records)


def test_fast_and_missing_pricing_skip_attribution_without_losing_failure() -> None:
    for tier, priced in (("fast", True), ("standard", False)):
        gateway = AttributionGateway(priced=priced)
        result = _optimize(gateway, tier=tier)

        assert gateway.attribution_batches == []
        assert result["report"]["history"][0]["candidate_failures"]
        attribution = result["report"]["failure_attribution"]
        assert attribution["skipped_count"] > 0


def test_pair_cap_applies_across_models_and_samples() -> None:
    gateway = AttributionGateway()
    result = _optimize(
        gateway,
        settings=Settings(attribution_pair_cap=1, attribution_dollar_cap=0.01),
    )

    attribution = result["report"]["failure_attribution"]
    assert attribution["requested_pair_count"] == 1
    assert attribution["skipped_count"] > 0
    assert len(gateway.attribution_batches) == 2  # One per Round.


def test_dollar_cap_skips_attribution_without_erasing_failures() -> None:
    gateway = AttributionGateway()
    result = _optimize(
        gateway,
        settings=Settings(attribution_pair_cap=10, attribution_dollar_cap=0.000001),
    )

    assert gateway.attribution_batches == []
    assert result["report"]["history"][0]["candidate_failures"]
    assert result["report"]["failure_attribution"]["requested_pair_count"] == 0
    assert {
        pair["reason"] for pair in result["report"]["failure_attribution"]["pairs"]
    } == {"dollar_budget_exhausted"}


def test_harness_counts_attributions_and_only_scores_independent_labels() -> None:
    dataset = Dataset.from_dict(
        {
            "name": "attribution-known-answer",
            "cases": [
                {
                    "id": "labeled",
                    "source": "synthetic",
                    "prompt": PROMPT,
                    "metadata": {
                        "attribution_labels": [
                            {
                                "prompt_digest": "digest-a",
                                "model": "weak-a",
                                "sample": 0,
                                "test_id": "t0",
                                "sentence_id": "s0002",
                                "kind": "ignored_constraint",
                                "provenance": "human",
                            }
                        ]
                    },
                }
            ],
        }
    )

    class KnownAnswerEngine:
        def optimize(self, prompt, _options):
            return {
                "status": "completed",
                "final_prompt": prompt,
                "original_kept": True,
                "report": {
                    "history": [
                        {
                            "evidence": {
                                "failure_attribution": {
                                    "pairs": [
                                        {
                                            "status": "supported",
                                            "prompt_digest": "digest-a",
                                            "model": "weak-a",
                                            "sample": 0,
                                            "test_id": "t0",
                                            "sentence_id": "s0002",
                                            "kind": "ignored_constraint",
                                        },
                                        {"status": "unresolved"},
                                        {"status": "skipped"},
                                    ]
                                }
                            }
                        }
                    ]
                },
            }

    report = EvaluationHarness(KnownAnswerEngine()).run(dataset).to_dict()

    attribution = report["failure_attribution"]
    assert attribution["attributed_count"] == 1
    assert attribution["unresolved_count"] == 1
    assert attribution["skipped_count"] == 1
    assert attribution["correctness"] == {
        "status": "available",
        "labeled_predictions": 1,
        "correct_predictions": 1,
        "accuracy": 1.0,
        "provenance": ["human"],
    }


def test_version_eight_attribution_replays_strictly(tmp_path: Path) -> None:
    path = tmp_path / "attribution.json"
    gateway = RecordingGateway(AttributionGateway(), path)
    gateway.writer_instruction_version = 8
    gateway.faithfulness_threshold = 0.8
    original = PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        config=Settings(),
        writer_instruction_version=8,
    ).optimize(PROMPT, {"tier": "standard", "clarification_allowed": False})

    replay = default_engine_factory(path)
    replay.store = RunStore(":memory:")
    reproduced = replay.optimize(
        PROMPT, {"tier": "standard", "clarification_allowed": False}
    )

    assert reproduced["final_prompt"] == original["final_prompt"]
    assert (
        reproduced["report"]["history"][0]["candidate_failures"]
        == original["report"]["history"][0]["candidate_failures"]
    )
