"""Weak-output failure hypotheses reach later Rounds only with source support."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from prompt_enhancer.catalog import JEV_MODEL, ModelInfo, StaticModelCatalog
from prompt_enhancer.config import Settings
from prompt_enhancer.evaluation import Dataset, EvaluationHarness
from prompt_enhancer.evaluation.harness import default_engine_factory
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.failure_attribution import AttributionBudget
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
        retry_count: int = 0,
        original_score_attempts_before_pass: int = 5,
    ) -> None:
        self.pointer = pointer
        self.kind = kind
        self.attributable = attributable
        self.confidence = confidence
        self.fail_attribution = fail_attribution
        self.malformed_attribution = malformed_attribution
        self.writer_states: list[dict[str, Any]] = []
        self.attribution_batches: list[list[dict[str, Any]]] = []
        self.original_score_attempts = 0
        self.original_score_attempts_before_pass = original_score_attempts_before_pass
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
        if retry_count:
            self.config = SimpleNamespace(max_retries=retry_count)

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
            if key.startswith("evaluate:compare:") and key.endswith(
                ":verbosity_direction"
            ):
                choice = "same"
            elif key == "task_type":
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
        elif key.startswith("score:"):
            state = request["state"]
            if state["candidate_prompt"] == state["original_prompt"]:
                self.original_score_attempts += 1
                # Fail one actual baseline sample, then let the original's own
                # floor-passing second-round vector provide a healthy stop.
                probability = (
                    0.99
                    if self.original_score_attempts
                    > self.original_score_attempts_before_pass
                    else 0.01
                )
            else:
                probability = 0.01
        elif key.startswith("evaluate:"):
            probability = 0.99
        else:
            probability = 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}


def _optimize(gateway: AttributionGateway, *, settings=None):
    return PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        config=settings or Settings(),
        writer_instruction_version=8,
    ).optimize(PROMPT, {"clarification_allowed": False})


def test_supported_source_attribution_reaches_next_round_writer() -> None:
    gateway = AttributionGateway()

    result = _optimize(gateway)

    assert result["original_kept"] is True
    assert len(gateway.writer_states) >= 2
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

        assert len(gateway.writer_states) >= 2
        assert not any(
            "attribution hypothesis" in item
            for item in gateway.writer_states[1]["previous_failures"]
        )
        records = result["report"]["history"][0]["candidate_failures"][0][
            "attributions"
        ]
        assert records
        assert all(record["status"] == "unresolved" for record in records)


def test_zero_pair_cap_and_missing_pricing_skip_attribution_without_losing_failure() -> (
    None
):
    for settings, priced, reason in (
        (Settings(attribution_pair_cap=0), True, "pair_budget_exhausted"),
        (Settings(), False, "missing_trustworthy_pricing"),
    ):
        gateway = AttributionGateway(priced=priced)
        result = _optimize(gateway, settings=settings)

        assert gateway.attribution_batches == []
        assert result["report"]["history"][0]["candidate_failures"]
        attribution = result["report"]["failure_attribution"]
        assert attribution["skipped_count"] > 0
        assert {pair["reason"] for pair in attribution["pairs"]} == {reason}


def test_attribution_budget_defaults_and_explicit_overrides_are_settings_backed():
    defaults = AttributionBudget.for_settings(Settings())
    custom = AttributionBudget.for_settings(Settings(), pair_cap=2, dollar_cap=0.004)

    assert (defaults.pair_cap, defaults.dollar_cap) == (30, 0.03)
    assert (custom.pair_cap, custom.dollar_cap) == (2, 0.004)


def test_pair_cap_applies_across_models_and_samples() -> None:
    gateway = AttributionGateway()
    result = _optimize(
        gateway,
        settings=Settings(attribution_pair_cap=1, attribution_dollar_cap=0.01),
    )

    attribution = result["report"]["failure_attribution"]
    assert attribution["requested_pair_count"] == 1
    assert attribution["skipped_count"] > 0
    assert len(gateway.attribution_batches) == len(
        result["report"]["history"]
    )  # One per round.


def test_dollar_cap_skips_attribution_without_erasing_failures() -> None:
    gateway = AttributionGateway()
    result = _optimize(
        gateway,
        settings=Settings(attribution_pair_cap=10, attribution_dollar_cap=0.000001),
    )

    assert gateway.attribution_batches == []
    assert result["report"]["history"][0]["candidate_failures"]
    attribution = result["report"]["history"][0]["evidence"]["failure_attribution"]
    assert attribution["requested_pair_count"] == 0
    assert {pair["reason"] for pair in attribution["pairs"]} == {
        "dollar_budget_exhausted"
    }


def test_recorded_retry_reservation_preserves_attribution_budget_on_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attribution-retry-budget.json"
    gateway = RecordingGateway(
        AttributionGateway(retry_count=3, original_score_attempts_before_pass=0), path
    )
    settings = Settings(attribution_dollar_cap=0.0003)
    original = PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        config=settings,
        writer_instruction_version=8,
    ).optimize(PROMPT, {"clarification_allowed": False})
    assert (
        json.loads(path.read_text())["cascade_settings"]["retry_reservation_multiplier"]
        == 4
    )
    original_attribution = original["report"]["history"][0]["evidence"][
        "failure_attribution"
    ]
    assert original_attribution["requested_pair_count"] == 0
    engine = default_engine_factory(path)
    engine.store = RunStore(":memory:")

    replayed = engine.optimize(PROMPT, {"clarification_allowed": False})

    assert (
        replayed["report"]["history"][0]["evidence"]["failure_attribution"]
        == original_attribution
    )


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
    gateway = RecordingGateway(
        AttributionGateway(original_score_attempts_before_pass=0), path
    )
    gateway.writer_instruction_version = 8
    gateway.faithfulness_threshold = 0.8
    original = PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        config=Settings(),
        writer_instruction_version=8,
    ).optimize(PROMPT, {"clarification_allowed": False})

    replay = default_engine_factory(path)
    replay.store = RunStore(":memory:")
    reproduced = replay.optimize(PROMPT, {"clarification_allowed": False})

    assert reproduced["final_prompt"] == original["final_prompt"]
    assert (
        reproduced["report"]["history"][0]["candidate_failures"]
        == original["report"]["history"][0]["candidate_failures"]
    )
