from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.diagnosis import (
    DEFAULT_RUBRIC,
    SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
    Diagnoser,
    DiagnosisRubric,
    ProblemKind,
    split_sentences,
)
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    DecisionPolicy,
    PolicyDecision,
    runtime_question_identity,
)
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.jev_questions import sentence_existence_question
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


class BatchScriptedGateway(ScriptedGateway):
    def __init__(self, *, decision):
        super().__init__(
            chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decision
        )
        self.decision_batches: list[list[str]] = []

    def decide_batch(self, requests, *, role="judge", run_id=None):
        self.decision_batches.append(
            [str(request.get("key", "")) for request in requests]
        )
        return super().decide_batch(requests, role=role, run_id=run_id)


def _gateway(
    *,
    existence: dict[str, Any] | None = None,
    pointers: dict[str, str] | None = None,
    confirmations: dict[str, float] | None = None,
    outside_reference: float = 0.01,
) -> BatchScriptedGateway:
    exists = existence or {}
    selected = pointers or {}
    confirmed = confirmations or {}

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if key == "task_type":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        if key.startswith("pointer:"):
            _, kind, window = key.split(":")
            sentence_id = selected.get(f"{kind}:{window}", "none")
            probs = (
                {sentence_id: 0.95, "none": 0.05}
                if sentence_id != "none"
                else {"none": 1.0}
            )
            return {
                "type": "choice",
                "choice": sentence_id,
                "probabilities": probs,
                "confidence": 0.95,
            }
        if key.startswith("existence:"):
            _, kind, window = key.split(":")
            answer = exists.get(f"{kind}:{window}", 0.01)
            if answer == "malformed":
                return {"type": "not-a-decision"}
            if answer is None:
                return None
            return {
                "type": "noul",
                "probability_true": answer,
                "confidence": 1.0,
            }
        if key.startswith("problem:"):
            sentence_id = key.rsplit(":", 1)[-1]
            kind = key.split(":")[1]
            return {
                "type": "noul",
                "probability_true": confirmed.get(f"{kind}:{sentence_id}", 0.95),
                "confidence": 1.0,
            }
        probability = outside_reference if key == "gap:outside_reference" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    return BatchScriptedGateway(decision=decide)


def _optimize(prompt: str, gateway: ScriptedGateway, **kwargs):
    return PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=gateway,
        speculative_diagnosis=False,
        **kwargs,
    ).optimize(prompt, {"tier": "fast", "clarification_allowed": False})


def test_confident_pointer_with_low_existence_does_not_report_or_hint_problem() -> None:
    prompt = "Tell the warehouse team about the vans. Keep it concise."

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if key == "task_type":
            return {
                "type": "choice",
                "choice": "general",
                "probabilities": {"general": 1.0},
                "confidence": 1.0,
            }
        if key == "pointer:vagueness:0":
            return {
                "type": "choice",
                "choice": "s0001",
                "probabilities": {"s0001": 0.95, "none": 0.05},
                "confidence": 0.95,
            }
        if key.startswith("pointer:"):
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        if key == "existence:vagueness:0":
            return {"type": "noul", "probability_true": 0.1, "confidence": 1.0}
        if key.startswith("problem:vagueness:"):
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        probability = 0.78 if key == "gap:outside_reference" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=ScriptedGateway(
            chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=decide
        ),
    ).optimize(prompt, {"tier": "fast", "clarification_allowed": False})

    diagnosis = result["report"]["diagnosis"]
    assert diagnosis["problem_sentences"] == []
    assert diagnosis["possible_gaps"] == [
        {
            "key": "outside_reference",
            "label": "details only you know",
            "missing_probability": 0.78,
            "threshold": 0.8,
            "sentence": None,
        }
    ]


def test_supported_existence_and_confirmation_persist_all_sentence_evidence() -> None:
    prompt = "The answer should fit in a tweet. Also explain the details."
    store = RunStore(":memory:")
    gateway = _gateway(
        existence={"vagueness:0": 0.95},
        pointers={"vagueness:0": "s0001"},
    )
    result = PromptOptimizer(store=store, gateway=gateway).optimize(
        prompt, {"tier": "fast", "clarification_allowed": False}
    )

    diagnosis = result["report"]["diagnosis"]
    assert [
        (item["kind"], item["sentence_id"]) for item in diagnosis["problem_sentences"]
    ] == [("vagueness", "s0001")]
    assert diagnosis["sentence_protocol_version"] == SENTENCE_DIAGNOSIS_PROTOCOL_VERSION
    record = store.get_run(result["run_id"])
    assert record is not None
    stored = record["result"]["report"]["diagnosis"]
    pair = next(
        item for item in stored["sentence_evidence"] if item["kind"] == "vagueness"
    )
    assert pair["existence"]["raw_answer"]["probability_true"] == 0.95
    assert pair["pointer"]["raw_answer"]["choice"] == "s0001"
    assert pair["confirmation"]["raw_answer"]["probability_true"] == 0.95
    for role in ("existence", "pointer", "confirmation"):
        assert pair[role]["snapshot"] == JEV_MODEL
        assert pair[role]["identity"]["rubric_version"] == "default-v1"
        assert pair[role]["identity"]["answering_snapshot"] == JEV_MODEL


@pytest.mark.parametrize("existence", [0.8, 0.95])
def test_existence_cutoff_accepts_the_boundary_and_supported_problem(
    existence: float,
) -> None:
    result = _optimize(
        "The answer should fit in a tweet.",
        _gateway(
            existence={"vagueness:0": existence},
            pointers={"vagueness:0": "s0001"},
        ),
    )

    assert [
        item["kind"] for item in result["report"]["diagnosis"]["problem_sentences"]
    ] == ["vagueness"]


def test_malformed_existence_for_one_kind_does_not_hide_an_independent_kind() -> None:
    result = _optimize(
        "The answer should fit in a tweet.",
        _gateway(
            existence={"vagueness:0": "malformed", "unresolved_reference:0": 0.95},
            pointers={
                "vagueness:0": "s0001",
                "unresolved_reference:0": "s0001",
            },
        ),
    )

    diagnosis = result["report"]["diagnosis"]
    assert [item["kind"] for item in diagnosis["problem_sentences"]] == [
        "unresolved_reference"
    ]
    vague = next(
        item for item in diagnosis["sentence_evidence"] if item["kind"] == "vagueness"
    )
    assert vague["existence"]["accepted"] is False
    assert vague["existence"]["reason"] == "missing_or_malformed_answer"
    assert vague["pointer"]["raw_answer"]["choice"] == "s0001"


def test_none_pointer_does_not_confirm_even_when_existence_is_high() -> None:
    result = _optimize(
        "The answer should fit in a tweet.",
        _gateway(existence={"vagueness:0": 0.99}),
    )

    assert result["report"]["diagnosis"]["problem_sentences"] == []
    vague = next(
        item
        for item in result["report"]["diagnosis"]["sentence_evidence"]
        if item["kind"] == "vagueness"
    )
    assert vague["pointer"]["reason"] == "none_selected"
    assert vague["confirmation"]["requested"] is False


def test_existence_cutoffs_are_per_kind_and_versioned_on_evidence() -> None:
    rubric = replace(
        DEFAULT_RUBRIC,
        existence_thresholds={"vagueness": 0.9},
        existence_threshold_version="test-existence-cutoffs-v2",
    )
    result = _optimize(
        "The answer should fit in a tweet.",
        _gateway(
            existence={"vagueness:0": 0.85, "unresolved_reference:0": 0.85},
            pointers={
                "vagueness:0": "s0001",
                "unresolved_reference:0": "s0001",
            },
        ),
        diagnosis_rubric=rubric,
    )

    assert [
        item["kind"] for item in result["report"]["diagnosis"]["problem_sentences"]
    ] == ["unresolved_reference"]
    vague = next(
        item
        for item in result["report"]["diagnosis"]["sentence_evidence"]
        if item["kind"] == "vagueness"
    )
    unresolved = next(
        item
        for item in result["report"]["diagnosis"]["sentence_evidence"]
        if item["kind"] == "unresolved_reference"
    )
    assert vague["existence"]["threshold"] == 0.9
    assert vague["existence"]["rubric_threshold_version"] == "test-existence-cutoffs-v2"
    assert unresolved["existence"]["threshold"] == 0.8


def test_pointer_and_existence_use_one_shared_batch_with_exact_window_context() -> None:
    prompt = "First. Second. Third."
    gateway = _gateway(
        existence={"vagueness:0": 0.95}, pointers={"vagueness:0": "s0001"}
    )
    _optimize(prompt, gateway)

    sentence_batch = next(
        keys
        for keys in gateway.decision_batches
        if any(key.startswith("pointer:") for key in keys)
    )
    assert len(sentence_batch) == 8
    assert {key.split(":")[1] for key in sentence_batch} == {
        "vagueness",
        "unresolved_reference",
        "contradiction",
        "embedded_instruction",
    }
    logged = {
        entry["question"]["key"]: entry["question"] for entry in gateway.decision_log
    }
    pointer = logged["pointer:vagueness:0"]
    exists = logged["existence:vagueness:0"]
    assert pointer["state"]["prompt"] == prompt
    assert pointer["state"]["sentences"] == [
        {"id": "s0001", "text": "First."},
        {"id": "s0002", "text": "Second."},
        {"id": "s0003", "text": "Third."},
    ]
    assert exists["state"]["candidate_sentence_ids"] == ["s0001", "s0002", "s0003"]
    assert "this exact candidate window" in exists["query"]
    assert exists["state"]["prompt"] == prompt


def test_cross_window_pointer_is_rejected_and_cannot_duplicate_a_sentence() -> None:
    prompt = " ".join(f"Sentence {index}." for index in range(1, 256))
    gateway = _gateway(
        existence={"vagueness:0": 0.95, "vagueness:1": 0.95},
        pointers={"vagueness:0": "s0001", "vagueness:1": "s0001"},
    )
    result = _optimize(prompt, gateway)

    assert [
        (item["kind"], item["sentence_id"])
        for item in result["report"]["diagnosis"]["problem_sentences"]
    ] == [("vagueness", "s0001")]
    windows = [
        item
        for item in result["report"]["diagnosis"]["sentence_evidence"]
        if item["kind"] == "vagueness"
    ]
    first = next(item for item in windows if item["window_index"] == 0)
    second = next(item for item in windows if item["window_index"] == 1)
    assert first["pointer"]["reason"] == "supported"
    assert second["pointer"]["reason"] == "selected_id_outside_window"
    assert (
        len(
            [
                entry
                for entry in gateway.decision_log
                if str(entry["question"].get("key", "")).startswith(
                    "problem:vagueness:"
                )
            ]
        )
        == 1
    )


def test_later_window_without_pointer_decision_has_no_earlier_calibration() -> None:
    class GatePolicy(DecisionPolicy):
        def apply(self, **_kwargs) -> PolicyDecision:
            return PolicyDecision(disposition="gate", verdict="gate", threshold=0.8)

    prompt = " ".join(f"Sentence {index}." for index in range(1, 256))
    gateway = _gateway(
        existence={"vagueness:0": 0.95, "vagueness:1": 0.95},
        pointers={"vagueness:0": "s0001", "vagueness:1": "none"},
    )
    result = _optimize(prompt, gateway, decision_policy=GatePolicy())
    windows = [
        item
        for item in result["report"]["diagnosis"]["sentence_evidence"]
        if item["kind"] == "vagueness"
    ]

    assert windows[0]["pointer"]["calibration"]["verdict"] == "gate"
    assert windows[1]["pointer"]["reason"] == "none_selected"
    assert "calibration" not in windows[1]["pointer"]


@pytest.mark.parametrize(
    ("probability", "verdict", "confirmed"),
    [(0.85, "gate", False), (0.9, "gate", True), (0.95, "ranker", False)],
)
def test_matching_existence_calibration_overrides_the_rubric_cutoff(
    probability: float, verdict: str, confirmed: bool
) -> None:
    request = {
        "model": JEV_MODEL,
        "query": sentence_existence_question("vagueness"),
        "state": {
            "prompt": "A sentence.",
            "sentences": [{"id": "s0001", "text": "A sentence."}],
        },
        "type": "noul",
        "key": "existence:vagueness:0",
        "question_schema": {
            "protocol": SENTENCE_DIAGNOSIS_PROTOCOL_VERSION,
            "question": 1,
        },
    }
    identity = runtime_question_identity(
        "existence:vagueness",
        request,
        family="existence",
        rubric_version="default-v1",
        snapshot=JEV_MODEL,
    )
    artifact = CalibrationArtifact.from_dict(
        {
            "name": "existence-cutoff",
            "questions": {
                identity.question_id: {
                    "identity": identity.to_dict(),
                    "verdict": verdict,
                    "threshold": 0.9,
                }
            },
        }
    )
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=_gateway(
            existence={"vagueness:0": probability},
            pointers={"vagueness:0": "s0001"},
        ),
        decision_policy=DecisionPolicy.from_artifact(artifact),
    )

    result = optimizer.optimize(
        "A sentence.", {"tier": "fast", "clarification_allowed": False}
    )

    assert bool(result["report"]["diagnosis"]["problem_sentences"]) is confirmed
    evidence = next(
        item
        for item in result["report"]["diagnosis"]["sentence_evidence"]
        if item["kind"] == "vagueness"
    )
    assert evidence["existence"]["threshold"] == 0.9
    assert (
        evidence["existence"]["identity"]["question_digest"] == identity.question_digest
    )
    if verdict == "ranker":
        assert evidence["existence"]["reason"] == "calibration_unavailable"


def test_pointer_abstention_is_reported_as_calibration_abstained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Diagnoser._policy_decision

    def abstain_pointer(self, *, question_id, **kwargs):
        if question_id == "pointer:vagueness":
            return PolicyDecision(disposition="abstain", reason="policy_abstained")
        return original(self, question_id=question_id, **kwargs)

    monkeypatch.setattr(Diagnoser, "_policy_decision", abstain_pointer)
    result = _optimize(
        "A sentence.",
        _gateway(
            existence={"vagueness:0": 0.95},
            pointers={"vagueness:0": "s0001"},
        ),
    )
    evidence = next(
        item
        for item in result["report"]["diagnosis"]["sentence_evidence"]
        if item["kind"] == "vagueness"
    )
    assert evidence["pointer"]["accepted"] is False
    assert evidence["pointer"]["reason"] == "calibration_abstained"


def test_legacy_diagnosis_payload_deserializes_without_inventing_existence_values() -> (
    None
):
    from prompt_enhancer.diagnosis import diagnosis_from_dict

    old = diagnosis_from_dict(
        {
            "task_type": "general",
            "task_type_label": "General",
            "task_type_confidence": 1.0,
            "confirmed_gaps": [],
            "problem_sentences": [],
        }
    )

    assert old.sentence_protocol_version == 1
    assert old.sentence_evidence == ()
    assert "existence" not in json.dumps(old.as_dict())
    assert DEFAULT_RUBRIC.existence_threshold_for(ProblemKind.VAGUENESS) == 0.8
    assert (
        DiagnosisRubric(task_types=()).existence_threshold_for("contradiction") == 0.8
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "Use e.g. Python 3.5 to parse v1.2. Dr. Smith wrote it.",
            ["Use e.g. Python 3.5 to parse v1.2.", "Dr. Smith wrote it."],
        ),
        (
            "You are a helpful assistant. Do not mention the U.S. government. Be concise.",
            [
                "You are a helpful assistant.",
                "Do not mention the U.S. government.",
                "Be concise.",
            ],
        ),
        (
            "Rules:\n- be brief\n- be kind\nWrite a poem",
            ["Rules:", "- be brief", "- be kind", "Write a poem"],
        ),
        (
            "Steps:\n1. Read the file.\n2. Summarize it.",
            ["Steps:", "1. Read the file.", "2. Summarize it."],
        ),
        (
            'Translate to French: "Hello. How are you?" Keep the tone friendly.',
            ['Translate to French: "Hello. How are you?"', "Keep the tone friendly."],
        ),
        (
            '"Yes." or "No." answers are not acceptable.',
            ['"Yes." or "No." answers are not acceptable.'],
        ),
        (
            "Use option A. Then do option B.",
            ["Use option A.", "Then do option B."],
        ),
        (
            "See https://a.com/x.html. Then reply.",
            ["See https://a.com/x.html.", "Then reply."],
        ),
        ("Email a@example.com. Then reply.", ["Email a@example.com.", "Then reply."]),
        (
            "Version 2.0 is out. Update your code to v2.1.3.",
            ["Version 2.0 is out.", "Update your code to v2.1.3."],
        ),
    ],
)
def test_split_sentences_preserves_natural_units_and_offsets(
    prompt: str, expected: list[str]
) -> None:
    sentences = split_sentences(prompt)

    assert [sentence.text for sentence in sentences] == expected
    assert all(
        prompt[sentence.start : sentence.end] == sentence.text for sentence in sentences
    )
    assert [sentence.id for sentence in sentences] == [
        f"s{index:04d}" for index in range(1, len(sentences) + 1)
    ]


def test_sentence_protocol_version_tracks_splitter_change() -> None:
    assert SENTENCE_DIAGNOSIS_PROTOCOL_VERSION == 3


def test_historical_sentence_protocols_keep_the_original_boundaries() -> None:
    prompt = "Use e.g. Python 3.5 to parse v1.2.\n- be brief\n- be kind"

    current = [sentence.text for sentence in split_sentences(prompt)]
    legacy = [sentence.text for sentence in split_sentences(prompt, protocol_version=2)]

    assert current == ["Use e.g. Python 3.5 to parse v1.2.", "- be brief", "- be kind"]
    assert legacy == ["Use e.g.", "Python 3.5 to parse v1.2.", "- be brief\n- be kind"]
