from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from prompt_enhancer import diagnosis as diagnosis_module
from prompt_enhancer.catalog import JEV_MODEL, ModelInfo, StaticModelCatalog
from prompt_enhancer.diagnosis import (
    DEFAULT_RUBRIC,
    ChecklistItem,
    Diagnoser,
    GapImpact,
)
from prompt_enhancer.evaluation.harness import default_engine_factory
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    ProviderError,
    ScriptedGateway,
)
from prompt_enhancer.jev import batch_decision_payload
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.rubric_revisions import (
    RubricQuestion,
    RubricVersion,
    SQLiteRubricStore,
)
from prompt_enhancer.store import RunStore


class BatchGateway(ScriptedGateway):
    def __init__(self, *, pointer: str = "none", root: str = "general") -> None:
        self.batches: list[list[dict]] = []
        self.pointer = pointer
        self.root = root
        super().__init__(
            chat=self._chat,
            decision=self._answer,
        )

    @staticmethod
    def _chat(_model, messages, *, role, **_kwargs):
        if role != "writer":
            return "pass"
        state = json.loads(messages[1]["content"])
        strategies = state.get("strategies")
        if isinstance(strategies, list):
            prompt = str(state.get("prompt", ""))
            # Echo each candidate so the run terminates only after the real
            # baseline vector and Accept judgment establish convergence.
            return json.dumps({strategy["name"]: prompt for strategy in strategies})
        return '{"tests":[]}'

    def decide_batch(self, requests, *, role="judge", run_id=None):
        self.batches.append([dict(request) for request in requests])
        return super().decide_batch(requests, role=role, run_id=run_id)

    def _answer(self, request, **_kwargs):
        key = str(request.get("key", ""))
        if key.endswith(":verbosity_direction"):
            selected = "same"
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": {selected: 1.0},
                "confidence": 1.0,
            }
        if key == "route:find":
            selected = "clearer"
        elif key == "strategy_choice":
            selected = "add_missing_context"
        elif key == "task_type":
            selected = self.root
        elif key.startswith("task_type:"):
            selected = "writing" if key.endswith("communication") else "unknown"
        elif key.startswith("pointer:"):
            selected = self.pointer if key == "pointer:vagueness:0" else "none"
        else:
            selected = None
        if selected is not None:
            return {
                "type": "choice",
                "choice": selected,
                "probabilities": {selected: 1.0},
                "confidence": 1.0,
            }
        if key.startswith("score:") or key.startswith(
            ("evaluate:", "fidelity:", "strategy_recheck:")
        ):
            probability = 0.99
        else:
            probability = 0.99 if key == "existence:vagueness:0" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}


def _diagnosis_batches(gateway: BatchGateway) -> list[list[dict]]:
    """Batches carrying diagnosis keys, excluding other pipeline judgments."""
    return [
        batch
        for batch in gateway.batches
        if batch
        and all(
            str(request.get("key", "")).startswith(
                (
                    "task_type",
                    "gap:",
                    "pointer:",
                    "existence:",
                    "problem:",
                    "rubric:",
                )
            )
            for request in batch
        )
    ]


def _run(
    gateway: BatchGateway,
    *,
    speculative: bool = True,
    observe_sequential: bool = False,
    rubric=DEFAULT_RUBRIC,
):
    return PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        diagnosis_rubric=rubric,
        speculative_diagnosis=speculative,
        observe_sequential_diagnosis=observe_sequential,
    ).optimize(
        "Write a brief note. Keep it clear.",
        {"clarification_allowed": False},
    )


def test_normal_fanout_uses_one_request_without_pointer_and_matches_sequential() -> (
    None
):
    fanout = BatchGateway(root="communication")
    sequential = BatchGateway(root="communication")

    current = _run(fanout)
    baseline = _run(sequential, speculative=False, observe_sequential=True)

    assert len(_diagnosis_batches(fanout)) == 1
    assert current["report"]["diagnosis"]["request_evidence"]["provider_requests"] == 1
    assert baseline["report"]["diagnosis"]["request_evidence"]["provider_requests"] > 1
    assert (
        current["report"]["diagnosis"]["task_type"]
        == baseline["report"]["diagnosis"]["task_type"]
    )
    assert (
        current["report"]["diagnosis"]["confirmed_gaps"]
        == baseline["report"]["diagnosis"]["confirmed_gaps"]
    )
    assert (
        current["report"]["diagnosis"]["problem_sentences"]
        == baseline["report"]["diagnosis"]["problem_sentences"]
    )
    keys = {str(request["key"]) for request in _diagnosis_batches(fanout)[0]}
    assert {
        "task_type",
        "task_type:communication",
        "task_type:investigation",
        "task_type:execution",
    } <= keys
    assert "existence:vagueness:0" in keys


def test_surviving_pointer_uses_one_confirmation_request() -> None:
    gateway = BatchGateway(pointer="s0001")
    sequential = BatchGateway(pointer="s0001")

    result = _run(gateway)
    baseline = _run(sequential, speculative=False)

    assert len(_diagnosis_batches(gateway)) == 2
    assert {str(request["key"]) for request in _diagnosis_batches(gateway)[1]} == {
        "problem:vagueness:s0001"
    }
    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert evidence["complete"] is True
    assert evidence["provider_requests"] == 2
    assert (
        result["report"]["diagnosis"]["problem_sentences"]
        == baseline["report"]["diagnosis"]["problem_sentences"]
    )


def test_unused_malformed_speculative_leaf_does_not_change_selected_branch() -> None:
    class MalformedUnusedLeafGateway(BatchGateway):
        def _answer(self, request, **kwargs):
            if request.get("key") == "task_type:investigation":
                return {"type": "not-a-decision"}
            return super()._answer(request, **kwargs)

    gateway = MalformedUnusedLeafGateway(root="communication")

    result = _run(gateway)

    assert result["report"]["diagnosis"]["task_type"] == "writing"
    assert result["report"]["diagnosis"]["request_evidence"]["complete"] is True


def test_same_gap_key_with_different_question_meaning_is_not_deduplicated() -> None:
    class WordingGateway(BatchGateway):
        def _answer(self, request, **kwargs):
            if str(request.get("key", "")).startswith("gap:shared"):
                return {
                    "type": "noul",
                    "probability_true": 0.99
                    if request.get("query") == "Is the writing audience missing?"
                    else 0.01,
                    "confidence": 1.0,
                }
            return super()._answer(request, **kwargs)

    rubric = replace(
        DEFAULT_RUBRIC,
        task_types=tuple(
            replace(
                task,
                checklist=task.checklist
                + (
                    ChecklistItem(
                        "shared",
                        "shared",
                        GapImpact.LOW,
                        question="Is the writing audience missing?"
                        if task.key == "writing"
                        else "Is the general setting missing?",
                    ),
                ),
            )
            if task.key in {"general", "writing"}
            else task
            for task in DEFAULT_RUBRIC.task_types
        ),
    )
    gateway = WordingGateway(root="communication")

    result = _run(gateway, rubric=rubric)

    matching = [
        request
        for request in _diagnosis_batches(gateway)[0]
        if str(request["key"]).startswith("gap:shared")
    ]
    assert len(matching) == 2
    assert len({request["query"] for request in matching}) == 2
    assert "shared" in {
        gap["key"] for gap in result["report"]["diagnosis"]["confirmed_gaps"]
    }


def test_active_rubric_question_joins_the_first_provider_request(
    tmp_path: Path,
) -> None:
    rubric_store = SQLiteRubricStore(tmp_path / "rubric.sqlite3")
    rubric_store.initialize(
        RubricVersion(
            "active",
            (RubricQuestion("missing-audience", "Is the audience missing?"),),
        )
    )
    gateway = BatchGateway()

    result = PromptOptimizer(
        gateway=gateway,
        store=RunStore(":memory:"),
        rubric_store=rubric_store,
    ).optimize(
        "Write a brief note.",
        {"clarification_allowed": False},
    )

    assert len(_diagnosis_batches(gateway)) == 1
    assert "rubric:missing-audience" in {
        str(request["key"]) for request in _diagnosis_batches(gateway)[0]
    }
    assert result["report"]["diagnosis"]["request_evidence"]["complete"] is True


def test_oversized_fanout_falls_back_with_a_hard_request_cap(monkeypatch) -> None:
    many = tuple(
        ChecklistItem(f"extra_{index}", f"extra criterion {index}", GapImpact.LOW)
        for index in range(50)
    )
    rubric = replace(
        DEFAULT_RUBRIC,
        task_types=tuple(
            replace(task, checklist=task.checklist + many)
            if task.key == "general"
            else task
            for task in DEFAULT_RUBRIC.task_types
        ),
    )
    monkeypatch.setattr(diagnosis_module, "MAX_DIAGNOSIS_PROVIDER_REQUESTS", 1)
    gateway = BatchGateway()

    result = _run(gateway, rubric=rubric)

    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert evidence["mode"] == "bounded_sequential_fallback"
    assert evidence["complete"] is False
    assert evidence["provider_requests"] == 1
    assert result["original_kept"] is True
    assert "incomplete" in result["report"]["summary"].lower()


def test_fanout_uses_selected_models_current_context_limit() -> None:
    gateway = BatchGateway()
    gateway.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=8_192),)
    )

    result = _run(gateway)

    evidence = result["report"]["diagnosis"]["request_evidence"]
    assert evidence["mode"] == "bounded_sequential_fallback"
    assert evidence["reason"] == "speculative_request_exceeds_provider_limits"
    assert (
        evidence["provider_requests"]
        <= diagnosis_module.MAX_DIAGNOSIS_PROVIDER_REQUESTS
    )
    for batch in _diagnosis_batches(gateway):
        _, envelope = batch_decision_payload(batch, model=JEV_MODEL)
        assert len(json.dumps(envelope, ensure_ascii=False).encode()) <= 7_168


def test_sequential_dispatch_splits_oversized_questions_and_keeps_request_cap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(diagnosis_module, "MAX_DIAGNOSIS_PROVIDER_REQUESTS", 4)
    gateway = BatchGateway(root="communication")
    gateway.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=3_000),)
    )

    report = Diagnoser(
        gateway, speculative_fanout=False, record_request_evidence=True
    ).diagnose("Write a brief note. Keep it clear.")

    assert report.request_evidence is not None
    assert report.request_evidence["mode"] == "bounded_sequential_fallback"
    assert report.request_evidence["provider_requests"] == 4
    assert report.request_evidence["complete"] is False
    for batch in gateway.batches:
        _, envelope = batch_decision_payload(batch, model=JEV_MODEL)
        assert len(json.dumps(envelope, ensure_ascii=False).encode()) <= 1_976


def test_unobserved_sequential_diagnosis_reports_incomplete_dispatch() -> None:
    complete = _run(BatchGateway(), speculative=False)
    assert "request_evidence" not in complete["report"]["diagnosis"]

    gateway = BatchGateway()
    gateway.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=1_100),)
    )
    incomplete = _run(gateway, speculative=False)

    assert incomplete["original_kept"] is True
    assert "incomplete" in incomplete["report"]["summary"].lower()
    assert incomplete["report"]["diagnosis"]["request_evidence"]["complete"] is False


def test_sequential_missing_answer_reports_incomplete_diagnosis() -> None:
    class MissingAnswerGateway(BatchGateway):
        def _answer(self, request, **kwargs):
            if request.get("key") == "task_type":
                return None
            return super()._answer(request, **kwargs)

    result = _run(MissingAnswerGateway(), speculative=False)

    assert result["original_kept"] is True
    assert "incomplete" in result["report"]["summary"].lower()
    assert result["report"]["diagnosis"]["request_evidence"]["complete"] is False


def test_request_sizing_uses_a_conservative_fallback_without_model_metadata() -> None:
    request = {"key": "large", "state": "x" * 100_000, "query": "Is it clear?"}
    known = BatchGateway()
    known.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=200_000),)
    )

    assert Diagnoser(known)._request_fits([request]) is True
    assert Diagnoser(BatchGateway())._request_fits([request]) is False


def test_over_character_cap_returns_incomplete_without_inference() -> None:
    gateway = BatchGateway()
    prompt = "x" * (diagnosis_module.MAX_DIAGNOSIS_INPUT_CHARACTERS + 1)

    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        prompt, {}
    )

    assert gateway.batches == []
    assert result["report"]["requirements"]["release_eligible"] is False
    assert result["original_kept"] is True
    assert result["report"]["diagnosis"]["request_evidence"]["complete"] is False
    assert result["report"]["diagnosis"]["sentence_protocol_version"] == (
        diagnosis_module.SENTENCE_DIAGNOSIS_PROTOCOL_VERSION
    )


def test_provider_failure_keeps_original_with_incomplete_diagnosis() -> None:
    class FailedGateway(BatchGateway):
        def decide_batch(self, requests, *, role="judge", run_id=None):
            self.batches.append([dict(request) for request in requests])
            raise ProviderError("scripted", JEV_MODEL, None, "unavailable")

    gateway = FailedGateway()

    result = _run(gateway)

    assert len(_diagnosis_batches(gateway)) == 1
    assert (
        result["report"]["requirements"]["whole_source_audit"]["status"] == "unresolved"
    )
    assert result["status"] == "failed"
    assert result["report"]["outcome"] == "failed_operational"
    assert result["report"]["failure"]["kind"] == "incomplete_diagnosis"
    assert result["original_kept"] is True
    assert (
        result["report"]["diagnosis"]["request_evidence"]["reason"]
        == "diagnosis_provider_error"
    )


def test_http_transport_counts_physical_diagnosis_requests_including_retry() -> None:
    class Transport:
        def __init__(self, retry_once: bool) -> None:
            self.requests: list[dict] = []
            self.retry_once = retry_once

        def request(self, url, **kwargs):
            self.requests.append({"url": url, **kwargs})
            if self.retry_once and len(self.requests) == 1:
                return {"status_code": 429, "json": {}, "headers": {}}
            answers = {}
            for key, question in kwargs["json"]["questions"].items():
                if question["type"] == "choice":
                    selected = "general" if key == "task_type" else "none"
                    answers[key] = {
                        "type": "choice",
                        "choice": selected,
                        "probabilities": {selected: 1.0},
                        "confidence": 1.0,
                    }
                else:
                    answers[key] = {"type": "noul", "noul": 0.01}
            return {
                "status_code": 200,
                "json": {
                    "model": JEV_MODEL,
                    "answers": answers,
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
                "headers": {},
            }

    for retry in (False, True):
        transport = Transport(retry)
        gateway = HttpGateway(
            transport,
            config=GatewayConfig(openrouter_api_key="test", max_retries=int(retry)),
            sleep=lambda _seconds: None,
        )

        diagnosis = Diagnoser(gateway).diagnose("Write a brief note.")

        assert len(transport.requests) == 1 + int(retry)
        assert diagnosis.request_evidence["provider_requests"] == 1 + int(retry)
        assert diagnosis.request_evidence["latency_source"] == "measured_provider"


def test_fanout_recording_strictly_replays_and_old_sequential_bundle_still_loads(
    tmp_path: Path,
    deterministic_active_clock,
) -> None:
    deterministic_active_clock.step_per_round_s = 80.0
    for speculative, observe_sequential in (
        (True, False),
        (False, False),
        (False, True),
    ):
        path = tmp_path / f"{speculative}-{observe_sequential}.json"
        recording = RecordingGateway(BatchGateway(pointer="s0001"), path)
        original = PromptOptimizer(
            gateway=recording,
            store=RunStore(":memory:"),
            speculative_diagnosis=speculative,
            observe_sequential_diagnosis=observe_sequential,
        ).optimize(
            "Write a brief note. Keep it clear.",
            {"clarification_allowed": False},
        )

        replay = default_engine_factory(path)
        replay.store = RunStore(":memory:")
        restored = replay.optimize(
            "Write a brief note. Keep it clear.",
            {"clarification_allowed": False},
        )

        assert replay.speculative_diagnosis is speculative
        assert replay.observe_sequential_diagnosis is observe_sequential
        assert restored["status"] == original["status"]
        assert restored["final_prompt"] == original["final_prompt"]
        assert (
            restored["report"]["diagnosis"]["problem_sentences"]
            == original["report"]["diagnosis"]["problem_sentences"]
        )


def test_recording_replays_the_provider_limit_and_diagnosis_protocols(
    tmp_path: Path,
    deterministic_active_clock,
) -> None:
    deterministic_active_clock.step_per_round_s = 80.0
    path = tmp_path / "bounded.json"
    gateway = BatchGateway()
    gateway.catalog = StaticModelCatalog(
        (), (ModelInfo(JEV_MODEL, "openrouter", context_window=8_192),)
    )
    recording = RecordingGateway(gateway, path)
    original = PromptOptimizer(
        gateway=recording,
        store=RunStore(":memory:"),
        sentence_diagnosis_version=2,
        task_taxonomy_version=2,
    ).optimize(
        "Write a brief note. Keep it clear.",
        {"clarification_allowed": False},
    )
    bundle = json.loads(path.read_text())
    assert bundle["diagnosis_request_byte_limit"] == 7_168
    assert bundle["sentence_diagnosis_version"] == 2
    assert bundle["task_taxonomy_version"] == 2

    replay = default_engine_factory(path)
    replay.store = RunStore(":memory:")
    restored = replay.optimize(
        "Write a brief note. Keep it clear.",
        {"clarification_allowed": False},
    )
    restored_evidence = restored["report"]["diagnosis"]["request_evidence"]
    original_evidence = original["report"]["diagnosis"]["request_evidence"]
    assert restored_evidence["mode"] == original_evidence["mode"]
    assert (
        restored_evidence["provider_requests"] == original_evidence["provider_requests"]
    )
    assert restored_evidence["reason"] == original_evidence["reason"]


def test_recorded_http_retry_reservation_matches_bounded_diagnosis_replay(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(diagnosis_module, "MAX_DIAGNOSIS_PROVIDER_REQUESTS", 4)

    class Transport:
        def request(self, _url, **kwargs):
            answers = {}
            for key, question in kwargs["json"]["questions"].items():
                if question["type"] == "choice":
                    selected = "general" if key == "task_type" else "none"
                    answers[key] = {
                        "type": "choice",
                        "choice": selected,
                        "probabilities": {selected: 1.0},
                    }
                else:
                    answers[key] = {"type": "noul", "noul": 0.01}
            return {
                "status_code": 200,
                "json": {
                    "model": JEV_MODEL,
                    "answers": answers,
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
                "headers": {},
            }

    path = tmp_path / "http-retry-budget.json"
    gateway = HttpGateway(
        Transport(),
        config=GatewayConfig(openrouter_api_key="test", max_retries=1),
        catalog=StaticModelCatalog(
            (), (ModelInfo(JEV_MODEL, "openrouter", context_window=3_000),)
        ),
        sleep=lambda _seconds: None,
    )
    original = PromptOptimizer(
        gateway=RecordingGateway(gateway, path),
        store=RunStore(":memory:"),
        speculative_diagnosis=False,
        writer_instruction_version=6,
    ).optimize(
        "Write a brief note. Keep it clear.",
        {"clarification_allowed": False},
    )
    bundle = json.loads(path.read_text())
    assert bundle["diagnosis_retry_reservation_multiplier"] == 2
    assert "cascade_settings" not in bundle
    assert original["report"]["diagnosis"]["request_evidence"]["complete"] is False

    replay = default_engine_factory(path)
    replay.store = RunStore(":memory:")
    restored = replay.optimize(
        "Write a brief note. Keep it clear.",
        {"clarification_allowed": False},
    )

    assert restored["status"] == original["status"]
    assert restored["final_prompt"] == original["final_prompt"]
    assert (
        restored["report"]["diagnosis"]["request_evidence"]["reason"]
        == original["report"]["diagnosis"]["request_evidence"]["reason"]
    )
