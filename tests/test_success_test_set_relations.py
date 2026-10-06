"""Version 13 observes relationships without altering success-test acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import ProviderError, ReplayGateway, ScriptedGateway
from prompt_enhancer.success_tests import (
    MAX_SET_RELATION_REQUEST_BYTES,
    MAX_SET_RELATION_TESTS,
    SET_RELATION_CRITERIA,
    SUCCESS_TEST_SET_RELATION_VERSION,
    SuccessTestCompiler,
)


def _writer(count: int = 3) -> dict[str, Any]:
    tests = [
        {
            "question": f"Does the answer satisfy observable check {index}?",
            "kind": "noul",
            "expected": "yes",
        }
        for index in range(count)
    ]
    return {"choices": [{"message": {"content": json.dumps({"tests": tests})}}]}


def _choice(relation: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": relation,
        "probabilities": {
            option: 0.91 if option == relation else 0.03
            for option in SET_RELATION_CRITERIA
        },
        "confidence": 0.91,
    }


@pytest.mark.parametrize("observe_set_relations", [False, True])
@pytest.mark.parametrize("incomplete_proposal", [False, True])
def test_empty_proposals_report_relation_status_without_jev_requests(
    observe_set_relations: bool, incomplete_proposal: bool
) -> None:
    tests = (
        [
            {
                "question": "Which requested format does the answer use?",
                "kind": "choice",
                "expected": "json",
                "options": ["json", "text"],
            }
        ]
        if incomplete_proposal
        else []
    )
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": json.dumps({"tests": tests})}}]
        },
        decision=_screen_or_relation,
    )

    compiled = SuccessTestCompiler(
        gateway, observe_set_relations=observe_set_relations
    ).compile("Answer in JSON.")
    report = compiled.as_dict()

    assert compiled.tests == ()
    assert gateway.decision_log == []
    assert len(compiled.rejected) == int(incomplete_proposal)
    if incomplete_proposal:
        assert compiled.rejected[0].reason == "missing Choice descriptions"
    if observe_set_relations:
        assert report["set_relation_version"] == SUCCESS_TEST_SET_RELATION_VERSION
        assert report["set_relations"] == []
        assert report["set_relation_observation"] == {
            "status": "not_applicable",
            "accepted_tests": 0,
            "compared_tests": 0,
            "compared_pairs": 0,
            "omitted_pairs": 0,
            "gateway_batch_calls": 0,
        }
    else:
        assert "set_relation_version" not in report
        assert "set_relations" not in report
        assert "set_relation_observation" not in report


def _screen_or_relation(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    key = request["key"]
    if key.startswith("success-test-set:"):
        left, right = key.removeprefix("success-test-set:").split(":")
        return _choice("duplicate" if (left, right) == ("t0", "t1") else "conflict")
    return {
        "type": "noul",
        "noul": 0.01 if key.endswith("evaluator_instructions") else 0.99,
        "confidence": 0.99,
    }


def test_relations_record_both_hazards_without_changing_accepted_tests() -> None:
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: _writer(), decision=_screen_or_relation
    )

    compiled = SuccessTestCompiler(gateway, observe_set_relations=True).compile(
        "Answer all three checks."
    )
    report = compiled.as_dict()

    assert [test.id for test in compiled.tests] == ["t0", "t1", "t2"]
    assert compiled.rejected == ()
    assert report["set_relation_version"] == SUCCESS_TEST_SET_RELATION_VERSION
    assert [row["relation"] for row in report["set_relations"]] == [
        "conflict",
        "conflict",
        "duplicate",
    ]
    assert report["set_relation_observation"]["status"] == "complete"
    assert all(
        row["answering_snapshot"] == gateway.jev_model
        for row in report["set_relations"]
    )
    requests = [
        entry["question"]
        for entry in gateway.decision_log
        if entry["question"]["key"].startswith("success-test-set:")
    ]
    assert len(requests) == 3
    assert all(
        request["state"]["prompt"] == "Answer all three checks." for request in requests
    )
    assert all(
        "Answer all three checks." not in json.dumps(request["query"])
        for request in requests
    )


def test_relation_failure_is_visible_and_does_not_discard_tests() -> None:
    def decide(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if request["key"].startswith("success-test-set:"):
            raise ProviderError("openrouter", "jev", None, kind="unavailable")
        return _screen_or_relation(request)

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: _writer(2), decision=decide
    )
    compiled = SuccessTestCompiler(gateway, observe_set_relations=True).compile(
        "Answer both checks."
    )

    assert len(compiled.tests) == 2
    assert compiled.set_relations[0]["status"] == "provider_error"
    assert compiled.set_relation_observation["status"] == "provider_error"
    assert compiled.set_relation_observation["error"] == "unavailable"


def test_relation_budget_and_legacy_path_are_explicit() -> None:
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: _writer(MAX_SET_RELATION_TESTS + 1),
        decision=_screen_or_relation,
    )
    observed = SuccessTestCompiler(gateway, observe_set_relations=True).compile(
        "Answer each check."
    )
    assert len(observed.tests) == MAX_SET_RELATION_TESTS + 1
    assert observed.set_relation_observation["status"] == "partial"
    assert observed.set_relation_observation["omitted_pairs"] == MAX_SET_RELATION_TESTS

    legacy_gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: _writer(2), decision=_screen_or_relation
    )
    legacy = SuccessTestCompiler(legacy_gateway).compile("Answer both checks.")
    assert "set_relations" not in legacy.as_dict()
    assert not any(
        entry["question"]["key"].startswith("success-test-set:")
        for entry in legacy_gateway.decision_log
    )


def test_relation_requests_replay_with_exact_keys_and_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "success-test-set.json"
    source = ScriptedGateway(
        chat=lambda *_args, **_kwargs: _writer(2), decision=_screen_or_relation
    )
    recording = RecordingGateway(source, path)
    recorded = SuccessTestCompiler(recording, observe_set_relations=True).compile(
        "Answer both checks."
    )
    bundle = json.loads(path.read_text())
    replay = ReplayGateway(
        bundle["responses"],
        decision_provenance=bundle["decision_provenance"],
        jev_model=bundle["jev_model"],
    )
    repeated = SuccessTestCompiler(replay, observe_set_relations=True).compile(
        "Answer both checks."
    )

    assert repeated.as_dict() == recorded.as_dict()
    assert len(replay.replayed_keys) == 10  # one writer, eight screens, one pair
    assert recorded.set_relations[0]["answering_snapshot"] == source.jev_model

    incomplete = dict(bundle["responses"])
    incomplete.pop(replay.replayed_keys[-1])
    missing = ReplayGateway(
        incomplete,
        decision_provenance=bundle["decision_provenance"],
        jev_model=bundle["jev_model"],
    )
    with pytest.raises(ProviderError, match="no recorded response"):
        SuccessTestCompiler(missing, observe_set_relations=True).compile(
            "Answer both checks."
        )


def test_unusable_relation_answer_is_reported_without_discarding_tests() -> None:
    def decide(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if request["key"].startswith("success-test-set:"):
            return {"type": "noul", "noul": 0.99, "confidence": 0.99}
        return _screen_or_relation(request)

    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: _writer(2), decision=decide
    )
    compiled = SuccessTestCompiler(gateway, observe_set_relations=True).compile(
        "Answer both checks."
    )

    assert len(compiled.tests) == 2
    assert compiled.set_relations[0]["status"] == "invalid_response"
    assert compiled.set_relation_observation["status"] == "incomplete"


def test_oversized_relation_request_is_not_sent() -> None:
    def writer(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        tests = [
            {
                "question": "Does the answer include " + "x" * 25_000 + "?",
                "expected": "yes",
            },
            {
                "question": "Does the answer include " + "y" * 25_000 + "?",
                "expected": "yes",
            },
        ]
        return {"choices": [{"message": {"content": json.dumps({"tests": tests})}}]}

    gateway = ScriptedGateway(chat=writer, decision=_screen_or_relation)
    compiled = SuccessTestCompiler(gateway, observe_set_relations=True).compile(
        "Answer both checks."
    )

    assert len(compiled.tests) == 2
    assert compiled.set_relations == ()
    assert compiled.set_relation_observation["status"] == "request_budget_exceeded"
    assert compiled.set_relation_observation["serialized_input_bytes_estimate"] > (
        MAX_SET_RELATION_REQUEST_BYTES
    )
    assert compiled.set_relation_observation["omitted_pairs"] == 1
    assert not any(
        entry["question"]["key"].startswith("success-test-set:")
        for entry in gateway.decision_log
    )
