"""Versioned recovery at the writer, run, and recording boundaries."""

from __future__ import annotations

import json

import pytest

from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.success_tests import SuccessTestCompiler

PROMPT = "Summarize the report in two sentences."
GOOD_TESTS = json.dumps(
    {"tests": [{"question": "Is it two sentences?", "kind": "noul", "expected": "yes"}]}
)


def accepting_decision(request, **_kwargs):
    return {
        "type": "noul",
        "noul": 0.0 if request["key"].endswith("evaluator_instructions") else 0.99,
        "confidence": 0.99,
    }


@pytest.mark.parametrize(
    "unreadable",
    [
        '{"tests": "bad\\escape"}',
        'Sure: {"outer": invalid, "nested": {"tests": []}}',
    ],
)
@pytest.mark.parametrize("wrapper", ["{}", "Here you go: {}", "```json\n{}\n```"])
def test_compiler_recovers_an_unreadable_reply_before_screening(
    unreadable: str, wrapper: str
) -> None:
    replies = iter([unreadable, wrapper.format(GOOD_TESTS)])
    requests = []

    def chat(_model, messages, **_kwargs):
        requests.append(messages)
        return next(replies)

    compiler = SuccessTestCompiler(
        ScriptedGateway(chat=chat, decision=accepting_decision), instruction_version=14
    )
    compiled = compiler.compile(PROMPT)

    assert [test.question for test in compiled.tests] == ["Is it two sentences?"]
    assert len(compiled.screening_checks) == 1
    assert [item["outcome"] for item in compiler.writer_attempts] == [
        "invalid_response",
        "success",
    ]
    first_state = json.loads(requests[0][1]["content"])
    retry_state = json.loads(requests[1][1]["content"])
    assert first_state == {"prompt": PROMPT}
    assert retry_state == {
        "prompt": PROMPT,
        "writer_reply_retry": {"operation": "success_tests", "attempt": 2},
    }
    assert requests[0][0] == requests[1][0]


def test_compiler_rejects_malformed_items_without_losing_valid_siblings() -> None:
    raw_items = [
        {"kind": "noul", "expected": "yes"},
        {"question": "Is it two sentences?", "kind": "noul", "expected": "yes"},
        "not an object",
        {"question": None},
        {"question": "Is it brief?", "kind": "invented"},
    ]
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: json.dumps({"tests": raw_items}),
        decision=accepting_decision,
    )
    compiler = SuccessTestCompiler(gateway, instruction_version=14)
    compiled = compiler.compile(PROMPT)

    assert [test.id for test in compiled.tests] == ["t1"]
    rejected = compiled.as_dict()["rejected"]
    assert [item["test_id"] for item in rejected] == ["t0", "t2", "t3", "t4"]
    assert [item["raw_item"] for item in rejected] == [
        raw_items[i] for i in (0, 2, 3, 4)
    ]
    assert [item["reason"] for item in rejected] == [
        "each success test must have a question",
        "each success test must be an object",
        "each success test must have a question",
        "unsupported success test kind",
    ]
    assert all(
        item["test"] is None and item["faithful_probability"] is None
        for item in rejected
    )
    assert [check.test_id for check in compiled.screening_checks] == ["t1"]
    assert len(compiler.writer_attempts) == 1


@pytest.mark.parametrize(
    "bad_reply", ["{}", "[]", "null", '{"clarify":""}', "not JSON", "", {"choices": []}]
)
def test_candidate_writer_recovers_an_unusable_reply(bad_reply) -> None:
    from prompt_enhancer.rewrite import CandidateWriter
    from prompt_enhancer.strategies import CandidateBatchRequest, RewriteStrategy

    replies = iter(
        [bad_reply, '{"clarify":"Summarize the report using two sentences."}']
    )
    writer = CandidateWriter(
        ScriptedGateway(chat=lambda *_args, **_kwargs: next(replies)),
        instruction_version=14,
    )
    result = writer.generate_candidates(
        CandidateBatchRequest(
            PROMPT, (RewriteStrategy("clarify", "safe", "Clarify the request."),)
        )
    )

    assert result == {"clarify": "Summarize the report using two sentences."}
    assert [item["outcome"] for item in writer.writer_attempts] == [
        "invalid_response",
        "success",
    ]
    assert writer.writer_attempts[0]["reason"] in {
        "invalid_reply_shape",
        "invalid_json",
        "empty_completion",
    }


def test_exhausted_writer_replies_remain_operational_and_are_saved() -> None:
    from test_always_attempt import _gateway

    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.store import RunStore

    gateway = _gateway()
    gateway.chat_handler = lambda *_args, **_kwargs: {"choices": []}
    store = RunStore(":memory:")
    result = PromptOptimizer(gateway=gateway, store=store).optimize(
        "whats 2 plus 2", {"clarification_allowed": False}
    )

    assert result["report"]["outcome"] == "failed_operational"
    assert result["report"]["failure"]["kind"] == "invalid_response"
    attempts = result["report"]["writer_attempts"]
    assert [item["attempt"] for item in attempts] == [1, 2]
    assert all(
        item["round"] == 1 and item["reason"] == "empty_completion" for item in attempts
    )
    saved = store.get_run(result["run_id"])
    assert saved["result"]["report"]["writer_attempts"] == attempts


def test_recovered_run_reports_attempts_and_cost_in_round_history() -> None:
    from test_always_attempt import _gateway

    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.store import RunStore

    gateway = _gateway()
    normal_chat = gateway.chat_handler

    def chat(model, messages, *, role, **kwargs):
        if role == "writer":
            gateway.usage.record(role=role, provider="scripted", model=model, cost=1.25)
            state = json.loads(messages[1]["content"])
            if (
                "state.strategies" in messages[0]["content"]
                and "writer_reply_retry" not in state
            ):
                return "not JSON"
        return normal_chat(model, messages, role=role, **kwargs)

    gateway.chat_handler = chat
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        "whats 2 plus 2", {"clarification_allowed": False}
    )

    assert result["status"] == "completed"
    attempts = result["report"]["writer_attempts"]
    assert [(item["operation"], item["outcome"]) for item in attempts] == [
        ("success_tests", "success"),
        ("candidates", "invalid_response"),
        ("candidates", "success"),
    ]
    assert result["cost"]["cost_by_role"]["writer"] == 5.0  # includes source extraction
    assert result["report"]["history"][0]["evidence"]["writer_attempts"] == attempts


def test_retry_answers_record_and_replay_with_distinct_request_keys(tmp_path) -> None:
    from prompt_enhancer.evaluation.capture_audit import validate_capture
    from prompt_enhancer.evaluation.recording import RecordingGateway
    from prompt_enhancer.gateway import ReplayGateway

    expected_keys = []
    replies = iter(["not JSON", '{"tests":[]}'])

    def chat(model, messages, *, role, **_kwargs):
        expected_keys.append(
            ReplayGateway.request_key(
                "chat", model, {"model": model, "messages": list(messages)}, role
            )
        )
        return next(replies)

    path = tmp_path / "recording.json"
    recorded = RecordingGateway(ScriptedGateway(chat=chat), path)
    original = SuccessTestCompiler(recorded, instruction_version=14).compile(PROMPT)
    bundle = json.loads(path.read_text())
    receipt = validate_capture(
        bundle["request_captures"], expected_keys, bundle["responses"]
    )
    assert receipt == {"status": "complete", "request_count": 2, "response_count": 2}
    assert expected_keys[0] != expected_keys[1]
    replay = ReplayGateway(bundle["responses"])
    repeated = SuccessTestCompiler(replay, instruction_version=14).compile(PROMPT)
    assert repeated.as_dict() == original.as_dict()
    assert replay.replayed_keys == expected_keys


def test_http_adapter_accounts_for_billed_empty_reply_and_retry() -> None:
    from test_gateway import QueueTransport, Response

    from prompt_enhancer.gateway import GatewayConfig, HttpGateway

    transport = QueueTransport(
        [
            Response(
                200,
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "cost": 0.02,
                    },
                },
            ),
            Response(
                200,
                {
                    "choices": [{"message": {"content": '{"tests":[]}'}}],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "cost": 0.03,
                    },
                },
            ),
        ]
    )
    gateway = HttpGateway(
        transport, config=GatewayConfig(openrouter_api_key="synthetic-test-key")
    )
    compiler = SuccessTestCompiler(
        gateway, writer_model="test-writer", instruction_version=14
    )
    compiled = compiler.compile(PROMPT)
    usage = gateway.usage_report()
    assert compiled.tests == ()
    assert usage["calls"] == 2
    assert usage["cost_by_role"]["writer"] == 0.05
    assert usage["tokens"]["input_tokens"] == 22
    assert usage["tokens"]["output_tokens"] == 7


@pytest.mark.parametrize(
    "bad_reply",
    [
        '{"tests": "bad\\escape"}',
        '{"tests": [] "extra": true}',
        "",
        "   ",
        {"choices": []},
        "null",
        '{"tests": null}',
        "{}",
    ],
)
def test_unusable_success_test_reply_has_one_recovery_call(bad_reply) -> None:
    replies = iter([bad_reply, GOOD_TESTS])
    compiler = SuccessTestCompiler(
        ScriptedGateway(
            chat=lambda *_args, **_kwargs: next(replies), decision=accepting_decision
        ),
        instruction_version=14,
    )
    assert compiler.compile(PROMPT).tests[0].question == "Is it two sentences?"
    assert [item["attempt"] for item in compiler.writer_attempts] == [1, 2]


@pytest.mark.parametrize("version", range(1, 14))
def test_older_writer_versions_do_not_retry_or_reject_items_locally(version) -> None:
    from prompt_enhancer.rewrite import CandidateWriter
    from prompt_enhancer.strategies import CandidateBatchRequest, RewriteStrategy

    calls = []

    def chat(_model, messages, **_kwargs):
        calls.append(messages)
        return '{"tests":[{"kind":"noul"}]}'

    compiler = SuccessTestCompiler(
        ScriptedGateway(chat=chat), instruction_version=version
    )
    with pytest.raises(ValueError, match="each success test must have a question"):
        compiler.compile(PROMPT)
    assert len(calls) == 1
    assert json.loads(calls[0][1]["content"]) == {"prompt": PROMPT}
    assert compiler.writer_attempts == []

    calls.clear()
    writer = CandidateWriter(ScriptedGateway(chat=chat), instruction_version=version)
    with pytest.raises(ValueError, match="omitted a selected strategy"):
        writer.generate_candidates(
            CandidateBatchRequest(
                PROMPT, (RewriteStrategy("clarify", "safe", "Clarify."),)
            )
        )
    assert len(calls) == 1
    assert "writer_reply_retry" not in json.loads(calls[0][1]["content"])
    assert writer.writer_attempts == []


def test_all_malformed_items_return_no_tests_without_retry_or_screening() -> None:
    raw_items = [
        {},
        {"question": "Choose?", "kind": "choice", "options": []},
        {"question": "Rate?", "kind": "score", "levels": []},
    ]
    compiler = SuccessTestCompiler(
        ScriptedGateway(
            chat=lambda *_args, **_kwargs: json.dumps({"tests": raw_items})
        ),
        instruction_version=14,
    )
    compiled = compiler.compile(PROMPT)
    assert compiled.tests == ()
    assert len(compiled.rejected) == 3
    assert compiled.screening_checks == ()
    assert [item["outcome"] for item in compiler.writer_attempts] == ["success"]


@pytest.mark.parametrize("fail_on_retry", [False, True])
def test_provider_failure_does_not_add_a_caller_retry(fail_on_retry) -> None:
    from prompt_enhancer.gateway import ProviderError

    failure = ProviderError("scripted", "writer", 503, role="writer")
    calls = 0

    def chat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if fail_on_retry and calls == 1:
            return "not JSON"
        raise failure

    compiler = SuccessTestCompiler(ScriptedGateway(chat=chat), instruction_version=14)
    with pytest.raises(ProviderError) as raised:
        compiler.compile(PROMPT)
    assert raised.value is failure
    assert calls == (2 if fail_on_retry else 1)
    assert compiler.writer_attempts[-1]["outcome"] == "provider_error"


def test_failed_candidate_retry_keeps_earlier_writer_attempts_in_run_report() -> None:
    from test_always_attempt import _gateway

    from prompt_enhancer.gateway import ProviderError
    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.store import RunStore

    gateway = _gateway()
    normal_chat = gateway.chat_handler

    def chat(model, messages, *, role, **kwargs):
        if role == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            if "writer_reply_retry" in state:
                raise ProviderError("scripted", model, 503, role="writer")
            return "not JSON"
        return normal_chat(model, messages, role=role, **kwargs)

    gateway.chat_handler = chat
    result = PromptOptimizer(gateway=gateway, store=RunStore(":memory:")).optimize(
        "whats 2 plus 2", {"clarification_allowed": False}
    )
    assert result["report"]["outcome"] == "failed_operational"
    assert [item["outcome"] for item in result["report"]["writer_attempts"]] == [
        "success",
        "invalid_response",
        "provider_error",
    ]
    assert result["report"]["failure"]["http_status"] == 503


def test_resuming_a_paused_retry_keeps_unfinished_round_attempts() -> None:
    from test_always_attempt import _gateway

    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.run_control import BudgetPaused
    from prompt_enhancer.store import RunStore

    gateway = _gateway()
    normal_chat = gateway.chat_handler
    paused_once = False

    def chat(model, messages, *, role, **kwargs):
        nonlocal paused_once
        if role == "writer":
            state = json.loads(messages[1]["content"])
            if not paused_once and "state.strategies" in messages[0]["content"]:
                if "writer_reply_retry" in state:
                    paused_once = True
                    raise BudgetPaused(
                        reason="spend_limit", history=(), spent_usd=2.5, elapsed_ms=1
                    )
                gateway.usage.record(
                    role=role, provider="scripted", model=model, cost=1.25
                )
                return "not JSON"
            gateway.usage.record(role=role, provider="scripted", model=model, cost=1.25)
        return normal_chat(model, messages, role=role, **kwargs)

    gateway.chat_handler = chat
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(gateway=gateway, store=store)
    paused = optimizer.optimize("whats 2 plus 2", {"clarification_allowed": False})
    assert paused["status"] == "needs_input"
    previous_attempts = paused["report"]["writer_attempts"]
    assert [item["outcome"] for item in previous_attempts] == [
        "success",
        "invalid_response",
    ]
    assert (
        paused["cost"]["cost_by_role"]["writer"] == 3.75
    )  # includes source extraction

    continued = optimizer.continue_run(paused["run_id"])
    assert continued["status"] == "completed", continued["report"]
    attempts = continued["report"]["writer_attempts"]
    assert attempts[: len(previous_attempts)] == previous_attempts
    assert [item["outcome"] for item in attempts] == [
        "success",
        "invalid_response",
        "success",
        "success",
    ]
    assert continued["cost"]["cost_by_role"]["writer"] == 6.25
    assert (
        store.get_run(paused["run_id"])["result"]["report"]["writer_attempts"]
        == attempts
    )
