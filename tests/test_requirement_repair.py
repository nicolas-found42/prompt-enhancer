"""Rejected drafts and replacement checks through optimize/jobs/history."""

import json

import pytest
from active_clock import TickingClock, advancing_chat
from test_compound_requirements import compound_gateway

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.jobs import RunJobs


@pytest.mark.parametrize("replacement_passes", [True, False])
def test_rejected_proposal_is_replaced_without_deleting_requirement(
    tmp_path, replacement_passes
):
    prompt = "Reply with exactly PING and nothing else."
    gateway = compound_gateway(
        prompt,
        [],
        output="PING",
        candidate="Respond with exactly PING and nothing else.",
    )
    chat, decide = gateway.chat_handler, gateway.decision_handler
    generations = []

    def generate(model, messages, **params):
        if params["role"] == "writer" and "Compile the user" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            generations.append(state)
            question = (
                "Does the answer contain PING?"
                if state.get("attempt", 1) == 1
                else "Does the complete answer equal PING?"
            )
            return json.dumps(
                {"tests": [{"question": question, "kind": "noul", "expected": "yes"}]}
            )
        return chat(model, messages, **params)

    def screen(request, **params):
        key = str(request.get("key", ""))
        if key.startswith("requirement:test-binding:"):
            return {
                "data": {
                    "noul": {
                        "probability": 0.99,
                        "certainty": 0.99,
                        "reasoning": "private diagnostics",
                    }
                }
            }
        if key.startswith("success-test-screen:"):
            dimension = request["dimension"]
            bad = not replacement_passes or any(
                "contain PING" in str(v)
                for v in request["state"]["success_tests"].values()
            )
            probability = (
                0.99
                if dimension in {"faithfulness", "no_invention", "assessability"}
                else 0.01
            )
            if dimension == "assessability" and bad:
                probability = 0.01
            return {"type": "noul", "probability_true": probability, "confidence": 1.0}
        if key.startswith("grade_"):
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        return decide(request, **params)

    gateway.chat_handler, gateway.decision_handler = generate, screen
    store = RunStore(tmp_path / "regeneration.sqlite")
    result = PromptOptimizer(store=store, gateway=gateway).optimize(
        prompt, {"time_limit_s": 0}
    )
    evidence = result["report"]["history"][0]["evidence"]["test_screening"][
        "source_checks"
    ]
    assert len(generations) == len(evidence["attempts"]) == 2
    assert generations[1]["rejected_proposals"][0]["requirement_ids"]
    assert evidence["attempts"][0]["rejected"]
    assert evidence["attempts"][0]["bindings"][0]["raw_decision"] == {
        "data": {"noul": {"probability": 0.99, "certainty": 0.99}}
    }
    assert evidence["coverage"][0]["status"] == (
        "tested" if replacement_passes else "unresolved"
    )
    assert result["report"]["requirements"]["requirements"][0]["source"] == prompt
    assert store.get_run(result["run_id"])["result"]["report"] == json.loads(
        json.dumps(result["report"])
    )


def test_draft_repair_carries_own_failure_evidence_and_retests_after_reload(tmp_path):
    prompt = "Write an invitation using {date}."
    bad = "Compose an invitation using tomorrow."
    good = "Compose an invitation using {date}."
    gateway = compound_gateway(prompt, [], candidate=bad, output="An invitation.")
    chat = gateway.chat_handler
    requests = []

    def repair(model, messages, **params):
        if params["role"] == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            requests.append(state)
            candidate = good if state.get("repair_evidence") else bad
            return json.dumps({item["name"]: candidate for item in state["strategies"]})
        return chat(model, messages, **params)

    clock = TickingClock()
    gateway.chat_handler = advancing_chat(repair, clock, 1)
    path = tmp_path / "repair.sqlite"
    store = RunStore(path)
    optimizer = PromptOptimizer(store=store, gateway=gateway, clock=clock)
    jobs = RunJobs(store=store, monotonic=clock)
    jobs.submit(
        "repair",
        "optimize",
        lambda progress, _cancel, _observe: optimizer.optimize(
            prompt, run_id="repair", progress=progress
        ),
        lambda _exc: {},
        prompt=prompt,
    )
    done = jobs.wait("repair", timeout=60)
    assert done["result"]["final_prompt"] == good
    repair_state = next(item for item in requests if item.get("repair_evidence"))
    failure = repair_state["repair_evidence"][0]
    assert failure["candidate_prompt"] == bad
    assert failure["requirement_findings"][0]["requirement_id"]
    assert any(
        item["kind"] == "repair" and item["draft"] == good for item in done["events"]
    )
    assert any(item["kind"] == "retest" and item["checks"] for item in done["events"])
    restored = RunJobs(store=RunStore(path)).get("repair")
    assert restored["events"] == done["events"]


def test_deadline_during_replacement_retains_the_rejected_physical_proposal(tmp_path):
    from prompt_enhancer.run_control import RunDeadlineReached

    clock = TickingClock()
    prompt = "Reply with exactly PING and nothing else."
    gateway = compound_gateway(
        prompt,
        [],
        output="PONG",
        candidate="Respond with exactly PING and nothing else.",
    )
    chat, decide = gateway.chat_handler, gateway.decision_handler

    def deadline(model, messages, **params):
        if params["role"] == "writer" and "Compile the user" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            if state.get("attempt") == 2:
                clock.now = 151
                raise RunDeadlineReached(
                    history=(), spent_usd=0, elapsed_ms=151_000, deadline_s=150
                )
            return '{"tests":[{"question":"Does the answer contain PING?","kind":"noul","expected":"yes"}]}'
        return chat(model, messages, **params)

    def binding(request, **params):
        if str(request.get("key", "")).startswith("requirement:test-binding:"):
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        return decide(request, **params)

    gateway.chat_handler, gateway.decision_handler = deadline, binding
    store = RunStore(tmp_path / "deadline.sqlite")
    result = PromptOptimizer(store=store, gateway=gateway, clock=clock).optimize(prompt)
    assert result["original_kept"] and result["report"]["outcome"] is None
    assert result["report"]["control_state"] == "deadline_reached"
    retained = result["report"]["test_screening"]
    assert retained["attempts"][0]["rejected"]
    assert retained["coverage"][0]["status"] == "unresolved"
    assert (
        store.get_run(result["run_id"])["result"]["report"]["test_screening"]
        == retained
    )


def test_repeated_failed_repairs_reach_deadline_and_keep_every_completed_round(
    tmp_path,
):
    prompt = "Write an invitation using {date}."
    clock = TickingClock()
    gateway = compound_gateway(
        prompt, [], candidate="Compose an invitation using tomorrow."
    )
    gateway.chat_handler = advancing_chat(gateway.chat_handler, clock, 10)
    store = RunStore(tmp_path / "failed-repair.sqlite")
    jobs = RunJobs(store=store, monotonic=clock)
    optimizer = PromptOptimizer(store=store, gateway=gateway, clock=clock)
    jobs.submit(
        "failed-repair",
        "optimize",
        lambda progress, _cancel, _observe: optimizer.optimize(
            prompt, run_id="failed-repair", progress=progress
        ),
        lambda _exc: {},
        prompt=prompt,
    )
    # This wall guard detects a hung worker; the injected clock still proves
    # the product's 150-second active deadline, even under coverage overhead.
    done = jobs.wait("failed-repair", timeout=60)
    result = done["result"]
    assert result["report"]["control_state"] == "deadline_reached"
    assert result["final_prompt"] == prompt and result["report"]["outcome"] is None
    assert len(result["report"]["history"]) > 1
    assert any(event["kind"] == "repair" for event in done["events"])
    assert not any(event["kind"] == "qualified" for event in done["events"])
    assert (
        RunJobs(store=RunStore(tmp_path / "failed-repair.sqlite")).get("failed-repair")[
            "events"
        ]
        == done["events"]
    )


@pytest.mark.parametrize("binding", [0.01, None])
def test_unbound_accepted_proposal_is_retained_but_excluded_from_grading(
    tmp_path, binding
):
    prompt = "Reply with exactly PING and nothing else."
    gateway = compound_gateway(
        prompt,
        [],
        output="PING",
        candidate="Respond with exactly PING and nothing else.",
    )
    chat, decide = gateway.chat_handler, gateway.decision_handler

    def generate(model, messages, **params):
        if params["role"] == "writer" and "Compile the user" in messages[0]["content"]:
            return '{"tests":[{"question":"Does the answer equal PING and mention bananas?","kind":"noul","expected":"yes"}]}'
        return chat(model, messages, **params)

    def screen(request, **params):
        key = str(request.get("key", ""))
        if key.startswith("requirement:test-binding:"):
            return (
                None
                if binding is None
                else {"type": "noul", "probability_true": binding, "confidence": 1.0}
            )
        if key.startswith("success-test-screen:"):
            probability = (
                0.99
                if request["dimension"]
                in {"faithfulness", "no_invention", "assessability"}
                else 0.01
            )
            return {"type": "noul", "probability_true": probability, "confidence": 1.0}
        return decide(request, **params)

    gateway.chat_handler, gateway.decision_handler = generate, screen
    result = PromptOptimizer(
        store=RunStore(tmp_path / "unbound.sqlite"), gateway=gateway
    ).optimize(prompt, {"time_limit_s": 0})
    evidence = result["report"]["history"][0]["evidence"]["test_screening"]
    assert evidence["tests"] == []
    source_checks = evidence["source_checks"]
    assert len(source_checks["attempts"]) == 2
    assert all(attempt["tests"] for attempt in source_checks["attempts"])
    assert all(attempt["binding_rejected"] for attempt in source_checks["attempts"])
    assert source_checks["coverage"][0]["status"] == "unresolved"
    assert evidence["rejected"]
    assert not any(
        str(entry["question"].get("key", "")).startswith("grade_")
        for entry in gateway.decision_log
    )


def test_source_bindings_are_size_bounded_batches_and_checkpoint_partial_answers():
    from prompt_enhancer.requirement_tests import compile_source_tests
    from prompt_enhancer.requirements import extract_requirements
    from prompt_enhancer.success_tests import SuccessTestCompiler

    prompt = "Write using {one}, {two}, {three}, {four}, {five}, {six}."
    requirements = extract_requirements(prompt)
    gateway = compound_gateway(prompt, [])
    chat, decide_batch = gateway.chat_handler, gateway.decide_batch
    decide = gateway.decision_handler

    def faithful(request, **params):
        if str(request.get("key", "")).startswith("faithful:"):
            return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}
        return decide(request, **params)

    gateway.decision_handler = faithful
    batches, checkpoints = [], []

    def generate(model, messages, **params):
        if "Compile the user" in messages[0]["content"]:
            return json.dumps(
                {
                    "tests": [
                        {
                            "question": f"Does the answer use {{{word}}}?",
                            "kind": "noul",
                            "expected": "yes",
                        }
                        for word in ("one", "two", "three", "four", "five", "six")
                    ]
                }
            )
        return chat(model, messages, **params)

    def batched(requests, **params):
        if all(
            str(item.get("key", "")).startswith("requirement:test-binding:")
            for item in requests
        ):
            batches.append(requests)
            return [{"type": "noul", "probability_true": 0.99, "confidence": 1.0}] * (
                len(requests) - 1
            )
        return decide_batch(requests, **params)

    gateway.chat_handler, gateway.decide_batch = generate, batched
    compiler = SuccessTestCompiler(
        gateway, writer_model="test", screen_protocol_version=1
    )
    compiled, evidence = compile_source_tests(
        compiler, prompt, requirements, on_evidence=checkpoints.append
    )
    assert compiled.tests
    assert len(batches) > 1 and all(len(batch) <= 8 for batch in batches)
    assert all(
        len(json.dumps(batch, ensure_ascii=False).encode()) < 48_000
        for batch in batches
    )
    bindings = evidence["attempts"][0]["bindings"]
    assert len(bindings) == 6 * len(requirements)
    assert len({item["request_id"] for item in bindings}) == len(bindings)
    assert any(
        item["status"] == "unresolved" and item["raw_decision"] is None
        for item in bindings
    )
    assert any(
        any(
            item["status"] == "unresolved" for item in record["attempts"][0]["bindings"]
        )
        for record in checkpoints
    )


def test_lossless_draft_does_not_claim_it_consumed_ordinary_repair_evidence():
    from dataclasses import replace

    from test_rounds import _gateway, _plan

    from prompt_enhancer.requirements import extract_requirements
    from prompt_enhancer.rounds import run_round

    prompt = "Read the background notes. Write an invitation using {date}."
    gateway = _gateway(tests='{"tests":[]}')
    decide = gateway.decision_handler

    def role(request, **params):
        if str(request.get("key", "")).startswith("restructure_lossless:role:"):
            choice = (
                "context"
                if "background"
                in next(
                    unit["text"]
                    for unit in request["state"]["source_units"]
                    if unit["id"] == request["key"].rsplit(":", 1)[-1]
                )
                else "task"
            )
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        return decide(request, **params)

    gateway.decision_handler = role
    events = []
    outcome = run_round(
        gateway,
        replace(
            _plan(diagnosis={"confirmed_gaps": [], "problem_sentences": []}),
            prompt=prompt,
            working_prompt=prompt,
            applied_style="faithful_transform",
            writer_instruction_version=15,
            route_strategies=("restructure_lossless",),
            requirements=extract_requirements(prompt),
            repair_evidence=(
                {
                    "candidate_id": "old",
                    "candidate_prompt": "Use tomorrow.",
                    "requirement_findings": [
                        {"reason": "The date placeholder was dropped."}
                    ],
                },
            ),
        ),
        on_activity=events.append,
    )
    lossless = [
        item["candidate_id"]
        for item in outcome.candidates
        if item["strategy"] == "restructure_lossless"
    ]
    assert lossless
    assert any(
        event["kind"] == "checks" and event.get("candidate_id") in lossless
        for event in events
    )
    assert not any(
        event["kind"] in {"repair", "retest"} and event.get("candidate_id") in lossless
        for event in events
    )


@pytest.mark.parametrize("missing_binding", [True, False])
def test_screened_out_unbound_proposal_still_informs_bounded_replacement(
    tmp_path, missing_binding
):
    prompt = "Reply with exactly PING and nothing else."
    gateway = compound_gateway(
        prompt,
        [],
        output="PING",
        candidate="Respond with exactly PING and nothing else.",
    )
    chat, decide = gateway.chat_handler, gateway.decision_handler
    states = []

    def generate(model, messages, **params):
        if params["role"] == "writer" and "Compile the user" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            states.append(state)
            return '{"tests":[{"question":"Does the answer contain PING?","kind":"noul","expected":"yes"}]}'
        return chat(model, messages, **params)

    def screen(request, **params):
        key = str(request.get("key", ""))
        if key.startswith("requirement:test-binding:"):
            return (
                None
                if missing_binding
                else {"type": "noul", "probability_true": 0.01, "confidence": 1.0}
            )
        if key.startswith("success-test-screen:"):
            probability = (
                0.01
                if request["dimension"] == "assessability"
                else 0.99
                if request["dimension"] in {"faithfulness", "no_invention"}
                else 0.01
            )
            return {"type": "noul", "probability_true": probability, "confidence": 1.0}
        return decide(request, **params)

    gateway.chat_handler, gateway.decision_handler = generate, screen
    result = PromptOptimizer(
        store=RunStore(tmp_path / "unbound-replacement.sqlite"), gateway=gateway
    ).optimize(prompt, {"time_limit_s": 0})
    source = result["report"]["history"][0]["evidence"]["test_screening"][
        "source_checks"
    ]
    assert len(states) == len(source["attempts"]) == 2
    feedback = states[1]["rejected_proposals"][0]
    assert feedback["test"]["question"] == "Does the answer contain PING?"
    assert feedback["reason"] and feedback["requirement_ids"] == []
    assert feedback["bindings"] and all(
        item["status"] == "unresolved" for item in feedback["bindings"]
    )
    assert source["coverage"][0]["status"] == "unresolved"


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("missing", "The binding judge returned no decision."),
        ("invalid", "The binding judge returned an invalid decision."),
        ("outage", "The binding judge could not be reached."),
        ("oversize", "The binding request exceeds the supported 48000-byte limit."),
        ("uncertain", "The binding decision does not establish source support."),
        ("unsupported", "The binding decision does not establish source support."),
    ],
)
def test_binding_uncertainty_retains_its_reason_and_physical_tests(mode, reason):
    from prompt_enhancer.gateway import ProviderError
    from prompt_enhancer.requirement_tests import compile_source_tests
    from prompt_enhancer.requirements import extract_requirements
    from prompt_enhancer.success_tests import SuccessTestCompiler

    prompt = "Reply with exactly PING and nothing else."
    requirements = extract_requirements(prompt)
    gateway = compound_gateway(prompt, [])
    batch = gateway.decide_batch

    def generate(_model, _messages, **_params):
        return '{"tests":[{"id":"t0","question":"Does the complete answer equal PING?","kind":"noul","expected":"yes"}]}'

    def decide(request, **_params):
        return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}

    binding_calls = []

    def bindings(requests, **params):
        if not all(
            str(item.get("key", "")).startswith("requirement:test-binding:")
            for item in requests
        ):
            return batch(requests, **params)
        binding_calls.extend(requests)
        if mode == "outage":
            raise ProviderError("scripted", "test", 503, "private diagnostics")
        answer = {
            "missing": None,
            "invalid": {"arbitrary": "private diagnostics"},
            "uncertain": {"type": "noul", "probability_true": 0.99, "confidence": 0.2},
            "unsupported": {
                "type": "noul",
                "probability_true": 0.01,
                "confidence": 1.0,
            },
        }.get(mode)
        return [answer] * len(requests)

    gateway.chat_handler, gateway.decision_handler = generate, decide
    gateway.decide_batch = bindings
    compiler = SuccessTestCompiler(
        gateway, writer_model="test", screen_protocol_version=1
    )
    compiled, evidence = compile_source_tests(
        compiler,
        prompt + ("\nContext: " + "x" * 48_000 if mode == "oversize" else ""),
        requirements,
    )
    assert not compiled.tests
    assert len(evidence["attempts"]) == 2
    assert all(attempt["tests"] for attempt in evidence["attempts"])
    assert all(
        binding["status"] == "unresolved" and binding.get("reason") == reason
        for attempt in evidence["attempts"]
        for binding in attempt["bindings"]
    )
    assert all(item["status"] == "unresolved" for item in evidence["coverage"])
    assert "private diagnostics" not in json.dumps(evidence)
    assert bool(binding_calls) is (mode != "oversize")
    assert [attempt["tests"][0]["id"] for attempt in evidence["attempts"]] == [
        "rNone:a1:t0",
        "rNone:a2:t0",
    ]


def test_worker_deadline_is_durable_before_the_completion_handoff(
    tmp_path, monkeypatch
):
    import threading

    from prompt_enhancer import publication

    clock = TickingClock()
    store = RunStore(tmp_path / "worker-deadline.sqlite")
    entered, release = threading.Event(), threading.Event()
    transaction = publication.checkpoint_transaction

    class CompletionHandoff:
        def set(self, value):
            return transaction.set(value)

        def get(self):
            return transaction.get()

        def reset(self, token):
            transaction.reset(token)
            entered.set()
            assert release.wait(5), "Completion handoff was not released."

    monkeypatch.setattr(publication, "checkpoint_transaction", CompletionHandoff())
    screening = {
        "attempts": [{"attempt": 1, "rejected": [{"reason": "Not assessable"}]}]
    }
    jobs = RunJobs(store=store, monotonic=clock)

    def expire(_progress, _cancel, _observe):
        clock.now = 151
        return {"status": "completed", "report": {}}

    jobs.submit(
        "worker-deadline",
        "optimize",
        expire,
        lambda _exc: {},
        prompt="Reply exactly PING.",
        evidence=lambda: {"report": {"test_screening": screening}},
    )
    try:
        assert entered.wait(5), "Worker did not reach the completion handoff."
        done = jobs.get("worker-deadline")
        assert done["state"] == "done"
        assert done["result"]["report"]["test_screening"] == screening
        assert store.get_run("worker-deadline")["result"] == done["result"]
    finally:
        release.set()
        jobs._executor.shutdown(wait=True)
