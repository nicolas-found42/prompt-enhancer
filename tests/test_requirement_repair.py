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
    done = jobs.wait("repair")
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
    done = jobs.wait("failed-repair")
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
