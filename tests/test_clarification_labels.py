"""Copied prompts use human labels while ledger identities remain durable."""

from active_clock import TickingClock, advancing_chat
from e2e_server import chat, decide

from prompt_enhancer import PromptOptimizer, RunStore
from prompt_enhancer.gateway import ScriptedGateway


def test_scoped_conflict_choice_uses_its_human_label_in_the_final_prompt():
    original = "Give Section A exactly two bullets and exactly three bullets. Preserve {topic}."
    clock = TickingClock()
    optimizer = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=ScriptedGateway(chat=advancing_chat(chat, clock, 20), decision=decide),
        clock=clock,
    )
    paused = optimizer.optimize(original)
    question = paused["questions"][0]
    result = optimizer.resume(
        paused["run_id"], {question["id"]: question["options"][0]["value"]}
    )

    assert result["report"]["outcome"] == "converged"
    assert "Resolve conflicting requirements: two bullets" in result["final_prompt"]
    assert "choice:requirement:" not in result["final_prompt"]
    assert "{topic}" in result["final_prompt"]
    assert "exactly three bullets" not in result["final_prompt"]
    ledger = result["report"]["requirements"]
    assert ledger["contradictions"][0]["id"] == question["id"]
    assert ledger["contradictions"][0]["status"] == "resolved_by_user"
    assert any(item.get("superseded_by") for item in ledger["requirements"])
    assert optimizer.store.get_run(result["run_id"])["prompt"] == original
