from prompt_enhancer import PromptOptimizer, RunStore


def test_clear_prompt_is_returned_unchanged_and_persisted() -> None:
    prompt = "Summarize this article in three concise bullets for a busy reader."
    store = RunStore(":memory:")
    optimizer = PromptOptimizer(store=store)

    result = optimizer.optimize(prompt, {"tier": "Fast"})

    assert result["status"] == "completed"
    assert result["final_prompt"] == prompt
    assert result["original_kept"] is True
    assert result["report"]["status"] == "no_change"
    assert "no rewrite" in result["report"]["summary"].casefold()
    assert result["run_id"]

    record = store.get_run(result["run_id"])
    assert record is not None
    assert record["prompt"] == prompt
    assert record["options"]["tier"] == "fast"
    assert record["result"] == result
    assert record["cost"] == result["cost"]
    assert record["timing"] == result["timing"]


def test_empty_prompt_is_rejected() -> None:
    optimizer = PromptOptimizer(store=RunStore(":memory:"))

    try:
        optimizer.optimize("   ")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty prompt should be rejected")
