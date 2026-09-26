from __future__ import annotations

import json
from collections import Counter

from prompt_enhancer.evaluation import Dataset, EvaluationHarness
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.lossless_restructuring import (
    RoleAssignment,
    render_lossless_candidate,
    segment_source_units,
    verify_lossless_proof,
)
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def test_lossless_renderer_preserves_exact_units_duplicates_and_fixed_order() -> None:
    prompt = (
        "  Summarize this report.\n\n"
        "Keep the dates exactly as written! Summarize this report.\n"
        "\nUse this example: ‘A. B!’"
    )
    units = segment_source_units(prompt)
    roles = [
        "constraint"
        if "Keep the dates" in unit.text
        else "example"
        if "Use this example" in unit.text or unit.text.strip().startswith("B!")
        else "task"
        for unit in units
    ]
    assignments = tuple(
        RoleAssignment(unit.id, role, role, 0.99, {role: 0.99})
        for unit, role in zip(units, roles, strict=True)
    )

    result = render_lossless_candidate(prompt, units, assignments)

    grouped = {
        role: "".join(
            unit.text
            for unit, assignment in zip(units, assignments, strict=True)
            if assignment.role == role
        )
        for role in ("task", "constraint", "example")
    }
    assert result.text == (
        "### Task\n\n"
        + grouped["task"]
        + "\n\n### Constraints\n\n"
        + grouped["constraint"]
        + "\n\n### Examples\n\n"
        + grouped["example"]
    )
    assert Counter(unit.text for unit in units) == Counter(
        unit.text for unit in segment_source_units(prompt)
    )
    assert len({unit.id for unit in units}) == len(units)
    assert result.evidence["source_preservation"]["status"] == "passed"
    assert verify_lossless_proof(prompt, result.text, result.proof).passed


def test_code_and_list_blocks_remain_indivisible_source_units() -> None:
    prompt = (
        "Complete these steps:\n\n"
        "- Keep this first item. It has two sentences.\n"
        "- Keep this second item!\n\n"
        "```python\nprint('Keep. Every! Character?')\n```\n\n"
        "Then summarize."
    )

    units = segment_source_units(prompt)

    assert "".join(unit.text for unit in units) == prompt
    assert (
        sum(unit.text.lstrip("\n").startswith("- Keep this first") for unit in units)
        == 1
    )
    assert sum(unit.text.lstrip("\n").startswith("```python") for unit in units) == 1
    assert next(
        unit.text
        for unit in units
        if unit.text.lstrip("\n").startswith("- Keep this first")
    ) == (
        "\n\n- Keep this first item. It has two sentences.\n- Keep this second item!\n"
    )
    code = next(
        unit.text for unit in units if unit.text.lstrip("\n").startswith("```python")
    )
    assert code == "\n```python\nprint('Keep. Every! Character?')\n```\n"


def test_unknown_low_confidence_and_missing_roles_fall_into_other() -> None:
    prompt = "A task. A constraint. Unclear wording."
    units = segment_source_units(prompt)
    assignments = (
        RoleAssignment(units[0].id, "task", "task", 0.99, {"task": 0.99}),
        RoleAssignment(
            units[1].id,
            "constraint",
            "other",
            0.79,
            {"constraint": 0.79},
        ),
        RoleAssignment(units[2].id, "unknown", "other", 0.99, {"unknown": 0.99}),
    )

    result = render_lossless_candidate(prompt, units, assignments)

    roles = result.evidence["roles"]
    assert [item["role"] for item in roles] == ["task", "other", "other"]
    assert result.evidence["unknowns"] == [units[1].id, units[2].id]
    assert "### Other" in result.text


def test_proof_rejects_changed_source_multiplicity_or_rendered_body() -> None:
    prompt = "Repeat this. Repeat this."
    units = segment_source_units(prompt)
    assignments = (
        RoleAssignment(units[0].id, "task", "task", 0.99, {"task": 0.99}),
        RoleAssignment(
            units[1].id, "constraint", "constraint", 0.99, {"constraint": 0.99}
        ),
    )
    result = render_lossless_candidate(prompt, units, assignments)

    assert result.text is not None and result.proof is not None
    invalid = verify_lossless_proof(prompt, result.text + " invented", result.proof)

    assert not invalid.passed
    assert "rendered candidate hash does not match" in invalid.reasons


def _run_lossless_round(
    *,
    meaning_probability: float = 1.0,
    strong_passes: bool = True,
    mixed_strategies: bool = False,
    invalid_role_id: bool = False,
):
    writer_requests: list[dict] = []
    decision_keys: list[str] = []

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer":
            state = json.loads(messages[1]["content"])
            writer_requests.append(state)
            if "strategies" in state:
                assert mixed_strategies
                assert all(
                    strategy["name"] != "restructure_lossless"
                    for strategy in state["strategies"]
                )
                return json.dumps(
                    {
                        strategy["name"]: state["prompt"]
                        for strategy in state["strategies"]
                    }
                )
            return '{"tests":[{"question":"Does the output summarize the report?","kind":"noul","expected":"yes"}]}'
        prompt = messages[0]["content"]
        if role == "strong_check":
            return "fail" if not strong_passes and "### Task" in prompt else "pass"
        return "pass" if "### Task" in prompt else "fail"

    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        decision_keys.append(key)
        if key == "strategy_choice":
            choice = "restructure_lossless"
        elif key.startswith("restructure_lossless:role:"):
            text = request["state"]["target_unit_text"]
            choice = "context" if "background" in text else "task"
        elif request.get("type") == "choice":
            choice = "general" if key == "task_type" else "none"
        else:
            probability = (
                1.0
                if key.startswith(("gap:goal", "faithful:"))
                or (
                    key.startswith("strategy_recheck:")
                    and (
                        mixed_strategies
                        or key == "strategy_recheck:restructure_lossless"
                    )
                )
                or (key.startswith("grade_") and request["state"]["output"] == "pass")
                else 0.0
            )
            if key == "fidelity:meaning":
                probability = meaning_probability
            if key.startswith("grade_") and key.endswith("_second"):
                probability = 1.0 - probability
            return {"type": "noul", "probability_true": probability}
        answer = {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: 1.0},
            "confidence": 1.0,
        }
        if invalid_role_id and key == "restructure_lossless:role:u0001":
            answer["unit_id"] = "not-a-source-unit"
        return answer

    prompt = "Read the background notes. Summarize the report."
    result = PromptOptimizer(
        store=RunStore(":memory:"),
        gateway=ScriptedGateway(chat=chat, decision=decide),
        writer_instruction_version=4,
    ).optimize(
        prompt,
        {
            "tier": "standard" if mixed_strategies else "fast",
            "clarification_allowed": False,
        },
    )

    return result, writer_requests, decision_keys


def test_optimizer_selects_lossless_candidate_without_candidate_writer_call() -> None:
    result, writer_requests, decision_keys = _run_lossless_round()

    assert result["original_kept"] is False
    assert "### Context" in result["final_prompt"]
    assert "### Task" in result["final_prompt"]
    assert writer_requests
    assert all("strategies" not in request for request in writer_requests)
    assert any(key.startswith("restructure_lossless:role:") for key in decision_keys)
    assert "fidelity:meaning" in decision_keys
    assert not any(key.startswith("fidelity:sentence:") for key in decision_keys)
    assert result["report"]["per_model"]["ranking"]["selected_candidate_id"]


def test_other_strategy_uses_writer_when_sharing_a_round_with_lossless() -> None:
    result, writer_requests, _ = _run_lossless_round(mixed_strategies=True)

    candidate_writes = [
        request for request in writer_requests if "strategies" in request
    ]
    assert len(candidate_writes) == 1
    assert candidate_writes[0]["strategies"]
    assert all(
        strategy["name"] != "restructure_lossless"
        for strategy in candidate_writes[0]["strategies"]
    )
    assert any(
        candidate["strategy"] == "restructure_lossless"
        for candidate in result["report"]["candidates"]
    )


def test_invalid_role_id_declines_and_reports_reason_without_running_candidate() -> (
    None
):
    result, writer_requests, keys = _run_lossless_round(invalid_role_id=True)

    assert result["original_kept"] is True
    assert all("strategies" not in request for request in writer_requests)
    assert any(key.startswith("restructure_lossless:role:") for key in keys)
    evidence = result["report"]["lossless_restructuring"]
    assert evidence["outcome"] == "declined"
    assert "invalid unit ID" in evidence["decline_reason"]
    assert evidence["source_preservation"]["status"] == "not_proven"
    assert result["report"]["candidates"] == []


def test_meaning_rejection_keeps_original_after_lossless_proof() -> None:
    result, _writers, keys = _run_lossless_round(meaning_probability=0.1)

    assert result["original_kept"] is True
    assert "fidelity:meaning" in keys
    candidate = next(
        item
        for item in result["report"]["candidates"]
        if item["strategy"] == "restructure_lossless"
    )
    assert (
        candidate["metadata"]["lossless_restructuring"]["source_preservation"]["status"]
        == "passed"
    )
    assert candidate["metadata"]["fidelity"]["meaning_preserved"] is False


def test_strong_regression_keeps_original_after_lossless_proof_and_weak_win() -> None:
    result, _writers, keys = _run_lossless_round(strong_passes=False)

    assert result["original_kept"] is True
    assert "fidelity:meaning" in keys
    assert result["report"]["strong_check"]["candidates"]


def test_harness_counts_structural_selection_win_and_strong_rejection() -> None:
    dataset = Dataset.from_dict(
        {
            "name": "structural-scripted",
            "cases": [{"id": "one", "prompt": "A task.", "source": "synthetic"}],
        }
    )

    class ResultEngine:
        def __init__(self, result):
            self.result = result

        def optimize(self, _prompt, _options):
            return self.result

    selected, _, _ = _run_lossless_round()
    selected_summary = (
        EvaluationHarness(ResultEngine(selected))
        .run(dataset)
        .to_dict()["restructuring"]
    )
    assert selected_summary["total_cases"] == 1
    assert selected_summary["selected_for_generation"] == 1
    assert selected_summary["built_candidates"] == 1
    assert selected_summary["wins"] == 1
    assert selected_summary["win_rate"] == 1.0
    assert selected_summary["strong_checked"] == 1
    assert selected_summary["strong_rejection_rate"] == 0.0

    rejected, _, _ = _run_lossless_round(strong_passes=False)
    rejected_summary = (
        EvaluationHarness(ResultEngine(rejected))
        .run(dataset)
        .to_dict()["restructuring"]
    )
    assert rejected_summary["wins"] == 0
    assert rejected_summary["strong_checked"] == 1
    assert rejected_summary["strong_regressions"] == 1
    assert rejected_summary["strong_rejection_rate"] == 1.0
