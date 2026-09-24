"""Public clarification behavior tests using no model gateway."""

from pathlib import Path

import pytest

from prompt_enhancer.clarification import (
    ClarificationService,
    GapAssessment,
    InMemoryClarificationRepository,
    InvalidAnswerError,
    SQLiteClarificationRepository,
    build_plan,
)


def high_unknown_gap() -> GapAssessment:
    return GapAssessment(
        id="output_format",
        label="output format",
        impact="high",
        present=None,
        question="Which output format should the assistant produce?",
        options=(
            {"value": "json", "label": "JSON", "preselected": True},
            {"value": "markdown", "label": "Markdown"},
        ),
    )


def test_confident_inference_and_low_impact_unknown_do_not_ask() -> None:
    plan = build_plan(
        [
            GapAssessment(
                id="audience",
                label="audience",
                impact="high",
                present=False,
                confidence=0.91,
                value="a busy product manager",
                inferred=True,
            ),
            GapAssessment(
                id="tone",
                label="tone",
                impact="low",
                present=None,
            ),
        ]
    )

    assert plan.questions == ()
    assert [(item.key, item.value, item.source) for item in plan.assumptions] == [
        ("audience", "a busy product manager", "inferred"),
        ("tone", "Not specified", "skipped_low_impact"),
    ]


def test_high_unknown_pauses_and_resume_continues_same_run() -> None:
    repository = InMemoryClarificationRepository()
    calls = []

    def continuation(state):
        calls.append(state["run_id"])
        return {"status": "completed", "run_id": state["run_id"], "prompt": "rewritten"}

    service = ClarificationService(repository, continuation=continuation)
    plan = build_plan([high_unknown_gap()])
    paused = service.start("run-7", "write a report", plan)

    assert paused["status"] == "needs_input"
    question = paused["questions"][0]
    assert question["id"] == "output_format"
    assert question["default"] == "json"
    assert question["allow_other"] is True
    assert question["options"][-1]["value"] == "other"
    assert calls == []

    completed = service.resume("run-7", {"output_format": "markdown"})
    assert completed["status"] == "completed"
    assert completed["run_id"] == "run-7"
    assert completed["assumptions"] == [
        {"key": "output_format", "value": "Markdown", "source": "answer", "label": "output format"}
    ]
    assert calls == ["run-7"]
    assert repository.load("run-7")["status"] == "completed"


def test_other_answer_and_explicit_skip_record_assumptions() -> None:
    service = ClarificationService(InMemoryClarificationRepository())
    service.start("run-8", "write a report", build_plan([high_unknown_gap()]))

    with pytest.raises(InvalidAnswerError):
        service.resume("run-8", {"output_format": {"value": "other"}})

    answered = service.resume("run-8", {"output_format": {"value": "other", "text": "CSV"}})
    assert answered["assumptions"][0]["value"] == "CSV"

    service = ClarificationService(InMemoryClarificationRepository())
    service.start("run-9", "write a report", build_plan([high_unknown_gap()]))
    skipped = service.skip("run-9")
    assert skipped["status"] == "completed"
    assert skipped["assumptions"] == [
        {"key": "output_format", "value": "JSON", "source": "skipped_clarification"}
    ]


def test_sqlite_repository_persists_paused_state(tmp_path: Path) -> None:
    repository = SQLiteClarificationRepository(tmp_path / "runs.sqlite")
    service = ClarificationService(repository)
    service.start("run-10", "write a report", build_plan([high_unknown_gap()]))

    reloaded = ClarificationService(repository)
    assert reloaded.repository.load("run-10")["status"] == "needs_input"
    assert reloaded.resume("run-10", {"output_format": "json"})["status"] == "completed"


def test_reference_to_unseen_details_is_asked_about_not_assumed() -> None:
    from prompt_enhancer.gateway import ScriptedGateway
    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.store import RunStore

    def chat(_model, _messages, *, role, **_kwargs):
        return '{"gaps":{"outside_reference":{"question":"What is \'the thing about the warranty\'?","options":[{"value":"leave_out","label":"Leave it out"}]}}}'

    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "writing" if request.get("key") == "task_type" else "unknown"
            return {"type": "choice", "choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0}
        probability = 0.97 if request.get("key") == "gap:outside_reference" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=ScriptedGateway(chat=chat, decision=decide))
    result = optimizer.optimize("Do the letter like last time. Mention the thing about the warranty.", {"tier": "fast"})

    assert result["status"] == "needs_input"
    question = result["questions"][0]
    assert question["id"] == "outside_reference"
    assert "warranty" in question["prompt"]
    assert question["allow_other"] is True


def test_answered_outside_reference_reads_as_plain_text_in_the_prompt() -> None:
    from prompt_enhancer.gateway import ScriptedGateway
    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.store import RunStore

    def chat(_model, _messages, *, role, **_kwargs):
        return '{"gaps":{"outside_reference":{"question":"What is \'the thing\'?","options":[{"value":"leave_out","label":"Leave it out"}]}},"tests":[]}'

    def decide(request, **_kwargs):
        if request.get("type") == "choice":
            choice = "writing" if request.get("key") == "task_type" else "unknown"
            return {"type": "choice", "choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0}
        probability = 0.97 if request.get("key") == "gap:outside_reference" else 0.01
        return {"type": "noul", "probability_true": probability, "confidence": 1.0}

    optimizer = PromptOptimizer(store=RunStore(":memory:"), gateway=ScriptedGateway(chat=chat, decision=decide))
    paused = optimizer.optimize("Mention the thing.", {"tier": "fast"})
    done = optimizer.resume(paused["run_id"], {"outside_reference": {"value": "other", "text": "the 5-year warranty"}})

    assert "outside_reference" not in done["final_prompt"]
    assert "Details: the 5-year warranty" in done["final_prompt"]
    assert done["report"]["assumptions"][0]["label"] == "details only you know"
    assert done["timing"]["total_ms"] >= 0
