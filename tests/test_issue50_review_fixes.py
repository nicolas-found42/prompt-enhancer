from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from prompt_enhancer import jev_questions
from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.diagnosis import Diagnoser, ProblemKind, default_gap_question
from prompt_enhancer.evaluation.__main__ import main as evaluation_main
from prompt_enhancer.evaluation.calibration import (
    CalibrationArtifact,
    DecisionPolicy,
    QuestionIdentity,
)
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def _pointer_artifact(verdict: str) -> CalibrationArtifact:
    identity = QuestionIdentity(
        question_id="pointer:vagueness",
        question=jev_questions.sentence_pointer_question("vagueness"),
        primitive="choice",
        criteria="sentence-id-options-with-none",
        event_mapping={"selected_correctness": True},
        family="pointer",
        rubric_version="default-v1",
        answering_snapshot=JEV_MODEL,
    )
    return CalibrationArtifact.from_dict(
        {
            "name": "pointer-review",
            "questions": {
                identity.question_id: {
                    "identity": identity.to_dict(),
                    "verdict": verdict,
                    "threshold": 0.8,
                }
            },
        }
    )


@pytest.mark.parametrize(
    ("prompt", "verdict", "confidence", "expected_count"),
    [
        ("First sentence. Second sentence.", "gate", 0.1, 1),
        ("First sentence. Second sentence. Third sentence.", "gate", 0.1, 1),
        ("First sentence. Second sentence.", "ranker", 0.1, 0),
        ("First sentence. Second sentence.", "ranker", 0.99, 1),
    ],
)
def test_pointer_artifact_matches_across_prompts_and_ranker_keeps_confidence_floor(
    prompt: str, verdict: str, confidence: float, expected_count: int
) -> None:
    def decide(request, **_kwargs):
        key = str(request.get("key", ""))
        if key.startswith("existence:"):
            return {
                "type": "noul",
                "probability_true": 0.95,
                "confidence": 0.95,
            }
        if key.startswith("pointer:vagueness:"):
            options = request["options"]
            return {
                "type": "choice",
                "choice": options[0],
                "probabilities": {options[0]: 0.95, "none": 0.05},
                "confidence": confidence,
            }
        if key.startswith("pointer:"):
            return {
                "type": "choice",
                "choice": "none",
                "probabilities": {"none": 1.0},
                "confidence": 1.0,
            }
        if key.startswith("problem:"):
            return {"type": "noul", "probability_true": 0.95, "confidence": 0.95}
        if request.get("type") == "choice":
            return {"type": "choice", "choice": "general", "confidence": 1.0}
        return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}

    diagnoser = Diagnoser(
        ScriptedGateway(decision=decide),
        decision_policy=DecisionPolicy.from_artifact(_pointer_artifact(verdict)),
    )
    report = diagnoser.diagnose(prompt)

    assert len(report.problem_sentences) == expected_count
    if expected_count:
        assert report.problem_sentences[0].kind == ProblemKind.VAGUENESS
    assert report.calibration is not None
    assert report.calibration["pointer:vagueness"]["disposition"] == verdict


@pytest.mark.parametrize(
    "policy_input", ["{broken", '["not an object"]', "/missing/policy.json"]
)
def test_invalid_calibration_policy_is_reported_as_cli_error(policy_input: str) -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "evaluation"
        / "calibration_known_answer.json"
    )
    stderr = StringIO()

    code = evaluation_main(
        ["calibrate", str(fixture), "--calibration-policy", policy_input],
        stderr=stderr,
    )

    assert code == 2
    assert stderr.getvalue().startswith("evaluation error:")


@pytest.mark.parametrize("form", ["object", "mapping", "path", "calibration", "policy"])
def test_optimizer_reads_custom_policy_version_from_artifact(
    form: str, tmp_path: Path
) -> None:
    identity = QuestionIdentity(
        question_id="gap:goal",
        question=default_gap_question("goal"),
        primitive="noul",
        criteria=("no", "yes"),
        event_mapping={"positive_class": "yes"},
        family="gap",
        rubric_version="default-v1",
        answering_snapshot=JEV_MODEL,
        policy_version="review-v2",
    )
    artifact = CalibrationArtifact.from_dict(
        {
            "name": "custom-policy",
            "metadata": {"verdict_policy": {"policy_version": "review-v2"}},
            "questions": {
                identity.question_id: {
                    "identity": identity.to_dict(),
                    "verdict": "gate",
                    "threshold": 0.8,
                }
            },
        }
    )
    source: object = artifact
    if form == "mapping":
        source = artifact.to_dict()
    elif form == "path":
        source = tmp_path / "calibration-artifact.json"
        artifact.save(source)
    elif form == "policy":
        source = DecisionPolicy.from_artifact(artifact, policy_version="review-v2")

    keyword = "calibration" if form == "calibration" else "decision_policy"
    optimizer = PromptOptimizer(
        gateway=ScriptedGateway(),
        store=RunStore(":memory:"),
        **{keyword: source},
    )

    assert optimizer.decision_policy is not None
    assert optimizer.decision_policy.resolve(
        identity.question_id, identity=identity, snapshot=JEV_MODEL
    ).may_gate
