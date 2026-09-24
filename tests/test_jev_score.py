from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.grading import grade_panel_with_jev
from prompt_enhancer.jev import ScoreDecision, parse_decision
from prompt_enhancer.runner import PanelResult


def test_scripted_gateway_parses_fractional_score_and_preserves_legend() -> None:
    answer = {
        "type": "score", "score": 2.45, "confidence": 0.53,
        "legend": {"0": "Neither", "1": "Partial", "2": "Good", "3": "Excellent"},
        "probabilities": {"0": 0, "1": 0.01, "2": 0.53, "3": 0.46},
    }
    gateway = ScriptedGateway(decision=lambda *_args, **_kwargs: answer)

    decision = parse_decision(gateway.decide({"type": "score", "query": "How well?", "levels": list(answer["legend"].values())}))

    assert isinstance(decision, ScoreDecision)
    assert decision.level == 2
    assert decision.score == 2.45
    assert decision.legend == answer["legend"]
    assert decision.probabilities == answer["probabilities"]


def test_score_level_tie_uses_lower_level() -> None:
    decision = parse_decision({"type": "score", "score": 1.5, "probabilities": {"0": 0, "1": 0.5, "2": 0.5}})

    assert isinstance(decision, ScoreDecision)
    assert decision.level == 1


def test_grading_mixed_tests_uses_correct_answers_and_one_noul_request_each() -> None:
    requests = []

    def decide(request, **_kwargs):
        requests.append(request)
        key = request["key"]
        if key == "grade_0_1_first" or key == "grade_0_1_second":
            return {"type": "choice", "probabilities": {"pass": 0.8, "fail": 0.2}, "choice": "pass"}
        return {"type": "noul", "probability_true": 0.9 if key == "grade_0_0_first" else 0.1}

    panel = [PanelResult(candidate_id="candidate", model="weak", sample=0, seed=1, output="answer")]
    tests = [
        {"kind": "noul", "question": "Is it correct?", "expected": "yes"},
        {"kind": "choice", "question": "Which outcome?", "expected": "pass", "options": ["pass", "fail"]},
        {"kind": "noul", "question": "Is it incorrect?", "expected": "no"},
    ]

    grades, evidence = grade_panel_with_jev(panel, tests, ScriptedGateway(decision=decide), judge_model="jev", run_id="run")

    assert [request["key"] for request in requests] == [
        "grade_0_0_first", "grade_0_1_first", "grade_0_1_second", "grade_0_2_first",
    ]
    assert len(evidence) == 4
    assert grades["candidate"].per_model_samples["weak"] == (0.8,)


def test_unusable_noul_answer_cannot_pass_expected_no() -> None:
    panel = [PanelResult(candidate_id="candidate", model="weak", sample=0, seed=1, output="answer")]
    tests = [{"kind": "noul", "question": "Is it incorrect?", "expected": "no"}]
    gateway = ScriptedGateway(decision=lambda request, **_kwargs: {"type": "noul", "probability_true": "invalid"})

    grades, _ = grade_panel_with_jev(panel, tests, gateway, judge_model="jev", run_id="run")

    assert grades["candidate"].per_model_samples["weak"] == (0.0,)
