from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.jev import ScoreDecision, parse_decision


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
