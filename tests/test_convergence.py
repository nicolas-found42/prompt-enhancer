from __future__ import annotations

from prompt_enhancer.convergence import best_candidate_vector, convergence_decision


def _vector(*, clarity: float = 0.8, passed: bool = True) -> dict:
    floors = {"fidelity": 0.8, "clarity": 0.6}
    scores = {"fidelity": 1.0, "clarity": clarity}
    return {
        "scores": scores,
        "floors": floors,
        "breaches": [] if passed else ["clarity"],
        "passed": passed,
    }


def test_converges_when_all_floors_pass_and_gain_is_at_epsilon() -> None:
    decision = convergence_decision(
        _vector(clarity=0.8),
        previous_vector=_vector(clarity=0.78),
        epsilon=0.01,
    )

    assert decision.converged is True
    assert decision.status == "converged"
    assert decision.gain == 0.010000000000000009
    assert decision.to_dict()["scores"] == {"fidelity": 1.0, "clarity": 0.8}
    assert decision.to_dict()["floors"] == {"fidelity": 0.8, "clarity": 0.6}


def test_below_floor_retries_even_when_gain_is_small() -> None:
    decision = convergence_decision(
        _vector(clarity=0.59, passed=False),
        previous_vector=_vector(clarity=0.58, passed=False),
        epsilon=0.01,
    )

    assert decision.converged is False
    assert decision.status == "continue"
    assert decision.breaches == ("clarity",)


def test_gain_above_epsilon_retries_even_after_all_floors_pass() -> None:
    decision = convergence_decision(
        _vector(clarity=0.9),
        previous_vector=_vector(clarity=0.8),
        epsilon=0.01,
    )

    assert decision.converged is False
    assert decision.status == "continue"
    assert decision.gain > decision.epsilon


def test_first_round_with_a_floor_passing_candidate_converges() -> None:
    decision = convergence_decision(_vector(), previous_vector=None, epsilon=0.01)

    assert decision.converged is True
    assert decision.gain is None


def test_convergence_uses_the_selected_candidate_not_a_rejected_ranked_vector() -> None:
    report = {
        "selection_evidence": {
            "selected_candidate": None,
            "ranking": [
                {
                    "selected": False,
                    "metadata": {
                        "score_vector": {
                            **_vector(),
                            "passed": True,
                        }
                    },
                }
            ],
        }
    }

    vector = best_candidate_vector(report)
    decision = convergence_decision(vector, previous_vector=None, epsilon=0.01)

    assert vector is not None
    assert vector["selected"] is False
    assert decision.converged is False


def test_rejected_changed_candidate_cannot_stand_in_for_original_baseline() -> None:
    original_scores = {"clarity": 0.4}
    vector = best_candidate_vector(
        {
            "selection_evidence": {
                "original_kept": True,
                "original": {
                    "candidate_id": "original",
                    "metadata": {
                        "score_vector": {
                            "scores": original_scores,
                            "floors": {"clarity": 0.6},
                            "passed": False,
                            "source": "original_baseline",
                        }
                    },
                },
                "ranking": [
                    {
                        "candidate_id": "changed-but-rejected",
                        "selected": False,
                        "metadata": {
                            "score_vector": {
                                **_vector(clarity=0.99),
                                "passed": True,
                            }
                        },
                    }
                ],
            }
        }
    )

    assert vector is not None
    assert vector["scores"] == original_scores
    assert vector["passed"] is False
    assert vector["selected"] is False
    assert vector["source"] == "original_baseline"
