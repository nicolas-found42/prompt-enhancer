from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from prompt_enhancer.config import Settings
from prompt_enhancer.repeat import RepeatCoordinator
from prompt_enhancer.rounds import CandidateFailure, RoundOutcome, RoundPlan

GAP = {"confirmed_gaps": [{"key": "goal"}], "problem_sentences": []}


def _outcome(
    request,
    *,
    failures: tuple[CandidateFailure, ...] = (),
    final_prompt: str = "original",
    diagnosis: dict = GAP,
    convergence: dict | None = None,
) -> RoundOutcome:
    plan = RoundPlan(
        prompt="original",
        working_prompt="original",
        run_id=request.run_id,
        seed=0,
        diagnosis=diagnosis,
        assumptions=(),
        settings=Settings(),
        faithfulness_threshold=0.8,
        writer_instruction_version=2,
        prior_failures=request.prior_failures,
        prior_vector=request.prior_vector,
    )
    return RoundOutcome(
        plan=plan,
        status=(
            "converged"
            if convergence is not None and convergence.get("status") == "converged"
            else "no_change"
            if final_prompt == "original"
            else "improved"
        ),
        summary="",
        final_prompt=final_prompt,
        original_kept=final_prompt == "original",
        cost={"total": 0.001},
        timing={"elapsed_ms": 12},
        failures=failures,
        convergence=convergence,
    )


def _failure(
    reason: str = "candidate-1 missed the required output format",
) -> CandidateFailure:
    return CandidateFailure(
        candidate_id="candidate-1",
        strategy="specify_output_format",
        reasons=(reason,),
        weak_pass_rates={"weak-a": 0.25},
        strong_pass_rate=1.0,
    )


def _run(execute_round, run_id: str = "run"):
    return RepeatCoordinator().run(
        run_id=run_id, prompt="original", execute_round=execute_round
    )


def test_loop_retries_without_a_round_cap_until_fixture_stops() -> None:
    requests = []

    def execute_round(request):
        requests.append(request)
        if request.round_number <= 5:
            return _outcome(request, failures=(_failure(),))
        return _outcome(request)

    result = _run(execute_round)

    assert len(requests) == 6
    assert [request.round_number for request in requests] == list(range(1, 7))
    assert len(result.history) == 6


def test_gain_above_epsilon_retries_after_a_winner_is_selected() -> None:
    requests = []
    passing = {
        "scores": {"clarity": 0.8},
        "floors": {"clarity": 0.6},
        "passed": True,
        "selected": True,
    }
    improving = {
        "scores": {"clarity": 0.9},
        "floors": {"clarity": 0.6},
        "passed": True,
        "selected": True,
    }

    def execute_round(request):
        requests.append(request)
        if request.round_number == 1:
            return _outcome(
                request,
                failures=(_failure("another candidate lost on clarity"),),
                final_prompt="first gated candidate",
                convergence={**passing, "status": "continue", "gain": None},
            )
        return _outcome(
            request,
            final_prompt="best gated candidate",
            convergence={**improving, "status": "converged", "gain": 0.1},
        )

    result = _run(execute_round)

    assert len(requests) == 2
    assert requests[1].prior_failures == (
        "candidate-1 (specify_output_format): another candidate lost on clarity; "
        "weak pass rates: weak-a=0.250; strong pass rate=1.000",
    )
    assert requests[1].prior_vector["scores"] == {"clarity": 0.8}
    assert result.final_prompt == "best gated candidate"


def test_converged_original_is_reported_as_success_in_report_and_history() -> None:
    vector = {
        "scores": {"clarity": 0.8},
        "floors": {"clarity": 0.6},
        "passed": True,
        "status": "converged",
        "gain": None,
    }
    payload = _run(
        lambda request: _outcome(request, convergence=vector),
    ).as_payload()

    assert payload["status"] == "completed"
    assert payload["final_prompt"] == "original"
    assert payload["converged"] is True
    assert payload["report"]["status"] == "converged"
    assert payload["report"]["history"][0]["status"] == "converged"


def test_final_report_and_history_identify_the_best_candidate_after_regression() -> (
    None
):
    requests = []

    class Candidate:
        def __init__(self, candidate_id: str, text: str) -> None:
            self.candidate_id = candidate_id
            self.text = text
            self.prompt = text
            self.strategy = "specify_output_format"

        def to_dict(self):
            return {
                "candidate_id": self.candidate_id,
                "text": self.text,
                "prompt": self.text,
                "strategy": self.strategy,
            }

    class Ranking:
        def __init__(self, candidate_id: str, text: str) -> None:
            self.selected = Candidate(candidate_id, text)
            self.selected_candidate_id = candidate_id
            self.original_kept = False
            self.ranked = ()

        def to_dict(self):
            return {
                "selected_candidate_id": self.selected_candidate_id,
                "selected_candidate": self.selected.to_dict(),
                "ranking": [],
            }

    scores_by_round = (0.8, 0.95, 0.9)
    prompts = (
        "round-one prompt",
        "best historical prompt",
        "regressed terminal prompt",
    )

    def execute_round(request):
        requests.append(request)
        index = request.round_number - 1
        score = scores_by_round[index]
        vector = {
            "scores": {"clarity": score},
            "floors": {"clarity": 0.6},
            "passed": True,
            "selected": True,
            "status": "converged" if index == 2 else "continue",
            "gain": 0.0 if index == 2 else None,
        }
        outcome = _outcome(
            request,
            final_prompt=prompts[index],
            convergence=vector,
        )
        return replace(
            outcome,
            ranking=Ranking(f"candidate-{index + 1}", prompts[index]),
            panel=SimpleNamespace(to_dict=lambda: {}),
            strategies=SimpleNamespace(to_dict=lambda: {}),
        )

    result = _run(execute_round)
    payload = result.as_payload()
    final_report = payload["report"]

    assert len(requests) == 3
    assert result.final_prompt == "best historical prompt"
    assert payload["final_prompt"] == "best historical prompt"
    assert result.outcome.final_prompt == "best historical prompt"
    assert result.outcome.selected_candidate_id == "candidate-2"
    assert final_report["selection_evidence"]["selected_candidate_id"] == "candidate-2"
    assert (
        final_report["selection_evidence"]["selected_candidate"]["text"]
        == "best historical prompt"
    )
    assert final_report["convergence"]["source_round"] == 2
    assert final_report["convergence"]["scores"] == {"clarity": 0.95}
    assert final_report["convergence"]["terminal_scores"] == {"clarity": 0.9}
    assert final_report["history"][-1]["selected_candidate_id"] == "candidate-3"


def test_resumed_convergence_restores_persisted_best_selection_evidence() -> None:
    from prompt_enhancer.repeat import RoundEvidence, _history_from_run
    from prompt_enhancer.selector import RankingCandidate, RankingResult
    from prompt_enhancer.store import RunStore

    def selected_outcome(request, prompt, score, status):
        candidate = RankingCandidate(
            f"candidate-{request.round_number}", prompt, "specify_output_format"
        )
        original = RankingCandidate("original", "original", "original")
        return replace(
            _outcome(
                request,
                final_prompt=prompt,
                convergence={
                    "scores": {"clarity": score},
                    "floors": {"clarity": 0.6},
                    "passed": True,
                    "selected": True,
                    "status": status,
                    "gain": -0.05 if status == "converged" else None,
                },
            ),
            ranking=RankingResult(candidate, original, False, (), {}),
            panel=SimpleNamespace(to_dict=lambda: {"source": prompt}),
            strategies=SimpleNamespace(to_dict=lambda: {}),
        )

    source = selected_outcome(
        SimpleNamespace(
            run_id="resumed",
            round_number=1,
            prior_failures=(),
            prior_vector=None,
        ),
        "persisted best prompt",
        0.95,
        "continue",
    )
    evidence = RoundEvidence.from_outcome(round_number=1, outcome=source)
    store = RunStore(":memory:")
    store.save_run(
        {
            "run_id": "resumed",
            "prompt": "original",
            "result": {"report": {"history": [evidence.to_dict()]}},
        }
    )
    history = _history_from_run(store.get_run("resumed")["result"])
    result = RepeatCoordinator().run(
        run_id="resumed",
        prompt="original",
        prior_history=history,
        execute_round=lambda request: selected_outcome(
            request, "regressed resumed prompt", 0.9, "converged"
        ),
    )
    payload = result.as_payload()
    assert payload["final_prompt"] == "persisted best prompt"
    assert result.outcome.final_prompt == "persisted best prompt"
    assert payload["selected_candidate_id"] == "candidate-1"
    assert (
        payload["report"]["selection_evidence"]["selected_candidate"]["text"]
        == "persisted best prompt"
    )
    assert payload["report"]["convergence"]["scores"] == {"clarity": 0.95}
    assert payload["report"]["convergence"]["source_round"] == 1
    assert payload["report"]["convergence"]["terminal_scores"] == {"clarity": 0.9}
    assert payload["report"]["history"][-1]["selected_candidate_id"] == "candidate-2"


def test_later_round_receives_previous_candidate_failures() -> None:
    requests = []

    def execute_round(request):
        requests.append(request)
        if request.round_number == 1:
            return _outcome(
                request,
                failures=(_failure("candidate-2 lost on the smallest weak model"),),
            )
        return _outcome(request)

    result = _run(execute_round)

    assert requests[0].prior_failures == ()
    assert requests[1].prior_round_failures[0].candidate_id == "candidate-1"
    assert (
        "candidate-2 lost on the smallest weak model" in requests[1].prior_failures[0]
    )
    assert requests[1].prior_failures == (requests[1].prior_round_failures[0].summary,)
    assert result.history[0].candidate_failures[0].weak_pass_rates == {"weak-a": 0.25}
    assert result.history[0].cost == {"total": 0.001}
    assert result.history[0].timing == {"elapsed_ms": 12}


def test_a_round_without_failures_ends_the_loop() -> None:
    requests = []

    def execute_round(request):
        requests.append(request)
        return _outcome(request)

    _run(execute_round)

    assert len(requests) == 1
