"""Bounded style presentation never authorizes new facts or task scope."""

from prompt_enhancer.config import Settings
from prompt_enhancer.fidelity import FidelityResult
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.score_vector import score_candidate


def test_adequate_self_baseline_passes_absolute_scores_but_deficient_one_fails():
    def decide(request, **_kwargs):
        state = request["state"]
        assert state["candidate_prompt"] == state["original_prompt"]
        query = request["query"]
        relative = "than the original" in query or "more specifically" in query
        sufficient = state["candidate_prompt"] == "What is 2 + 2?"
        return {"type": "noul", "probability_true": float(sufficient and not relative)}

    gateway = ScriptedGateway(decision=decide)
    vectors = [
        score_candidate(
            gateway,
            prompt,
            prompt,
            fidelity=FidelityResult(True, True, True),
            applied_style="clearer",
            style_bundle=(),
            floors=Settings().score_floors,
            judge_model="judge",
            run_id="baseline",
        )
        for prompt in ("What is 2 + 2?", "Do the thing we discussed.")
    ]
    assert vectors[0].passed
    assert not vectors[1].passed
    assert {"clarity", "specificity"}.issubset(vectors[1].breaches)


def test_presentation_support_requires_matching_catalog_authorization():
    from prompt_enhancer.fidelity import check_candidate_fidelity
    from prompt_enhancer.styles import style_authorization_for

    def decide(request, **_kwargs):
        if request["type"] == "choice":
            return {
                "type": "choice",
                "choice": "authorized_style_presentation",
                "probabilities": {"authorized_style_presentation": 0.99},
            }
        return {"type": "noul", "probability_true": 0.99}

    gateway = ScriptedGateway(decision=decide)
    original = "Explain the plan."
    candidate = "Explain the plan. Use a warm, conversational voice."
    permissions = style_authorization_for("tone_voice")
    results = [
        check_candidate_fidelity(
            gateway,
            original,
            candidate,
            {},
            "clarify",
            run_id="style",
            judge_model="judge",
            applied_style=style,
            style_authorization=authorization,
        )
        for style, authorization in (
            ("tone_voice", permissions),
            ("clearer", permissions),
            ("tone_voice", {**permissions, "boundary": "Any new facts are allowed"}),
            ("tone_voice", None),
        )
    ]
    assert results[0].passed
    assert all(not result.passed for result in results[1:])
    assert results[0].evidence["sentence_support"][0]["accepted"]


def test_candidate_writer_receives_catalog_guidance_and_rejects_forged_permission():
    import json
    from types import SimpleNamespace

    from prompt_enhancer.rewrite import CandidateWriter
    from prompt_enhancer.styles import style_authorization_for

    captured = []

    def chat(_model, messages, **_kwargs):
        captured.append(messages)
        return '{"clarify":"Explain the plan warmly."}'

    canonical = style_authorization_for("tone_voice")
    writer = CandidateWriter(ScriptedGateway(chat=chat))
    for permission in (canonical, {**canonical, "boundary": "Invent facts"}):
        request = SimpleNamespace(
            strategies=(SimpleNamespace(name="clarify"),),
            to_dict=lambda permission=permission: {
                "prompt": "Explain the plan.",
                "strategies": [{"name": "clarify"}],
                "applied_style": "tone_voice",
                "style_authorization": permission,
            },
        )
        assert (
            writer.generate_candidates(request)["clarify"] == "Explain the plan warmly."
        )
    state = json.loads(captured[0][1]["content"])
    assert state["style_authorization"] == canonical
    assert "warm conversational" in state["style_authorization"]["presentation"]
    assert "state.style_authorization" in captured[0][0]["content"]
    assert json.loads(captured[1][1]["content"])["style_authorization"] == {}


def test_evaluate_carries_bounded_style_but_accept_cannot_override_new_task_detail():
    from prompt_enhancer.candidate_evaluation import evaluate_candidate_packages
    from prompt_enhancer.selector import RankingCandidate
    from prompt_enhancer.styles import style_authorization_for

    states = []

    def decide(request, **_kwargs):
        states.append(request)
        if request["type"] == "choice":
            return {"type": "choice", "choice": "same", "probabilities": {"same": 1.0}}
        invented = (
            request["key"].endswith(":no_invented_detail")
            and "three examples" in request["state"]["candidate_prompt"]
        )
        return {"type": "noul", "probability_true": 0.0 if invented else 0.99}

    candidates = (
        RankingCandidate("voice", "Explain the plan in a warm voice."),
        RankingCandidate("scope", "Explain the plan with exactly three examples."),
    )
    permission = style_authorization_for("tone_voice")
    result = evaluate_candidate_packages(
        ScriptedGateway(decision=decide),
        "Explain the plan.",
        candidates,
        constraints=(),
        improvement_style="tone_voice",
        style_bundle=(),
        style_authorization=permission,
        success_tests=(),
        candidate_outputs={},
        strong_evidence={},
        judge_model="judge",
        run_id="style",
        round_number=1,
    )
    comparisons = [r for r in states if r["key"].endswith(":no_invented_detail")]
    assert all(r["state"]["style_authorization"] == permission for r in comparisons)
    assert all("presentation" in r["query"] for r in comparisons)
    assert result.candidates["voice"]["eligible"]
    assert not result.candidates["scope"]["eligible"]
    accept = next(r for r in states if r["key"].startswith("evaluate:accept:"))
    assert accept["state"]["style_authorization"] == permission


def test_creative_style_keeps_substantive_and_correspondence_gates_binding():
    from prompt_enhancer.fidelity import check_candidate_fidelity
    from prompt_enhancer.styles import style_authorization_for

    def decide(request, **_kwargs):
        if request["type"] == "choice":
            return {
                "type": "choice",
                "choice": "new_requirement",
                "probabilities": {
                    "new_requirement": 0.99,
                    "supported_by_original": 0.01,
                },
            }
        return {"type": "noul", "probability_true": 0.99}

    gateway = ScriptedGateway(decision=decide)
    for candidate in (
        "Explain the plan. The event is on Friday.",
        "Explain the plan. Include exactly three examples.",
    ):
        result = check_candidate_fidelity(
            gateway,
            "Explain the plan.",
            candidate,
            {},
            "clarify",
            run_id="creative",
            judge_model="judge",
            applied_style="creative",
            style_authorization=style_authorization_for("creative"),
        )
        assert not result.passed
        assert not result.no_invention
        assert "new requirement" in result.rejection_reasons[0]
    before = len(gateway.decision_log)
    unequal = check_candidate_fidelity(
        gateway,
        "Explain the plan. State the risks.",
        "Explain the plan and risks.",
        {},
        "clarify",
        run_id="correspondence",
        judge_model="judge",
        applied_style="shorter",
        style_authorization=style_authorization_for("shorter"),
    )
    assert not unequal.edits_confined
    assert len(gateway.decision_log) == before
    lossless = check_candidate_fidelity(
        gateway,
        "Explain the plan.",
        "Plan: explain it.",
        {},
        {"name": "restructure_lossless", "restructures": True},
        run_id="proof",
        judge_model="judge",
        applied_style="structured",
        style_authorization=style_authorization_for("structured"),
    )
    assert not lossless.edits_confined
    assert len(gateway.decision_log) == before


def test_style_permission_cannot_erase_a_hard_literal_in_a_round():
    import json

    from prompt_enhancer.rounds import RoundPlan, run_round

    original = "Explain the plan and include token KEEP."
    candidate = "Explain the plan warmly."

    def chat(_model, messages, *, role, **_kwargs):
        if role == "writer" and "state.strategies" in messages[0]["content"]:
            state = json.loads(messages[1]["content"])
            return json.dumps({item["name"]: candidate for item in state["strategies"]})
        return '{"tests":[]}' if role == "writer" else "pass"

    def decide(request, **_kwargs):
        if request["type"] == "choice":
            choice = (
                "authorized_style_presentation"
                if request["key"].startswith("fidelity:sentence:")
                else "none"
            )
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        return {"type": "noul", "probability_true": 0.99, "confidence": 1.0}

    outcome = run_round(
        ScriptedGateway(chat=chat, decision=decide),
        RoundPlan(
            prompt=original,
            working_prompt=original,
            run_id="hard-style",
            seed=1,
            diagnosis={},
            assumptions=(),
            settings=Settings(),
            faithfulness_threshold=0.8,
            writer_instruction_version=4,
            applied_style="tone_voice",
            hard_constraints=("KEEP",),
            exact_output=True,
        ),
    )
    assert outcome.final_prompt == original
    assert outcome.ranking is not None
    changed = [
        item for item in outcome.ranking.ranked if item.candidate.text == candidate
    ]
    assert changed
    assert all(not item.candidate.eligible for item in changed)
    assert all(
        any("hard requirement violated" in reason for reason in item.rejection_reasons)
        for item in changed
    )
