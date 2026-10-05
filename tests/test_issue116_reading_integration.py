"""Version 12 reads success criteria with a batched Jev request, not regexes.

Every test drives a ScriptedGateway: no live call reaches a provider. The
reading questions are answered by the scripted decision handler in the exact
shape ``jev.parse_decision`` expects, so the tests exercise the production path
(``CriterionReader`` -> cascade -> run report) end to end.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from prompt_enhancer import criterion_reading
from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.config import Settings
from prompt_enhancer.criterion_checks import check_criterion
from prompt_enhancer.criterion_reading import (
    CRITERION_READING_MIN_VERSION,
    CURRENT_READING_POLICY,
    CriterionReader,
    CriterionReading,
    Reading,
    ReadingPolicy,
    questions_digest,
    resolve_with_policy,
)
from prompt_enhancer.gateway import ProviderError, ScriptedGateway
from prompt_enhancer.grading import grade_panel_with_jev
from prompt_enhancer.grading_cascade import CascadeBudget
from prompt_enhancer.rewrite import (
    CURRENT_WRITER_INSTRUCTION_VERSION,
    WRITER_INSTRUCTION_VERSIONS,
)
from prompt_enhancer.runner import PanelResult


def _choice(selected: str, options: Sequence[str] = ()) -> dict[str, Any]:
    values = tuple(dict.fromkeys((selected, *options)))
    return {
        "type": "choice",
        "choice": selected,
        "probabilities": {value: 1 / len(values) for value in values},
        "confidence": 1.0,
    }


def _noul(probability: float) -> dict[str, Any]:
    return {
        "type": "noul",
        "probability_true": probability,
        "confidence": 0.9,
    }


def reading_handler(
    *,
    kind: str,
    op: str = "none",
    bound: str = "none",
    low: str = "none",
    high: str = "none",
    judgments: Mapping[str, float] | None = None,
    negated: float = 0.0,
) -> Any:
    """A decision handler that answers the eight reading questions."""
    probabilities = dict(
        {"partial": 0.0, "conditional": 0.0, "approximate": 0.0} | dict(judgments or {})
    )
    answers = {
        "kind": kind,
        "op": op,
        "bound": bound,
        "low": low,
        "high": high,
        "partial": probabilities["partial"],
        "conditional": probabilities["conditional"],
        "approximate": probabilities["approximate"],
        "negated": negated,
    }

    def handler(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        name = str(request["key"]).rsplit(":", 1)[-1]
        answer = answers.get(name)
        if answer is None:
            # Any other question (e.g. a grading ask) lands on the uncertain
            # band so the cascade still considers the pair eligible.
            return _noul(0.5)
        if isinstance(answer, float):
            return _noul(answer)
        return _choice(str(answer), tuple(request.get("criteria", ()) or ()))

    return handler


def _panel() -> list[PanelResult]:
    return [PanelResult("candidate", "weak", 0, 7, "word " * 500, "prompt")]


def _reading_gateway(handler: Any) -> ScriptedGateway:
    return ScriptedGateway(decision=handler, jev_model=JEV_MODEL)


# --- Acceptance: the listed criteria resolve to the intended check ----------


@pytest.mark.parametrize(
    ("criterion", "handler", "output", "passed"),
    [
        (
            "Is the letter itself 200 words or fewer, with no unnecessary "
            "preamble or explanation?",
            reading_handler(kind="word_count", op="at most", bound="200"),
            "word " * 500,
            False,
        ),
        (
            "Is the letter itself 200 words or fewer, with no unnecessary "
            "preamble or explanation?",
            reading_handler(kind="word_count", op="at most", bound="200"),
            "word " * 200,
            True,
        ),
        (
            "The reply is no longer than 100 words",
            reading_handler(kind="word_count", op="at most", bound="100"),
            "word " * 500,
            False,
        ),
        (
            "The response is not valid JSON",
            reading_handler(kind="valid_json", negated=0.9),
            '{"a": 1}',
            False,
        ),
        (
            "The response is not valid JSON",
            reading_handler(kind="valid_json", negated=0.9),
            "plain prose",
            True,
        ),
    ],
)
def test_listed_criteria_resolve_to_the_intended_check_on_a_scripted_gateway(
    criterion: str, handler: Any, output: str, passed: bool
) -> None:
    gateway = _reading_gateway(handler)
    reader = CriterionReader(gateway)

    reading = reader.check(criterion, output)

    assert reading.reading is not None
    assert reading.resolved is not None
    assert reading.exact is not None
    assert reading.exact["passed"] is passed
    assert reading.as_check().exact == reading.exact
    # One batched request of every reading question, cached for the run.
    assert reading.request_count == READING_QUESTION_COUNT
    assert len(gateway.calls) == READING_QUESTION_COUNT
    assert reader.request_count == 1


def test_the_regex_would_have_left_the_writer_phrasing_unresolved() -> None:
    """The regression: the same criterion was unresolved before version 12."""
    criterion = "Is the letter itself 200 words or fewer, with no unnecessary preamble?"
    legacy = check_criterion(criterion, "word " * 500)

    assert legacy.exact is None
    assert legacy.unsupported == ("words",)

    reading = CriterionReader(
        _reading_gateway(reading_handler(kind="word_count", op="at most", bound="200"))
    ).check(criterion, "word " * 500)

    assert reading.exact is not None
    assert reading.exact["operator"] == "at most"
    assert reading.exact["bound"] == 200
    assert reading.unsupported == ()


# --- Acceptance: conditional and part-scoped criteria never use whole output --


@pytest.mark.parametrize(
    ("criterion", "handler"),
    [
        (
            "Under 100 words, or a table if longer",
            reading_handler(
                kind="word_count",
                op="under",
                bound="100",
                judgments={"conditional": 0.9},
            ),
        ),
        (
            "The intro paragraph is under 50 words",
            reading_handler(
                kind="word_count",
                op="under",
                bound="50",
                judgments={"partial": 0.8},
            ),
        ),
    ],
)
def test_conditional_and_part_scoped_criteria_are_never_checked_against_the_output(
    criterion: str, handler: Any
) -> None:
    output = "word " * 500
    # The regex's known failure: it checks these against the whole output.
    assert check_criterion(criterion, output).exact is not None

    reading = CriterionReader(_reading_gateway(handler)).check(criterion, output)

    assert reading.exact is None
    assert reading.resolved is None
    assert reading.reason == "criterion_reading_unresolved"


def test_part_scoped_criterion_vetoes_a_regex_check_the_reading_abstains_on() -> None:
    criterion = "The intro paragraph is under 50 words"
    output = "word " * 500
    reader = CriterionReader(
        _reading_gateway(
            reading_handler(
                kind="word_count", op="under", bound="50", judgments={"partial": 0.9}
            )
        )
    )

    reading = reader.check(criterion, output)
    merged = reading.as_check(regex=check_criterion(criterion, output))

    assert merged.exact is None
    assert merged.unsupported == ("criterion_reading_unresolved",)


# --- Acceptance: the audited policy and the shared resolve policy -----------


def test_default_policy_matches_the_audited_cutoffs() -> None:
    assert CURRENT_READING_POLICY.version == "issue-116-v1"
    assert CURRENT_READING_POLICY.cutoffs() == {
        "partial": 0.40,
        "conditional": 0.50,
        "approximate": 0.50,
    }
    assert CURRENT_READING_POLICY.band() == (0.35, 0.65)


def test_partial_between_the_default_and_the_old_cutoff_flips_the_outcome() -> None:
    """`partial` 0.40 is a real change: 0.5 would have resolved this."""
    reading = Reading(
        kind="word_count",
        op="under",
        bound="100",
        low="none",
        high="none",
        noul={"partial": 0.45, "conditional": 0.0, "negated": 0.0, "approximate": 0.0},
    )

    assert resolve_with_policy(reading, {"100": 100}) is None
    legacy = criterion_reading.resolve(reading, {"100": 100}, (0.5, 0.5))
    assert legacy is not None


def test_production_reader_uses_the_shared_resolve_and_question_digest() -> None:
    reader = CriterionReader(
        _reading_gateway(reading_handler(kind="word_count", op="under", bound="100"))
    )

    reading = reader.check("The answer is under 100 words", "word " * 500)

    assert reading.policy_version == CURRENT_READING_POLICY.version
    assert reading.questions_digest == questions_digest()
    # The reading is what the measurement script's `resolve` would build.
    assert reading.resolved == criterion_reading.resolve(
        reading.reading,
        reading.candidates,
        CURRENT_READING_POLICY.band(),
        cutoffs=CURRENT_READING_POLICY.cutoffs(),
    )


def test_a_range_resolves_only_when_both_ends_are_candidate_numbers() -> None:
    reader = CriterionReader(
        _reading_gateway(
            reading_handler(kind="word_count", op="between", low="100", high="150")
        )
    )

    reading = reader.check("The answer is 100-150 words long", "word " * 120)

    assert reading.exact is not None
    assert reading.exact["operator"] == "between"
    assert (reading.exact["bound"], reading.exact["upper_bound"]) == (100, 150)


def test_sentence_count_stays_unresolved_because_code_cannot_count_sentences() -> None:
    reader = CriterionReader(
        _reading_gateway(
            reading_handler(kind="sentence_count", op="at most", bound="3")
        )
    )

    reading = reader.check("The reply is at most 3 sentences", "One. Two. Three.")

    assert reading.reading is not None
    assert reading.exact is None
    assert reading.reason == "criterion_reading_sentence_count_unsupported"


# --- Acceptance: a failed or unavailable reading leaves it unresolved --------


def test_provider_failure_leaves_the_criterion_unresolved_and_is_recorded() -> None:
    def failing(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        raise ProviderError("scripted", JEV_MODEL, None, "reading failed")

    reader = CriterionReader(_reading_gateway(failing))

    reading = reader.check("The answer is under 100 words", "word " * 500)

    assert reading.reading is None
    assert reading.exact is None
    assert reading.reason == "criterion_reading_provider_network"
    assert reading.request_count == 0
    assert reading.to_evidence()["reason"] == reading.reason


def test_incomplete_answers_leave_the_criterion_unresolved() -> None:
    def partial(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if str(request["key"]).endswith(":kind"):
            return _choice("word_count")
        raise ProviderError("scripted", JEV_MODEL, None, "no answer")

    gateway = ScriptedGateway(decision=partial, jev_model=JEV_MODEL)
    gateway.decide_batch = lambda requests, **_kwargs: [  # type: ignore[method-assign]
        _choice("word_count")
    ]

    reading = CriterionReader(gateway).check("The answer is under 100 words", "word")

    assert reading.reading is None
    assert reading.reason == "criterion_reading_incomplete"


def test_unusable_answer_shape_leaves_the_criterion_unresolved() -> None:
    def wrong_shape(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        return {"type": "unknown", "answer": "yes"}

    reading = CriterionReader(_reading_gateway(wrong_shape)).check(
        "The answer is under 100 words", "word " * 500
    )

    assert reading.reading is None
    assert reading.reason == "criterion_reading_unusable"


def test_a_failed_reading_keeps_the_pre_version_12_answer() -> None:
    def failing(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        return {"type": "unknown", "answer": "yes"}

    criterion = "The answer is under 100 words"
    output = "word " * 500
    reading = CriterionReader(_reading_gateway(failing)).check(criterion, output)

    assert reading.reading is None
    assert reading.as_check(regex=check_criterion(criterion, output)) == (
        check_criterion(criterion, output)
    )


# --- Acceptance: the conflict rule is unchanged -----------------------------


def test_code_check_disagreeing_with_jev_stays_unresolved() -> None:
    """Jev says pass; the counted reading says fail; the pair is unresolved."""
    cascade = grade_recorded(
        version=12,
        reading=reading_handler(kind="word_count", op="under", bound="100"),
        confirmation=(0.95, 0.95, 0.05),
    )

    pair = cascade["pairs"][0]
    assert pair["status"] == "unresolved"
    assert pair["deterministic_conflict"]["passed"] is False
    # The reading resolved a check whose count disagrees with Jev's "pass".
    reading = pair["criterion_reading"]
    assert reading["resolved"] is not None
    assert reading["check"]["passed"] is False


def test_conflict_rule_accepts_evidence_that_agrees_with_the_count() -> None:
    """The same evidence with a failing verdict is not a conflict."""
    from prompt_enhancer.grading_cascade import _strong_evidence

    run = PanelResult("candidate", "weak", 0, 7, "word " * 30, "prompt")
    reader = CriterionReader(
        _reading_gateway(reading_handler(kind="word_count", op="under", bound="100"))
    )
    raw = json.dumps(
        {
            "suggested_verdict": "pass",
            "prompt_quote": "prompt",
            "output_quote": run.output,
            "rationale": "The answer is well inside the limit.",
        }
    )

    proposed, reason = _strong_evidence(raw, run, CRITERION, reader)

    assert reason == "valid_evidence"
    assert proposed is not None
    assert proposed["suggested_verdict"] == "pass"


def test_conflict_rule_rejects_strong_evidence_that_contradicts_the_count() -> None:
    from prompt_enhancer.grading_cascade import _strong_evidence

    run = _panel()[0]
    reader = CriterionReader(
        _reading_gateway(reading_handler(kind="word_count", op="under", bound="100"))
    )
    raw = json.dumps(
        {
            "suggested_verdict": "pass",
            "prompt_quote": "prompt",
            "output_quote": "word word",
            "rationale": "The answer looks well under the limit.",
        }
    )

    proposed, reason = _strong_evidence(raw, run, CRITERION, reader)

    assert proposed is None
    assert reason == "deterministic_evidence_conflict"


# --- Acceptance: pre-version-12 recordings reproduce -------------------------


def test_reading_gate_version_is_supported_by_the_current_writer() -> None:
    assert CRITERION_READING_MIN_VERSION == 12
    assert CRITERION_READING_MIN_VERSION in WRITER_INSTRUCTION_VERSIONS
    assert CURRENT_WRITER_INSTRUCTION_VERSION >= CRITERION_READING_MIN_VERSION


@pytest.mark.parametrize("version", [7, 11])
def test_pre_version_12_recordings_make_no_reading_requests(
    version: int, tmp_path: Path
) -> None:
    """Recordings made before version 12 keep the regex path exactly."""
    path = tmp_path / f"cascade-{version}.json"
    grade_recorded(record_path=path, version=version)
    recorded = json.loads(path.read_text(encoding="utf-8"))

    assert recorded["writer_instruction_version"] == version
    assert "criterion_reading" not in recorded
    requests = json.dumps(list(recorded["responses"].values()))
    assert "criterion-reading" not in requests


def test_version_12_recordings_replay_reproduce_exactly(tmp_path: Path) -> None:
    path = tmp_path / "cascade-12.json"
    original, _ = grade_recorded(version=12, record_path=path, return_batches=True)
    recorded = json.loads(path.read_text(encoding="utf-8"))

    assert recorded["writer_instruction_version"] == 12
    assert recorded["criterion_reading"]["policy_version"] == (
        CURRENT_READING_POLICY.version
    )
    assert recorded["criterion_reading"]["questions_digest"] == questions_digest()
    # A strict ReplayGateway answers only an exactly-recorded request, so a
    # successful replay proves the eight reading questions were recorded with
    # the same keys, state and criteria as the live path sent.
    replayed = replay_grade(recorded)
    assert replayed == original

    legacy = grade_recorded(version=11, record_path=tmp_path / "cascade-11.json")
    legacy_recorded = json.loads((tmp_path / "cascade-11.json").read_text())
    assert len(recorded["responses"]) > len(legacy_recorded["responses"])
    assert "criterion_reading" not in legacy


# --- Acceptance: the run report keeps per-criterion evidence ----------------


def test_run_report_records_the_reading_and_probabilities_per_criterion() -> None:
    cascade = grade_recorded(version=12)
    summary = cascade["criterion_reading"]

    assert summary["policy_version"] == CURRENT_READING_POLICY.version
    assert summary["questions_digest"] == questions_digest()
    assert summary["cutoffs"] == CURRENT_READING_POLICY.cutoffs()
    assert summary["json_negation_band"] == [0.35, 0.65]
    assert summary["distinct_criteria"] >= 1
    assert summary["request_count"] == summary["distinct_criteria"]
    read = summary["reads"][0]
    assert read["criterion"]
    assert set(read["reading"]) == {"kind", "op", "bound", "low", "high"}
    assert set(read["judgment_probabilities"]) == {
        "partial",
        "conditional",
        "negated",
        "approximate",
    }
    assert read["answered_by"] == JEV_MODEL
    assert read["reason"]


def test_run_report_usage_includes_the_reading_requests() -> None:
    _, batches = grade_recorded(version=12, return_batches=True)
    reading_batches = [
        batch
        for batch in batches
        if batch and str(batch[0]["key"]).startswith("criterion-reading:")
    ]

    assert reading_batches
    assert all(len(batch) == READING_QUESTION_COUNT for batch in reading_batches)
    assert grade_recorded(version=12)["criterion_reading"]["request_count"] == len(
        reading_batches
    )


# --- Acceptance: a failed or unavailable reading leaves it unresolved --------


def test_unresolved_reading_does_not_fail_the_run() -> None:
    """A reading that abstains leaves the pair unresolved; the run completes."""
    cascade, _ = grade_recorded(
        version=12,
        reading=reading_handler(
            kind="word_count", op="under", bound="100", judgments={"partial": 0.9}
        ),
        return_batches=True,
    )

    assert cascade["criterion_reading"]["unresolved_count"] == 1
    assert cascade["criterion_reading"]["failed_count"] == 0
    assert cascade["criterion_reading"]["request_count"] == 1
    pair = cascade["pairs"][0]
    assert "criterion-reading:partial" in pair["criterion_reading"]["reason"] or (
        pair["criterion_reading"]["resolved"] is None
    )
    assert pair["criterion_reading"]["reason"] == "criterion_reading_unresolved"


def test_failed_reading_does_not_fail_the_run_and_is_visible() -> None:
    def failing(request: Mapping[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if str(request["key"]).startswith("criterion-reading:"):
            raise ProviderError("scripted", JEV_MODEL, None, "reading down")
        return _noul(0.5)

    cascade = grade_recorded(version=12, reading=failing)
    reported = cascade["criterion_reading"]

    assert reported["failed_count"] == 1
    assert reported["failure_reasons"] == ["criterion_reading_provider_network"]
    assert reported["resolved_count"] == 0
    assert cascade["pairs"][0]["criterion_reading"]["reason"] == (
        "criterion_reading_provider_network"
    )


# --- Acceptance: one reading request per distinct criterion -----------------


def test_reading_is_cached_per_criterion_across_pairs() -> None:
    gateway = _reading_gateway(
        reading_handler(kind="word_count", op="under", bound="100")
    )
    reader = CriterionReader(gateway)

    first = reader.check(CRITERION, "word " * 10)
    second = reader.check(CRITERION, "word " * 500)
    other = reader.check("The answer is under 50 words", "word " * 10)

    assert len(gateway.calls) == 2 * READING_QUESTION_COUNT  # one batch per criterion
    assert reader.request_count == 2
    assert first.exact is not None and first.exact["passed"] is True
    assert second.exact is not None and second.exact["passed"] is False
    assert other.exact is None  # "50" is not the selected candidate


def test_reader_can_be_built_with_a_different_policy() -> None:
    gateway = _reading_gateway(
        reading_handler(
            kind="word_count", op="under", bound="100", judgments={"partial": 0.45}
        )
    )
    lenient = CriterionReader(gateway, policy=ReadingPolicy(partial=0.50))

    unresolved = CriterionReader(gateway).check(CRITERION, "word " * 10)
    resolved = lenient.check(CRITERION, "word " * 10)

    assert unresolved.exact is None
    assert resolved.exact is not None
    assert resolved.policy_version == "issue-116-v1"


def test_criterion_reading_evidence_round_trips_through_json() -> None:
    reader = CriterionReader(
        _reading_gateway(reading_handler(kind="word_count", op="under", bound="100"))
    )
    reading: CriterionReading = reader.check(CRITERION, "word " * 10)

    restored = json.loads(json.dumps(reading.to_evidence()))

    assert restored["check"]["operator"] == "under"
    assert restored["judgment_probabilities"]["partial"] == 0.0
    assert restored["resolved"]["bound"] == 100


def test_recorded_policy_round_trips_and_rejects_malformed_metadata() -> None:
    from prompt_enhancer.criterion_reading import (
        CURRENT_READING_POLICY,
        policy_from_metadata,
    )

    policy = policy_from_metadata(criterion_reading.recording_metadata())

    assert policy == CURRENT_READING_POLICY
    for malformed in (
        None,
        {},
        {"cutoffs": {"partial": 0.4}},
        {"cutoffs": {}, "json_negation_band": []},
        {"cutoffs": {}, "json_negation_band": [0.9, 0.1]},
        {"cutoffs": {}, "json_negation_band": [0.1, 2.0]},
        {"cutoffs": {"partial": "high"}, "json_negation_band": [0.3, 0.7]},
    ):
        assert policy_from_metadata(malformed) == CURRENT_READING_POLICY


# --- The scripted grading harness -------------------------------------------


PROMPT = "Read the background notes. Summarize the report."
CRITERION = "The answer is under 100 words"
# The reader asks the five choice questions plus the four yes/no judgments.
READING_QUESTION_COUNT = len(criterion_reading.reading_questions(["100"]))


def _grading_decide(
    reading: Any, *, confirmation: tuple[float, float, float] = (0.95, 0.95, 0.05)
) -> Any:
    probabilities = dict(
        zip(("sufficient", "meets", "violation"), confirmation, strict=True)
    )

    def decide(request: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        key = str(request.get("key", ""))
        if key.startswith("criterion-reading:"):
            return reading(request, **kwargs)
        if key.startswith("grade-confirm:"):
            return _noul(probabilities[key.rsplit(":", 1)[-1]])
        return _noul(0.5)

    return decide


def grade_recorded(
    *,
    version: int,
    reading: Any = None,
    record_path: Path | None = None,
    return_batches: bool = False,
    confirmation: tuple[float, float, float] = (0.95, 0.95, 0.05),
) -> Any:
    """Grade one pair with the cascade, optionally recording every request."""
    batches: list[list[dict[str, Any]]] = []
    handler = _grading_decide(
        reading or reading_handler(kind="word_count", op="under", bound="100"),
        confirmation=confirmation,
    )

    class CountingGateway(ScriptedGateway):
        def decide_batch(
            self,
            requests: Sequence[Mapping[str, Any]],
            *,
            role: str = "judge",
            run_id: str | None = None,
        ) -> list[Any]:
            batches.append([dict(request) for request in requests])
            return super().decide_batch(requests, role=role, run_id=run_id)

    gateway: Any = CountingGateway(decision=handler, jev_model=JEV_MODEL)
    if record_path is not None:
        from prompt_enhancer.evaluation.recording import RecordingGateway

        recording = RecordingGateway(gateway, record_path)
        recording.writer_instruction_version = version
        recording.faithfulness_threshold = 0.8
        if version >= CRITERION_READING_MIN_VERSION:
            recording.criterion_reading = criterion_reading.recording_metadata()
        gateway = recording
    observation: dict[str, Any] = {}
    grade_panel_with_jev(
        _panel(),
        [{"id": "t0", "question": CRITERION, "kind": "noul"}],
        gateway,
        judge_model=JEV_MODEL,
        run_id="issue-116",
        shared_state=True,
        cascade_budget=CascadeBudget.for_settings(Settings()),
        cascade_observation=observation,
        cascade_strong_model="strong",
        read_criteria=version >= CRITERION_READING_MIN_VERSION,
    )
    if return_batches:
        return observation, batches
    return observation


def replay_grade(recorded: Mapping[str, Any]) -> dict[str, Any]:
    """Grade the same pair with a ReplayGateway built from a recording."""
    from prompt_enhancer.gateway import ReplayGateway

    gateway = ReplayGateway(
        recorded["responses"],
        decision_provenance=recorded.get("decision_provenance", {}),
        jev_model=recorded["jev_model"],
        allow_snapshot_mismatch=True,
    )
    observation: dict[str, Any] = {}
    grade_panel_with_jev(
        _panel(),
        [{"id": "t0", "question": CRITERION, "kind": "noul"}],
        gateway,
        judge_model=JEV_MODEL,
        run_id="issue-116",
        shared_state=True,
        cascade_budget=CascadeBudget.for_settings(Settings()),
        cascade_observation=observation,
        cascade_strong_model="strong",
        read_criteria=True,
    )
    return observation
