"""The labelled criterion set and the offline half of reading criteria with a model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import measure_criterion_reading as measurement
from measure_criterion_reading import (
    OPERATORS,
    Reading,
    load_cases,
    number_candidates,
    outcome,
    read_jev,
    regex_check,
    resolve,
    score,
)
from measure_criterion_reading import _reading as parse_recorded_reading

from prompt_enhancer.jev import NoulDecision


def _reading(**overrides: object) -> Reading:
    fields: dict[str, object] = {
        "kind": "word_count",
        "op": "under",
        "bound": "100",
        "low": "none",
        "high": "none",
        "noul": {
            "partial": 0.1,
            "conditional": 0.1,
            "negated": 0.1,
            "approximate": 0.1,
        },
    }
    return Reading(**{**fields, **overrides})  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_fixture_has_50_development_and_100_heldout_unique_criteria() -> None:
    cases = load_cases("all")

    assert Counter(case["split"] for case in cases) == {
        "development": 50,
        "heldout": 100,
    }
    assert len({case["criterion"] for case in cases}) == len(cases)
    assert len({case["id"] for case in cases}) == len(cases)


def test_every_label_is_a_supported_check_or_null() -> None:
    for case in load_cases("all"):
        expected = case["expected"]
        if expected is None:
            continue
        if expected["kind"] == "valid_json":
            assert set(expected) == {"kind", "negated"}
            continue
        assert expected["kind"] in {"word_count", "sentence_count"}
        assert expected["operator"] in OPERATORS
        assert (expected["operator"] == "between") == (
            expected["upper_bound"] is not None
        )
        if expected["upper_bound"] is not None:
            assert expected["bound"] < expected["upper_bound"]


def test_regex_baseline_reproduces_the_issue_on_the_development_split() -> None:
    cases = load_cases("development")

    tally = score(cases, {c["id"]: regex_check(c["criterion"]) for c in cases})

    assert (
        tally["correct_check"],
        tally["correct_abstain"],
        tally["missed"],
        tally["wrong_confident"],
    ) == (8, 11, 24, 7)


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("at most 1,000 words", {"1,000": 1000}),
        ("under one hundred words", {"one hundred": 100}),
        ("twenty-five words or fewer", {"twenty-five": 25}),
        ("one hundred and fifty words", {"one hundred and fifty": 150}),
        (
            "between two hundred and three hundred words",
            {"two hundred": 200, "three hundred": 300},
        ),
        ("between three and six sentences", {"three": 3, "six": 6}),
        ("the 5-paragraph essay in 500 words", {"5": 5, "500": 500}),
        ("no one is someone", {"one": 1}),
    ],
)
def test_number_candidates_are_valued_in_code(text: str, spans: dict[str, int]) -> None:
    assert number_candidates(text) == spans


def test_resolve_builds_a_check_from_the_selected_span_and_operator() -> None:
    check = resolve(
        _reading(op="at most", bound="two hundred"),
        {"two hundred": 200},
        (0.5, 0.5),
    )

    assert check == {
        "kind": "word_count",
        "operator": "at most",
        "bound": 200,
        "upper_bound": None,
    }


def test_resolve_orders_a_range_and_needs_two_distinct_ends() -> None:
    candidates = {"3": 3, "5": 5}

    swapped = resolve(_reading(op="between", low="5", high="3"), candidates, (0.5, 0.5))
    same = resolve(_reading(op="between", low="5", high="5"), candidates, (0.5, 0.5))

    assert swapped is not None
    assert (swapped["bound"], swapped["upper_bound"]) == (3, 5)
    assert same is None


@pytest.mark.parametrize("judgment", ["partial", "conditional", "approximate"])
def test_resolve_leaves_a_flagged_criterion_unresolved(judgment: str) -> None:
    noul = {"partial": 0.1, "conditional": 0.1, "negated": 0.1, "approximate": 0.1}
    reading = _reading(noul={**noul, judgment: 0.5})

    assert resolve(reading, {"100": 100}, (0.5, 0.5)) is None
    assert resolve(reading, {"100": 100}, (0.6, 0.6)) is not None


def test_resolve_never_invents_a_bound_the_criterion_does_not_contain() -> None:
    assert resolve(_reading(bound="none"), {"100": 100}, (0.5, 0.5)) is None
    assert resolve(_reading(bound="250"), {"100": 100}, (0.5, 0.5)) is None
    assert resolve(_reading(kind="other"), {"100": 100}, (0.5, 0.5)) is None


def test_resolve_negates_json_and_abstains_between_the_band_edges() -> None:
    noul = {"partial": 0.0, "conditional": 0.0, "approximate": 0.0}
    reading = _reading(kind="valid_json", op="none", bound="none")

    def with_negation(probability: float) -> Reading:
        return _reading(
            kind=reading.kind,
            op=reading.op,
            bound=reading.bound,
            noul={**noul, "negated": probability},
        )

    band = (0.2, 0.8)
    assert resolve(with_negation(0.1), {}, band) == {
        "kind": "valid_json",
        "negated": False,
    }
    assert resolve(with_negation(0.48), {}, band) is None
    assert resolve(with_negation(0.9), {}, band) == {
        "kind": "valid_json",
        "negated": True,
    }


def test_outcome_separates_wrong_from_missed_and_correctly_unchecked() -> None:
    expected = {"kind": "valid_json", "negated": False}

    assert outcome(expected, expected) == "correct_check"
    assert outcome(None, None) == "correct_abstain"
    assert outcome(None, expected) == "missed"
    assert outcome(expected, None) == "wrong_confident"
    assert outcome({"kind": "valid_json", "negated": True}, expected) == (
        "wrong_confident"
    )


@pytest.mark.parametrize("reply", ["[]", "null", "1", "{}"])
def test_malformed_chat_replies_are_unusable_rows(reply: str) -> None:
    assert parse_recorded_reading({"reader": "cheap"}, {"reply": reply}) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "bogus"),
        ("op", "bogus"),
        ("bound", "bogus"),
        ("low", "bogus"),
        ("high", "bogus"),
    ],
)
def test_chat_replies_with_out_of_schema_choices_are_unusable_rows(
    field: str, value: str
) -> None:
    data = {
        "kind": "word_count",
        "op": "under",
        "bound": "100",
        "low": "none",
        "high": "none",
        "partial": False,
        "conditional": False,
        "negated": False,
        "approximate": False,
    }
    data[field] = value

    row = {"reply": json.dumps(data)}
    assert parse_recorded_reading({"reader": "cheap"}, row, ["100"]) is None
    with pytest.raises(ValueError):
        measurement.read_chat(row["reply"], ["100"])


def test_read_jev_rejects_an_unexpected_decision_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        measurement,
        "parse_decision",
        lambda _payload: NoulDecision(probability=0.25, confidence=0.75),
    )

    with pytest.raises(TypeError, match="expected choice"):
        read_jev({"kind": {}})

    assert parse_recorded_reading({"reader": "jev"}, {"answers": {"kind": {}}}) is None


def test_record_refuses_to_resume_with_a_different_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reader": "jev",
                "model": "old-model",
                "questions_digest": measurement.questions_digest(),
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        measurement,
        "HttpGateway",
        lambda **_kwargs: SimpleNamespace(jev_model="current-model"),
    )

    with pytest.raises(ValueError, match="model"):
        measurement.record("jev", [], output)


def test_record_refuses_to_resume_cheap_recording_with_a_different_token_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reader": "cheap",
                "model": measurement.CHEAP_MODEL,
                "max_tokens": 256,
                "questions_digest": measurement.questions_digest(),
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(measurement, "HttpGateway", lambda **_kwargs: SimpleNamespace())

    with pytest.raises(ValueError, match="max_tokens"):
        measurement.record("cheap", [], output, max_tokens=300)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_non_positive_max_tokens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_criterion_reading.py",
            "run",
            "--reader",
            "cheap",
            "--max-tokens",
            value,
            "--output",
            "recording.json",
        ],
    )
    monkeypatch.setattr(measurement, "record", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit) as error:
        measurement.main()

    assert error.value.code == 2
    assert "--max-tokens must be a positive integer" in capsys.readouterr().err


def test_cli_rejects_max_tokens_for_jev_reader(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_criterion_reading.py",
            "run",
            "--reader",
            "jev",
            "--max-tokens",
            "250",
            "--output",
            "recording.json",
        ],
    )
    monkeypatch.setattr(measurement, "record", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit) as error:
        measurement.main()

    assert error.value.code == 2
    assert (
        "--max-tokens applies only to the cheap chat reader" in capsys.readouterr().err
    )


def test_read_chat_parses_a_well_formed_reply() -> None:
    reply = json.dumps(
        {
            "kind": "word_count",
            "op": "under",
            "bound": "100",
            "low": "none",
            "high": "none",
            "partial": True,
            "conditional": False,
            "negated": False,
            "approximate": False,
        }
    )

    reading = measurement.read_chat(reply, ["100"])

    assert reading == _reading(
        noul={
            "partial": 1.0,
            "conditional": 0.0,
            "negated": 0.0,
            "approximate": 0.0,
        }
    )


def test_report_warns_when_question_wording_digest_is_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = tmp_path / "recording.json"
    recording.write_text(
        json.dumps(
            {
                "reader": "cheap",
                "model": "test-model",
                "questions_digest": "old-digest",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    measurement.report(recording, show_errors=False)
    output = capsys.readouterr().out

    assert "WARNING" in output
    assert "question wording" in output


def test_report_scores_recording_with_truncation_and_fixture_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {
            "id": "dev-case",
            "split": "development",
            "criterion": "The answer is under 100 words",
            "expected": {
                "kind": "word_count",
                "operator": "under",
                "bound": 100,
                "upper_bound": None,
            },
        },
        {
            "id": "extra-case",
            "split": "challenge",
            "criterion": "Write a friendly answer",
            "expected": None,
        },
    ]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    monkeypatch.setattr(measurement, "FIXTURE", fixture)

    reply = json.dumps(
        {
            "kind": "word_count",
            "op": "under",
            "bound": "100",
            "low": "none",
            "high": "none",
            "partial": False,
            "conditional": False,
            "negated": False,
            "approximate": False,
        }
    )
    recording = tmp_path / "recording.json"
    recording.write_text(
        json.dumps(
            {
                "reader": "cheap",
                "model": measurement.CHEAP_MODEL,
                "questions_digest": measurement.questions_digest(),
                "max_tokens": 50,
                "rows": [
                    {
                        "case_id": "dev-case",
                        "reply": reply,
                        "usage": {"output_tokens": 50},
                    },
                    {
                        "case_id": "extra-case",
                        "reply": "[]",
                        "usage": {"completion_tokens": 2},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    measurement.report(
        recording,
        show_errors=False,
        cutoffs=measurement.parse_cutoffs(
            "partial=0.2,conditional=0.5,approximate=0.5"
        ),
    )
    output = capsys.readouterr().out

    assert "| challenge |" in output
    assert "unusable answers: extra-case" in output
    assert "truncation-suspect: dev-case" in output
    assert "partial: n=1 median=0.000 p90=0.000 max=0.000" in output
    assert "conditional: n=1 median=0.000 p90=0.000 max=0.000" in output
    assert "approximate: n=1 median=0.000 p90=0.000 max=0.000" in output
    assert "| development | per-judgment cutoffs partial >= 0.2" in output


def test_report_includes_negation_distribution_only_for_valid_json_readings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {
            "id": "json-false",
            "split": "development",
            "criterion": "The output must be valid JSON",
            "expected": {"kind": "valid_json", "negated": False},
        },
        {
            "id": "json-true",
            "split": "development",
            "criterion": "The output must not be valid JSON",
            "expected": {"kind": "valid_json", "negated": True},
        },
        {
            "id": "json-other",
            "split": "development",
            "criterion": "The output should be parseable as JSON",
            "expected": {"kind": "valid_json", "negated": False},
        },
        {
            "id": "word-count",
            "split": "development",
            "criterion": "The answer is under 100 words",
            "expected": {
                "kind": "word_count",
                "operator": "under",
                "bound": 100,
                "upper_bound": None,
            },
        },
        {
            "id": "json-unusable",
            "split": "development",
            "criterion": "The response is not valid JSON",
            "expected": {"kind": "valid_json", "negated": False},
        },
    ]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    monkeypatch.setattr(measurement, "FIXTURE", fixture)

    def reply(kind: str, negated: bool) -> str:
        return json.dumps(
            {
                "kind": kind,
                "op": "none" if kind == "valid_json" else "under",
                "bound": "none" if kind == "valid_json" else "100",
                "low": "none",
                "high": "none",
                "partial": False,
                "conditional": False,
                "negated": negated,
                "approximate": False,
            }
        )

    recording = tmp_path / "recording.json"
    rows = [
        {"case_id": case_id, "reply": reply(kind, negated)}
        for case_id, kind, negated in (
            ("json-false", "valid_json", False),
            ("json-true", "valid_json", True),
            ("json-other", "valid_json", False),
            ("word-count", "word_count", True),
        )
    ]
    # An unparseable reply yields no Reading, so json-unusable exercises the
    # usable-readings guard in the negation distribution: five JSON rows are
    # recorded but only three usable checkable readings are counted.
    rows.append({"case_id": "json-unusable", "reply": "not-json"})
    recording.write_text(
        json.dumps(
            {
                "reader": "cheap",
                "model": measurement.CHEAP_MODEL,
                "questions_digest": measurement.questions_digest(),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

    measurement.report(recording, show_errors=False)
    output = capsys.readouterr().out

    assert "unusable answers: json-unusable" in output
    assert "negated: n=3 median=0.000 p90=1.000 max=1.000" in output

    cases[:] = [cases[-1]]
    fixture.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    # The single fixture case now has one usable word_count-kind reading, so
    # the n=0 line reflects kind exclusion, not an empty case set: dropping
    # the valid_json filter would count this negated=True reading and fail
    # the assertion below.
    recording.write_text(
        json.dumps(
            {
                "reader": "cheap",
                "model": measurement.CHEAP_MODEL,
                "questions_digest": measurement.questions_digest(),
                "rows": [
                    {
                        "case_id": "json-unusable",
                        "reply": json.dumps(
                            {
                                "kind": "word_count",
                                "op": "none",
                                "bound": "none",
                                "low": "none",
                                "high": "none",
                                "partial": False,
                                "conditional": False,
                                "negated": True,
                                "approximate": False,
                            }
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    measurement.report(recording, show_errors=False)
    output = capsys.readouterr().out

    assert "negated: n=0 (no usable valid_json readings)" in output


def test_unknown_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        load_cases("not-in-fixture")


def test_per_judgment_cutoffs_parse_and_change_resolution() -> None:
    cutoffs = measurement.parse_cutoffs("partial=0.7,conditional=0.5,approximate=0.5")
    reading = _reading(
        noul={
            "partial": 0.6,
            "conditional": 0.4,
            "negated": 0.1,
            "approximate": 0.4,
        }
    )

    assert cutoffs == {"partial": 0.7, "conditional": 0.5, "approximate": 0.5}
    assert resolve(reading, {"100": 100}, (0.5, 0.5)) is None
    assert resolve(reading, {"100": 100}, (0.5, 0.5), cutoffs=cutoffs) == {
        "kind": "word_count",
        "operator": "under",
        "bound": 100,
        "upper_bound": None,
    }


@pytest.mark.parametrize("invalid_field", ["partial", "unexpected"])
def test_nonconforming_chat_fields_are_unusable_rows(invalid_field: str) -> None:
    data: dict[str, object] = {
        "kind": "word_count",
        "op": "under",
        "bound": "100",
        "low": "none",
        "high": "none",
        "partial": False,
        "conditional": False,
        "negated": False,
        "approximate": False,
    }
    if invalid_field == "partial":
        data["partial"] = "yes"
    else:
        data[invalid_field] = True

    # Candidates keep bound/low/high in-schema so the partial and unexpected
    # variants reach the field validation they target instead of failing on
    # the bound enum with empty candidates.
    assert (
        parse_recorded_reading(
            {"reader": "cheap"}, {"reply": json.dumps(data)}, ["100"]
        )
        is None
    )


def test_report_flags_legacy_cheap_recording_at_default_token_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = tmp_path / "legacy-recording.json"
    recording.write_text(
        json.dumps(
            {
                "reader": "cheap",
                "model": "test-model",
                "questions_digest": measurement.questions_digest(),
                "rows": [
                    {
                        "case_id": "dev-001",
                        "reply": "{}",
                        "usage": {"completion_tokens": 300},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    measurement.report(recording, show_errors=False)

    assert "truncation-suspect: dev-001" in capsys.readouterr().out


def _recording_cases() -> list[dict[str, object]]:
    return [
        {
            "id": f"case-{index}",
            "split": "development",
            "criterion": f"The answer is under {100 + index} words",
            "expected": None,
        }
        for index in range(2)
    ]


class _RecordingGateway:
    def __init__(
        self,
        *,
        config: measurement.GatewayConfig,
        fail_on_second_chat: bool = False,
        shared_calls: list[tuple[str, str]] | None = None,
    ) -> None:
        self.jev_model = config.jev_model
        self.decision_log: list[dict[str, object]] = []
        self.calls = shared_calls if shared_calls is not None else []
        self.chat_calls = 0
        self.fail_on_second_chat = fail_on_second_chat

    def decide_batch(
        self, requests: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        self.calls.append(("jev", self.jev_model))
        self.decision_log.append(
            {"answered_by": self.jev_model, "usage": {"input_tokens": 7}}
        )
        answers: list[dict[str, object]] = []
        for request in requests:
            key = request["key"]
            selected = (
                "word_count"
                if key == "kind"
                else "under"
                if key == "op"
                else "100"
                if key == "bound"
                else "none"
            )
            answers.append(
                {
                    "type": "choice",
                    "selected": selected,
                    "options": [{"value": selected, "probability": 1}],
                }
                if key in ("kind", "op", "bound", "low", "high")
                else {"type": "noul", "probability_true": 0}
            )
        return answers

    def chat(self, model: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        self.calls.append(("cheap", model))
        self.chat_calls += 1
        if self.fail_on_second_chat and self.chat_calls == 2:
            raise RuntimeError("synthetic provider failure")
        return {
            "model": model,
            "usage": {"completion_tokens": 12},
            "finish_reason": "stop",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "kind": "word_count",
                                "op": "under",
                                "bound": "100",
                                "low": "none",
                                "high": "none",
                                "partial": False,
                                "conditional": False,
                                "negated": False,
                                "approximate": False,
                            }
                        )
                    }
                }
            ],
        }


@pytest.mark.parametrize("reader", ["jev", "cheap"])
def test_record_saves_raw_answers_and_resumes_without_repeating_cases(
    reader: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    cases = _recording_cases()
    calls: list[tuple[str, str]] = []

    def make_gateway(*, config: measurement.GatewayConfig) -> _RecordingGateway:
        return _RecordingGateway(config=config, shared_calls=calls)

    monkeypatch.setattr(measurement, "HttpGateway", make_gateway)
    measurement.record(reader, cases, output)
    recorded = json.loads(output.read_text(encoding="utf-8"))
    assert len(recorded["rows"]) == 2
    assert all(
        row["case_id"] == case["id"]
        for row, case in zip(recorded["rows"], cases, strict=True)
    )
    if reader == "jev":
        assert all("answers" in row and "usage" in row for row in recorded["rows"])
        assert recorded["rows"][0]["answers"]["kind"]["selected"] == "word_count"
    else:
        assert all("reply" in row and "usage" in row for row in recorded["rows"])
        assert json.loads(recorded["rows"][0]["reply"])["kind"] == "word_count"
        assert all(row["finish_reason"] == "stop" for row in recorded["rows"])

    calls_before_resume = len(calls)
    measurement.record(reader, cases, output)
    assert len(calls) == calls_before_resume


def test_provider_failure_keeps_completed_rows_and_resume_finishes_remaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    cases = _recording_cases()
    instances: list[_RecordingGateway] = []

    def make_gateway(*, config: measurement.GatewayConfig) -> _RecordingGateway:
        gateway = _RecordingGateway(config=config, fail_on_second_chat=not instances)
        instances.append(gateway)
        return gateway

    monkeypatch.setattr(measurement, "HttpGateway", make_gateway)
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        measurement.record("cheap", cases, output)
    interrupted = json.loads(output.read_text(encoding="utf-8"))
    assert [row["case_id"] for row in interrupted["rows"]] == ["case-0"]
    assert interrupted["rows"][0]["reply"]

    measurement.record("cheap", cases, output)
    resumed = json.loads(output.read_text(encoding="utf-8"))
    assert [row["case_id"] for row in resumed["rows"]] == ["case-0", "case-1"]
    assert len(instances[-1].calls) == 1


@pytest.mark.parametrize(
    ("finish_reason", "usage", "expected"),
    [
        ("length", {"completion_tokens": 12}, True),
        ("stop", {"completion_tokens": 50}, False),
        (None, {"completion_tokens": 50}, True),
    ],
)
def test_report_uses_finish_reason_before_legacy_token_cap_heuristic(
    finish_reason: str | None,
    usage: dict[str, int],
    expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "dev-001",
                        "split": "development",
                        "criterion": "Write a friendly answer",
                        "expected": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(measurement, "FIXTURE", fixture)
    recording = tmp_path / "recording.json"
    recording.write_text(
        json.dumps(
            {
                "reader": "cheap",
                "model": "test-model",
                "questions_digest": measurement.questions_digest(),
                "max_tokens": 50,
                "rows": [
                    {
                        "case_id": "dev-001",
                        "reply": "{}",
                        "usage": usage,
                        **(
                            {"finish_reason": finish_reason}
                            if finish_reason is not None
                            else {}
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    measurement.report(recording, show_errors=False)

    output = capsys.readouterr().out
    assert ("truncation-suspect: dev-001" in output) is expected
    assert "unusable answers: dev-001" in output
    assert "| development | unresolved if p >= 0.5 | 0 | 1 | 0 | 0 |" in output


@pytest.mark.parametrize("reader", ["jev", "cheap"])
def test_model_override_is_used_recorded_and_protected_by_resume_guard(
    reader: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    calls: list[tuple[str, str]] = []

    def make_gateway(*, config: measurement.GatewayConfig) -> _RecordingGateway:
        return _RecordingGateway(config=config, shared_calls=calls)

    monkeypatch.setattr(measurement, "HttpGateway", make_gateway)
    measurement.record(reader, _recording_cases()[:1], output, model="custom/model")

    recorded = json.loads(output.read_text(encoding="utf-8"))
    assert recorded["model"] == "custom/model"
    assert calls == [(reader, "custom/model")]
    with pytest.raises(ValueError, match="different model"):
        measurement.record(reader, _recording_cases(), output, model="other/model")


def test_model_override_has_a_bounded_length() -> None:
    assert measurement.parse_model("  custom/model  ") == "custom/model"
    with pytest.raises(argparse.ArgumentTypeError, match="at most"):
        measurement.parse_model("m" * (measurement.MAX_MODEL_LENGTH + 1))


@pytest.mark.parametrize("value", ["", "   "])
def test_cli_rejects_empty_model_override(
    value: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_criterion_reading.py",
            "run",
            "--reader",
            "cheap",
            "--model",
            value,
            "--output",
            "recording.json",
        ],
    )

    with pytest.raises(SystemExit) as error:
        measurement.main()

    assert error.value.code == 2
    assert "--model must not be empty" in capsys.readouterr().err


def test_cli_rejects_completion_token_cap_above_configured_bound(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_criterion_reading.py",
            "run",
            "--reader",
            "cheap",
            "--max-tokens",
            str(measurement.MAX_COMPLETION_TOKENS + 1),
            "--output",
            "recording.json",
        ],
    )

    with pytest.raises(SystemExit) as error:
        measurement.main()

    assert error.value.code == 2
    assert (
        f"--max-tokens must be at most {measurement.MAX_COMPLETION_TOKENS}"
        in capsys.readouterr().err
    )


def test_report_cli_scores_a_temporary_recording_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "report-case",
                        "split": "development",
                        "criterion": "Write a friendly answer",
                        "expected": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(measurement, "FIXTURE", fixture)
    recording = tmp_path / "recording.json"
    recording.write_text(
        json.dumps(
            {
                "reader": "cheap",
                "model": "test-model",
                "questions_digest": measurement.questions_digest(),
                "max_tokens": 300,
                "rows": [
                    {
                        "case_id": "report-case",
                        "reply": "{}",
                        "usage": {"completion_tokens": 4},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["measure_criterion_reading.py", "report", "--recording", str(recording)],
    )

    measurement.main()

    output = capsys.readouterr().out
    assert "# cheap (test-model): 1 criteria" in output
    assert "| development | unresolved if p >= 0.5 | 0 | 1 | 0 | 0 |" in output
