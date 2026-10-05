"""Reject the batch-snapshot reuse that previously invalidated an audit."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from prompt_enhancer.evaluation.capture_audit import validate_capture
from prompt_enhancer.evaluation.recording import RecordingGateway
from prompt_enhancer.gateway import ScriptedGateway


def requests() -> list[dict]:
    return [
        {"key": f"item-{i}", "type": "noul", "state": {"item": i}} for i in range(3)
    ]


def test_batch_capture_keeps_distinct_payloads_correlations_and_answers(
    tmp_path: Path,
) -> None:
    gateway = RecordingGateway(
        ScriptedGateway(
            decision=lambda request, **_: {
                "type": "noul",
                "probability_true": request["state"]["item"] / 3,
            }
        ),
        tmp_path / "record.json",
    )
    original = requests()
    answers = gateway.decide_batch(original)
    original[0]["state"]["item"] = 999
    gateway.save()
    bundle = json.loads(gateway.path.read_text())
    captures = bundle["request_captures"]
    assert [row["payload"]["state"]["item"] for row in captures] == [0, 1, 2]
    assert [row["answer"] for row in captures] == answers
    assert len({row["correlation_id"] for row in captures}) == 3
    assert bundle["capture_audit"]["status"] == "complete"
    assert bundle["capture_audit"]["request_count"] == 3


@pytest.mark.parametrize(
    "fault",
    [
        "reused_snapshot",
        "reused_id",
        "missing_record",
        "wrong_answer",
        "wrong_hash",
        "reordered",
    ],
)
def test_audit_rejects_corrupt_capture_before_success(
    tmp_path: Path, fault: str
) -> None:
    gateway = RecordingGateway(
        ScriptedGateway(decision=lambda request, **_: request["key"]),
        tmp_path / "record.json",
    )
    gateway.decide_batch(requests())
    records = deepcopy(gateway.request_captures)
    if fault == "reused_snapshot":
        records[0]["payload"] = records[-1]["payload"]
    elif fault == "reused_id":
        records[0]["correlation_id"] = records[-1]["correlation_id"]
    elif fault == "missing_record":
        records.pop()
    elif fault == "wrong_answer":
        records[0]["answer"] = records[-1]["answer"]
    elif fault == "wrong_hash":
        records[0]["request_key"] = "incorrect"
    else:
        records.reverse()
    with pytest.raises(ValueError):
        validate_capture(records, gateway.expected_request_keys, gateway.responses)
    gateway.request_captures = records
    before = gateway.path.read_bytes()
    with pytest.raises(ValueError):
        gateway.save()
    assert gateway.path.read_bytes() == before


def test_overwritten_provider_batch_snapshot_is_rejected_before_recording(
    tmp_path: Path,
) -> None:
    scripted = ScriptedGateway(decision=lambda request, **_: request["key"])
    original = scripted.decide_batch

    def broken_batch(batch, **kwargs):
        answers = original(batch, **kwargs)
        for entry in scripted.decision_log:
            entry["question"] = deepcopy(batch[-1])
        return answers

    scripted.decide_batch = broken_batch
    gateway = RecordingGateway(scripted, tmp_path / "record.json")
    with pytest.raises(ValueError, match="per-request association"):
        gateway.decide_batch(requests())
    assert not gateway.path.exists()
    assert not gateway.responses


def test_audit_cli_requires_independent_expected_sequence(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    gateway = RecordingGateway(
        ScriptedGateway(decision=lambda request, **_: request["key"]),
        tmp_path / "record.json",
    )
    gateway.decide_batch(requests())
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(gateway.expected_request_keys))
    script = Path(__file__).resolve().parents[1] / "scripts/audit_gateway_capture.py"
    env = {**os.environ, "PYTHONPATH": str(script.parent.parent / "src")}
    command = [
        sys.executable,
        str(script),
        str(gateway.path),
        "--expected-keys",
        str(expected),
    ]
    passed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(passed.stdout)["status"] == "complete"
    expected.write_text(json.dumps(list(reversed(gateway.expected_request_keys))))
    failed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failed.returncode == 1
    assert not failed.stdout
    assert json.loads(failed.stderr)["status"] == "failed"


def test_batch_answer_count_mismatch_writes_no_partial_receipt(tmp_path: Path) -> None:
    scripted = ScriptedGateway(decision=lambda request, **_: request["key"])
    original = scripted.decide_batch

    def truncated(batch, **kwargs):
        return original(batch, **kwargs)[:-1]

    scripted.decide_batch = truncated
    gateway = RecordingGateway(scripted, tmp_path / "record.json")
    with pytest.raises(ValueError, match="counts do not reconcile"):
        gateway.decide_batch(requests())
    assert not gateway.path.exists()


@pytest.mark.parametrize("records", [[None], "invalid", [dict(correlation_id="x")]])
def test_malformed_capture_raises_validation_error(records) -> None:
    with pytest.raises(ValueError):
        validate_capture(records, ["expected"], {})
