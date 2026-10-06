"""Exercise interrupted checks, late review, budgets and isolated configuration."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.config import Settings
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from bootstrap import require_quality_tools
from isolated_run import prepare, verify_settings
from review_readiness import assess, fingerprint
from validation.bundles import partition
from validation.receipts import receipt_status

ROOT = Path(__file__).parents[1]


def test_sigterm_marks_interrupted_receipt_and_stops_child_group(tmp_path):
    child = tmp_path / "child.py"
    marker = tmp_path / "ready"
    child.write_text(
        "import pathlib,time,sys\npathlib.Path(sys.argv[1]).write_text('ready')\ntime.sleep(60)\n"
    )
    code = f"import sys\nsys.path.insert(0,{str(ROOT / 'scripts')!r})\nfrom validation.receipts import Receipt\nfrom pathlib import Path\nr=Receipt(Path({str(tmp_path / 'receipt')!r}),kind='validation',phase='final',sha='test')\nr.run('wait',[sys.executable,{str(child)!r},{str(marker)!r}],cwd=Path({str(tmp_path)!r}))\n"
    runner = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        receipt_path = tmp_path / "receipt/receipt.json"
        before = json.loads(receipt_path.read_text())
        assert receipt_status(before) == "active"
        assert before["checks"][0]["process"]["pid"] != runner.pid
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=8) != 0
        after = json.loads(receipt_path.read_text())
        assert after["status"] == "interrupted", runner.stderr.read().decode()
        assert after["checks"][0]["status"] == "interrupted"
        with pytest.raises(ProcessLookupError):
            os.kill(before["checks"][0]["process"]["pid"], 0)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()


def test_liveness_does_not_trust_reused_pid_or_legacy_receipt(monkeypatch):
    import socket
    from datetime import UTC, datetime

    import validation.receipts as receipts

    value = {
        "status": "running",
        "runner": {"host": socket.gethostname(), "pid": 42, "identity": "original"},
        "heartbeat_at": datetime.now(UTC).isoformat(),
    }
    monkeypatch.setattr(receipts, "process_identity", lambda pid: "replacement")
    assert receipt_status(value) == "abandoned"
    assert receipt_status({"status": "running"}) == "unknown"
    monkeypatch.setattr(receipts, "process_identity", lambda pid: "original")
    value["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    assert receipt_status(value) == "unresponsive"


def test_latest_head_review_requires_every_comment_disposition():
    reviewer = "qodo-code-review[bot]"
    reviews = [
        {"id": 1, "commit_id": "old", "state": "COMMENTED", "user": {"login": reviewer}}
    ]
    assert assess("new", reviews, [], {}, reviewer=reviewer)["status"] == "pending"
    reviews.append(
        {"id": 2, "commit_id": "new", "state": "COMMENTED", "user": {"login": reviewer}}
    )
    comments = [
        {"id": 10, "commit_id": "new", "user": {"login": reviewer}},
        {"id": 11, "commit_id": "new", "user": {"login": reviewer}},
    ]
    dispositions = {
        "10": {
            "status": "dismissed",
            "evidence": "test shows contract preserved",
            "fingerprint": fingerprint(comments[0]),
        }
    }
    result = assess("new", reviews, comments, dispositions, reviewer=reviewer)
    assert result["status"] == "needs_triage"
    assert result["unresolved"] == ["11"]
    dispositions["11"] = {
        "status": "fixed",
        "evidence": "regression passes at fix SHA",
        "fingerprint": fingerprint(comments[1]),
    }
    assert (
        assess("new", reviews, comments, dispositions, reviewer=reviewer)["status"]
        == "triaged"
    )
    dispositions["11"]["evidence"] = ""
    assert assess("new", reviews, comments, dispositions, reviewer=reviewer)[
        "unresolved"
    ] == ["11"]


def test_edited_review_comment_invalidates_disposition_and_settling_signature():
    reviewer = "qodo-code-review[bot]"
    reviews = [
        {
            "id": 1,
            "commit_id": "head",
            "state": "COMMENTED",
            "user": {"login": reviewer},
        }
    ]
    comment = {
        "id": 10,
        "commit_id": "head",
        "user": {"login": reviewer},
        "body": "Original finding",
        "updated_at": "2026-10-06T12:00:00Z",
    }
    dispositions = {
        "10": {
            "status": "fixed",
            "evidence": "test passes",
            "fingerprint": fingerprint(comment),
        }
    }
    before = assess("head", reviews, [comment], dispositions, reviewer=reviewer)
    assert before["status"] == "triaged"
    comment.update(body="Changed finding", updated_at="2026-10-06T12:00:01Z")
    after = assess("head", reviews, [comment], dispositions, reviewer=reviewer)
    assert after["status"] == "needs_triage"
    assert after["unresolved"] == ["10"]
    assert after["comment_ids"] == before["comment_ids"]
    assert after["comment_fingerprints"] != before["comment_fingerprints"]
    dispositions["10"]["fingerprint"] = fingerprint(comment)
    assert (
        assess("head", reviews, [comment], dispositions, reviewer=reviewer)["status"]
        == "triaged"
    )


def test_readiness_wait_settles_again_after_same_id_comment_edit(tmp_path, monkeypatch):
    import review_readiness as readiness

    clock = [0.0]
    reviewer = "qodo-code-review[bot]"
    comment = {
        "id": 10,
        "commit_id": "head",
        "user": {"login": reviewer},
        "body": "Original",
    }
    dispositions = tmp_path / "dispositions.json"
    dispositions.write_text(
        json.dumps(
            {
                "10": {
                    "status": "fixed",
                    "evidence": "original finding checked",
                    "fingerprint": fingerprint(comment),
                }
            }
        )
    )
    output = tmp_path / "receipt.json"

    def fetch(endpoint, deadline):
        if endpoint.endswith("/reviews"):
            return [
                [
                    {
                        "id": 1,
                        "commit_id": "head",
                        "state": "COMMENTED",
                        "user": {"login": reviewer},
                    }
                ]
            ]
        if endpoint.endswith("/comments"):
            return [[{**comment, "body": "Edited" if clock[0] >= 5 else "Original"}]]
        return [{"head": {"sha": "head"}}]

    monkeypatch.setattr(readiness, "gh_json", fetch)
    monkeypatch.setattr(readiness.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        readiness.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_readiness.py",
            "1",
            "--head",
            "head",
            "--timeout",
            "30",
            "--dispositions",
            str(dispositions),
            "--output",
            str(output),
        ],
    )
    assert readiness.main() == 1
    assert clock[0] == 15
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "needs_triage"
    assert receipt["unresolved"] == ["10"]


def test_partition_preserves_all_unicode_evidence_and_complete_patches():
    patches = [
        {"path": "a.py", "diff": "complete a"},
        {"path": "b.py", "diff": "complete b"},
    ]
    text = "🦉abc\n" * 1000
    plan = partition(
        patches,
        [{"id": "test-log", "text": text}],
        context_tokens=200,
        reserve_tokens=40,
        dependencies={"a.py": [{"id": "helper.py", "text": "entire helper"}]},
    )
    assert plan["patch_paths"] == ["a.py", "b.py"]
    assert plan["bundles"][0]["diff"] == "complete a"
    assert plan["bundles"][0]["evidence"][0]["text"] == "entire helper"
    coverage = plan["evidence_coverage"][0]
    assert (
        "".join(
            plan["bundles"][i]["evidence"][0]["text"]
            for i in coverage["bundle_indices"]
        )
        == text
    )
    assert all(b["estimated_input_tokens"] <= 160 for b in plan["bundles"])
    with pytest.raises(ValueError, match="Complete patch"):
        partition(
            [{"path": "large.py", "diff": "x" * 1000}],
            [],
            context_tokens=200,
            reserve_tokens=40,
        )


def test_quality_dependency_is_required_when_the_package_is_present(tmp_path):
    package = tmp_path / "tools/quality/package.json"
    package.parent.mkdir(parents=True)
    package.write_text("{}")
    with pytest.raises(ValueError, match="Pinned quality CLI"):
        require_quality_tools(tmp_path)
    cli = package.parent / "node_modules/jev-lint/dist/cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// installed")
    require_quality_tools(tmp_path)


def test_isolated_settings_are_verified_before_submission_and_exclude_secrets(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "models": {
                    "writer": "chosen-writer",
                    "strong": "chosen-strong",
                    "weak": [f"weak-{i}" for i in range(5)],
                },
                "score_floors": {"clarity": 0.7},
                "api_key": "synthetic-secret",
                "floor_calibration": {"note": "synthetic-secret"},
            }
        )
    )
    output = tmp_path / "isolated"
    plan = prepare(
        source, output, judge="judge-1", operation_timeout=30, request_timeout=10
    )
    actual = {
        "judge_model": "judge-1",
        "writer_model": "chosen-writer",
        "strong_check_model": "chosen-strong",
        "weak_models": [f"weak-{i}" for i in range(5)],
        "score_floors": {"clarity": 0.7},
        "gateway_limits": {"operation_timeout_s": 30, "request_timeout_s": 10},
    }
    verify_settings(plan, actual)
    assert "synthetic-secret" not in (output / "runs.settings.json").read_text()
    assert "synthetic-secret" not in (output / "provenance.json").read_text()
    with pytest.raises(ValueError, match="Effective model settings"):
        verify_settings(plan, {**actual, "writer_model": "environment-default"})
    with pytest.raises(ValueError, match="Effective Gateway bounds"):
        verify_settings(
            plan,
            {
                **actual,
                "gateway_limits": {"operation_timeout_s": 99, "request_timeout_s": 10},
            },
        )
    gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    store = RunStore(output / "runs.sqlite3")
    optimizer = PromptOptimizer(store=store, config=Settings(), gateway=gateway)
    assert optimizer.get_model_settings()["writer_model"] == "chosen-writer"
    client = TestClient(create_app(optimizer=optimizer))
    result = client.post(
        "/api/optimize",
        json={
            "prompt": "Explain recursion.",
            "model_overrides": {"writer": "run-writer"},
        },
    ).json()
    record = store.get_run(result["run_id"])
    assert record["configuration"]["models"]["writer"] == "run-writer"
    store.close()
    reopened = RunStore(output / "runs.sqlite3")
    assert (
        reopened.get_run(result["run_id"])["configuration"]["models"]["writer"]
        == "run-writer"
    )
    reopened.close()


def test_cleanup_kills_ignoring_descendant_after_leader_exits(tmp_path):
    from validation.receipts import stop_process_group

    marker = tmp_path / "descendant-ready"
    descendant = "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid())); time.sleep(60)"
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(60)",
            descendant,
            str(marker),
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        pid = int(marker.read_text())
        stop_process_group(leader)
        state = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True
        ).stdout.strip()
        assert not state or state.startswith("Z")
        assert leader.returncode is not None
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait()


def test_github_calls_share_the_overall_deadline(monkeypatch):
    import review_readiness as readiness

    monkeypatch.setattr(readiness.time, "monotonic", lambda: 100)
    seen = []

    def invoke(command, **kwargs):
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="[]")

    monkeypatch.setattr(readiness.subprocess, "run", invoke)
    assert readiness.gh_json("endpoint", 100.25) == []
    assert seen == [0.25]
    with pytest.raises(readiness.ReadinessTimeout):
        readiness.gh_json("endpoint", 100)
    assert seen == [0.25]


def test_zero_review_budget_records_timeout_without_network(tmp_path, monkeypatch):
    import review_readiness as readiness

    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_readiness.py",
            "1",
            "--head",
            "head",
            "--timeout",
            "0",
            "--output",
            str(output),
        ],
    )
    assert readiness.main() == 1
    assert json.loads(output.read_text())["status"] == "timed_out"


def test_background_override_provenance_survives_interruption_recovery(tmp_path):
    import threading

    from prompt_enhancer.jobs import RunJobs

    entered = threading.Event()
    release = threading.Event()

    class BlockingOptimizer(PromptOptimizer):
        def _optimize_started(self, *args, **kwargs):
            entered.set()
            assert release.wait(10)
            raise RuntimeError("offline")

    database = tmp_path / "runs.sqlite3"
    store = RunStore(database)
    optimizer = BlockingOptimizer(store=store, gateway=ScriptedGateway())
    client = TestClient(create_app(optimizer=optimizer))
    run_id = None
    saved = None
    recovered = None
    try:
        response = client.post(
            "/api/jobs/optimize",
            json={
                "prompt": "Explain recursion.",
                "model_overrides": {"writer": "background-writer"},
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        assert entered.wait(10)
        saved = RunStore(database)
        assert (
            saved.get_run(run_id)["configuration"]["models"]["writer"]
            == "background-writer"
        )
        recovered = RunJobs(store=saved)
        assert recovered.get(run_id)["state"] == "interrupted"
        assert (
            recovered.get(run_id)["result"]["report"]["outcome"] == "failed_operational"
        )
        assert (
            saved.get_run(run_id)["configuration"]["models"]["writer"]
            == "background-writer"
        )
    finally:
        release.set()
        if run_id is not None:
            client.app.state.jobs.wait(run_id)
        client.app.state.jobs._executor.shutdown(wait=True)
        if recovered is not None:
            recovered._executor.shutdown(wait=True)
        if saved is not None:
            saved.close()
        store.close()
