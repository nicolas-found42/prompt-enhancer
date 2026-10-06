"""Validation CLIs record completed commands and fail closed on bad evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def command(args, **kwargs):
    environment = kwargs.pop("env", os.environ)
    clean = {
        key: value for key, value in environment.items() if not key.startswith("GIT_")
    }
    return subprocess.run(
        args, env=clean, capture_output=True, text=True, check=False, **kwargs
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (
        ["init"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Test"],
    ):
        assert command(["git", "-C", str(root), *args]).returncode == 0
    (root / "source.py").write_text("print('tracked')\n")
    (root / ".gitignore").write_text(".local/\ncoverage/\n")
    assert command(["git", "-C", str(root), "add", "."]).returncode == 0
    assert command(["git", "-C", str(root), "commit", "-m", "initial"]).returncode == 0
    return root


def fake_codeql(tmp_path, mode="valid"):
    tool = tmp_path / "codeql"
    tool.write_text(f"""#!{sys.executable}
import json, pathlib, sys
args = sys.argv[1:]
if args[0] == 'version':
    print(json.dumps({{'version':'test-1'}}))
elif args[:2] == ['database', 'create']:
    source = pathlib.Path(next(a.split('=', 1)[1] for a in args if a.startswith('--source-root=')))
    assert (source / 'source.py').read_text() == "print('tracked')\\n"
    assert not (source / 'coverage').exists()
    pathlib.Path(args[2]).mkdir()
else:
    target = pathlib.Path(next(a.split('=', 1)[1] for a in args if a.startswith('--output=')))
    target.write_text(json.dumps({{'version':'2.1.0', 'runs':[{{'tool':{{'driver':{{'name':'CodeQL'}}}}, 'results':[]}}]}}) if {mode!r} != 'malformed' else '{{}}')
    print('analysis finished')
    if {mode!r} == 'failed':
        sys.exit(7)
""")
    tool.chmod(0o755)
    return tool


def codeql_command(repo, tool, output):
    return [
        sys.executable,
        str(SCRIPTS / "run_codeql.py"),
        "--repo",
        str(repo),
        "--codeql",
        str(tool),
        "--language",
        "python",
        "--output",
        str(output),
    ]


def test_codeql_uses_committed_snapshot_and_records_completed_analysis(repo, tmp_path):
    (repo / "source.py").write_text("print('dirty')\n")
    (repo / "coverage").mkdir()
    (repo / "coverage" / "generated.js").write_text("generated")
    output = tmp_path / "results"
    result = command(codeql_command(repo, fake_codeql(tmp_path), output))
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["codeql_version"] == "test-1"
    assert receipt["analyses"][0]["finding_count"] == 0
    assert receipt["checks"][-1]["exit_code"] == 0
    assert "analysis finished" in (output / receipt["checks"][-1]["log"]).read_text()
    assert (repo / "source.py").read_text() == "print('dirty')\n"


@pytest.mark.parametrize("mode", ["failed", "malformed"])
def test_codeql_never_reports_success_from_failed_or_malformed_output(
    repo, tmp_path, mode
):
    output = tmp_path / "results"
    result = command(codeql_command(repo, fake_codeql(tmp_path, mode), output))
    assert result.returncode != 0
    assert json.loads((output / "receipt.json").read_text())["status"] == "failed"


def test_validation_stops_before_tests_on_fast_failure_and_keeps_success_logs(
    repo, tmp_path
):
    (repo / ".pre-commit-config.yaml").write_text("""repos:
  - repo: local
    hooks:
      - id: python-tests
      - id: python-format
""")
    tool = tmp_path / "pre-commit"
    tool.write_text(f"""#!{sys.executable}
import os, sys
print('output from ' + sys.argv[2])
if os.environ.get('FAIL_FAST') and sys.argv[2] == 'python-format':
    sys.exit(1)
""")
    tool.chmod(0o755)
    output = tmp_path / "failed"
    args = [
        sys.executable,
        str(SCRIPTS / "run_validation.py"),
        "--repo",
        str(repo),
        "--pre-commit",
        str(tool),
    ]
    result = command(
        [*args, "--output", str(output)], env={**os.environ, "FAIL_FAST": "1"}
    )
    assert result.returncode == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert [check["name"] for check in receipt["checks"]] == ["python-format"]
    output = tmp_path / "passed"
    result = command([*args, "--output", str(output)])
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert [check["name"] for check in receipt["checks"]] == [
        "python-format",
        "python-tests",
    ]
    assert (
        "output from python-tests"
        in (output / receipt["checks"][-1]["log"]).read_text()
    )


def make_validation_receipt(repo, tmp_path, *, pending=False):
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: python-tests\n"
    )
    assert command(["git", "-C", str(repo), "add", "."]).returncode == 0
    assert command(["git", "-C", str(repo), "commit", "-m", "config"]).returncode == 0
    if pending:
        (repo / "source.py").write_text("print('pending commit')\n")
    tool = tmp_path / "pre-commit"
    tool.write_text(f'#!{sys.executable}\nprint("pytest: 3 passed")\n')
    tool.chmod(0o755)
    output = tmp_path / "checks"
    result = command(
        [
            sys.executable,
            str(SCRIPTS / "run_validation.py"),
            "--repo",
            str(repo),
            "--pre-commit",
            str(tool),
            "--output",
            str(output),
        ]
    )
    assert result.returncode == 0, result.stdout
    return output / "receipt.json"


@pytest.mark.parametrize("pending", [False, True])
def test_evidence_includes_patch_and_verified_final_logs(repo, tmp_path, pending):
    receipt = make_validation_receipt(repo, tmp_path, pending=pending)
    if pending:
        assert (
            command(
                ["git", "-C", str(repo), "commit", "-am", "validated contents"]
            ).returncode
            == 0
        )
    output = tmp_path / "evidence.json"
    result = command(
        [
            sys.executable,
            str(SCRIPTS / "build_review_evidence.py"),
            "--repo",
            str(repo),
            "--base",
            "HEAD~2" if pending else "HEAD~1",
            "--receipt",
            str(receipt),
            "--output",
            str(output),
        ]
    )
    assert result.returncode == 0, result.stdout
    bundle = json.loads(output.read_text())
    assert ".pre-commit-config.yaml" in bundle["diff"]
    assert bundle["evidence"][0]["text"] == bundle["diff"]
    assert len(bundle["evidence"]) == 2
    assert any("pytest: 3 passed" in item["text"] for item in bundle["evidence"])


@pytest.mark.parametrize(
    "fault", ["reproduction", "failed", "tampered_log", "stale_source", "missing_check"]
)
def test_evidence_rejects_invalid_completion_proof(repo, tmp_path, fault):
    receipt = make_validation_receipt(repo, tmp_path)
    data = json.loads(receipt.read_text())
    if fault == "reproduction":
        data["phase"] = "reproduction"
    elif fault == "failed":
        data["checks"][0]["exit_code"] = 1
    elif fault == "tampered_log":
        (receipt.parent / data["checks"][0]["log"]).write_text("all passed")
    elif fault == "stale_source":
        (repo / "source.py").write_text("print('new')\n")
        assert command(["git", "-C", str(repo), "commit", "-am", "new"]).returncode == 0
    else:
        data["checks"] = []
    receipt.write_text(json.dumps(data))
    output = tmp_path / "evidence.json"
    result = command(
        [
            sys.executable,
            str(SCRIPTS / "build_review_evidence.py"),
            "--repo",
            str(repo),
            "--base",
            "HEAD~1",
            "--receipt",
            str(receipt),
            "--output",
            str(output),
        ]
    )
    assert result.returncode == 1
    assert "Evidence rejected:" in result.stdout
    assert not output.exists()


def test_codeql_honors_repo_despite_inherited_hook_git_context(repo, tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    for args in (
        ["init"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Test"],
    ):
        assert command(["git", "-C", str(foreign), *args]).returncode == 0
    (foreign / "foreign.txt").write_text("unrelated repository")
    assert command(["git", "-C", str(foreign), "add", "."]).returncode == 0
    assert (
        command(["git", "-C", str(foreign), "commit", "-m", "foreign"]).returncode == 0
    )
    # Deliberately bypass the test helper's isolation to exercise the CLI boundary.
    result = subprocess.run(
        codeql_command(repo, fake_codeql(tmp_path), tmp_path / "results"),
        env={
            **os.environ,
            "GIT_DIR": str(foreign / ".git"),
            "GIT_WORK_TREE": str(foreign),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert command(["git", "-C", str(foreign), "status", "--porcelain"]).stdout == ""


def test_codeql_evidence_requires_reviewed_commit_even_with_identical_tree(
    repo, tmp_path
):
    make_validation_receipt(repo, tmp_path)
    output = tmp_path / "codeql-results"
    assert command(codeql_command(repo, fake_codeql(tmp_path), output)).returncode == 0
    args = [
        sys.executable,
        str(SCRIPTS / "build_review_evidence.py"),
        "--repo",
        str(repo),
        "--base",
        "HEAD~1",
        "--receipt",
        str(output / "receipt.json"),
    ]
    assert command([*args, "--output", str(tmp_path / "accepted.json")]).returncode == 0
    assert (
        command(
            ["git", "-C", str(repo), "commit", "--allow-empty", "-m", "same tree"]
        ).returncode
        == 0
    )
    rejected = tmp_path / "rejected.json"
    result = command([*args, "--output", str(rejected)])
    assert result.returncode == 1
    assert "commit differs" in result.stdout
    assert not rejected.exists()
