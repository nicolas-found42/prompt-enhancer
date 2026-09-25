"""The PR check uses GitHub's parsed closing links, not PR-body text."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pr_issue_link.py"


def run_check(branch: str, linked: list[int]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), branch],
        input=json.dumps(
            {"closingIssuesReferences": [{"number": number} for number in linked]}
        ),
        text=True,
        capture_output=True,
        check=False,
    )


def test_issue_branch_needs_a_link_that_github_recognizes() -> None:
    missing = run_check("codex/issue-50-per-question-calibration", [])
    assert missing.returncode == 1
    assert "Add Closes #50" in missing.stderr

    linked = run_check("codex/issue-50-per-question-calibration", [49, 50])
    assert linked.returncode == 0
    assert "Issue #50 is linked" in linked.stdout


def test_follow_up_branch_does_not_claim_to_close_an_issue() -> None:
    result = run_check("codex/calibration-report-clarity", [])
    assert result.returncode == 0
