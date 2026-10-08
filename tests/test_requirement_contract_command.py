"""The maintainer command fails closed and leaves evidence at its CLI seam."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_command_rejects_partial_coverage_and_retains_failed_receipt(tmp_path):
    report = tmp_path / "run.json"
    report.write_text(
        json.dumps(
            {
                "prompt": "Explain photosynthesis.",
                "result": {
                    "report": {
                        "requirements": {
                            "coverage": "partial",
                            "release_eligible": False,
                        }
                    }
                },
            }
        )
    )
    output = tmp_path / "evidence"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/check_requirement_contract.py",
            "--output",
            str(output),
            "--release-report",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert "Unresolved" in process.stdout
    assert receipt["source_digest"]
    assert receipt["profile"] == "controlled-gateway-clock-v1"


def test_campaign_timeout_is_not_a_success_and_retains_the_attempt_log(tmp_path):
    output = tmp_path / "timeout"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/check_requirement_contract.py",
            "--output",
            str(output),
            "--timeout",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert len(receipt["checks"]) == 1
    assert receipt["checks"][0]["exit_code"] != 0
