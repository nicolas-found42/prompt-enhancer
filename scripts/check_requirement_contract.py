"""Run bounded public requirement controls, retaining logs and durable history."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from validation.receipts import Receipt, commit, worktree_digest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_TESTS = (
    "tests/test_compound_requirements.py",
    "tests/test_compound_review_regressions.py",
    "tests/test_issue187_requirements.py",
    "tests/test_issue187_protected_blocks.py",
    "tests/test_issue187_jobs.py",
)


def release_report(path: Path) -> None:
    """A high aggregate score cannot replace an audited source contract."""
    value = json.loads(path.read_text())
    result = value.get("result", value)
    ledger = result.get("report", {}).get("requirements", {})
    source = value.get("prompt", result.get("original_prompt"))
    if (
        not isinstance(source, str)
        or ledger.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest()
        or ledger.get("coverage") != "audited"
        or ledger.get("release_eligible") is not True
        or ledger.get("gaps")
        or ledger.get("whole_source_audit", {}).get("status") != "accepted"
        or any(
            item.get("audit", {}).get("status") != "accepted"
            for item in ledger.get("requirements", [])
            if item.get("source_kind") == "original_prompt"
        )
        or any(
            item.get("status") != "resolved_by_user"
            for item in ledger.get("contradictions", [])
        )
    ):
        raise ValueError(f"Unresolved or mismatched source coverage: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New artifact directory"
    )
    parser.add_argument("--release-report", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be between 1 and 600 seconds")
    receipt = Receipt(
        args.output.resolve(),
        kind="requirement-contract",
        phase="final",
        sha=commit(ROOT, "HEAD"),
    )
    try:
        with receipt.interruptions():
            receipt.value.update(
                source_digest=worktree_digest(ROOT),
                profile="controlled-gateway-clock-v1",
                oracle="public-contract-controls-v1",
                contract_issues=list(range(189, 196)),
                contract_sources={
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in CONTRACT_TESTS
                },
            )
            receipt.save()
            for path in args.release_report:
                release_report(path)
            xml = receipt.output / "tests.xml"
            # Keep pytest's SQLite stores under the artifact directory so a
            # maintainer can reopen the same saved histories after the controls.
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *CONTRACT_TESTS,
                f"--junitxml={xml}",
                f"--basetemp={receipt.output / 'histories'}",
            ]
            bounded = [
                sys.executable,
                "-c",
                "import subprocess,sys; "
                "sys.exit(subprocess.run(sys.argv[2:], timeout=int(sys.argv[1])).returncode)",
                str(args.timeout),
                *command,
            ]
            receipt.run("public-requirement-controls", bounded, cwd=ROOT)
            cases = ET.parse(xml).getroot().findall(".//testcase")
            if not cases or any(
                case.find("failure") is not None
                or case.find("error") is not None
                or case.find("skipped") is not None
                for case in cases
            ):
                raise ValueError("Public controls were missing, skipped or incomplete")
            if worktree_digest(ROOT) != receipt.value["source_digest"]:
                raise ValueError("Source changed during the contract controls")
            receipt.value["tested_controls"] = len(cases)
            receipt.finish()
            print(f"Requirement controls passed: {receipt.path}")
            return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        ET.ParseError,
    ) as exc:
        receipt.fail(exc)
        print(f"Requirement controls failed: {receipt.path}: {exc}")
        return 1
    except KeyboardInterrupt:
        print(f"Requirement controls interrupted: {receipt.path}")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
