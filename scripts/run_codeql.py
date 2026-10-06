"""Analyze a committed Git snapshot and report completed CodeQL results."""

import argparse
import json
from pathlib import Path

from validation.receipts import Receipt, commit, digest, snapshot

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "python": "python-code-scanning.qls",
    "javascript-typescript": "javascript-code-scanning.qls",
}


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or data.get("version") != "2.1.0"
        or not isinstance(data.get("runs"), list)
        or not data["runs"]
    ):
        raise ValueError("missing SARIF 2.1.0 analysis runs")
    results = []
    for run in data["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("results"), list):
            raise ValueError("malformed SARIF analysis run")
        tool = run.get("tool")
        driver = tool.get("driver") if isinstance(tool, dict) else None
        if not isinstance(driver, dict) or not isinstance(driver.get("name"), str):
            raise ValueError("missing SARIF tool metadata")
        invocations = run.get("invocations", [])
        if not isinstance(invocations, list):
            raise ValueError("malformed SARIF invocations")
        for invocation in invocations:
            if (
                not isinstance(invocation, dict)
                or invocation.get("executionSuccessful") is False
            ):
                raise ValueError("SARIF reports unsuccessful or malformed analysis")
        for result in run["results"]:
            if not isinstance(result, dict) or not isinstance(
                result.get("ruleId"), str
            ):
                raise ValueError("malformed SARIF result")
            results.append(result)
    return {
        "finding_count": len(results),
        "rule_ids": sorted({result["ruleId"] for result in results}),
        "sarif_sha256": digest(path.read_bytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--codeql", default="codeql")
    parser.add_argument("--language", action="append", choices=SUITES)
    parser.add_argument("--phase", choices=["reproduction", "final"], default="final")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for this run"
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = Receipt(
        args.output.resolve(),
        kind="codeql",
        phase=args.phase,
        sha=commit(repo, args.ref),
    )
    try:
        with receipt.interruptions():
            source = receipt.output / "source"
            receipt.value["source_digest"] = snapshot(
                repo, receipt.value["commit"], source
            )
            version = receipt.run(
                "version", [args.codeql, "version", "--format=json"], cwd=source
            )
            receipt.value["codeql_version"] = json.loads(version.read_text())["version"]
            if (
                not isinstance(receipt.value["codeql_version"], str)
                or not receipt.value["codeql_version"]
            ):
                raise ValueError("missing CodeQL CLI version")
            receipt.value["analyses"] = []
            for language in dict.fromkeys(args.language or SUITES):
                database = receipt.output / f"database-{language}"
                sarif = receipt.output / f"{language}.sarif"
                suite = SUITES[language]
                receipt.run(
                    f"create-{language}",
                    [
                        args.codeql,
                        "database",
                        "create",
                        str(database),
                        f"--language={language}",
                        "--build-mode=none",
                        f"--source-root={source}",
                    ],
                    cwd=source,
                )
                receipt.run(
                    f"analyze-{language}",
                    [
                        args.codeql,
                        "database",
                        "analyze",
                        str(database),
                        suite,
                        "--format=sarif-latest",
                        f"--output={sarif}",
                    ],
                    cwd=source,
                )
                receipt.value["analyses"].append(
                    {
                        "language": language,
                        "query_suite": suite,
                        "sarif": sarif.name,
                        **summarize(sarif),
                    }
                )
                receipt.save()
            receipt.finish()
            for analysis in receipt.value["analyses"]:
                print(
                    f"{analysis['language']}: analysis completed; {analysis['finding_count']} findings"
                )
            print(f"Receipt: {receipt.path}")
            return 0
    except KeyboardInterrupt:
        print(f"CodeQL interrupted. Receipt: {receipt.path}")
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        receipt.fail(exc)
        print(f"CodeQL failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
