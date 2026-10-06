"""Build final review evidence from matching snapshots and validated receipts."""

import argparse
import json
import tempfile
from pathlib import Path

import yaml
from run_codeql import summarize
from validation.receipts import commit, digest, git, snapshot, write_json

ROOT = Path(__file__).resolve().parents[1]


def artifact(directory: Path, name: str, expected: str) -> str:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("artifact is missing or escapes its receipt directory")
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"artifact changed: {name}")
    return data.decode()


def read_receipt(path: Path, source: str, hooks: list[str]) -> list[dict]:
    data = json.loads(path.read_text())
    if (
        data.get("schema_version") != 1
        or data.get("status") != "complete"
        or data.get("phase") != "final"
    ):
        raise ValueError(f"not a completed final receipt: {path}")
    if data.get("source_digest") != source:
        raise ValueError(f"receipt source differs from reviewed head: {path}")
    checks = data.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("receipt has no completed checks")
    names = [check["name"] for check in checks]
    if len(set(names)) != len(names):
        raise ValueError("duplicate receipt checks")
    if data.get("kind") == "validation":
        skipped = data.get("skipped_hooks", [])
        if (
            set(skipped) - {"no-commit-to-branch"}
            or data.get("configured_hooks") != hooks
            or set(names) != set(hooks)
        ):
            raise ValueError("receipt does not cover the reviewed hook configuration")
    elif data.get("kind") == "codeql":
        analyses = data.get("analyses", [])
        if not analyses or not data.get("codeql_version"):
            raise ValueError("missing CodeQL analysis metadata")
        expected_checks = ["version"]
        for analysis in analyses:
            language = analysis["language"]
            expected_checks.extend([f"create-{language}", f"analyze-{language}"])
            artifact(path.parent, analysis["sarif"], analysis["sarif_sha256"])
            if (
                summarize(path.parent / analysis["sarif"])["finding_count"]
                != analysis["finding_count"]
            ):
                raise ValueError("CodeQL finding count differs from SARIF")
        if names != expected_checks:
            raise ValueError("CodeQL receipt has missing or unexpected checks")
    else:
        raise ValueError("unknown receipt kind")
    sections = [json.dumps(data, ensure_ascii=False)]
    for check in checks:
        if check.get("status") != "passed" or check.get("exit_code") != 0:
            raise ValueError("receipt includes a failed or unfinished check")
        log = artifact(path.parent, check["log"], check["log_sha256"])
        sections.append(f"Completed check: {check['name']}\n{log}")
    return [{"id": f"receipt:{path}", "text": "\n\n".join(sections)}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=200_000)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("output already exists; use a fresh evidence file")
        if len(args.receipt) > 15:
            raise ValueError("select at most 15 receipts plus the reviewed patch")
        repo = args.repo.resolve()
        base, head = commit(repo, args.base), commit(repo, args.head)
        patch = git(repo, "diff", "--no-ext-diff", "--no-textconv", base, head).decode()
        if len(patch) > 50_000:
            raise ValueError(
                "patch exceeds Jev whole-change input limit; use per-file review"
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            source = snapshot(repo, head, root)
            config = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
            hooks = [
                hook["id"] for origin in config["repos"] for hook in origin["hooks"]
            ]
            evidence = [{"id": "reviewed-patch", "text": patch}]
            for path in args.receipt:
                evidence.extend(read_receipt(path.resolve(), source, hooks))
        bundle = {
            "schema_version": 1,
            "base": base,
            "head": head,
            "source_digest": source,
            "diff": patch,
            "evidence": evidence,
        }
        if len(json.dumps(bundle).encode()) > args.max_bytes:
            raise ValueError(
                "evidence exceeds --max-bytes; select fewer receipts or raise explicitly"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, bundle)
        print(f"Final evidence for {head}: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evidence rejected: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
