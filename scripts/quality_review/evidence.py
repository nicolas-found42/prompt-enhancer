"""Build bounded review evidence from immutable Git objects, never execute it."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

MAX_FILE_BYTES = 100_000
MAX_STATE_BYTES = 32_000
SOURCE_ROOTS = ("src/", "tests/", "scripts/", "web/src/", "web/e2e/")
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx"}
ADR = "docs/adr/0001-gateway-returns-raw-answers.md"


def git_environment() -> dict[str, str]:
    """Git hooks export repository overrides; -C alone does not isolate a repo."""
    return {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class GitEvidence:
    def __init__(self, root: Path, base: str, head: str):
        self.root = root.resolve()
        self.base = self.git(
            "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}"
        ).strip()
        self.head = self.git(
            "rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}"
        ).strip()
        self.merge_base = self.git("merge-base", self.base, self.head).strip()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "--no-pager", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            env=git_environment(),
        ).stdout

    def source(self, sha: str, path: str) -> str:
        parts = PurePosixPath(path)
        if parts.is_absolute() or ".." in parts.parts or str(parts) != path:
            raise ValueError("source path must be a normalized repository path")
        entries = self.git("ls-tree", "-z", sha, "--", path).split("\0")
        exact = [entry for entry in entries if entry.endswith("\t" + path)]
        if len(exact) != 1 or not exact[0].startswith(("100644 blob ", "100755 blob ")):
            raise ValueError(f"missing or non-regular source: {path}")
        oid = exact[0].split("\t")[0].split()[-1]
        if int(self.git("cat-file", "-s", oid)) > MAX_FILE_BYTES:
            raise ValueError(f"source exceeds byte budget: {path}")
        return self.git("cat-file", "blob", oid)

    def changed(self) -> list[str]:
        return [
            p
            for p in self.git(
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                self.merge_base,
                self.head,
            ).split("\0")
            if p
        ]

    def changed_lines(self, path: str) -> list[tuple[int, int]]:
        diff = self.git(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--unified=0",
            self.merge_base,
            self.head,
            "--",
            path,
        )
        return [
            (int(start), int(start) + max(1, int(count or 1)) - 1)
            for start, count in re.findall(r"^@@ .* \+(\d+)(?:,(\d+))? @@", diff, re.M)
        ]


def is_source(path: str) -> bool:
    return (
        path.startswith(SOURCE_ROOTS)
        and PurePosixPath(path).suffix in SOURCE_SUFFIXES
        and "/fixtures/" not in path
    )


def functions(source: str) -> dict[str, tuple[int, int, str]]:
    result = {}
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in tree.body:
        children = (
            [(node.name + ".", child) for child in node.body]
            if isinstance(node, ast.ClassDef)
            else [("", node)]
        )
        for prefix, child in children:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno or child.lineno
                result[prefix + child.name] = (
                    start,
                    end,
                    "\n".join(lines[start - 1 : end]),
                )
    return result


def gateway_findings(git: GitEvidence) -> list[dict[str, Any]]:
    path = "src/prompt_enhancer/gateway.py"
    if path not in git.changed():
        return []
    try:
        source = git.source(git.head, path)
        units = functions(source)
    except (ValueError, SyntaxError):
        return [
            {
                "id": "gateway:unavailable",
                "rule_id": "gateway-raw-answers",
                "path": path,
                "line": 1,
                "end_line": 1,
                "quote": "",
                "claim": "Gateway evidence is unavailable or invalid.",
                "evidence_paths": [],
            }
        ]
    changed = git.changed_lines(path)
    return [
        {
            "id": f"gateway:{name}",
            "rule_id": "gateway-raw-answers",
            "path": path,
            "line": start,
            "end_line": end,
            "quote": quote,
            "claim": f"{name} violates the Gateway raw-answer contract.",
            "evidence_paths": [],
        }
        for name, (start, end, quote) in units.items()
        if name.rsplit(".", 1)[-1] in {"decide", "decide_batch", "chat"}
        and any(a <= end and b >= start for a, b in changed)
    ]


def build_plan(
    git: GitEvidence, rules: dict, findings: list[dict], max_cases: int = 12
) -> dict:
    if not 1 <= max_cases <= 100:
        raise ValueError("max_cases must be between 1 and 100")
    cases, omitted, seen = [], [], set()
    changed = set(git.changed())
    for finding in findings:
        case_id = str(finding.get("id", ""))
        case = {"id": case_id, "finding": finding, "status": "ready"}
        try:
            if not case_id or case_id in seen:
                raise ValueError("finding IDs must be unique and nonempty")
            seen.add(case_id)
            if len(cases) >= max_cases:
                omitted.append({"id": case_id, "reason": "case budget"})
                continue
            path = finding["path"]
            if not is_source(path) or path not in changed:
                raise ValueError("finding must cite an eligible changed source file")
            start, end = finding["line"], finding["end_line"]
            if type(start) is not int or type(end) is not int or not 1 <= start <= end:
                raise ValueError("invalid source span")
            source = git.source(git.head, path)
            lines = source.splitlines()
            if (
                end > len(lines)
                or not finding["quote"]
                or finding["quote"] != "\n".join(lines[start - 1 : end])
            ):
                raise ValueError(
                    "quote does not exactly match the reviewed source span"
                )
            if not any(a <= end and b >= start for a, b in git.changed_lines(path)):
                raise ValueError("finding span does not overlap a changed line")
            rule = rules["rules"][finding["rule_id"]]
            contracts = {p: git.source(git.base, p) for p in rule.get("contracts", [])}
            related = {}
            for p in finding.get("evidence_paths", []):
                if not is_source(p):
                    raise ValueError("related evidence must be eligible source")
                related[p] = git.source(git.head, p)
            try:
                before = git.source(git.merge_base, path)
            except ValueError as exc:
                if "missing or non-regular" not in str(exc):
                    raise
                before = ""
            # Python units keep relevant complete functions instead of slicing lines.
            after = source
            if path.endswith(".py"):
                units = functions(source)
                matched = [
                    (name, unit)
                    for name, unit in units.items()
                    if unit[0] <= start <= end <= unit[1]
                ]
                if matched:
                    name, (_, _, after) = min(
                        matched, key=lambda item: item[1][1] - item[1][0]
                    )
                    before = functions(before).get(name, (0, 0, ""))[2]
            state = {
                "rule": rule,
                "finding": finding,
                "before": before,
                "after": after,
                "related_contracts": contracts,
                "related_sources": related,
                "context_complete": True,
            }
            if len(json.dumps(state, ensure_ascii=False).encode()) > MAX_STATE_BYTES:
                raise ValueError(
                    "evidence exceeds state budget; provide a smaller review unit"
                )
            case["state"] = state
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            case.update(status="incomplete", reason=str(exc))
        cases.append(case)
    plan = {
        "schema_version": 1,
        "base_sha": git.base,
        "merge_base_sha": git.merge_base,
        "head_sha": git.head,
        "rules_digest": digest(rules),
        "cases": cases,
        "omitted": omitted,
    }
    return {**plan, "plan_digest": digest(plan)}
