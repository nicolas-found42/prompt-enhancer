"""Run the pinned linter on an isolated snapshot of changed source files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .evidence import GitEvidence, git_environment, is_source
from .review import MODEL


def lint_snapshot(
    git: GitEvidence,
    trusted_root: Path,
    *,
    live: bool = False,
    max_files: int = 12,
    max_tokens: int = 50_000,
    selected_paths: list[str] | None = None,
) -> dict:
    report = {
        "status": "skipped",
        "head_sha": git.head,
        "omitted": [],
        "requested_model": MODEL,
        "scope": "changed lines in immutable before/after snapshots",
        "selected_paths": selected_paths,
    }
    if not 1 <= max_files <= 100 or not 1 <= max_tokens <= 250_000:
        raise ValueError("invalid linter budget")
    cli = trusted_root / "tools/quality/node_modules/jev-lint/dist/cli.js"
    if not cli.exists():
        return {
            **report,
            "reason": "install pinned tools with npm ci --prefix tools/quality --ignore-scripts",
        }
    with tempfile.TemporaryDirectory(prefix="jev-review-") as temp:
        root = Path(temp)
        paths = []
        after_sources = {}
        total_bytes = 0
        changed = git.changed()
        for path in selected_paths or []:
            if path not in changed or not is_source(path):
                report["omitted"].append(
                    {
                        "path": path,
                        "reason": "selected path is not eligible changed source",
                    }
                )
        for path in changed:
            if not is_source(path) or selected_paths and path not in selected_paths:
                continue
            try:
                source = git.source(git.head, path)
                total_bytes += len(source.encode())
                if len(paths) >= max_files or total_bytes > 250_000:
                    raise ValueError("snapshot budget")
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                after_sources[path] = source
                try:
                    before = git.source(git.merge_base, path)
                except ValueError:
                    before = ""
                destination.write_text(before)
                paths.append(path)
            except ValueError as exc:
                report["omitted"].append({"path": path, "reason": str(exc)})
        if not paths:
            return {
                **report,
                "reason": "no eligible files",
                "status": "partial" if report["omitted"] else "complete",
            }
        shutil.copy(trusted_root / ".jev-lint.yaml", root / ".jev-lint.yaml")
        shutil.copytree(trusted_root / ".jev-lint/rules", root / ".jev-lint/rules")

        def snapshot_git(*args: str) -> str:
            return subprocess.run(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "commit.gpgSign=false",
                    "-c",
                    "user.name=Quality snapshot",
                    "-c",
                    "user.email=quality@localhost",
                    *args,
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
                env=git_environment(),
            ).stdout.strip()

        snapshot_git("init", "--quiet")
        snapshot_git("add", ".")
        snapshot_git("commit", "--quiet", "-m", "before")
        before_sha = snapshot_git("rev-parse", "HEAD")
        for path, source in after_sources.items():
            (root / path).write_text(source)
        snapshot_git("add", ".")
        snapshot_git("commit", "--quiet", "--allow-empty", "-m", "after")
        command = [
            "node",
            str(cli),
            "review",
            "--base",
            before_sha,
            "--config",
            str(root / ".jev-lint.yaml"),
            "--cache",
            "none",
            "--model",
            MODEL,
            "--format",
            "json",
            "--fail-on",
            "error",
            "--show-missing",
            "--concurrency",
            "1",
        ]
        # Only the explicit live subprocess receives an API credential.
        env = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "TMPDIR", "SYSTEMROOT")
            if key in os.environ
        }

        def invoke(args: list[str], environment: dict) -> tuple[int, dict]:
            result = subprocess.run(
                command + args,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=180,
            )
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict):
                raise ValueError("linter response must be an object")
            return result.returncode, payload

        try:
            code, planned = invoke(["--dry-run", "--show-subjects"], env)
            report["plan"] = planned
            if code or not planned.get("dryRun"):
                return {
                    **report,
                    "status": "failed",
                    "reason": "linter planning failed",
                }
            if planned.get("tokens", 0) > max_tokens:
                return {
                    **report,
                    "status": "partial",
                    "reason": "planned input token budget exceeded",
                }
            if not live or not os.environ.get("OPENROUTER_API_KEY"):
                return {
                    **report,
                    "reason": "dry run; no inference requested"
                    if not live
                    else "OPENROUTER_API_KEY unavailable",
                }
            env.update(
                TYPESAFE_API_KEY=os.environ["OPENROUTER_API_KEY"],
                TYPESAFE_BASE_URL="https://openrouter.ai/api",
                JEV_LINT_MODEL=MODEL,
            )
            record_path = root / "recording.json"
            code, result = invoke(["--record", str(record_path)], env)
            report["result"] = result
            if record_path.exists():
                report["recording"] = json.loads(record_path.read_text())
            incomplete = bool(
                result.get("errors") or result.get("degraded") or report["omitted"]
            )
            # An absent/malformed stats object never counts as a complete review.
            stats = result.get("stats", {})
            incomplete = incomplete or bool(stats.get("missing", 0)) or not stats
            report["status"] = (
                "failed"
                if code not in (0, 1)
                else "partial"
                if incomplete
                else "complete"
            )
            report["cost_note"] = (
                "jev-lint spent.usd is a fixed-price estimate, not provider billing"
            )
            return report
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return {
                **report,
                "status": "failed",
                "reason": "linter failed, timed out, or returned invalid JSON",
            }
