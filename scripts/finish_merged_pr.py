"""Sync main and remove a merged PR's matching branch and worktree.

Run from any worktree in this repository: python3 scripts/finish_merged_pr.py NUMBER
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


class CleanupError(Exception):
    """A safety check failed before cleanup could finish."""


def command(*args: str, cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CleanupError(f"{' '.join(args)} failed: {detail}")
    return result.stdout


def git(*args: str, cwd: Path) -> str:
    return command("git", *args, cwd=cwd)


def worktrees(repo: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for record in git("worktree", "list", "--porcelain", "-z", cwd=repo).split("\0\0"):
        fields = dict(
            field.split(" ", 1)
            for field in record.strip("\0").split("\0")
            if " " in field
        )
        if "branch" in fields and "worktree" in fields:
            result[fields["branch"]] = Path(fields["worktree"])
    return result


def checked_pr(repo: Path, number: int) -> dict[str, str]:
    repository = json.loads(
        command("gh", "repo", "view", "--json", "nameWithOwner", cwd=repo)
    )
    repo_name = repository["nameWithOwner"]
    pr = json.loads(
        command(
            "gh",
            "pr",
            "view",
            str(number),
            "--repo",
            repo_name,
            "--json",
            "state,baseRefName,headRefName,headRefOid,isCrossRepository",
            cwd=repo,
        )
    )
    if pr["state"] != "MERGED":
        raise CleanupError(
            f"PR #{number} is {pr['state']}; only merged PRs can be cleaned"
        )
    if pr["baseRefName"] != "main" or pr["isCrossRepository"]:
        raise CleanupError("PR must target this repository's main branch")
    if not pr.get("headRefName") or not pr.get("headRefOid"):
        raise CleanupError("PR head branch or commit is unavailable")
    if pr["headRefName"] == "main":
        raise CleanupError("Refusing to delete main")
    return pr


def branch_tip(repo: Path, ref: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def finish(repo: Path, number: int, *, discard_ignored: bool = False) -> None:
    pr = checked_pr(repo, number)
    branch = pr["headRefName"]
    expected = pr["headRefOid"]

    git("fetch", "origin", "--prune", cwd=repo)
    checked_out = worktrees(repo)
    main = checked_out.get("refs/heads/main")
    if main is None:
        raise CleanupError("No local main worktree is checked out")
    if git("status", "--porcelain", "--untracked-files=no", cwd=main).strip():
        raise CleanupError(f"Main worktree has tracked changes: {main}")

    local_ref = f"refs/heads/{branch}"
    remote_ref = f"refs/remotes/origin/{branch}"
    local_tip = branch_tip(repo, local_ref)
    remote_tip = branch_tip(repo, remote_ref)
    if local_tip is not None and local_tip != expected:
        raise CleanupError(f"Local {branch} has commits beyond PR head {expected}")
    if remote_tip is not None and remote_tip != expected:
        raise CleanupError(f"Remote {branch} moved beyond PR head {expected}")

    branch_worktree = checked_out.get(local_ref)
    if (
        branch_worktree is not None
        and git(
            "status", "--porcelain", "--untracked-files=all", cwd=branch_worktree
        ).strip()
    ):
        raise CleanupError(f"Branch worktree has uncommitted files: {branch_worktree}")
    if branch_worktree is not None and not discard_ignored:
        ignored = [
            line[3:]
            for line in git(
                "status",
                "--porcelain",
                "--ignored",
                "--untracked-files=normal",
                cwd=branch_worktree,
            ).splitlines()
            if line.startswith("!! ")
        ]
        if ignored:
            sample = ", ".join(ignored[:5])
            raise CleanupError(
                f"Branch worktree contains ignored files ({sample}); move them or "
                "rerun with --discard-ignored if disposable"
            )

    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", "main", "origin/main"],
        cwd=repo,
        check=False,
    ).returncode:
        raise CleanupError("Local main has diverged from origin/main")

    git("merge", "--ff-only", "origin/main", cwd=main)
    print(f"Updated main to {branch_tip(repo, 'refs/heads/main')}")

    if remote_tip is not None:
        git(
            "push",
            f"--force-with-lease=refs/heads/{branch}:{expected}",
            "origin",
            f":refs/heads/{branch}",
            cwd=repo,
        )
        print(f"Deleted remote branch {branch}")
    if branch_worktree is not None:
        os.chdir(main)
        git("worktree", "remove", "--", str(branch_worktree), cwd=main)
        print(f"Removed worktree {branch_worktree}")
    if local_tip is not None:
        if branch_tip(main, local_ref) != expected:
            raise CleanupError(f"Local {branch} changed during cleanup")
        git("branch", "-D", "--", branch, cwd=main)
        print(f"Deleted local branch {branch}")
    git("fetch", "origin", "--prune", cwd=main)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="merged pull request number")
    parser.add_argument(
        "--discard-ignored",
        action="store_true",
        help="also remove ignored files in the branch worktree",
    )
    args = parser.parse_args()
    try:
        repo = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()).strip())
        finish(repo, args.number, discard_ignored=args.discard_ignored)
    except CleanupError as exc:
        print(f"Cleanup stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
