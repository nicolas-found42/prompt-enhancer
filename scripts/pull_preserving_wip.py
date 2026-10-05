"""Rehearse a fast-forward and WIP restoration before changing this checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


class PullError(RuntimeError):
    """The checkout cannot be updated with complete WIP restoration."""


def git(repo: Path, *args: str, check: bool = True) -> bytes:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    if check and result.returncode:
        raise PullError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def file_bytes(path: Path) -> bytes:
    return (
        path.readlink().as_posix().encode() if path.is_symlink() else path.read_bytes()
    )


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target.symlink_to(source.readlink())
    else:
        shutil.copy2(source, target)


def fingerprint(repo: Path, files: list[str]) -> dict[str, object]:
    return {
        "head": git(repo, "rev-parse", "HEAD").decode().strip(),
        "status": git(repo, "status", "--porcelain=v1", "-z").hex(),
        "index": hashlib.sha256(git(repo, "diff", "--cached", "--binary")).hexdigest(),
        "diff": hashlib.sha256(git(repo, "diff", "--binary")).hexdigest(),
        "files": {
            name: hashlib.sha256(file_bytes(repo / name)).hexdigest()
            if (repo / name).exists() or (repo / name).is_symlink()
            else None
            for name in files
        },
    }


def pull(
    repo: Path, backup: Path, *, remote: str = "origin", branch: str = "main"
) -> dict:
    repo = repo.resolve()
    backup = backup.resolve()
    if backup.is_relative_to(repo) or backup.exists():
        raise PullError("Use a new backup directory outside this checkout")
    if git(repo, "ls-files", "--unmerged"):
        raise PullError("Resolve existing conflicts before pulling")
    if git(repo, "branch", "--show-current").decode().strip() != branch:
        raise PullError(f"Checkout must be on {branch}")
    tracked = git(repo, "diff", "--name-only", "HEAD", "-z").decode().split("\0")
    untracked = (
        git(repo, "ls-files", "--others", "--exclude-standard", "-z")
        .decode()
        .split("\0")
    )
    tracked = [name for name in tracked if name]
    untracked = [name for name in untracked if name]
    files = sorted(set(tracked + untracked))
    before = fingerprint(repo, files)
    backup.mkdir(parents=True)
    (backup / "tracked.patch").write_bytes(git(repo, "diff", "--binary", "HEAD"))
    (backup / "index.patch").write_bytes(git(repo, "diff", "--cached", "--binary"))
    for name in files:
        source = repo / name
        if source.exists() or source.is_symlink():
            copy_file(source, backup / "files" / name)
    receipt: dict = {
        "before": before,
        "status": "blocked",
        "files": {name: "preserved_for_reconciliation" for name in files},
        "stash": None,
    }
    receipt_path = backup / "receipt.json"

    def save() -> None:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        git(repo, "fetch", remote, branch)
        target = git(repo, "rev-parse", "FETCH_HEAD").decode().strip()
        receipt["target"] = target
        ignored = [
            name
            for name in git(
                repo, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
            )
            .decode()
            .split("\0")
            if name
        ]
        incoming = [
            name
            for name in git(repo, "ls-tree", "-r", "--name-only", "-z", target)
            .decode()
            .split("\0")
            if name
        ]
        for name in ignored:
            if any(
                name == tracked_name
                or name.startswith(tracked_name + "/")
                or tracked_name.startswith(name + "/")
                for tracked_name in incoming
            ):
                raise PullError(f"Ignored local data would be overwritten: {name}")
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(before["head"]), target],
            cwd=repo,
            capture_output=True,
        )
        if result.returncode:
            raise PullError("Local branch is not an ancestor of the remote head")
        # This creates a tracked-only stash object without touching the checkout/index.
        tracked_stash = git(repo, "stash", "create").decode().strip()
        receipt["tracked_snapshot"] = tracked_stash or None
        with tempfile.TemporaryDirectory(prefix="pull-rehearsal-") as directory:
            rehearsal = Path(directory) / "checkout"
            git(repo, "worktree", "add", "--detach", str(rehearsal), target)
            try:
                if tracked_stash:
                    git(rehearsal, "stash", "apply", "--index", tracked_stash)
                for name in untracked:
                    source, destination = backup / "files" / name, rehearsal / name
                    if destination.exists() or destination.is_symlink():
                        if (
                            destination.is_symlink() != source.is_symlink()
                            or file_bytes(destination) != file_bytes(source)
                        ):
                            raise PullError(
                                f"Untracked file conflicts with remote: {name}"
                            )
                        receipt["files"][name] = "already_incorporated"
                    else:
                        copy_file(source, destination)
                        receipt["files"][name] = "restored"
                for name in tracked:
                    receipt["files"][name] = (
                        "already_incorporated"
                        if not git(rehearsal, "diff", "HEAD", "--", name)
                        else "restored"
                    )
            finally:
                # Only our disposable rehearsal is removed, including its copied WIP.
                git(repo, "worktree", "remove", "--force", str(rehearsal))
        if fingerprint(repo, files) != before:
            raise PullError("Checkout changed during rehearsal; no update performed")
        if files:
            git(
                repo,
                "stash",
                "push",
                "--include-untracked",
                "-m",
                "pull-preserving-wip backup",
            )
            receipt["stash"] = git(repo, "rev-parse", "stash@{0}").decode().strip()
            save()
        git(repo, "merge", "--ff-only", "--no-overwrite-ignore", target)
        if tracked_stash:
            git(repo, "stash", "apply", "--index", tracked_stash)
        for name in untracked:
            destination = repo / name
            source = backup / "files" / name
            if not (destination.exists() or destination.is_symlink()):
                copy_file(source, destination)
            elif destination.is_symlink() != source.is_symlink() or file_bytes(
                destination
            ) != file_bytes(source):
                raise PullError(f"File changed after rehearsal: {name}")
        receipt["status"] = "complete"
        receipt["head"] = git(repo, "rev-parse", "HEAD").decode().strip()
        save()
        return receipt
    except (PullError, OSError) as exc:
        receipt["error"] = str(exc)
        if receipt["stash"]:
            receipt["status"] = "reconciliation_required"
        else:
            receipt["files"] = {name: "preserved_for_reconciliation" for name in files}
        save()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    try:
        repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
        pull(repo, args.backup, remote=args.remote, branch=args.branch)
    except (PullError, OSError) as exc:
        print(f"Pull stopped: {exc}")
        return 1
    print(f"Pull and restoration complete; receipt: {args.backup / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
