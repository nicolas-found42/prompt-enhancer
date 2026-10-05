"""Exercise branch cleanup against a disposable local Git remote."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finish_merged_pr.py"


def clean_git_env() -> dict[str, str]:
    # Git commit hooks export repository-local variables; child test repos need their own.
    names = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {key: value for key, value in os.environ.items() if key not in names}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=clean_git_env(),
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> dict[str, Path | str]:
    remote = tmp_path / "remote.git"
    main = tmp_path / "main"
    feature = tmp_path / "feature"
    integrator = tmp_path / "integrator"
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(tmp_path, "clone", str(remote), str(main))
    git(main, "config", "user.name", "Test")
    git(main, "config", "user.email", "test@example.com")
    (main / "README.md").write_text("base\n")
    (main / ".gitignore").write_text("cache/\n")
    git(main, "add", "README.md", ".gitignore")
    git(main, "commit", "-m", "base")
    git(main, "push", "-u", "origin", "main")

    git(main, "worktree", "add", "-b", "feature/example", str(feature))
    (feature / "feature.txt").write_text("merged content\n")
    git(feature, "add", "feature.txt")
    git(feature, "commit", "-m", "feature")
    head = git(feature, "rev-parse", "HEAD")
    git(feature, "push", "-u", "origin", "feature/example")

    git(tmp_path, "clone", str(remote), str(integrator))
    git(integrator, "config", "user.name", "Test")
    git(integrator, "config", "user.email", "test@example.com")
    (integrator / "feature.txt").write_text("merged content\n")
    git(integrator, "add", "feature.txt")
    git(integrator, "commit", "-m", "squash feature")
    merged_main = git(integrator, "rev-parse", "HEAD")
    git(integrator, "push", "origin", "main")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = repo ]; then\n'
        "  printf '%s\\n' '{\"nameWithOwner\":\"owner/repo\"}'\n"
        "else\n"
        "  printf '%s\\n' \"$FAKE_PR_JSON\"\n"
        "fi\n"
    )
    gh.chmod(0o755)
    return {
        "main": main,
        "feature": feature,
        "head": head,
        "merged_main": merged_main,
        "fake_bin": fake_bin,
    }


def cleanup(
    repository: dict[str, Path | str],
    *,
    discard_ignored: bool = False,
    cleanup_only: bool = False,
    hook_python: Path | None = None,
    receipt: Path | None = None,
    **changes: object,
) -> subprocess.CompletedProcess[str]:
    pr: dict[str, object] = {
        "state": "MERGED",
        "baseRefName": "main",
        "headRefName": "feature/example",
        "headRefOid": repository["head"],
        "isCrossRepository": False,
    }
    pr.update(changes)
    env = clean_git_env()
    env["PATH"] = f"{repository['fake_bin']}{os.pathsep}{env['PATH']}"
    env["FAKE_PR_JSON"] = json.dumps(pr)
    args = [sys.executable, str(SCRIPT), "7"]
    if discard_ignored:
        args.append("--discard-ignored")
    if cleanup_only:
        args.append("--cleanup-only")
    if hook_python:
        args.extend(["--hook-python", str(hook_python)])
    if receipt:
        args.extend(["--receipt", str(receipt)])
    return subprocess.run(
        args,
        cwd=repository["feature"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_finishes_squash_merged_pr_with_clean_worktree(
    repository: dict[str, Path | str],
) -> None:
    main = repository["main"]
    feature = repository["feature"]
    assert isinstance(main, Path) and isinstance(feature, Path)
    (main / "local.settings").write_text("keep me\n")

    result = cleanup(repository)

    assert result.returncode == 0, result.stderr
    assert git(main, "rev-parse", "main") == repository["merged_main"]
    assert git(main, "branch", "--list", "feature/example") == ""
    assert git(main, "ls-remote", "--heads", "origin", "feature/example") == ""
    assert not feature.exists()
    assert (main / "local.settings").read_text() == "keep me\n"


def test_open_pr_cannot_be_cleaned(repository: dict[str, Path | str]) -> None:
    main = repository["main"]
    feature = repository["feature"]
    assert isinstance(main, Path) and isinstance(feature, Path)
    old_main = git(main, "rev-parse", "main")

    result = cleanup(repository, state="OPEN")

    assert result.returncode == 1
    assert "only merged PRs" in result.stderr
    assert git(main, "rev-parse", "main") == old_main
    assert feature.exists()


def test_new_local_commit_stops_before_changes(
    repository: dict[str, Path | str],
) -> None:
    main = repository["main"]
    feature = repository["feature"]
    assert isinstance(main, Path) and isinstance(feature, Path)
    old_main = git(main, "rev-parse", "main")
    (feature / "later.txt").write_text("local work\n")
    git(feature, "add", "later.txt")
    git(feature, "commit", "-m", "later work")

    result = cleanup(repository)

    assert result.returncode == 1
    assert "Local feature/example has commits beyond PR head" in result.stderr
    assert git(main, "rev-parse", "main") == old_main
    assert feature.exists()


def test_dirty_worktree_stops_before_changes(repository: dict[str, Path | str]) -> None:
    main = repository["main"]
    feature = repository["feature"]
    assert isinstance(main, Path) and isinstance(feature, Path)
    old_main = git(main, "rev-parse", "main")
    (feature / "draft.txt").write_text("uncommitted work\n")

    result = cleanup(repository)

    assert result.returncode == 1
    assert "uncommitted files" in result.stderr
    assert git(main, "rev-parse", "main") == old_main
    assert feature.exists()


def test_ignored_files_need_explicit_discard(repository: dict[str, Path | str]) -> None:
    main = repository["main"]
    feature = repository["feature"]
    assert isinstance(main, Path) and isinstance(feature, Path)
    old_main = git(main, "rev-parse", "main")
    (feature / "cache").mkdir()
    (feature / "cache" / "local.db").write_text("private data\n")

    stopped = cleanup(repository)

    assert stopped.returncode == 1
    assert "contains ignored files (cache/)" in stopped.stderr
    assert git(main, "rev-parse", "main") == old_main
    assert (feature / "cache" / "local.db").exists()

    discarded = cleanup(repository, discard_ignored=True)

    assert discarded.returncode == 0, discarded.stderr
    assert not feature.exists()


def test_cleanup_only_preserves_dirty_main_and_reports_deferred_sync(
    repository: dict, tmp_path: Path
) -> None:
    main = repository["main"]
    (main / "README.md").write_text("primary WIP\n")
    git(main, "add", "README.md")
    (main / "notes.txt").write_text("untracked WIP\n")
    before = (
        git(main, "rev-parse", "HEAD"),
        git(main, "diff", "--cached"),
        git(main, "status", "--porcelain"),
    )
    receipt = tmp_path / "receipt.json"
    result = cleanup(repository, cleanup_only=True, receipt=receipt)
    assert result.returncode == 0, result.stderr
    assert before == (
        git(main, "rev-parse", "HEAD"),
        git(main, "diff", "--cached"),
        git(main, "status", "--porcelain"),
    )
    assert (main / "README.md").read_text() == "primary WIP\n"
    assert (main / "notes.txt").read_text() == "untracked WIP\n"
    assert not repository["feature"].exists()
    record = json.loads(receipt.read_text())
    assert record["synchronization"] == "deferred"
    assert record["main_before"] == record["main_after"]
    assert record["cleanup"] == "complete"


def test_default_cleanup_still_protects_dirty_main(repository: dict) -> None:
    main = repository["main"]
    (main / "README.md").write_text("WIP\n")
    result = cleanup(repository)
    assert result.returncode == 1
    assert "Main worktree has tracked changes" in result.stderr
    assert repository["feature"].exists()


def test_cleanup_repairs_shared_hook_runtime_before_removal(
    repository: dict, tmp_path: Path
) -> None:
    main, feature = repository["main"], repository["feature"]
    # Use a configured hook location to exercise core.hooksPath as well.
    hooks = tmp_path / "shared-hooks"
    hooks.mkdir()
    git(main, "config", "core.hooksPath", str(hooks))
    hook = hooks / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n# File generated by pre-commit\nINSTALL_PYTHON="
        + str(feature / ".venv/bin/python")
        + '\nexec "$INSTALL_PYTHON" -c "import pre_commit; print(42)"\n'
    )
    hook.chmod(0o755)
    result = cleanup(repository, hook_python=Path(sys.executable))
    assert result.returncode == 0, result.stderr
    assert not feature.exists()
    executed = subprocess.run([str(hook)], capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout.strip() == "42"


def test_missing_replacement_preserves_worktree_and_remote_branch(
    repository: dict, tmp_path: Path
) -> None:
    main, feature = repository["main"], repository["feature"]
    hook = main / ".git/hooks/pre-commit"
    hook.write_text(
        "#!/bin/sh\n# File generated by pre-commit\nINSTALL_PYTHON="
        + str(feature / ".venv/bin/python")
        + "\n"
    )
    old_head = git(main, "rev-parse", "HEAD")
    result = cleanup(repository, hook_python=tmp_path / "missing-python")
    assert result.returncode == 1
    assert "No replacement Python" in result.stderr
    assert feature.exists()
    assert git(main, "rev-parse", "HEAD") == old_head
    assert git(main, "ls-remote", "--heads", "origin", "feature/example")


def test_custom_hook_dependency_is_preserved_for_manual_repair(
    repository: dict,
) -> None:
    main, feature = repository["main"], repository["feature"]
    hook = main / ".git/hooks/pre-commit"
    hook.write_text(f"#!/bin/sh\nexec {feature}/custom-check\n")
    result = cleanup(repository, cleanup_only=True)
    assert result.returncode == 1
    assert "repair its dependencies" in result.stderr
    assert feature.exists()


def test_hook_interpreter_symlink_inside_removed_worktree_is_not_reused(
    repository: dict, tmp_path: Path
) -> None:
    main, feature = repository["main"], repository["feature"]
    (feature / "cache").mkdir()
    python = feature / "cache/python"
    python.symlink_to(sys.executable)
    hook = main / ".git/hooks/pre-commit"
    hook.write_text(
        f"#!/bin/sh\n# File generated by pre-commit\nINSTALL_PYTHON={python}\n"
    )
    result = cleanup(repository, discard_ignored=True, hook_python=python)
    assert result.returncode == 1
    assert "No replacement Python" in result.stderr
    assert feature.exists()


def test_remote_branch_movement_blocks_cleanup_before_main_sync(
    repository: dict,
) -> None:
    main = repository["main"]
    integrator = main.parent / "integrator"
    git(integrator, "switch", "feature/example")
    (integrator / "later.txt").write_text("remote new work\n")
    git(integrator, "add", "later.txt")
    git(integrator, "commit", "-m", "new remote work")
    git(integrator, "push", "origin", "feature/example")
    old_head = git(main, "rev-parse", "HEAD")
    result = cleanup(repository)
    assert result.returncode == 1
    assert "Remote feature/example moved beyond PR head" in result.stderr
    assert git(main, "rev-parse", "HEAD") == old_head
    assert repository["feature"].exists()
