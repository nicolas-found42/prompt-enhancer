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
    git(main, "add", "README.md")
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
    repository: dict[str, Path | str], **changes: object
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
    return subprocess.run(
        [sys.executable, str(SCRIPT), "7"],
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
