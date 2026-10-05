"""Exercise pull rehearsal with real disposable repositories and collisions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_finish_merged_pr import clean_git_env, git
from test_finish_merged_pr import repository as repository

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pull_preserving_wip.py"


def run_pull(repository: dict, backup: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--backup", str(backup)],
        cwd=repository["main"],
        env=clean_git_env(),
        capture_output=True,
        text=True,
    )


def test_pull_restores_staged_changes_and_untracked_files(
    repository: dict, tmp_path: Path
) -> None:
    main = repository["main"]
    (main / "README.md").write_text("local staged work\n")
    git(main, "add", "README.md")
    (main / "draft.bin").write_bytes(b"\x00\xffdraft")
    backup = tmp_path / "backup"
    result = run_pull(repository, backup)
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(main, "rev-parse", "HEAD") == repository["merged_main"]
    assert (main / "README.md").read_text() == "local staged work\n"
    assert git(main, "diff", "--cached", "--name-only") == "README.md"
    assert (main / "draft.bin").read_bytes() == b"\x00\xffdraft"
    receipt = json.loads((backup / "receipt.json").read_text())
    assert receipt["files"] == {"README.md": "restored", "draft.bin": "restored"}
    assert receipt["stash"] == git(main, "rev-parse", "stash@{0}")


def test_newly_tracked_file_collision_blocks_before_primary_changes(
    repository: dict, tmp_path: Path
) -> None:
    main = repository["main"]
    (main / "feature.txt").write_text("different untracked implementation\n")
    old_head = git(main, "rev-parse", "HEAD")
    old_status = git(main, "status", "--porcelain")
    result = run_pull(repository, tmp_path / "backup")
    assert result.returncode == 1
    assert "Untracked file conflicts with remote: feature.txt" in result.stdout
    assert git(main, "rev-parse", "HEAD") == old_head
    assert git(main, "status", "--porcelain") == old_status
    assert (main / "feature.txt").read_text() == "different untracked implementation\n"
    receipt = json.loads((tmp_path / "backup/receipt.json").read_text())
    assert receipt["files"]["feature.txt"] == "preserved_for_reconciliation"
    assert receipt["stash"] is None


def test_identical_newly_tracked_file_is_already_incorporated(
    repository: dict, tmp_path: Path
) -> None:
    main = repository["main"]
    (main / "feature.txt").write_text("merged content\n")
    result = run_pull(repository, tmp_path / "backup")
    assert result.returncode == 0, result.stdout
    receipt = json.loads((tmp_path / "backup/receipt.json").read_text())
    assert receipt["files"]["feature.txt"] == "already_incorporated"
    assert git(main, "status", "--porcelain") == ""


def test_tracked_conflict_leaves_head_index_and_content_unchanged(
    repository: dict, tmp_path: Path
) -> None:
    main = repository["main"]
    integrator = main.parent / "integrator"
    (integrator / "README.md").write_text("remote replacement\n")
    git(integrator, "add", "README.md")
    git(integrator, "commit", "-m", "remote change")
    git(integrator, "push", "origin", "main")
    (main / "README.md").write_text("local replacement\n")
    git(main, "add", "README.md")
    before = (
        git(main, "rev-parse", "HEAD"),
        git(main, "diff", "--cached"),
        git(main, "status", "--porcelain"),
    )
    result = run_pull(repository, tmp_path / "backup")
    assert result.returncode == 1
    assert before == (
        git(main, "rev-parse", "HEAD"),
        git(main, "diff", "--cached"),
        git(main, "status", "--porcelain"),
    )
    assert (main / "README.md").read_text() == "local replacement\n"
    assert git(main, "ls-files", "--unmerged") == ""


def test_ignored_data_collision_blocks_without_reading_or_stashing_it(
    repository: dict, tmp_path: Path
) -> None:
    main = repository["main"]
    (main / "cache").mkdir()
    (main / "cache/data.bin").write_bytes(b"local ignored data")
    integrator = main.parent / "integrator"
    (integrator / "cache").mkdir()
    (integrator / "cache/data.bin").write_bytes(b"remote data")
    git(integrator, "add", "--force", "cache/data.bin")
    git(integrator, "commit", "-m", "track previously ignored path")
    git(integrator, "push", "origin", "main")
    old_head = git(main, "rev-parse", "HEAD")
    result = run_pull(repository, tmp_path / "backup")
    assert result.returncode == 1
    assert "Ignored local data would be overwritten" in result.stdout
    assert git(main, "rev-parse", "HEAD") == old_head
    assert (main / "cache/data.bin").read_bytes() == b"local ignored data"
    assert not (tmp_path / "backup/files/cache/data.bin").exists()
