"""Record terminal subprocess results, their logs, and exact source identity."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path


def local_environment(environment: dict | None = None) -> dict:
    # Hooks export repository/index selectors. Every operation here targets cwd
    # or --repo explicitly, so an inherited selector must not redirect it.
    return {
        key: value
        for key, value in (os.environ if environment is None else environment).items()
        if not key.startswith("GIT_")
    }


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], env=local_environment()
    )


def commit(repo: Path, ref: str) -> str:
    return (
        git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
        .decode()
        .strip()
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_digest(root: Path, paths: list[str]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(set(paths)):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            kind, data = "symlink", os.readlink(path).encode()
        else:
            kind = "executable" if path.stat().st_mode & 0o111 else "file"
            data = path.read_bytes()
        hasher.update(json.dumps([name, kind, digest(data)]).encode() + b"\n")
    return hasher.hexdigest()


def worktree_digest(repo: Path) -> str:
    paths = (
        git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        .decode()
        .split("\0")
    )
    return source_digest(repo, [name for name in paths if name])


def snapshot(repo: Path, sha: str, destination: Path) -> str:
    destination.mkdir()
    archive = destination.parent / "source.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", sha],
            stdout=stream,
            env=local_environment(),
            check=True,
        )
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()
    paths = git(repo, "ls-tree", "-r", "--name-only", "-z", sha).decode().split("\0")
    return source_digest(destination, [name for name in paths if name])


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class Receipt:
    def __init__(self, output: Path, *, kind: str, phase: str, sha: str):
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.path = output / "receipt.json"
        self.value: dict = {
            "schema_version": 1,
            "kind": kind,
            "phase": phase,
            "commit": sha,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "checks": [],
        }
        self.save()

    def save(self) -> None:
        write_json(self.path, self.value)

    def run(
        self, name: str, command: list[str], *, cwd: Path, env: dict | None = None
    ) -> Path:
        log = self.output / f"{len(self.value['checks']):02d}-{name}.log"
        check = {"name": name, "command": command, "status": "running", "log": log.name}
        self.value["checks"].append(check)
        self.save()
        started = time.monotonic()
        with log.open("wb") as stream:
            try:
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    env=local_environment(env),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                exit_code = result.returncode
            except OSError as exc:
                stream.write(str(exc).encode())
                exit_code = 127
        check.update(
            exit_code=exit_code,
            duration_seconds=round(time.monotonic() - started, 3),
            log_sha256=digest(log.read_bytes()),
            status="passed" if exit_code == 0 else "failed",
        )
        self.save()
        if exit_code:
            raise RuntimeError(f"{name} exited {exit_code}; see {log}")
        return log

    def finish(self) -> None:
        checks = self.value["checks"]
        if not checks or any(check["status"] != "passed" for check in checks):
            raise ValueError("receipt has incomplete checks")
        self.value["status"] = "complete"
        self.save()

    def fail(self, error: Exception) -> None:
        self.value.update(status="failed", error=str(error))
        self.save()
