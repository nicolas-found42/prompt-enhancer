"""Record terminal subprocess results, their logs, and exact source identity."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import tarfile
import time
from contextlib import contextmanager
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


def process_identity(pid: int) -> str | None:
    """Include start time so a reused PID cannot impersonate this runner."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True
    )
    return result.stdout.strip() or None


def receipt_status(value: dict, *, stale_seconds: float = 10) -> str:
    if value.get("status") != "running":
        return value.get("status", "unknown")
    runner = value.get("runner", {})
    if runner.get("host") != socket.gethostname() or not runner.get("identity"):
        return "unknown"
    if process_identity(runner["pid"]) != runner["identity"]:
        return "abandoned"
    heartbeat = datetime.fromisoformat(value["heartbeat_at"])
    age = (datetime.now(UTC) - heartbeat).total_seconds()
    return "unresponsive" if age > stale_seconds else "active"


def interrupt_signal(signum, frame) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def stop_process_group(process: subprocess.Popen) -> None:
    # The child owns a separate session; terminating it cannot kill the agent.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        # Keep the leader unreaped until escalation, reserving its PID/PGID even
        # when it exits before a descendant that ignores SIGTERM.
        time.sleep(2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except PermissionError:
            # macOS can return EPERM for a group containing only zombies.
            # Do not hide a permissions error while a live member remains.
            members = subprocess.check_output(["ps", "-axo", "pgid=,stat="], text=True)
            if any(
                group == str(process.pid) and not state.startswith("Z")
                for line in members.splitlines()
                if len(fields := line.split()) == 2
                for group, state in [fields]
            ):
                raise
        process.wait(timeout=2)
    except ProcessLookupError:
        process.wait()


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
            "runner": {
                "pid": os.getpid(),
                "identity": process_identity(os.getpid()),
                "host": socket.gethostname(),
            },
            "checks": [],
        }
        self.save()

    def save(self) -> None:
        self.value["heartbeat_at"] = datetime.now(UTC).isoformat()
        write_json(self.path, self.value)

    @contextmanager
    def interruptions(self):
        """Cover setup, bookkeeping and gaps between child checks as well."""
        previous = signal.signal(signal.SIGTERM, interrupt_signal)
        try:
            yield
        except KeyboardInterrupt as exc:
            if self.value["status"] != "interrupted":
                self.value.update(status="interrupted", error=str(exc) or "interrupted")
                self.save()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)

    def run(
        self, name: str, command: list[str], *, cwd: Path, env: dict | None = None
    ) -> Path:
        log = self.output / f"{len(self.value['checks']):02d}-{name}.log"
        check = {"name": name, "command": command, "status": "running", "log": log.name}
        self.value["checks"].append(check)
        self.save()
        started = time.monotonic()
        previous_handler = signal.signal(signal.SIGTERM, interrupt_signal)
        with log.open("wb") as stream:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=local_environment(env),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                check["process"] = {
                    "pid": process.pid,
                    "identity": process_identity(process.pid),
                }
                while True:
                    self.save()
                    try:
                        exit_code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except KeyboardInterrupt as exc:
                check.update(
                    status="interrupted",
                    duration_seconds=round(time.monotonic() - started, 3),
                )
                self.value.update(status="interrupted", error=str(exc))
                self.save()
                if "process" in locals():
                    try:
                        stop_process_group(process)
                    except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                        check["cleanup_error"] = type(cleanup_error).__name__
                        self.save()
                        raise
                raise
            except OSError as exc:
                stream.write(str(exc).encode())
                exit_code = 127
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
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
