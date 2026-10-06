"""Prepare an isolated run with key-free configuration; live execution is explicit."""

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from validation.receipts import commit, stop_process_group, worktree_digest, write_json

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.config import Settings
from prompt_enhancer.settings import ModelDefaults

ROOT = Path(__file__).resolve().parents[1]


def prepare(
    source: Path,
    output: Path,
    *,
    judge: str,
    operation_timeout: float,
    request_timeout: float,
) -> dict:
    value = json.loads(source.read_text())
    models = ModelDefaults.from_dict(value["models"]).to_dict()
    # Export the public model fields only. Arbitrary sidecar fields never reach
    # the copied settings or provenance (including calibration free text).
    public = {"models": models, "judge": judge}
    floors = value.get("score_floors", {})
    if floors:
        public["score_floors"] = {
            key: float(floors[key]) for key in Settings().score_floors if key in floors
        }
        if any(
            not math.isfinite(x) or not 0 <= x <= 1
            for x in public["score_floors"].values()
        ):
            raise ValueError("Invalid score floors")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "runs.settings.json", public)
    plan = {
        "source_settings_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "exported_settings_sha256": hashlib.sha256(
            (output / "runs.settings.json").read_bytes()
        ).hexdigest(),
        "models": models,
        "judge": judge,
        "score_floors": public.get("score_floors", {}),
        "operation_timeout_s": operation_timeout,
        "request_timeout_s": request_timeout,
        "source_commit": commit(ROOT, "HEAD"),
        "source_digest": worktree_digest(ROOT),
        "status": "prepared",
    }
    write_json(output / "provenance.json", plan)
    return plan


def verify_settings(plan: dict, actual: dict) -> None:
    expected = {
        "judge_model": plan["judge"],
        "writer_model": plan["models"]["writer"],
        "strong_check_model": plan["models"]["strong"],
        "weak_models": plan["models"]["weak"],
    }
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Effective model settings differ from the isolated run plan")
    if any(
        actual.get("score_floors", {}).get(key) != value
        for key, value in plan.get("score_floors", {}).items()
    ):
        raise ValueError("Effective score floors differ from the isolated run plan")
    limits = actual.get("gateway_limits", {})
    if any(
        limits.get(key) != plan[key]
        for key in ("operation_timeout_s", "request_timeout_s")
    ):
        raise ValueError("Effective Gateway bounds differ from the isolated run plan")


def request(base: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=2) as response:
        return json.load(response)


def execute(
    plan: dict, output: Path, prompt: str, *, wall_seconds: float, drain_seconds: float
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "PROMPT_ENHANCER_DB": str(output / "runs.sqlite3"),
        "PROMPT_ENHANCER_PORT": str(port),
        "PROMPT_ENHANCER_HOST": "127.0.0.1",
        "PROMPT_ENHANCER_JEV_MODEL": plan["judge"],
        "PROMPT_ENHANCER_OPERATION_TIMEOUT": str(plan["operation_timeout_s"]),
        "PROMPT_ENHANCER_TIMEOUT": str(plan["request_timeout_s"]),
    }
    with (output / "server.log").open("wb") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "prompt_enhancer"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            startup = time.monotonic() + 15
            while True:
                try:
                    actual = request(base, "/api/settings")
                    break
                except OSError:
                    if server.poll() is not None or time.monotonic() >= startup:
                        raise RuntimeError(
                            "Isolated server did not become ready"
                        ) from None
                    time.sleep(0.1)
            verify_settings(plan, actual)
            plan.update(
                status="configuration_verified",
                effective_models={
                    k: actual[k]
                    for k in (
                        "judge_model",
                        "writer_model",
                        "strong_check_model",
                        "weak_models",
                    )
                },
            )
            write_json(output / "provenance.json", plan)
            job = request(base, "/api/jobs/optimize", {"prompt": prompt})
            run_id = job["run_id"]
            deadline = time.monotonic() + wall_seconds
            plan.update(
                run_id=run_id,
                wall_seconds=wall_seconds,
                cancel_drain_seconds=drain_seconds,
            )
            write_json(output / "provenance.json", plan)
            while job["state"] in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.1)
                job = request(base, f"/api/jobs/{run_id}")
            if job["state"] in {"queued", "running"}:
                request(base, f"/api/jobs/{run_id}/cancel", {})
                deadline = time.monotonic() + drain_seconds
                while (
                    job["state"] in {"queued", "running"}
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.1)
                    job = request(base, f"/api/jobs/{run_id}")
            write_json(output / "final-job.json", job)
            write_json(output / "history.json", request(base, f"/api/runs/{run_id}"))
            plan["status"] = "terminal" if job["state"] == "done" else "incomplete"
            write_json(output / "provenance.json", plan)
            if plan["status"] == "incomplete":
                raise RuntimeError(
                    "Isolated run remained active after cancellation drain"
                )
        finally:
            stop_process_group(server)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default=JEV_MODEL)
    parser.add_argument("--operation-timeout", type=float, default=30)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--wall-seconds", type=float, default=120)
    parser.add_argument("--drain-seconds", type=float, default=40)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Make live model calls using the server runtime environment",
    )
    parser.add_argument("--prompt")
    args = parser.parse_args()
    if any(
        not 0 < x <= 3600
        for x in (
            args.operation_timeout,
            args.request_timeout,
            args.wall_seconds,
            args.drain_seconds,
        )
    ) or (args.run and not args.prompt):
        parser.error(
            "Use positive finite bounds <=3600 seconds and supply a prompt for --run"
        )
    try:
        output = args.output.resolve()
        plan = prepare(
            args.source_settings,
            output,
            judge=args.judge_model,
            operation_timeout=args.operation_timeout,
            request_timeout=args.request_timeout,
        )
        if args.run:
            execute(
                plan,
                output,
                args.prompt,
                wall_seconds=args.wall_seconds,
                drain_seconds=args.drain_seconds,
            )
        print(f"Isolated run evidence: {output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Isolated run failed: {type(exc).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
