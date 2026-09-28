"""Run labeled draft-health cases through strict replay and report diagnostics.

Without --recording, first capture deterministic fixture answers, then replay
them. These synthetic figures verify the measurement path, not Jev accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prompt_enhancer.catalog import JEV_MODEL
from prompt_enhancer.gateway import ReplayGateway, ScriptedGateway
from prompt_enhancer.prompt_health import (
    PromptHealthPolicy,
    PromptHealthService,
    PromptHealthStore,
)


def _fixture_answer(
    request: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    key = str(request["key"])
    if request["type"] == "score":
        return {
            "type": "score",
            "score": 3,
            "levels": {str(index): float(index == 3) for index in range(4)},
        }
    if key.endswith(":applicable"):
        probability = 0.95 if ":task:" in key else 0.05
    else:
        state = request["state"]
        prompt = state.get("prompt", state.get("sentence", ""))
        expected = next(
            (case["expected_flags"] for case in cases if case["prompt"] == prompt),
            [],
        )
        if key.endswith(":vagueness") and state.get("sentence") == "Do this.":
            probability = 0.95
        else:
            sentence_flag = key.removeprefix("sentence:")
            probability = 0.95 if sentence_flag in expected else 0.05
    return {"type": "noul", "probability_true": probability}


class _CaptureGateway(ScriptedGateway):
    def __init__(self, cases: list[dict[str, Any]]) -> None:
        super().__init__(decision=lambda request, **_: _fixture_answer(request, cases))
        self.recordings: dict[str, Any] = {}
        self.provenance: dict[str, dict[str, Any]] = {}

    def decide(
        self,
        payload: Mapping[str, Any],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> Any:
        answer = super().decide(payload, role=role, run_id=run_id)
        key = ReplayGateway.request_key("decide", self.jev_model, dict(payload), role)
        self.recordings[key] = answer
        self.provenance[key] = {"answered_by": self.jev_model, "usage": {}}
        return answer


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_provenance() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, bool(dirty)


def benchmark(
    cases: list[dict[str, Any]], recording: dict[str, Any] | None = None
) -> dict[str, Any]:
    source_revision, source_dirty = _source_provenance()
    with tempfile.TemporaryDirectory(prefix="prompt-health-benchmark-") as folder:
        root = Path(folder)
        benchmark_policy = PromptHealthPolicy(
            max_refreshes_per_minute=max(20, len(cases))
        )
        if recording is None:
            capture = _CaptureGateway(cases)
            capture_service = PromptHealthService(
                capture,
                PromptHealthStore(root / "capture.sqlite3"),
                policy=benchmark_policy,
            )
            for index, case in enumerate(cases):
                result = capture_service.assess(case["prompt"], index, "capture")
                if result["status"] != "complete":
                    raise RuntimeError(f"fixture capture failed: {case['id']}")
            recording = {
                "responses": capture.recordings,
                "decision_provenance": capture.provenance,
                "jev_model": JEV_MODEL,
            }
            mode = "scripted_strict_replay"
        else:
            mode = "provided_strict_replay"
        replay = ReplayGateway(
            recording["responses"],
            decision_provenance=recording["decision_provenance"],
            jev_model=recording.get("jev_model", JEV_MODEL),
        )
        service = PromptHealthService(
            replay, PromptHealthStore(root / "replay.sqlite3"), policy=benchmark_policy
        )
        false_flags = missed_flags = labeled_flags = 0
        hits = misses = 0
        latencies: dict[int, list[float]] = {0: [], 1: []}
        for pass_number in range(2):
            for index, case in enumerate(cases):
                start = time.perf_counter()
                result = service.assess(
                    case["prompt"], index + pass_number * len(cases), "replay"
                )
                latencies[pass_number].append((time.perf_counter() - start) * 1000)
                if result["status"] != "complete":
                    raise RuntimeError(f"strict replay incomplete for {case['id']}")
                hits += result["cache"]["hits"]
                misses += result["cache"]["misses"]
                if pass_number == 0:
                    expected = set(case["expected_flags"])
                    predicted = {
                        f"{flag['sentence_id']}:{flag['kind']}"
                        for flag in result["flags"]
                    }
                    false_flags += len(predicted - expected)
                    missed_flags += len(expected - predicted)
                    labeled_flags += len(expected)
        return {
            "mode": mode,
            "cases": len(cases),
            "false_flags": false_flags,
            "missed_flags": missed_flags,
            "labeled_flags": labeled_flags,
            "cache_hit_rate": hits / (hits + misses) if hits + misses else None,
            "latency_ms": {
                name: {
                    "p50": _percentile(values, 0.5),
                    "p95": _percentile(values, 0.95),
                    "samples": values,
                }
                for name, values in (
                    ("first_pass", latencies[0]),
                    ("cached_pass", latencies[1]),
                )
            },
            "latency_provenance": "local_replay_execution",
            "source_revision": source_revision,
            "source_dirty": source_dirty,
            "case_set_digest": _digest(cases),
            "recording_digest": _digest(recording),
            "cost_usd": None,
            "cost_provenance": "unavailable_for_replay",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("tests/fixtures/prompt_health_cases.json"),
    )
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    recording = (
        json.loads(args.recording.read_text(encoding="utf-8"))
        if args.recording is not None
        else None
    )
    report = benchmark(cases, recording)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
