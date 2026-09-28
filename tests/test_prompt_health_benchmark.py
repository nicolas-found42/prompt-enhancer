"""The strict replay benchmark keeps enough evidence to reproduce its figures."""

from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path

import pytest


def test_replay_latency_passes_and_input_identity_are_reported() -> None:
    script = runpy.run_path(
        str(Path(__file__).parent.parent / "scripts/benchmark_prompt_health.py")
    )
    benchmark = script["benchmark"]
    digest = script["_digest"]
    percentile = script["_percentile"]
    cases = json.loads(
        (Path(__file__).parent / "fixtures/prompt_health_cases.json").read_text()
    )

    report = benchmark(cases)

    assert report["case_set_digest"] == digest(cases)
    assert len(report["recording_digest"]) == 64
    assert len(report["source_revision"]) == 40
    assert report["latency_provenance"] == "local_replay_execution"
    for name in ("first_pass", "cached_pass"):
        pass_latency = report["latency_ms"][name]
        samples = pass_latency["samples"]
        assert len(samples) == len(cases)
        assert pass_latency["p50"] == percentile(samples, 0.5)
        assert pass_latency["p95"] == percentile(samples, 0.95)


def test_replay_report_survives_missing_git_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = runpy.run_path(
        str(Path(__file__).parent.parent / "scripts/benchmark_prompt_health.py")
    )
    cases = json.loads(
        (Path(__file__).parent / "fixtures/prompt_health_cases.json").read_text()
    )

    def missing_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(script["subprocess"], "check_output", missing_git)
    report = script["benchmark"](cases)

    assert report["source_revision"] is None
    assert report["source_dirty"] is None
    assert report["cases"] == len(cases)
