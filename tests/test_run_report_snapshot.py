"""Pin the public results, reports included, of the scripted gateway scenarios.

The web app, run history, training and the evaluation harness all read the run
report, so a refactor must leave it byte-identical. Run IDs are random and
timing varies, so both are normalized away. The full results run to megabytes
(each round's history repeats its panel evidence), so the fixture pins a
SHA-256 digest; on a mismatch the actual JSON is written out for diffing. If a
report changes on purpose, regenerate with
``UPDATE_REPORT_SNAPSHOT=1 uv run pytest tests/test_run_report_snapshot.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from test_gateway_request_keys import SCENARIOS

from prompt_enhancer import PromptOptimizer, RunStore

FIXTURE = Path(__file__).parent / "fixtures" / "run_report_snapshot.json"


def _normalized(value: Any, run_ids: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalized(item, run_ids)
            for key, item in value.items()
            if key not in {"timing", "classification_latency_ms"}
        }
    if isinstance(value, list):
        return [_normalized(item, run_ids) for item in value]
    if isinstance(value, str):
        for run_id in run_ids:
            value = value.replace(run_id, "<run-id>")
    return value


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_results_are_unchanged(name: str, tmp_path: Path) -> None:
    make_gateway, run = SCENARIOS[name]
    results = run(
        PromptOptimizer(
            gateway=make_gateway(),
            store=RunStore(":memory:"),
            writer_instruction_version=4,
        )
    )
    rendered = json.dumps(
        _normalized(results, {result["run_id"] for result in results}),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    if os.environ.get("UPDATE_REPORT_SNAPSHOT"):
        pinned = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {}
        pinned[name] = digest
        FIXTURE.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")

    actual = tmp_path / f"{name}.json"
    actual.write_text(rendered + "\n")
    assert digest == json.loads(FIXTURE.read_text())[name], (
        f"results changed; actual results are in {actual}"
    )
