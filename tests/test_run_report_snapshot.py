"""Pin the public results, reports included, of the scripted gateway scenarios.

The web app, run history, training and the evaluation harness all read the run
report, so a refactor must leave it byte-identical. Run IDs are random and
timing varies, so both are normalized away. If a report changes on purpose,
regenerate with ``UPDATE_REPORT_SNAPSHOT=1 uv run pytest tests/test_run_report_snapshot.py``.
"""

from __future__ import annotations

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
        return {key: _normalized(item, run_ids) for key, item in value.items() if key != "timing"}
    if isinstance(value, list):
        return [_normalized(item, run_ids) for item in value]
    if isinstance(value, str):
        for run_id in run_ids:
            value = value.replace(run_id, "<run-id>")
    return value


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_results_are_unchanged(name: str) -> None:
    make_gateway, run = SCENARIOS[name]
    results = run(PromptOptimizer(gateway=make_gateway(), store=RunStore(":memory:")))
    snapshot = json.loads(json.dumps(_normalized(results, {result["run_id"] for result in results})))

    if os.environ.get("UPDATE_REPORT_SNAPSHOT"):
        pinned = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {}
        pinned[name] = snapshot
        FIXTURE.write_text(json.dumps(pinned, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    assert snapshot == json.loads(FIXTURE.read_text())[name]
