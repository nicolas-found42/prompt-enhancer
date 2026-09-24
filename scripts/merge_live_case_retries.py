"""Merge a targeted live rerun into a full recorded evaluation.

The strict full replay must be compared with the live reports afterward:
rerun case outcomes should match the targeted live run, and unaffected cases
should match the base run. This script never invokes a provider.

The merged recording takes the retry's writer instruction version. Base cases
that called the writer under an older version then fail strict replay, so the
full replay is the check that the merge is sound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation_review_common import save_json


def merge(base: dict[str, Any], retry: dict[str, Any], *, retry_case_ids: set[str]) -> dict[str, Any]:
    if not retry_case_ids or not retry_case_ids <= set(base["case_costs"]):
        raise ValueError("retry case IDs must be a nonempty subset of base cases")
    if set(retry["case_costs"]) != retry_case_ids or set(retry["case_latency_ms"]) != retry_case_ids:
        raise ValueError("retry recording case metrics differ from retry report")
    if set(base["case_costs"]) != set(base["case_latency_ms"]):
        raise ValueError("base recording has incomplete case metrics")
    if base["rubric_thresholds"] != retry["rubric_thresholds"]:
        raise ValueError("recordings use different rubric thresholds")
    if base.get("faithfulness_threshold", 0.9) != retry.get("faithfulness_threshold", 0.9):
        raise ValueError("recordings use different faithfulness thresholds")
    return {
        "responses": {**base["responses"], **retry["responses"]},
        "case_costs": {**base["case_costs"], **retry["case_costs"]},
        "case_latency_ms": {**base["case_latency_ms"], **retry["case_latency_ms"]},
        "rubric_thresholds": base["rubric_thresholds"],
        "writer_instruction_version": retry.get("writer_instruction_version", 1),
        "faithfulness_threshold": retry.get("faithfulness_threshold", 0.9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-record", type=Path, required=True)
    parser.add_argument("--retry-record", type=Path, required=True)
    parser.add_argument("--retry-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retry_report = json.loads(args.retry_report.read_text(encoding="utf-8"))
    if any(case["status"] in {"failed", "error"} for case in retry_report["cases"]):
        raise ValueError("retry report contains failed cases")
    result = merge(
        json.loads(args.base_record.read_text(encoding="utf-8")),
        json.loads(args.retry_record.read_text(encoding="utf-8")),
        retry_case_ids={case["case_id"] for case in retry_report["cases"]},
    )
    save_json(args.output, result)


if __name__ == "__main__":
    main()
