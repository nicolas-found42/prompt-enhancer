"""Compare matched sequential and speculative diagnosis harness reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stage(
    report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    requests: list[int] = []
    latencies: list[float] = []
    fallbacks = 0
    complete = 0
    for case in cases:
        evidence = case.get("diagnosis_request_evidence")
        if not isinstance(evidence, Mapping):
            continue
        count = evidence.get("provider_requests")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            requests.append(count)
        complete += evidence.get("complete") is True
        fallbacks += evidence.get("mode") == "bounded_sequential_fallback"
        values = evidence.get("request_latencies_ms")
        if (
            evidence.get("latency_source") == "measured_provider"
            and isinstance(values, list)
            and all(isinstance(value, (int, float)) and value >= 0 for value in values)
        ):
            latencies.append(sum(float(value) for value in values))
    micro = report.get("diagnosis", {})
    micro = micro.get("micro", {}) if isinstance(micro, Mapping) else {}
    return {
        "cases": len(cases),
        "complete_count": complete,
        "provider_request_count": sum(requests)
        if len(requests) == len(cases)
        else None,
        "provider_request_mean": statistics.mean(requests)
        if len(requests) == len(cases) and requests
        else None,
        "fallback_count": fallbacks,
        "fallback_frequency": fallbacks / len(cases) if cases else None,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_source": "measured_provider"
        if len(latencies) == len(cases) and cases
        else "unavailable_or_deterministic",
        "gap_precision": micro.get("precision") if isinstance(micro, Mapping) else None,
        "gap_recall": micro.get("recall") if isinstance(micro, Mapping) else None,
    }


def compare_diagnosis_reports(
    sequential: Mapping[str, Any], speculative: Mapping[str, Any]
) -> dict[str, Any]:
    old = {
        str(case["case_id"]): case
        for case in sequential.get("cases", [])
        if isinstance(case, Mapping) and case.get("case_id") is not None
    }
    new = {
        str(case["case_id"]): case
        for case in speculative.get("cases", [])
        if isinstance(case, Mapping) and case.get("case_id") is not None
    }
    matched = sorted(old.keys() & new.keys())
    parity = {
        case_id: all(
            old[case_id].get(field) == new[case_id].get(field)
            for field in (
                "predicted_task_type",
                "predicted_gaps",
                "predicted_problem_sentences",
            )
        )
        for case_id in matched
    }
    return {
        "matched_cases": len(matched),
        "unmatched_sequential": sorted(old.keys() - new.keys()),
        "unmatched_speculative": sorted(new.keys() - old.keys()),
        "deterministic_parity_count": sum(parity.values()),
        "different_cases": [case_id for case_id in matched if not parity[case_id]],
        "parity_note": "Matched scripted answers test implementation parity; live differences can include Jev answer noise.",
        "sequential": _stage(sequential, [old[case_id] for case_id in matched]),
        "speculative": _stage(speculative, [new[case_id] for case_id in matched]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequential", type=Path)
    parser.add_argument("speculative", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_diagnosis_reports(
        json.loads(args.sequential.read_text(encoding="utf-8")),
        json.loads(args.speculative.read_text(encoding="utf-8")),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
