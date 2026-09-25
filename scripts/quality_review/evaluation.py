"""Labeled semantic evaluation, separate from offline plumbing tests."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .evidence import MAX_STATE_BYTES, digest


def pilot_plan(corpus: dict) -> dict:
    cases: list[dict[str, Any]] = []
    for item in corpus["cases"]:
        state = item["state"]
        if len(json.dumps(state).encode()) > MAX_STATE_BYTES:
            raise ValueError("pilot case exceeds state budget")
        cases.append(
            {
                "id": item["id"],
                "finding": state["finding"],
                "state": state,
                "status": "ready",
            }
        )
    plan = {
        "schema_version": 1,
        "base_sha": "synthetic",
        "merge_base_sha": "synthetic",
        "head_sha": "synthetic",
        "rules_digest": digest([c["state"]["rule"] for c in cases]),
        "cases": cases,
        "omitted": [],
    }
    return {**plan, "plan_digest": digest(plan)}


def evaluate(report: dict, corpus: dict) -> dict:
    expected = pilot_plan(corpus)
    if report["plan_digest"] != expected["plan_digest"]:
        raise ValueError("report does not belong to this evaluation corpus")
    rows = {r["id"]: r for r in report["results"]}
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "cases": 0,
            "judged": 0,
            "abstained": 0,
            "missing": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "false_suppressions": 0,
            "expected_positives": 0,
            "correct_relation": 0,
        }
    )
    for item in corpus["cases"]:
        group = counts[f"{item['rule_family']}/{item['split']}"]
        group["cases"] += 1
        group["expected_positives"] += item["expected"] == "supported"
        row = rows.get(item["id"], {})
        if row.get("status") != "judged":
            group["missing"] += 1
            continue
        group["judged"] += 1
        choice = row["support"]["choice"]
        group["correct_relation"] += choice == item["expected"]
        if choice == "insufficient_evidence":
            group["abstained"] += 1
        elif choice == "supported":
            group["tp" if item["expected"] == "supported" else "fp"] += 1
        else:
            group["fn" if item["expected"] == "supported" else "tn"] += 1
            if item["expected"] == "supported":
                group["false_suppressions"] += 1
    for group in counts.values():
        group["precision"] = (
            group["tp"] / (group["tp"] + group["fp"])
            if group["tp"] + group["fp"]
            else None
        )
        group["recall_answered"] = (
            group["tp"] / (group["tp"] + group["fn"])
            if group["tp"] + group["fn"]
            else None
        )
        group["recall_all_positives"] = (
            group["tp"] / group["expected_positives"]
            if group["expected_positives"]
            else None
        )
    return {
        "status": report["status"],
        "groups": dict(counts),
        "warning": "Small synthetic pilot; not repository precision or permission to block merges.",
        "usage": report.get("usage"),
        "latency_seconds": sum(r.get("elapsed_seconds", 0) for r in rows.values()),
    }
