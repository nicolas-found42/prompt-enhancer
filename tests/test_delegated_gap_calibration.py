"""Delegated labels stay explicit and related prompts stay in one split."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_delegated_gaps import calibrate

from prompt_enhancer.diagnosis import default_gap_question


def test_calibration_keeps_source_groups_together_and_records_provenance(tmp_path: Path) -> None:
    groups: dict[int, str] = {}
    for index in range(100):
        group = f"participant-{index}"
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5
        groups.setdefault(bucket, group)
    train_group = next(group for bucket, group in groups.items() if bucket != 0)
    holdout_group = groups[0]
    dataset = {
        "name": "reviewed", "metadata": {"reviewer_kind": "user_delegated_model"},
        "cases": [
            {"id": f"case-{i}", "task_stratum": "chat", "source_group": group,
             "expected_gaps": ["context"] if i % 2 else [],
             "label_provenance": "user_delegated_model"}
            for i, group in enumerate([train_group, train_group, holdout_group, holdout_group])
        ],
    }
    decisions = {"dataset_name": "reviewed", "rows": [
        {"case_id": f"case-{i}", "source_group": case["source_group"],
         "probabilities": {"goal": 0.2, "context": 0.9 if i % 2 else 0.1}}
        for i, case in enumerate(dataset["cases"])
    ]}
    dataset_path, decisions_path = tmp_path / "cases.json", tmp_path / "decisions.json"
    dataset_path.write_text(json.dumps(dataset))
    decisions_path.write_text(json.dumps(decisions))

    report = calibrate(dataset_path, decisions_path)

    assert report["label_provenance"] == "user_delegated_model"
    assert report["questions"]["context"]["question"] == default_gap_question("context")
    assert report["questions"]["context"]["train_count"] == 2
    assert report["questions"]["context"]["holdout_count"] == 2
    assert report["questions"]["context"]["selected_holdout"]["tp"] == 1
    assert report["questions"]["goal"]["selected_threshold"] is None


def test_calibration_rejects_labels_without_delegated_provenance(tmp_path: Path) -> None:
    dataset_path, decisions_path = tmp_path / "cases.json", tmp_path / "decisions.json"
    dataset_path.write_text(json.dumps({"name": "reviewed", "metadata": {}, "cases": []}))
    decisions_path.write_text(json.dumps({"dataset_name": "reviewed", "rows": []}))

    with pytest.raises(ValueError, match="homogeneous delegated-model"):
        calibrate(dataset_path, decisions_path)
