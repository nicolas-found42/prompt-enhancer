import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from calibrate_faithfulness import calibrate
from compare_writer_reports import compare
from human_gap_review import batch_digest, import_review, make_review_html


def _batch():
    return {"cases": [
        {"id": "one", "prompt": "Write an outline", "result": "Outline", "source": "codex", "session": "one.jsonl"},
        {"id": "two", "prompt": "Fix it", "result": "Fixed", "source": "claude", "session": "two.jsonl"},
    ]}


def _review(batch):
    return {"batch_digest": batch_digest(batch), "reviewer": "Reviewer A", "reviews": [
        {"id": "one", "judgment": "labeled", "task_type": "writing", "gaps": [], "context_mode": "standalone", "context_text": "", "source_session_reviewed": True},
        {"id": "two", "judgment": "labeled", "task_type": "coding", "gaps": ["context"], "context_mode": "reconstructed", "context_text": "The prior turn names src/app.py.", "source_session_reviewed": True},
    ]}


def test_review_import_preserves_human_negatives_and_supplied_context():
    batch = _batch()
    dataset = import_review(batch, _review(batch), minimum=2)

    assert dataset["cases"][0]["expected_gaps"] == []
    assert dataset["cases"][1]["expected_gaps"] == ["context"]
    assert dataset["cases"][1]["prompt"].endswith("Current user request:\nFix it")
    assert dataset["metadata"]["task_counts"] == {"writing": 1, "coding": 1}


def test_review_import_rejects_missing_provenance_or_shortfall():
    batch = _batch()
    review = _review(batch)
    review["reviews"][1]["source_session_reviewed"] = False
    with pytest.raises(ValueError, match="source session review required"):
        import_review(batch, review, minimum=2)
    review["reviews"][1]["judgment"] = "uncertain"
    with pytest.raises(ValueError, match="only 1 usable human judgments"):
        import_review(batch, review, minimum=2)


def test_review_import_rejects_gap_outside_task_checklist():
    batch = _batch()
    review = _review(batch)
    review["reviews"][0]["gaps"] = ["language"]

    with pytest.raises(ValueError, match="gap label does not apply to task"):
        import_review(batch, review, minimum=2)


def test_review_html_escapes_prompt_markup():
    batch = _batch()
    batch["cases"][0]["prompt"] = "</script><script>alert(1)</script>"

    html = make_review_html(batch)

    assert html.count("</script>") == 2
    assert "\\u003c/script>" in html


def test_writer_comparison_excludes_unavailable_pairs():
    def report(writer, deltas):
        return {
            "run_identity": {"dataset_digest": "same"},
            "options": {"tier": "fast", "seed": 0, "model_overrides": {"writer": writer}},
            "cases": [{"case_id": str(i), "status": "completed", "score_delta": value} for i, value in enumerate(deltas)],
            "improvement": {"comparable_cases": sum(value is not None for value in deltas)},
            "cost": {"total": 0.01}, "latency_ms": {"p50": 100},
        }

    result = compare(report("a", [0.2, None, 0.1]), report("b", [0.1, 0.3, 0.1]))

    assert result["both_scored"] == 2
    assert result["unavailable_pairwise"] == 1
    assert (result["left_wins"], result["right_wins"], result["ties"]) == (1, 0, 1)


def test_faithfulness_review_checks_provenance_and_keeps_conversations_together(tmp_path):
    train_id = next(f"case-{i}" for i in range(100) if int(hashlib.sha256(f"case-{i}".encode()).hexdigest()[:8], 16) % 5 != 0)
    holdout_id = next(f"case-{i}" for i in range(100) if int(hashlib.sha256(f"case-{i}".encode()).hexdigest()[:8], 16) % 5 == 0)
    rows = []
    evidence = {"question": "faithful?", "rows": {}}
    for case_id, answer, probability in [(train_id, "yes", 0.95), (holdout_id, "no", 0.1)]:
        row_id = f"{case_id}/writer/test-1"
        row = {"row_id": row_id, "source_case_id": case_id, "writer": "writer", "prompt": "Write a report", "proposed_test": '{"question":"Report?"}', "human_faithful": answer, "human_notes": "", "human_reviewer": "A", "reviewed_at": "2026-09-23"}
        rows.append(row)
        evidence["rows"][row_id] = {"source_case_id": case_id, "writer": "writer", "probability": probability, "content_digest": hashlib.sha256((row["prompt"] + "\n" + row["proposed_test"]).encode()).hexdigest()}
    review_path = tmp_path / "review.csv"
    evidence_path = tmp_path / "evidence.json"
    with review_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    evidence_path.write_text(json.dumps(evidence))

    result = calibrate(review_path, evidence_path, minimum=2)

    assert (result["train_count"], result["holdout_count"]) == (1, 1)
    assert result["selected_holdout"]["tn"] == 1
    rows[0]["proposed_test"] = "changed"
    with review_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="source text or provenance changed"):
        calibrate(review_path, evidence_path, minimum=2)
