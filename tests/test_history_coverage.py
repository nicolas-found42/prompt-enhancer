"""Search completeness and pagination over a known persistent corpus."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.config import Settings
from prompt_enhancer.history import RunHistory
from prompt_enhancer.store import RunStore


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "history.sqlite3"
    store = RunStore(path)
    base = datetime(2026, 10, 1, tzinfo=UTC)
    for index in range(260):
        store.save_run(
            {
                "run_id": f"run-{index:03}",
                "created_at": (base + timedelta(seconds=index)).isoformat(),
                "prompt": f"Reference {index}",
                "metadata": {"batch": "match" if index % 17 == 0 else "other"},
            }
        )
    store.close()
    reopened = RunStore(path)
    yield reopened
    reopened.close()


def test_native_history_search_covers_old_records_and_metadata(corpus):
    history = RunHistory(corpus)
    assert [r["run_id"] for r in history.list_runs("Reference 0")] == ["run-000"]
    expected = [f"run-{i:03}" for i in reversed(range(260)) if i % 17 == 0]
    matches = history.list_runs(metadata={"batch": "match"}, limit=0)
    assert [r["run_id"] for r in matches] == expected
    assert history.list_runs("no such source") == []


def test_native_history_pages_have_no_missing_or_duplicate_records(corpus):
    history = RunHistory(corpus)
    pages = [history.list_runs(limit=37, offset=i) for i in range(0, 260, 37)]
    ids = [r["run_id"] for page in pages for r in page]
    assert ids == [f"run-{i:03}" for i in reversed(range(260))]
    assert len(set(ids)) == 260
    assert len(history.list_runs(limit=0)) == 260
    assert history.list_runs(limit=37, offset=260) == []
    # Public storage listing remains bounded; complete history uses its own seam.
    assert len(corpus.list_runs(limit=100000)) == 200


def test_app_history_search_returns_oldest_record_after_restart(corpus):
    client = TestClient(create_app(store=corpus, settings=Settings()))
    response = client.get("/api/runs", params={"search": "Reference 0"})
    assert response.status_code == 200
    assert [r["run_id"] for r in response.json()] == ["run-000"]


def test_complete_snapshot_does_not_reread_each_native_record(corpus, monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Complete snapshot must not load each run again")

    monkeypatch.setattr(corpus, "get_run", unexpected_read)
    assert len(RunHistory(corpus).list_runs(limit=0)) == 260


def test_legacy_three_method_store_can_still_search_full_details():
    class LegacyStore:
        def save_run(self, record):
            return record["run_id"]

        def list_runs(self):
            return [{"run_id": "legacy"}]

        def get_run(self, run_id):
            assert run_id == "legacy"
            return {"run_id": run_id, "prompt": "Legacy searchable text"}

    assert [r["run_id"] for r in RunHistory(LegacyStore()).list_runs("searchable")] == [
        "legacy"
    ]


def test_snapshot_summary_matches_detail_and_search_preserves_nested_evidence(corpus):
    history = RunHistory(corpus)
    run = corpus.get_run("run-000")
    run["result"] = {
        "status": "failed",
        "final_prompt": run["prompt"],
        "report": {
            "status": "failed",
            "failure": {"kind": "provider", "message": "synthetic failure"},
            "outputs": [{"text": "Unique nested evidence phrase"}],
        },
    }
    corpus.save_run(run)
    before = corpus.get_run("run-000")
    detail = history.get_run("run-000")
    assert history.list_runs("Unique nested evidence phrase") == [
        history._summary(detail)
    ]
    assert corpus.get_run("run-000") == before
