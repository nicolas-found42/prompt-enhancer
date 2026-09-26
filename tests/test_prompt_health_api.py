"""Public draft-health API behavior with a scripted Gateway and local SQLite."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prompt_enhancer.api import create_app
from prompt_enhancer.catalog import JEV_MODEL, ModelInfo, StaticModelCatalog
from prompt_enhancer.config import Settings
from prompt_enhancer.evaluation.calibration import DecisionPolicy, PolicyDecision
from prompt_enhancer.gateway import (
    GatewayConfig,
    HttpGateway,
    ProviderError,
    ScriptedGateway,
)
from prompt_enhancer.prompt_health import (
    PromptHealthPolicy,
    PromptHealthService,
    PromptHealthStore,
)


def _answer(request: dict, **_kwargs):
    key = request["key"]
    if request["type"] == "score":
        level = 3 if ":task:" in key else 2
        return {
            "type": "score",
            "score": level,
            "levels": {str(i): float(i == level) for i in range(4)},
            "confidence": 1.0,
        }
    if key.endswith(":applicable"):
        probability = 0.99 if ":task:" in key else 0.01
    elif key.endswith(":vagueness"):
        probability = 0.95 if request["state"]["sentence"] == "Do this." else 0.01
    elif key.endswith(":unresolved_reference"):
        state = request["state"]
        probability = (
            0.95
            if key.startswith("sentence:s0002:") and "Meet a person" in state["prompt"]
            else 0.01
        )
    elif key.endswith(":contradiction"):
        probability = (
            0.95 if "Do not use bullets" in request["state"]["prompt"] else 0.01
        )
    else:
        probability = 0.01
    return {"type": "noul", "probability_true": probability, "confidence": 1.0}


def _client(tmp_path: Path, gateway: ScriptedGateway | None = None) -> TestClient:
    return TestClient(
        create_app(
            settings=Settings(database_path=str(tmp_path / "runs.sqlite3")),
            health_gateway=gateway or ScriptedGateway(decision=_answer),
        )
    )


def test_health_api_scores_applicable_dimensions_and_separate_flags(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    prompt = "Write a clear notice. Do not use bullets."
    response = client.post(
        "/api/prompt-health",
        json={"prompt": prompt, "revision": 4, "session_id": "tab-a"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["draft"]["revision"] == 4
    assert body["status"] == "complete"
    assert body["composite"] == 1.0
    assert body["coverage"] == {
        "assessed": 1,
        "applicable": 1,
        "unknown": 0,
        "questions_answered": 16,
        "questions_total": 16,
    }
    assert [item["applicable"] for item in body["dimensions"]] == [
        True,
        False,
        False,
        False,
    ]
    assert any(flag["kind"] == "contradiction" for flag in body["flags"])
    assert body["usage"]["provider_requests"] == 2
    assert "secret" not in response.text


def test_exact_cache_reuses_context_free_sentence_but_invalidates_relations(
    tmp_path: Path,
) -> None:
    gateway = ScriptedGateway(decision=_answer)
    client = _client(tmp_path, gateway)
    first = "Write a notice. Refer to that rule."
    second = "Write a short notice. Refer to that rule."

    def assess(prompt: str, revision: int) -> dict:
        return client.post(
            "/api/prompt-health",
            json={"prompt": prompt, "revision": revision, "session_id": "tab-a"},
        ).json()

    initial = assess(first, 1)
    repeated = assess(first, 2)
    edited = assess(second, 3)

    assert initial["status"] == repeated["status"] == edited["status"] == "complete"
    assert repeated["cache"]["misses"] == 0
    assert repeated["usage"]["provider_requests"] == 0
    assert edited["cache"]["hits"] >= 1
    assert edited["cache"]["misses"] >= 8
    assert edited["usage"]["provider_requests"] >= 1
    assert edited["draft"]["hash"] != initial["draft"]["hash"]

    reopened = _client(tmp_path)
    from_disk = reopened.post(
        "/api/prompt-health",
        json={"prompt": second, "revision": 4, "session_id": "tab-b"},
    ).json()
    assert from_disk["cache"]["misses"] == 0
    assert from_disk["usage"]["provider_requests"] == 0


def test_duplicate_sentences_remap_cached_text_checks_to_current_ids(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    prompt = "Do this. Do this."
    first = client.post(
        "/api/prompt-health",
        json={"prompt": prompt, "revision": 1, "session_id": "one"},
    ).json()
    assert first["cache"]["misses"] < first["coverage"]["questions_total"]
    assert first["status"] == "complete"
    assert [
        item["sentence_id"] for item in first["flags"] if item["kind"] == "vagueness"
    ] == ["s0001", "s0002"]

    reordered = client.post(
        "/api/prompt-health",
        json={
            "prompt": "Do this. First explain why. Do this.",
            "revision": 2,
            "session_id": "one",
        },
    ).json()
    assert reordered["status"] == "complete"
    assert reordered["cache"]["hits"] >= 2
    assert reordered["cache"]["misses"] > 0
    assert [
        item["sentence_id"]
        for item in reordered["flags"]
        if item["kind"] == "vagueness"
    ] == ["s0001", "s0003"]


def test_flag_offsets_use_browser_utf16_positions(tmp_path: Path) -> None:
    client = _client(tmp_path)
    prompt = "😀. Do this."

    result = client.post(
        "/api/prompt-health",
        json={"prompt": prompt, "revision": 1, "session_id": "emoji"},
    ).json()

    flag = next(item for item in result["flags"] if item["text"] == "Do this.")
    assert flag["start"] == len("😀. ".encode("utf-16-le")) // 2
    assert flag["end"] == len(prompt.encode("utf-16-le")) // 2


def test_changed_antecedent_rechecks_unchanged_sentence(tmp_path: Path) -> None:
    client = _client(tmp_path)

    def assess(prompt: str, revision: int) -> dict:
        return client.post(
            "/api/prompt-health",
            json={"prompt": prompt, "revision": revision, "session_id": "same"},
        ).json()

    resolved = assess("Meet Ada. Tell her the date.", 1)
    unresolved = assess("Meet a person. Tell her the date.", 2)

    assert not any(flag["kind"] == "unresolved_reference" for flag in resolved["flags"])
    assert any(
        flag["kind"] == "unresolved_reference" and flag["sentence_id"] == "s0002"
        for flag in unresolved["flags"]
    )
    assert unresolved["cache"]["hits"] >= 1


def test_size_pricing_provider_and_shared_rate_limits_quietly_hold(
    tmp_path: Path,
) -> None:
    clock = [1000.0]
    store = PromptHealthStore(tmp_path / "limits.sqlite3", clock=lambda: clock[0])
    gateway = ScriptedGateway(decision=_answer)
    policy = PromptHealthPolicy(max_draft_characters=30, max_refreshes_per_minute=1)
    service = PromptHealthService(gateway, store, policy=policy)

    assert service.assess("", 0, "a")["status"] == "empty"
    assert service.assess("x" * 31, 1, "a")["status"] == "unavailable"
    assert service.assess("Write a note.", 2, "a")["status"] == "complete"
    assert service.assess("Write a letter.", 3, "b")["status"] == "paused"
    assert store.usage()["refreshes_last_minute"] == 1
    clock[0] += 61
    assert service.assess("Write a letter.", 4, "b")["status"] == "complete"

    def failure(_request, **_kwargs):
        raise ProviderError("scripted", "jev", None)

    failed = PromptHealthService(ScriptedGateway(decision=failure), store)
    assert failed.assess("A new task.", 5, "c")["status"] == "unavailable"


def test_health_gateway_does_not_reset_optimizer_ledger(tmp_path: Path) -> None:
    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.store import RunStore

    optimizer_gateway = ScriptedGateway(decision=_answer)
    optimizer_gateway.decision_log.append({"existing": "optimization"})
    optimizer = PromptOptimizer(
        gateway=optimizer_gateway,
        store=RunStore(str(tmp_path / "runs.sqlite3")),
    )
    client = TestClient(create_app(optimizer=optimizer))

    response = client.post(
        "/api/prompt-health",
        json={"prompt": "Write a note.", "revision": 1, "session_id": "tab-a"},
    )

    assert response.json()["status"] == "complete"
    assert optimizer_gateway.decision_log == [{"existing": "optimization"}]


def test_busy_health_check_returns_paused_without_waiting(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    started = Event()
    release = Event()

    def slow_answer(request: dict, **_kwargs):
        if not started.is_set():
            started.set()
            assert release.wait(10)
        return _answer(request)

    service = PromptHealthService(
        ScriptedGateway(decision=slow_answer),
        PromptHealthStore(tmp_path / "busy.sqlite3"),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.assess, "Do this.", 1, "one")
        assert started.wait(10)
        try:
            second = pool.submit(service.assess, "Write a note.", 2, "two")
            assert second.result(timeout=2)["status"] == "paused"
        finally:
            release.set()
        assert first.result(timeout=10)["status"] == "complete"


def test_health_request_beside_active_optimization_keeps_separate_costs(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from prompt_enhancer.optimizer import PromptOptimizer
    from prompt_enhancer.store import RunStore

    started = Event()
    release = Event()

    def optimize_answer(request, **_kwargs):
        if not started.is_set():
            optimizer_gateway.usage.record(
                role="judge",
                provider="scripted",
                model=JEV_MODEL,
                cost=0.02,
            )
            started.set()
            assert release.wait(10)
        if request["type"] == "choice":
            choice = "general" if request.get("key") == "task_type" else "none"
            return {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 1.0,
            }
        return {"type": "noul", "probability_true": 0.01, "confidence": 1.0}

    optimizer_gateway = ScriptedGateway(
        chat=lambda *_args, **_kwargs: '{"tests":[]}', decision=optimize_answer
    )
    optimizer = PromptOptimizer(
        gateway=optimizer_gateway,
        store=RunStore(str(tmp_path / "active.sqlite3")),
    )
    client = TestClient(
        create_app(
            optimizer=optimizer, health_gateway=ScriptedGateway(decision=_answer)
        )
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            optimizer.optimize,
            "Write a note.",
            {"clarification_allowed": False},
        )
        assert started.wait(10)
        try:
            health = client.post(
                "/api/prompt-health",
                json={"prompt": "Write a note.", "revision": 1, "session_id": "tab"},
            ).json()
            assert health["status"] == "complete"
            assert health["usage"]["request_usd"] == 0
            assert optimizer_gateway.usage_report()["total"] == 0.02
        finally:
            release.set()
        optimized = future.result(timeout=10)
    assert optimized["cost"]["total"] >= 0.02


def test_missing_or_expensive_pricing_skips_before_transport(tmp_path: Path) -> None:
    class NoTransport:
        def request(self, *_args, **_kwargs):
            raise AssertionError("health budget must stop before transport")

    gateway = HttpGateway(
        transport=NoTransport(),
        config=GatewayConfig(openrouter_api_key="local-test-key", max_retries=0),
        catalog=StaticModelCatalog(
            (),
            (
                ModelInfo(
                    JEV_MODEL,
                    "openrouter",
                    input_cost_per_token=1.0,
                    output_cost_per_token=1.0,
                ),
            ),
        ),
    )
    service = PromptHealthService(gateway, PromptHealthStore(tmp_path / "cost.sqlite3"))
    report = service.assess("Write a note.", 1, "tab-a")
    assert report["status"] == "paused"
    assert "allowance" in report["reason"]
    assert service.store.usage()["rolling_hour_usd"] == 0

    gateway.catalog = StaticModelCatalog((), (ModelInfo(JEV_MODEL, "openrouter"),))
    missing = service.assess("Write a note.", 2, "tab-a")
    assert missing["status"] == "unavailable"
    assert "pricing" in missing["reason"]


def test_live_http_refresh_batches_requests_and_reconciles_usage(
    tmp_path: Path,
) -> None:
    class DecisionTransport:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def request(self, _url, *, json, **_kwargs):
            self.calls.append(json)
            answers = {}
            for key, question in json["questions"].items():
                if question["type"] == "score":
                    answers[key] = {
                        "type": "score",
                        "score": 3,
                        "levels": {str(i): float(i == 3) for i in range(4)},
                    }
                else:
                    answers[key] = {
                        "type": "noul",
                        "probability_true": 0.99
                        if key == "dimension:task:applicable"
                        else 0.01,
                    }
            return {
                "model": JEV_MODEL,
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

    transport = DecisionTransport()
    gateway = HttpGateway(
        transport=transport,
        config=GatewayConfig(openrouter_api_key="local-test-key", max_retries=0),
        catalog=StaticModelCatalog(
            (),
            (
                ModelInfo(
                    JEV_MODEL,
                    "openrouter",
                    input_cost_per_token=0.0000001,
                    output_cost_per_token=0.0000002,
                ),
            ),
        ),
    )
    store = PromptHealthStore(tmp_path / "live.sqlite3")
    service = PromptHealthService(gateway, store)

    first = service.assess("Write a note.", 1, "one")
    second = service.assess("Write a note.", 2, "two")

    assert first["status"] == second["status"] == "complete"
    assert len(transport.calls) == 2
    assert first["usage"]["provider_requests"] == 2
    assert second["usage"]["provider_requests"] == 0
    assert 0 < store.usage()["rolling_hour_usd"] < 0.05
    assert first["usage"]["rolling_hour_usd"] == store.usage()["rolling_hour_usd"]
    assert first["composite"] == 1.0


def test_failed_later_batch_keeps_unmeasured_reservation(tmp_path: Path) -> None:
    class FailingTransport:
        def __init__(self) -> None:
            self.calls = 0

        def request(self, _url, *, json, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                return {"status_code": 503, "json": {}, "headers": {}}
            answers = {}
            for key, question in json["questions"].items():
                if question["type"] == "score":
                    answers[key] = {
                        "type": "score",
                        "score": 3,
                        "levels": {str(i): float(i == 3) for i in range(4)},
                    }
                else:
                    answers[key] = {"type": "noul", "probability_true": 0.01}
            return {
                "model": JEV_MODEL,
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

    gateway = HttpGateway(
        transport=FailingTransport(),
        config=GatewayConfig(openrouter_api_key="local-test-key", max_retries=0),
        catalog=StaticModelCatalog(
            (),
            (
                ModelInfo(
                    JEV_MODEL,
                    "openrouter",
                    input_cost_per_token=1e-7,
                    output_cost_per_token=2e-7,
                ),
            ),
        ),
    )
    store = PromptHealthStore(tmp_path / "failed-batch.sqlite3")
    result = PromptHealthService(gateway, store).assess("Write a note.", 1, "tab")

    assert result["status"] == "unavailable"
    assert result["usage"]["provider_requests"] == 2
    reserved, measured, charged = store._db.execute(
        "SELECT reserved_usd, measured_usd, charged_usd FROM prompt_health_spend"
    ).fetchone()
    assert 0 < measured < reserved <= charged


def test_unverified_snapshot_does_not_cache_an_answer(tmp_path: Path) -> None:
    class MovingGateway(ScriptedGateway):
        def decide(self, payload, *, role="judge", run_id=None):
            answer = super().decide(payload, role=role, run_id=run_id)
            self.decision_log[-1]["answered_by"] = "typesafe/jev-moving-alias"
            return answer

    gateway = MovingGateway(decision=_answer)
    store = PromptHealthStore(tmp_path / "snapshot.sqlite3")
    service = PromptHealthService(gateway, store)
    first = service.assess("Write a note.", 1, "tab-a")
    second = service.assess("Write a note.", 2, "tab-a")
    assert first["status"] == second["status"] == "unavailable"
    assert first["cache"]["hits"] == second["cache"]["hits"] == 0


def test_display_policy_change_reuses_answers_but_question_version_does_not(
    tmp_path: Path, monkeypatch
) -> None:
    import prompt_enhancer.prompt_health as health

    service = PromptHealthService(
        ScriptedGateway(decision=_answer),
        PromptHealthStore(tmp_path / "versions.sqlite3"),
    )
    first = service.assess("Write a note.", 1, "tab")
    monkeypatch.setattr(health, "DISPLAY_VERSION", "prompt-health-display-2")
    display_only = service.assess("Write a note.", 2, "tab")
    changed_levels = {
        **health.SCORE_LEVELS,
        "task": (*health.SCORE_LEVELS["task"][:3], "3: clearer task definition"),
    }
    monkeypatch.setattr(health, "SCORE_LEVELS", changed_levels)
    new_criteria = service.assess("Write a note.", 3, "tab")
    monkeypatch.setattr(health, "QUESTION_VERSION", "prompt-health-2")
    new_question = service.assess("Write a note.", 4, "tab")

    assert first["cache"]["misses"] > 0
    assert display_only["cache"]["misses"] == 0
    assert display_only["display_version"] == "prompt-health-display-2"
    assert new_criteria["cache"]["misses"] == 1
    assert new_question["cache"]["misses"] > 0


def test_hourly_allowance_survives_store_reopen_and_new_tab(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite3"
    policy = PromptHealthPolicy(hourly_allowance_usd=0.05)
    first = PromptHealthStore(path)
    reservation, reason = first.reserve("tab-a", 0.03, policy)
    assert reservation is not None and reason is None
    first.settle(
        reservation,
        measured=0.03,
        dispatched=1,
        measured_complete=True,
        status="complete",
    )

    reopened = PromptHealthStore(path)
    second, reason = reopened.reserve("tab-b", 0.03, policy)
    assert second is None
    assert reason == "rolling-hour health allowance reached"
    assert reopened.usage()["rolling_hour_usd"] == 0.03


def test_rank_only_calibration_cannot_become_an_applicability_or_flag_gate(
    tmp_path: Path,
) -> None:
    class RankOnly(DecisionPolicy):
        def has_candidate(self, _question_id: str) -> bool:
            return True

        def apply(self, **_kwargs) -> PolicyDecision:
            return PolicyDecision(disposition="ranker", verdict="ranker")

    service = PromptHealthService(
        ScriptedGateway(decision=_answer),
        PromptHealthStore(tmp_path / "ranker.sqlite3"),
        decision_policy=RankOnly(),
    )
    result = service.assess("Do this.", 1, "tab")

    assert result["status"] == "partial"
    assert result["composite"] is None
    assert result["coverage"]["unknown"] == 4
    assert result["flags"] == []
