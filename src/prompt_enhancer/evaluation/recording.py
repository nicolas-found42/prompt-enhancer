"""Capture live gateway answers for strict, request-keyed evaluation replay."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..catalog import JEV_MODEL
from ..gateway import ReplayGateway, _completion_chat_request


class RecordingGateway:
    def __init__(self, gateway: Any, path: Path) -> None:
        self.gateway = gateway
        self.path = path
        self.responses: dict[str, Any] = {}
        self.case_latency_ms: dict[str, float] = {}
        self.case_costs: dict[str, dict[str, Any]] = {}
        self.rubric_thresholds: dict[str, float] | None = None

    def _record(self, operation: str, model: str, payload: Any, role: str, answer: Any) -> Any:
        key = ReplayGateway.request_key(operation, model, payload, role)
        previous = self.responses.get(key)
        if previous is not None and previous != answer:
            raise ValueError("identical gateway request produced different responses; strict replay cannot represent it")
        self.responses[key] = answer
        self.save()
        return answer

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "responses": self.responses,
            "case_latency_ms": self.case_latency_ms,
            "case_costs": self.case_costs,
            "rubric_thresholds": self.rubric_thresholds,
        }, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def attach_case_metrics(self, report: Any) -> None:
        for case in report.cases:
            if case.latency_ms is not None:
                self.case_latency_ms[case.case_id] = case.latency_ms
            if case.cost is not None:
                self.case_costs[case.case_id] = {
                    "total": case.cost,
                    "cost_by_role": dict(case.cost_by_role),
                }
        self.save()

    def new_run(self, run_id: str | None = None) -> str:
        return self.gateway.new_run(run_id)

    set_run = new_run

    def chat(self, model: str, messages: Any, *, role: str = "writer", run_id: str | None = None, **params: Any) -> Any:
        normalized = list(messages) if not isinstance(messages, str) else [{"role": "user", "content": messages}]
        answer = self.gateway.chat(model, messages, role=role, run_id=run_id, **params)
        return self._record("chat", model, {"model": model, "messages": normalized, **params}, role, answer)

    def complete(self, model: str | Mapping[str, Any], messages: Any = None, **kwargs: Any) -> Any:
        if isinstance(model, Mapping) and messages is None:
            model_id, messages, role = _completion_chat_request(model)
            kwargs.setdefault("role", role)
            return self.chat(model_id, messages, **kwargs)
        return self.chat(str(model), messages, **kwargs)

    def decide(self, payload: Mapping[str, Any], *, role: str = "judge", run_id: str | None = None) -> Any:
        request = dict(payload)
        if "state" not in request and "prompt" in request:
            request["state"] = request.pop("prompt")
        answer = self.gateway.decide(payload, role=role, run_id=run_id)
        return self._record("decide", JEV_MODEL, request, role, answer)

    decision = decide
    jev = decide

    def jev_batch(self, requests: Sequence[Mapping[str, Any]], *, role: str = "judge", run_id: str | None = None) -> list[Any]:
        answers = self.gateway.jev_batch(requests, role=role, run_id=run_id)
        for request, answer in zip(requests, answers, strict=True):
            self._record("decide", JEV_MODEL, dict(request), role, answer)
        return answers

    def list_models(self, *, refresh: bool = False) -> Any:
        return self.gateway.list_models(refresh=refresh)

    def usage_report(self) -> Any:
        return self.gateway.usage_report()

    @property
    def decision_log(self) -> Any:
        return self.gateway.decision_log

    @property
    def usage(self) -> Any:
        return self.gateway.usage
