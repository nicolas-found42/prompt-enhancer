"""Capture live gateway answers for strict, request-keyed evaluation replay."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..diagnosis import DEFAULT_RUBRIC, checklist_impacts, checklist_keys
from ..gateway import Gateway, ReplayGateway


class RecordingGateway:
    def __init__(self, gateway: Gateway, path: Path) -> None:
        self.gateway = gateway
        self.path = path
        self.responses: dict[str, Any] = {}
        self.decision_provenance: dict[str, dict[str, Any]] = {}
        self.case_latency_ms: dict[str, float] = {}
        self.case_costs: dict[str, dict[str, Any]] = {}
        self.rubric_thresholds: dict[str, float] | None = None
        self.writer_instruction_version: int | None = None
        self.faithfulness_threshold: float | None = None
        # Bundles written by this code carry the checklist their recordings saw.
        self.checklist_keys: list[str] | None = list(checklist_keys(DEFAULT_RUBRIC))
        self.checklist_impacts: dict[str, str] | None = checklist_impacts(
            DEFAULT_RUBRIC
        )
        # The weak-model panel calls the gateway from worker threads.
        self._lock = threading.RLock()

    def _record(
        self, operation: str, model: str, payload: Any, role: str, answer: Any
    ) -> Any:
        key = ReplayGateway.request_key(operation, model, payload, role)
        with self._lock:
            previous = self.responses.get(key)
            if previous is not None and previous != answer:
                raise ValueError(
                    "identical gateway request produced different responses; strict replay cannot represent it"
                )
            self.responses[key] = answer
            self.save()
        return answer

    def save(self) -> None:
        with self._lock:
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        bundle: dict[str, Any] = {
            "responses": self.responses,
            "decision_provenance": self.decision_provenance,
            "jev_model": self.gateway.jev_model,
            "case_latency_ms": self.case_latency_ms,
            "case_costs": self.case_costs,
            "rubric_thresholds": self.rubric_thresholds,
        }
        if self.writer_instruction_version is not None:
            bundle["writer_instruction_version"] = self.writer_instruction_version
        if self.faithfulness_threshold is not None:
            bundle["faithfulness_threshold"] = self.faithfulness_threshold
        if self.checklist_keys is not None:
            bundle["checklist_keys"] = self.checklist_keys
        if self.checklist_impacts is not None:
            bundle["checklist_impacts"] = self.checklist_impacts
        temporary.write_text(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
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

    def chat(
        self,
        model: str,
        messages: Any,
        *,
        role: str = "writer",
        run_id: str | None = None,
        **params: Any,
    ) -> Any:
        normalized = (
            list(messages)
            if not isinstance(messages, str)
            else [{"role": "user", "content": messages}]
        )
        answer = self.gateway.chat(model, messages, role=role, run_id=run_id, **params)
        return self._record(
            "chat",
            model,
            {"model": model, "messages": normalized, **params},
            role,
            answer,
        )

    def decide(
        self,
        payload: Mapping[str, Any],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> Any:
        request = dict(payload)
        if "state" not in request and "prompt" in request:
            request["state"] = request.pop("prompt")
        answer = self.gateway.decide(payload, role=role, run_id=run_id)
        self._record_provenance(request, role)
        return self._record("decide", self.gateway.jev_model, request, role, answer)

    def decide_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        role: str = "judge",
        run_id: str | None = None,
    ) -> list[Any]:
        answers = self.gateway.decide_batch(requests, role=role, run_id=run_id)
        for request, answer in zip(requests, answers, strict=True):
            self._record_provenance(request, role)
            self._record("decide", self.gateway.jev_model, dict(request), role, answer)
        return answers

    def _record_provenance(self, request: Mapping[str, Any], role: str) -> None:
        entry = next(
            (
                item
                for item in reversed(self.gateway.decision_log)
                if item["question"] == dict(request)
            ),
            None,
        )
        if entry is None or not entry.get("answered_by"):
            raise ValueError("Jev decision has no answering snapshot")
        key = ReplayGateway.request_key(
            "decide", self.gateway.jev_model, dict(request), role
        )
        self.decision_provenance[key] = {
            "answered_by": entry["answered_by"],
            "usage": entry.get("usage", {}),
        }

    def list_models(self, *, refresh: bool = False) -> Any:
        # The model catalog and ledger are not part of the Gateway interface.
        return cast(Any, self.gateway).list_models(refresh=refresh)

    def usage_report(self) -> dict[str, Any]:
        return self.gateway.usage_report()

    @property
    def decision_log(self) -> list[dict[str, Any]]:
        return self.gateway.decision_log

    @property
    def jev_model(self) -> str:
        return self.gateway.jev_model

    @property
    def usage(self) -> Any:
        return cast(Any, self.gateway).usage


if TYPE_CHECKING:
    _ADAPTER: type[Gateway] = RecordingGateway
