"""Per-role model usage and cost accounting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_response(cls, response: Any) -> Usage:
        payload = response
        if isinstance(response, Mapping) and "usage" in response:
            payload = response["usage"]
        if not isinstance(payload, Mapping):
            return cls()

        def integer(*names: str) -> int:
            for name in names:
                value = payload.get(name)
                try:
                    if value is not None:
                        return max(0, int(value))
                except (TypeError, ValueError):
                    continue
            return 0

        return cls(
            input_tokens=integer("input_tokens", "prompt_tokens", "input"),
            output_tokens=integer("output_tokens", "completion_tokens", "output"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class RoleUsage:
    role: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    calls: int = 0
    cap: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self,
        usage: Usage,
        *,
        input_cost_per_token: float | None = None,
        output_cost_per_token: float | None = None,
        cost: float | None = None,
        cap: float | None = None,
    ) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.calls += 1
        if cap is not None:
            self.cap = cap
        if cost is None:
            cost = 0.0
            if input_cost_per_token is not None:
                cost += usage.input_tokens * input_cost_per_token
            if output_cost_per_token is not None:
                cost += usage.output_tokens * output_cost_per_token
        self.cost += cost

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "calls": self.calls,
        }
        if self.cap is not None:
            result["cap"] = self.cap
            result["cap_used"] = self.cost
            result["cap_remaining"] = max(0.0, self.cap - self.cost)
        return result


class UsageLedger:
    """Thread-safe-enough aggregate for a single local optimization run.

    Accounting is keyed by role/provider/model, preventing a strong-check call
    from being attributed to the writer role.  Failed calls without usage do
    not add phantom cost; successful retries are each visible in ``calls``.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], RoleUsage] = {}
        self._by_role: defaultdict[str, float] = defaultdict(float)

    def record(
        self,
        *,
        role: str,
        provider: str,
        model: str,
        response: Any = None,
        usage: Usage | None = None,
        input_cost_per_token: float | None = None,
        output_cost_per_token: float | None = None,
        cost: float | None = None,
        cap: float | None = None,
    ) -> RoleUsage:
        actual_usage = usage or Usage.from_response(response)
        key = (role, provider, model)
        entry = self._entries.get(key)
        if entry is None:
            entry = RoleUsage(role=role, provider=provider, model=model)
            self._entries[key] = entry
        before = entry.cost
        entry.add(
            actual_usage,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
            cost=cost,
            cap=cap,
        )
        self._by_role[role] += entry.cost - before
        return entry

    def for_role(self, role: str) -> tuple[RoleUsage, ...]:
        return tuple(entry for key, entry in self._entries.items() if key[0] == role)

    def role_cost(self, role: str) -> float:
        return self._by_role.get(role, 0.0)

    @property
    def total_cost(self) -> float:
        return sum(self._by_role.values())

    @property
    def calls(self) -> int:
        return sum(entry.calls for entry in self._entries.values())

    def to_dict(self) -> dict[str, Any]:
        by_role: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        costs: dict[str, float] = {}
        tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for entry in self._entries.values():
            by_role[entry.role].append(entry.to_dict())
            costs[entry.role] = costs.get(entry.role, 0.0) + entry.cost
            tokens["input_tokens"] += entry.input_tokens
            tokens["output_tokens"] += entry.output_tokens
        tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]
        return {
            "total": self.total_cost,
            "total_cost": self.total_cost,
            "cost_by_role": costs,
            "tokens": tokens,
            "calls": self.calls,
            "by_role": dict(by_role),
        }

    summary = to_dict

    def merge(self, other: UsageLedger) -> None:
        for entry in other._entries.values():
            self.record(
                role=entry.role,
                provider=entry.provider,
                model=entry.model,
                usage=Usage(entry.input_tokens, entry.output_tokens),
                cost=entry.cost,
                cap=entry.cap,
            )
            # ``record`` increments calls once; preserve the other call count.
            target = self._entries[(entry.role, entry.provider, entry.model)]
            target.calls += entry.calls - 1


__all__ = ["RoleUsage", "Usage", "UsageLedger"]
