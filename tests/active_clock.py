"""Deterministic active-time clock for run-budget tests (issue #187).

The clock advances only when the scripted writer is called, so the 150-second
active budget can be crossed without real waiting.
"""

from collections.abc import Callable
from typing import Any


class TickingClock:
    """Callable clock advanced by scripted writer calls and, optionally, reads."""

    def __init__(
        self, step_per_read_s: float = 0.0, step_per_round_s: float = 30.0
    ) -> None:
        self.now = 0.0
        self.step_per_read_s = step_per_read_s
        self.step_per_round_s = step_per_round_s

    def __call__(self) -> float:
        self.now += self.step_per_read_s
        return self.now


def advancing_chat(
    chat: Callable[..., Any], clock: TickingClock, step_s: float
) -> Callable[..., Any]:
    """Wrap a scripted chat handler so each writer call advances the clock."""

    def advancing(model: Any, messages: Any, **kwargs: Any) -> Any:
        if kwargs.get("role") == "writer":
            clock.now += step_s
        return chat(model, messages, **kwargs)

    return advancing


def echo_candidates_for_diagnosis(chat: Callable[..., Any]) -> Callable[..., Any]:
    """Give diagnosis fixtures a valid rejected draft in the always-attempt lane.

    Other replies pass through unchanged. These fixtures have no adequate rewrite:
    echoes fail eligibility but allow public diagnosis/approval assertions to run.
    """
    import json

    def reply(model: Any, messages: Any, **kwargs: Any) -> Any:
        if kwargs.get("role") == "writer" and len(messages) > 1:
            try:
                state = json.loads(messages[1]["content"])
            except (ValueError, KeyError, TypeError):
                state = {}
            if isinstance(state, dict) and isinstance(state.get("strategies"), list):
                return json.dumps(
                    {item["name"]: state["prompt"] for item in state["strategies"]}
                )
        return chat(model, messages, **kwargs)

    return reply
