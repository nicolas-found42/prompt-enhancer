"""Shared test configuration.

Optimizer runs are bounded by a 150-second active budget (issue #187). Scripted
runs that never converge would otherwise spend that much real time, so every
test gets a deterministic active clock that advances after each Round.
"""

import pytest
from active_clock import TickingClock

import prompt_enhancer.optimizer as optimizer


@pytest.fixture(autouse=True)
def deterministic_active_clock(monkeypatch):
    clock = TickingClock()
    monkeypatch.setattr(optimizer, "_active_clock", clock)
    round_executor = optimizer.PromptOptimizer._round_executor

    def timed_round_executor(self, *args, **kwargs):
        execute = round_executor(self, *args, **kwargs)

        def timed_round(request):
            outcome = execute(request)
            if self._clock is clock:
                clock.now += clock.step_per_round_s
            return outcome

        return timed_round

    monkeypatch.setattr(
        optimizer.PromptOptimizer, "_round_executor", timed_round_executor
    )
    return clock
