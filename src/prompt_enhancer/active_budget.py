"""One active allowance shared by all operations of a logical run."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

ACTIVE_BUDGET_S = 150.0


class ActiveDeadlineExceeded(TimeoutError):
    """The logical run has no processing time left."""


@dataclass(frozen=True)
class ActiveBudget:
    clock: Callable[[], float]
    started: float
    elapsed_base_ms: int = 0
    parent: "ActiveBudget | None" = None

    def elapsed_ms(self) -> int:
        local = self.elapsed_base_ms + max(
            0, round((self.clock() - self.started) * 1000)
        )
        return max(local, self.parent.elapsed_ms() if self.parent is not None else 0)

    def remaining_s(self) -> float:
        return max(0.0, ACTIVE_BUDGET_S - self.elapsed_ms() / 1000)

    def check(self) -> None:
        if self.remaining_s() <= 0:
            raise ActiveDeadlineExceeded(
                "The 150-second active processing budget ended."
            )


current_budget: ContextVar[ActiveBudget | None] = ContextVar(
    "active_budget", default=None
)


@contextmanager
def budget_scope(budget: ActiveBudget) -> Iterator[None]:
    token = current_budget.set(budget)
    try:
        budget.check()
        yield
    finally:
        current_budget.reset(token)


def check_active_budget() -> None:
    budget = current_budget.get()
    if budget is not None:
        budget.check()
