"""Worker ownership check for durable publication after a terminal deadline."""

from collections.abc import Callable
from contextvars import ContextVar

publication_allowed: ContextVar[Callable[[], bool] | None] = ContextVar(
    "publication_allowed", default=None
)

# Jobs serialize completed-Round publication with the watchdog. The callable
# runs the supplied transition only while the worker still owns publication.
checkpoint_transaction: ContextVar[Callable[[Callable[[], None]], None] | None] = (
    ContextVar("checkpoint_transaction", default=None)
)
