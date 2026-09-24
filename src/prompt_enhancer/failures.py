"""Plain-language descriptions of why a run stopped.

Every failed run carries one of these in ``report.failure`` so the web app and
API clients can say what went wrong and what to do next, without parsing the
sanitized provider message.
"""

from __future__ import annotations

from typing import Any

from .gateway import ACCESS_DENIED_STATUS, ProviderError


class RunCancelled(RuntimeError):
    """Raised at the next stage boundary after a caller cancels a run."""


_PROVIDER_NAMES = {"go": "OpenCode Go", "openrouter": "OpenRouter"}


def describe_failure(exc: BaseException) -> dict[str, Any]:
    """Return a JSON-safe failure description with a user-facing hint."""
    if isinstance(exc, RunCancelled):
        return {
            "kind": "cancelled",
            "headline": "Run cancelled",
            "hint": "You cancelled this run. Your prompt was not changed.",
            "message": "cancelled by the user",
        }
    if isinstance(exc, ProviderError):
        details = exc.to_dict()
        name = _PROVIDER_NAMES.get(exc.provider, exc.provider)
        role = (exc.role or "model").replace("_", " ")
        if exc.kind == "invalid_response":
            headline = f"The {role} model gave an unusable reply"
            hint = f"{exc.model} answered, but not in the format the optimizer needs. Try again, or choose a different {role} model in Model choices."
        elif exc.status in ACCESS_DENIED_STATUS and exc.provider == "go":
            headline = "OpenCode Go refused the request"
            hint = f"Go models such as {exc.model} need an active OpenCode Go subscription. Choose OpenRouter models for the writer and strong check in Model choices, or renew the subscription."
        elif exc.status in ACCESS_DENIED_STATUS:
            headline = f"{name} refused the request"
            hint = f"Check that the {name} API key is valid and the account has credit, or choose a different model for the {role}."
        elif exc.status == 429:
            headline = f"{name} is rate limiting requests"
            hint = "Wait a minute and try again, or choose a different model."
        elif exc.status is not None and exc.status >= 500:
            headline = f"{name} had a server error"
            hint = "Try again shortly, or choose a different model."
        elif exc.status is not None:
            headline = f"{name} rejected the request"
            hint = f"{exc.model} returned HTTP {exc.status}. Choose a different model for the {role} and try again."
        else:
            headline = f"{name} could not be reached"
            hint = "Check your internet connection and try again."
        return {**details, "headline": headline, "hint": hint}
    return {
        "kind": "internal",
        "headline": "The optimizer hit an unexpected error",
        "hint": "Try again. If it keeps happening, the details below will help when reporting it.",
        "message": f"{type(exc).__name__}: {exc}",
    }
