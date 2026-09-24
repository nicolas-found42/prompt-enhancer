"""Runtime configuration.

Credentials are read only by the engine/server process. The public settings
view deliberately excludes them so the browser cannot receive provider keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .catalog import DEFAULT_GO_WRITER, JEV_MODEL


@dataclass(slots=True)
class Settings:
    database_path: str = "prompt_enhancer.sqlite3"
    openrouter_api_key: str | None = field(default=None, repr=False)
    opencode_go_key: str | None = field(default=None, repr=False)
    judge_model: str = JEV_MODEL
    writer_model: str = DEFAULT_GO_WRITER
    strong_check_model: str = "glm-5.3-flash"
    weak_models: tuple[str, ...] = (
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-nemo",
        "meta-llama/llama-3.2-3b-instruct",
    )
    # Offered by the web app when OpenCode Go refuses requests, so a user
    # without an active subscription can switch in one click.
    fallback_writer_model: str = "~deepseek/deepseek-flash-latest"
    fallback_strong_check_model: str = "deepseek/deepseek-v4.1-flash"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_path=os.getenv("PROMPT_ENHANCER_DB", "prompt_enhancer.sqlite3"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            opencode_go_key=os.getenv("OPENCODE_GO_KEY") or None,
            judge_model=os.getenv("PROMPT_ENHANCER_JEV_MODEL", JEV_MODEL),
            writer_model=os.getenv("PROMPT_ENHANCER_WRITER_MODEL", DEFAULT_GO_WRITER),
            strong_check_model=os.getenv(
                "PROMPT_ENHANCER_STRONG_MODEL", "glm-5.3-flash"
            ),
            fallback_writer_model=os.getenv(
                "PROMPT_ENHANCER_FALLBACK_WRITER_MODEL",
                "~deepseek/deepseek-flash-latest",
            ),
            fallback_strong_check_model=os.getenv(
                "PROMPT_ENHANCER_FALLBACK_STRONG_MODEL", "deepseek/deepseek-v4.1-flash"
            ),
        )

    def model_roles(self) -> dict[str, Any]:
        return {
            "judge": self.judge_model,
            "writer": self.writer_model,
            "strong": self.strong_check_model,
            "weak": list(self.weak_models),
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "judge_model": self.judge_model,
            "writer_model": self.writer_model,
            "strong_check_model": self.strong_check_model,
            "weak_models": list(self.weak_models),
        }
