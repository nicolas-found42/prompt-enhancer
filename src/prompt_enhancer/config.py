"""Runtime configuration.

Credentials are read only by the engine/server process. The public settings
view deliberately excludes them so the browser cannot receive provider keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Settings:
    database_path: str = "prompt_enhancer.sqlite3"
    openrouter_api_key: str | None = field(default=None, repr=False)
    opencode_go_key: str | None = field(default=None, repr=False)
    judge_model: str = "typesafe/jev-1.13"
    writer_model: str = "deepseek-v4.1-flash"
    strong_check_model: str = "glm-5.3-flash"
    weak_models: tuple[str, ...] = (
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-nemo",
        "meta-llama/llama-3.2-3b-instruct",
    )

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_path=os.getenv("PROMPT_ENHANCER_DB", "prompt_enhancer.sqlite3"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            opencode_go_key=os.getenv("OPENCODE_GO_KEY") or None,
            judge_model=os.getenv("PROMPT_ENHANCER_JUDGE_MODEL", "typesafe/jev-1.13"),
            writer_model=os.getenv(
                "PROMPT_ENHANCER_WRITER_MODEL", "deepseek-v4.1-flash"
            ),
            strong_check_model=os.getenv(
                "PROMPT_ENHANCER_STRONG_MODEL", "glm-5.3-flash"
            ),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "judge_model": self.judge_model,
            "writer_model": self.writer_model,
            "strong_check_model": self.strong_check_model,
            "weak_models": list(self.weak_models),
        }
