"""Local HTTP API with a scripted gateway for browser integration tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import uvicorn

from prompt_enhancer.api import create_app
from prompt_enhancer.gateway import ScriptedGateway
from prompt_enhancer.optimizer import PromptOptimizer
from prompt_enhancer.store import RunStore


def chat(_model: str, messages: Any, *, role: str, **_kwargs: Any) -> str:
    if role != "writer":
        return "A useful answer."
    instruction = messages[0]["content"]
    if "Suggest two or three plausible values" in instruction:
        return '{"gaps":{"goal":{"question":"What should the assistant do?","options":[{"value":"summarize","label":"Summarize"},{"value":"analyze","label":"Analyze"}]}}}'
    if "Revise only the stated assumption" in instruction:
        return "Analyze this."
    return '{"tests":[]}'


def decide(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    key = request.get("key")
    if key == "task_type":
        return {
            "type": "choice",
            "choice": "general",
            "probabilities": {"general": 1.0},
            "confidence": 1.0,
        }
    if key == "infer:goal":
        return {
            "type": "choice",
            "choice": "unknown",
            "probabilities": {"summarize": 0.6, "analyze": 0.2, "unknown": 0.2},
            "confidence": 0.6,
        }
    if request.get("type") == "choice":
        return {
            "type": "choice",
            "choice": "none",
            "probabilities": {"none": 1.0},
            "confidence": 1.0,
        }
    probability = (
        0.99
        if key == "gap:goal"
        else 1.0
        if "preserve" in str(request.get("question", ""))
        else 0.01
    )
    return {"type": "noul", "probability_true": probability, "confidence": 1.0}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="prompt-enhancer-e2e-") as directory:
        store = RunStore(Path(directory) / "runs.sqlite3")
        optimizer = PromptOptimizer(
            store=store, gateway=ScriptedGateway(chat=chat, decision=decide)
        )
        port = int(os.environ.get("E2E_API_PORT", "8765"))
        uvicorn.run(
            create_app(optimizer=optimizer),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
