"""Runtime configuration.

Credentials are read only by the engine/server process. The public settings
view deliberately excludes them so the browser cannot receive provider keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .catalog import DEFAULT_GO_WRITER, DEFAULT_WEAK_PANEL, JEV_MODEL

MAX_CANDIDATES = 6
DEFAULT_FIXED_WEAK_PANEL = DEFAULT_WEAK_PANEL + (
    "mimo-v2.6-flash",
    "google/gemma-3-4b-it",
)


@dataclass(slots=True)
class Settings:
    database_path: str = "prompt_enhancer.sqlite3"
    openrouter_api_key: str | None = field(default=None, repr=False)
    opencode_go_key: str | None = field(default=None, repr=False)
    judge_model: str = JEV_MODEL
    writer_model: str = DEFAULT_GO_WRITER
    strong_check_model: str = "glm-5.3-flash"
    weak_models: tuple[str, ...] = DEFAULT_FIXED_WEAK_PANEL
    candidate_count: int = MAX_CANDIDATES
    weak_model_count: int = 5
    weak_samples: int = 3
    # Offered by the web app when OpenCode Go refuses requests, so a user
    # without an active subscription can switch in one click.
    fallback_writer_model: str = "~deepseek/deepseek-flash-latest"
    fallback_strong_check_model: str = "deepseek/deepseek-v4.1-flash"
    grading_cascade_pair_cap: int | None = 30
    grading_cascade_dollar_cap: float | None = 0.05
    grading_confirmation_reservation_usd: float = 0.001
    attribution_pair_cap: int | None = 30
    attribution_dollar_cap: float | None = 0.03
    # Quality score-vector floors (#168): per-dimension minima on the 0-1
    # scale. Any dimension below its floor rejects the candidate outright
    # (max-gate: a breach is never averaged away by strong siblings). These
    # ship conservative — high enough to reject degenerate rewrites — and
    # recalibrate from keep/reject feedback (#170), which is why they live on
    # Settings instead of as module constants.
    #
    # Rationale: fidelity mirrors the fidelity gate threshold (0.80), so any
    # fidelity failure breaches; safety matches it because an unsafe rewrite
    # is as costly as an unfaithful one; clarity/specificity/coherence sit at
    # 0.60 so genuine rewrites pass while muddled ones fail; style fit is
    # lowest (0.50) because the style applies only where compatible with the
    # user's explicit constraints.
    score_floor_fidelity: float = 0.8
    score_floor_style_fit: float = 0.5
    score_floor_clarity: float = 0.6
    score_floor_specificity: float = 0.6
    score_floor_coherence: float = 0.6
    score_floor_safety: float = 0.8
    # Convergence epsilon (#169): the smallest mean-vector gain a further
    # Round must promise to be worth running. When every dimension meets its
    # floor and the best vector's marginal gain over the previous Round's best
    # has fallen to or below this value, the run stops as converged. It is a
    # documented policy knob (the spec's "small epsilon") so the practical
    # value can be recalibrated without a code change; a larger epsilon stops
    # sooner, a smaller one hunts longer for hard-won gains.
    convergence_epsilon: float = 0.01

    @property
    def score_floors(self) -> dict[str, float]:
        """Per-dimension floors keyed by score-vector dimension name."""
        return {
            "fidelity": self.score_floor_fidelity,
            "style_fit": self.score_floor_style_fit,
            "clarity": self.score_floor_clarity,
            "specificity": self.score_floor_specificity,
            "coherence": self.score_floor_coherence,
            "safety": self.score_floor_safety,
        }

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
            "score_floors": self.score_floors,
        }
