"""The provisional, opt-in #187 Llama evaluation contract."""

from dataclasses import replace

from .catalog import JEV_MODEL
from .config import Settings

PROFILE_ID = "llama-tuning-v1"
WEAK_MODEL = "meta-llama/llama-3.1-8b-instruct"
QWEN_MODEL = "qwen/qwen3.7-flash"
ALLOWED_PROVIDERS = ("novita", "groq")
# A pilot starting point, not a measured normal target or a frozen release profile.
PILOT_MAX_OUTPUT_TOKENS = 4096


def apply_tuning_profile(
    settings: Settings, name: str, provider: str | None = None
) -> Settings:
    if name != PROFILE_ID:
        raise ValueError("unknown evaluation profile")
    selected_provider = (
        provider if provider is not None else settings.evaluation_provider
    )
    if selected_provider is None:
        selected_provider = ALLOWED_PROVIDERS[0]
    provider_policy(selected_provider)
    return replace(
        settings,
        evaluation_profile=PROFILE_ID,
        evaluation_provider=selected_provider,
        judge_model=JEV_MODEL,
        writer_model=QWEN_MODEL,
        strong_check_model=QWEN_MODEL,
        weak_models=(WEAK_MODEL,),
        weak_model_count=1,
        weak_samples=3,
        weak_max_output_tokens=PILOT_MAX_OUTPUT_TOKENS,
        fallback_writer_model=QWEN_MODEL,
        fallback_strong_check_model=QWEN_MODEL,
    )


def provider_policy(provider: str) -> dict[str, object]:
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError("tuning comparisons allow only Novita or Groq")
    # Fallback is a comparison-level operation: the baseline must be rerun too.
    return {
        "only": [provider],
        "order": [provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
