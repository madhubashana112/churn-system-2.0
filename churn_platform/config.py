"""Runtime settings, read from the environment or a `.env` beside the repo root.

Field names are the environment variable names: `DASHSCOPE_API_KEY` has to keep
working under the name the README documents, so no prefix is applied.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, NamedTuple, Optional, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

# churn_platform/config.py -> churn_platform -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# The repo shipped `api_key.env`; `.env` is the conventional name. Both are
# read, resolved against the repo root so the CWD does not matter. Later files
# win, so `.env` overrides the legacy name.
ENV_FILES = (str(REPO_ROOT / "api_key.env"), str(REPO_ROOT / ".env"))

QwenMode = Literal["auto", "mock", "live"]


class Provider(NamedTuple):
    """A host that speaks the OpenAI chat-completions dialect.

    ``model`` is one that host actually serves on its free plan, so that
    dropping in a key is enough: no second lookup of which model ids exist.
    """

    name: str
    base_url: str
    model: str
    key_fields: Tuple[str, ...]
    # Entities per live call, or 0 to use the global BATCH_SIZE default. A host
    # whose free plan meters output tokens per minute needs a smaller batch than
    # one that meters credits, and the caller should not have to know which.
    batch_size: int = 0


PROVIDERS: Dict[str, Provider] = {
    "dashscope": Provider(
        name="dashscope",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",
        key_fields=("dashscope_api_key", "alibaba_api_key"),
    ),
    # Free plan: 30 requests/min and, per model, an output-tokens-per-minute cap.
    # Measured on 2026-09-05: the Qwen models fail json_object validation here
    # (qwen3.6-27b returns an empty generation, qwen3.8-27b needs ~1,880 output
    # tokens for ten customers against a 1,000 ceiling) and gpt-oss-20b degenerates
    # into stringified array elements. gpt-oss-120b scored every customer with an
    # AUC of 1.000 at six per call, which fits the same 1,000-token ceiling.
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        key_fields=("groq_api_key",),
        batch_size=6,
    ),
    # Free plan is a monthly credit rather than a rate limit, so the router's
    # :cheapest policy is what makes it last.
    "huggingface": Provider(
        name="huggingface",
        base_url="https://router.huggingface.co/v1",
        model="openai/gpt-oss-120b:cheapest",
        key_fields=("hf_token",),
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-oss-120b:free",
        key_fields=("openrouter_api_key",),
    ),
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-3.5-flash-lite",
        key_fields=("gemini_api_key",),
        batch_size=25,
    ),
    # A generic key with no host attached: the caller has to say where it goes.
    "custom": Provider(name="custom", base_url="", model="", key_fields=("ai_api_key",)),
}

# Highest precedence first, so a key already in use keeps winning after more
# providers are configured.
PROVIDER_ORDER = ("dashscope", "groq", "huggingface", "openrouter", "gemini", "custom")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # "auto" is the default: run offline when no key is configured rather than
    # failing at the first request, which is what makes the demo runnable as-is.
    qwen_mode: QwenMode = "auto"
    qwen_model: str = "qwen-max"
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    dashscope_api_key: Optional[str] = None
    alibaba_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    ai_api_key: Optional[str] = None

    batch_size: int = 10
    max_entities: int = 0  # 0 means no cap

    # Free tiers answer slowly and throttle hard. A request with no timeout
    # hangs the upload forever, and a 429 without a retry drops a whole batch
    # of customers from the analysis instead of waiting its turn.
    qwen_timeout_seconds: float = 120.0
    qwen_max_retries: int = 4

    def _credential(self, field: str) -> Optional[str]:
        value = getattr(self, field, None)
        # An empty value in .env is indistinguishable from an unset one here,
        # and would otherwise be passed to the client as a real key.
        return value.strip() if value and value.strip() else None

    @property
    def api_provider(self) -> str:
        """Name of the host that supplied a key, or "none"."""
        for name in PROVIDER_ORDER:
            if any(self._credential(field) for field in PROVIDERS[name].key_fields):
                return name
        return "none"

    @property
    def api_key(self) -> Optional[str]:
        provider = PROVIDERS.get(self.api_provider)
        if provider is None:
            return None
        for field in provider.key_fields:
            key = self._credential(field)
            if key:
                return key
        return None

    @property
    def resolved_base_url(self) -> str:
        """Where requests go: an explicit QWEN_BASE_URL, else the key's host."""
        if "qwen_base_url" in self.model_fields_set:
            return self.qwen_base_url
        provider = PROVIDERS.get(self.api_provider)
        if provider is None:
            return self.qwen_base_url
        if not provider.base_url:
            raise ValueError(
                "AI_API_KEY names no host on its own. Set QWEN_BASE_URL (and "
                "QWEN_MODEL) to the OpenAI-compatible endpoint the key is for."
            )
        return provider.base_url

    @property
    def resolved_model(self) -> str:
        """Which model to call: an explicit QWEN_MODEL, else the key's host default."""
        if "qwen_model" in self.model_fields_set:
            return self.qwen_model
        provider = PROVIDERS.get(self.api_provider)
        if provider is None or not provider.model:
            return self.qwen_model
        return provider.model

    @property
    def resolved_batch_size(self) -> int:
        """Entities per live call: an explicit BATCH_SIZE, else the key's host default."""
        if "batch_size" in self.model_fields_set:
            return self.batch_size
        provider = PROVIDERS.get(self.api_provider)
        if provider is None or not provider.batch_size:
            return self.batch_size
        return provider.batch_size


@lru_cache
def get_settings() -> Settings:
    return Settings()
