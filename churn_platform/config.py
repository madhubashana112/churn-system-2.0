"""Runtime settings, read from the environment or a `.env` beside the repo root.

Field names are the environment variable names: `DASHSCOPE_API_KEY` has to keep
working under the name the README documents, so no prefix is applied.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# churn_platform/config.py -> churn_platform -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# The repo shipped `api_key.env`; `.env` is the conventional name. Both are
# read, resolved against the repo root so the CWD does not matter. Later files
# win, so `.env` overrides the legacy name.
ENV_FILES = (str(REPO_ROOT / "api_key.env"), str(REPO_ROOT / ".env"))

QwenMode = Literal["auto", "mock", "live"]


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

    batch_size: int = 10
    max_entities: int = 0  # 0 means no cap

    @property
    def api_key(self) -> Optional[str]:
        key = self.dashscope_api_key or self.alibaba_api_key
        # An empty value in .env is indistinguishable from an unset one here,
        # and would otherwise be passed to the client as a real key.
        return key.strip() if key and key.strip() else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
