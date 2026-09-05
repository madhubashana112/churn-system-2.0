"""Live gateway for Alibaba Cloud Model Studio (Qwen).

Speaks the OpenAI-compatible endpoint, so the standard async client is used
unchanged with a different base URL.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from churn_platform.config import get_settings
from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway

logger = logging.getLogger(__name__)


class QwenGateway(IAIGateway):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        api_key = api_key or settings.api_key
        if not api_key:
            # Failing here beats failing later: a placeholder key produces an
            # opaque 401 from the API with no hint that configuration is missing.
            raise EnvironmentError(
                "No Alibaba Cloud API key found. Set DASHSCOPE_API_KEY (or "
                "ALIBABA_API_KEY) in the environment or in .env, or set "
                "QWEN_MODE=mock to run the platform offline."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or settings.qwen_base_url,
        )
        self.model = model or settings.qwen_model

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            logger.error("Qwen returned a response that was not valid JSON")
            raise
        except Exception:
            logger.exception("Error calling the Qwen API")
            raise
