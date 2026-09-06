"""Live gateway for any OpenAI-compatible chat endpoint.

The names say Qwen because Alibaba Cloud Model Studio was the first host, but
the client only needs a base URL and a key. The free hosted tiers (Groq, the
Hugging Face router, OpenRouter) speak the same dialect, and `Settings` maps
whichever key is present onto that host's endpoint and a model its free plan
actually serves.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from churn_platform.config import get_settings
from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway

logger = logging.getLogger(__name__)


def parse_json_payload(content: Optional[str]) -> dict:
    """Read one JSON object out of a model reply.

    Open-weight models on the free tiers wrap their answer in a ```json fence
    or preface it with a sentence even with JSON mode on, so the payload is
    located by its outermost braces instead of being assumed to be the whole
    reply. A reply that parses to something else is rejected here rather than
    surfacing as an AttributeError in a caller that indexes into it.
    """
    if not content or not content.strip():
        raise ValueError("the model returned an empty reply")

    text = content.strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("the reply contains no JSON object") from None
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise ValueError("the reply contains no parseable JSON object") from None

    if not isinstance(parsed, dict):
        raise ValueError(f"the reply was a JSON {type(parsed).__name__}, not an object")
    return parsed


class QwenGateway(IAIGateway):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        api_key = api_key or settings.api_key
        if not api_key:
            # Failing here beats failing later: a placeholder key produces an
            # opaque 401 from the API with no hint that configuration is missing.
            raise EnvironmentError(
                "No AI provider key found. Set GROQ_API_KEY (free tier, no card "
                "required), HF_TOKEN, OPENROUTER_API_KEY or DASHSCOPE_API_KEY in "
                "the environment or in .env, or set QWEN_MODE=mock to run the "
                "platform offline."
            )

        # The client's own retries cover 429 and 5xx with backoff, which is what
        # a throttled free tier needs: without them the caller skips the batch
        # and those customers silently vanish from the analysis.
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or settings.resolved_base_url,
            timeout=settings.qwen_timeout_seconds if timeout is None else timeout,
            max_retries=settings.qwen_max_retries if max_retries is None else max_retries,
        )
        self.model = model or settings.resolved_model

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
        except Exception:
            logger.exception("Error calling the %s API", self.model)
            raise

        content = completion.choices[0].message.content
        try:
            return parse_json_payload(content)
        except ValueError:
            # The raw reply is the only evidence of what went wrong, and it is
            # gone by the time the failed batch is logged upstream.
            logger.error("Model %s returned no usable JSON object: %.500r", self.model, content)
            raise
