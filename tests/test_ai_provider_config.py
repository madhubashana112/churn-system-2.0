"""Provider resolution and reply parsing for the hosted free tiers.

The platform has to work against whichever OpenAI-compatible host the user can
get a free key for, so two things are pinned down here: that a key on its own
selects the right endpoint and model, and that a reply which is JSON *plus*
decoration still scores customers instead of dropping the batch.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from churn_platform.config import PROVIDERS, Settings, get_settings
from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway
from churn_platform.infrastructure.ai.normalise_prediction import (
    normalise_probability,
    normalise_tier,
)
from churn_platform.infrastructure.ai.qwen_gateway import QwenGateway, parse_json_payload
from churn_platform.presentation.api.dependencies import build_ai_gateway, is_offline_gateway

KEY_FIELDS = (
    "DASHSCOPE_API_KEY",
    "ALIBABA_API_KEY",
    "GROQ_API_KEY",
    "HF_TOKEN",
    "OPENROUTER_API_KEY",
    "AI_API_KEY",
    "GEMINI_API_KEY",
)


def settings(**overrides: Any) -> Settings:
    """Settings with no env files and no ambient keys, so cases stay isolated."""
    return Settings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in KEY_FIELDS + ("QWEN_BASE_URL", "QWEN_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_a_working_key_on_disk_cannot_reach_the_suite():
    # conftest disables the env files before pytest imports a test module, because
    # dependencies.py composes its gateway at import time. Without that, one real
    # key in api_key.env turns every TestClient upload into a metered live call.
    assert Settings.model_config["env_file"] == ()
    assert get_settings().api_provider == "none"
    assert is_offline_gateway() is True


# -- which host a key selects -------------------------------------------------


def test_groq_key_selects_the_groq_endpoint_and_a_free_plan_model():
    resolved = settings(groq_api_key="gsk_test")

    assert resolved.api_provider == "groq"
    assert resolved.api_key == "gsk_test"
    assert resolved.resolved_base_url == "https://api.groq.com/openai/v1"
    assert resolved.resolved_model == PROVIDERS["groq"].model


def test_hf_token_selects_the_inference_providers_router():
    resolved = settings(hf_token="hf_test")

    assert resolved.api_provider == "huggingface"
    assert resolved.resolved_base_url == "https://router.huggingface.co/v1"


def test_openrouter_key_selects_openrouter():
    resolved = settings(openrouter_api_key="sk-or-test")

    assert resolved.api_provider == "openrouter"
    assert resolved.resolved_base_url == "https://openrouter.ai/api/v1"


def test_dashscope_key_keeps_the_original_host():
    resolved = settings(dashscope_api_key="sk-dashscope")

    assert resolved.api_provider == "dashscope"
    assert resolved.resolved_base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert resolved.resolved_model == "qwen-max"


def test_alibaba_alias_still_resolves_to_dashscope():
    resolved = settings(alibaba_api_key="sk-alibaba")

    assert resolved.api_provider == "dashscope"
    assert resolved.api_key == "sk-alibaba"


def test_dashscope_wins_when_several_keys_are_present():
    resolved = settings(dashscope_api_key="sk-dashscope", groq_api_key="gsk_test")

    assert resolved.api_provider == "dashscope"
    assert resolved.api_key == "sk-dashscope"


def test_groq_wins_over_a_hf_token():
    resolved = settings(groq_api_key="gsk_test", hf_token="hf_test")

    assert resolved.api_provider == "groq"


def test_blank_key_is_not_a_key():
    resolved = settings(groq_api_key="   ")

    assert resolved.api_provider == "none"
    assert resolved.api_key is None


def test_no_key_falls_back_to_the_documented_defaults():
    resolved = settings()

    assert resolved.api_provider == "none"
    assert resolved.api_key is None
    assert resolved.resolved_base_url == PROVIDERS["dashscope"].base_url
    assert resolved.resolved_model == "qwen-max"


# -- how many customers fit in one call ---------------------------------------


def test_groq_defaults_to_a_batch_its_free_plan_can_answer():
    # Ten customers need ~1,900 output tokens; this host's free plan caps output
    # at 1,000 per minute and rejects the request outright rather than queueing
    # it, so retrying the same batch can never succeed. Six fit.
    assert PROVIDERS["groq"].batch_size == 6
    assert settings(groq_api_key="gsk_test").resolved_batch_size == 6


def test_a_host_with_no_output_cap_keeps_the_global_batch_size():
    assert settings(dashscope_api_key="sk-dashscope").resolved_batch_size == 10
    assert settings().resolved_batch_size == 10


def test_an_explicit_batch_size_beats_the_host_default():
    resolved = settings(groq_api_key="gsk_test", batch_size=3)

    assert resolved.resolved_batch_size == 3


def test_explicit_endpoint_and_model_beat_the_provider_defaults():
    resolved = settings(
        groq_api_key="gsk_test",
        qwen_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        qwen_model="gemini-2.5-flash",
    )

    assert resolved.api_provider == "groq"
    assert resolved.resolved_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert resolved.resolved_model == "gemini-2.5-flash"


def test_explicit_model_alone_keeps_the_provider_endpoint():
    resolved = settings(groq_api_key="gsk_test", qwen_model="openai/gpt-oss-20b")

    assert resolved.resolved_base_url == "https://api.groq.com/openai/v1"
    assert resolved.resolved_model == "openai/gpt-oss-20b"


def test_generic_key_without_an_endpoint_is_refused():
    resolved = settings(ai_api_key="some-key")

    assert resolved.api_provider == "custom"
    with pytest.raises(ValueError, match="QWEN_BASE_URL"):
        resolved.resolved_base_url


def test_generic_key_with_an_endpoint_is_accepted():
    resolved = settings(ai_api_key="some-key", qwen_base_url="https://example.test/v1")

    assert resolved.resolved_base_url == "https://example.test/v1"


# -- reply parsing ------------------------------------------------------------


def test_plain_object_parses():
    assert parse_json_payload('{"predictions": []}') == {"predictions": []}


def test_fenced_reply_parses():
    reply = '```json\n{"predictions": [{"entity_id": "usr_1"}]}\n```'

    assert parse_json_payload(reply) == {"predictions": [{"entity_id": "usr_1"}]}


def test_prose_around_the_object_parses():
    reply = 'Here is the analysis you asked for:\n{"predictions": []}\nHope that helps.'

    assert parse_json_payload(reply) == {"predictions": []}


def test_surrounding_whitespace_parses():
    assert parse_json_payload('  \n {"a": 1} \n ') == {"a": 1}


def test_braces_inside_strings_do_not_confuse_the_slice():
    reply = 'Sure!\n{"reason": "plan {pro} lapsed", "score": 0.5}\nDone.'

    assert parse_json_payload(reply) == {"reason": "plan {pro} lapsed", "score": 0.5}


def test_a_json_array_is_refused_because_callers_index_into_an_object():
    with pytest.raises(ValueError, match="not an object"):
        parse_json_payload('[{"entity_id": "usr_1"}]')


def test_reply_with_no_json_is_refused():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_payload("I cannot help with that.")


def test_unparseable_json_is_refused():
    with pytest.raises(ValueError, match="no parseable JSON object"):
        parse_json_payload('{"predictions": [}')


def test_empty_reply_is_refused():
    with pytest.raises(ValueError, match="empty reply"):
        parse_json_payload("")
    with pytest.raises(ValueError, match="empty reply"):
        parse_json_payload(None)


# -- gateway construction -----------------------------------------------------


def test_gateway_names_every_supported_key_when_none_is_set():
    with pytest.raises(EnvironmentError) as excinfo:
        QwenGateway(base_url="https://example.test/v1")

    message = str(excinfo.value)
    for name in ("GROQ_API_KEY", "HF_TOKEN", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY"):
        assert name in message
    assert "QWEN_MODE=mock" in message


def test_gateway_uses_the_provider_endpoint_model_and_retry_settings():
    gateway = QwenGateway(api_key="gsk_test", base_url=None, model=None, timeout=7.5, max_retries=2)

    assert gateway.model == "qwen-max"  # no key in settings, so no provider default
    assert str(gateway.client.base_url).rstrip("/").endswith("/compatible-mode/v1")
    assert float(gateway.client.timeout) == 7.5
    assert gateway.client.max_retries == 2


def test_gateway_follows_the_key_host_when_settings_supply_it(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    from churn_platform.config import get_settings

    get_settings.cache_clear()
    try:
        gateway = QwenGateway()

        assert gateway.model == PROVIDERS["groq"].model
        assert "api.groq.com" in str(gateway.client.base_url)
        assert gateway.client.max_retries == 4
    finally:
        get_settings.cache_clear()


# -- gateway call -------------------------------------------------------------


class _StubCompletions:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def gateway_replying_with(content: Any) -> QwenGateway:
    gateway = QwenGateway(api_key="gsk_test", base_url="https://example.test/v1", model="test-model")
    gateway.client = SimpleNamespace(chat=SimpleNamespace(completions=_StubCompletions(content)))
    return gateway


def test_generate_json_requests_json_mode_and_returns_the_payload():
    gateway = gateway_replying_with('```json\n{"predictions": [{"entity_id": "usr_7"}]}\n```')

    payload = asyncio.run(gateway.generate_json("system", "user"))

    assert payload == {"predictions": [{"entity_id": "usr_7"}]}
    call = gateway.client.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_generate_json_raises_when_the_reply_is_not_an_object():
    gateway = gateway_replying_with("sorry, I cannot answer that")

    with pytest.raises(ValueError):
        asyncio.run(gateway.generate_json("system", "user"))


# -- gateway selection --------------------------------------------------------


def test_build_ai_gateway_goes_live_on_a_groq_key():
    gateway = build_ai_gateway(settings(groq_api_key="gsk_test"))

    assert isinstance(gateway, QwenGateway)
    assert gateway.model == PROVIDERS["groq"].model
    assert "api.groq.com" in str(gateway.client.base_url)


def test_build_ai_gateway_stays_offline_without_a_key():
    assert isinstance(build_ai_gateway(settings()), MockQwenGateway)


def test_build_ai_gateway_honours_mock_mode_even_with_a_key():
    gateway = build_ai_gateway(settings(groq_api_key="gsk_test", qwen_mode="mock"))

    assert isinstance(gateway, MockQwenGateway)


def test_build_ai_gateway_raises_in_live_mode_without_a_key():
    with pytest.raises(EnvironmentError):
        build_ai_gateway(settings(qwen_mode="live"))


# -- repairing a model's drift ------------------------------------------------


def test_known_tier_names_are_uppercased_when_they_match_the_probability():
    assert normalise_tier(" Critical ", 0.9) == "CRITICAL"
    assert normalise_tier("low", 0.05) == "LOW"
    assert normalise_tier("Medium", 0.4) == "MEDIUM"


def test_a_tier_that_contradicts_its_own_probability_is_replaced_by_the_band():
    # Measured, not hypothetical: gpt-oss-120b on Groq labelled 10 of 30 customers
    # against its own score, which would show a CRITICAL customer as MEDIUM and
    # drop them out of the at-risk KPI.
    assert normalise_tier("high", 0.2) == "LOW"
    assert normalise_tier("CRITICAL", 0.4) == "MEDIUM"
    assert normalise_tier("low", 0.95) == "CRITICAL"


def test_unknown_or_missing_tier_is_derived_from_the_probability():
    assert normalise_tier("at risk", 0.62) == "HIGH"
    assert normalise_tier(None, 0.1) == "LOW"
    assert normalise_tier(3, 0.95) == "CRITICAL"


def test_probabilities_are_coerced_and_clamped():
    assert normalise_probability("0.42") == 0.42
    assert normalise_probability(1.4) == 1.0
    assert normalise_probability(-0.2) == 0.0


def test_a_probability_that_is_not_a_number_fails_the_batch():
    for value in ("high", None, float("nan")):
        with pytest.raises(ValueError):
            normalise_probability(value)


def test_a_sector_core_survives_a_fenced_reply_with_lowercase_tiers():
    """The whole live path: fence stripped, tier uppercased, string number parsed."""
    reply = (
        "```json\n"
        + json.dumps({
            "predictions": [{
                "entity_id": "usr_1",
                "churn_prediction": {
                    "churn_probability": "0.91",
                    "risk_tier": "critical",
                    "primary_drivers": ["usage collapse"],
                },
                "retention_playbook": {
                    "action_type": "SAVE_OFFER",
                    "action_payload": "30% off for 3 months",
                    "channel": "Email",
                },
            }],
        })
        + "\n```"
    )
    core = SaasCore(gateway_replying_with(reply))

    (prediction, playbook), = asyncio.run(core.analyze([CustomerFeatures(entity_id="usr_1", features={})]))

    assert prediction.entity_id == "usr_1"
    assert prediction.churn_probability == 0.91
    assert prediction.risk_tier == "CRITICAL"
    assert prediction.primary_drivers == ["usage collapse"]
    assert playbook.channel == "Email"


def test_gemini_key_uses_google_endpoint_and_model():
    resolved = settings(gemini_api_key="gemini-test-key")
    assert resolved.api_provider == "gemini"
    assert resolved.api_key == "gemini-test-key"
    assert resolved.resolved_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert resolved.resolved_model == "gemini-3.5-flash-lite"
    assert resolved.resolved_batch_size == 25


def test_gemini_builds_live_gateway_and_all_sector_cores():
    from churn_platform.presentation.api.dependencies import build_engine_registry
    registry = build_engine_registry(settings(gemini_api_key="gemini-test-key"))
    assert registry.default == "ai"
    assert set(registry.cores["ai"]) == {"saas", "telecom", "fintech"}
    assert registry.gateways["ai"].model == "gemini-3.5-flash-lite"
    assert str(registry.gateways["ai"].client.base_url).startswith("https://generativelanguage.googleapis.com/")
