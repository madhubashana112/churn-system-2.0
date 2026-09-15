"""The live gateway over real HTTP, without leaving the test process.

Config and parsing are covered in isolation elsewhere; what is left is the part
that only fails in production: the openai client actually talking to a host that
throttles and decorates its replies. The stub app below is mounted into the
client's own transport, so requests and retries go through the real stack.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from churn_platform.domain.ai_errors import AIServiceError

from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
from churn_platform.infrastructure.ai.qwen_gateway import QwenGateway

REPLY = {
    "predictions": [
        {
            "entity_id": "usr_1",
            "churn_prediction": {"churn_probability": 0.87, "risk_tier": "critical"},
            "retention_playbook": {
                "action_type": "SAVE_OFFER",
                "action_payload": "30% off for 3 months",
                "channel": "Email",
            },
        }
    ]
}


def stub_host(state: Dict[str, Any]) -> FastAPI:
    """A chat-completions host that throttles first and fences its JSON."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        state["calls"] = state.get("calls", 0) + 1
        state["bodies"] = state.get("bodies", []) + [body]

        if state["calls"] <= state.get("throttle_first", 0):
            return JSONResponse(
                {"error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}},
                status_code=429,
                headers={"Retry-After": "0"},
            )

        content = json.dumps(REPLY)
        if state.get("fence", True):
            content = f"Here is the analysis:\n```json\n{content}\n```"
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    return app


def gateway_against(state: Dict[str, Any], max_retries: int = 3) -> QwenGateway:
    """A gateway whose client talks to the stub instead of the internet."""
    gateway = QwenGateway(
        api_key="test-key",
        base_url="https://stub.test/v1",
        model="test-model",
        timeout=5.0,
        max_retries=max_retries,
    )
    gateway.client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://stub.test/v1",
        timeout=5.0,
        max_retries=max_retries,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=stub_host(state)),
            base_url="https://stub.test",
        ),
    )
    return gateway


def test_a_throttled_call_is_retried_until_it_lands():
    state: Dict[str, Any] = {"throttle_first": 2}
    gateway = gateway_against(state)

    payload = asyncio.run(gateway.generate_json("system", "user"))

    assert payload == REPLY
    assert state["calls"] == 3


def test_the_request_asks_for_json_mode_and_sends_both_prompts():
    state: Dict[str, Any] = {}
    gateway = gateway_against(state)

    asyncio.run(gateway.generate_json("you are a churn model", "score these customers"))

    body = state["bodies"][0]
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [
        {"role": "system", "content": "you are a churn model"},
        {"role": "user", "content": "score these customers"},
    ]


def test_a_fenced_reply_with_prose_is_parsed_and_repaired_by_the_core():
    state: Dict[str, Any] = {"fence": True}
    predictions: List[Any] = asyncio.run(
        SaasCore(gateway_against(state)).analyze([CustomerFeatures(entity_id="usr_1", features={})])
    )

    (prediction, playbook), = predictions
    assert prediction.risk_tier == "CRITICAL"  # "critical" in the reply
    assert prediction.churn_probability == 0.87
    assert playbook.action_type == "SAVE_OFFER"


def test_exhausted_retries_surface_as_an_error_rather_than_a_silent_empty_run():
    state: Dict[str, Any] = {"throttle_first": 99}
    gateway = gateway_against(state, max_retries=1)

    with pytest.raises(AIServiceError) as failure:
        asyncio.run(gateway.generate_json("system", "user"))
    assert failure.value.status_code == 429
    assert state["calls"] == 2  # the attempt plus one retry

def test_gemini_json_requests_use_low_reasoning_without_changing_other_hosts():
    async def exercise():
        state = {'fence': False}
        gateway = QwenGateway(api_key='synthetic-test-key', base_url='https://generativelanguage.googleapis.com/v1', model='gemini-3.6-flash', max_retries=0)
        await gateway.client.close()
        gateway.client = AsyncOpenAI(api_key='synthetic-test-key', base_url='https://generativelanguage.googleapis.com/v1',
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=stub_host(state))), max_retries=0)
        try:
            result = await gateway.generate_json('Return JSON', 'Synthetic test')
            assert result == REPLY
            assert state['bodies'][0]['reasoning_effort'] == 'low'
            assert state['bodies'][0]['response_format'] == {'type':'json_object'}
        finally:
            await gateway.client.close()
    asyncio.run(exercise())


@pytest.mark.parametrize('status,body,expected_status,code', [
    (429, {'error': {'message':'private provider details', 'details':[{'quotaId':'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}},429,'AI_DAILY_QUOTA'),
    (429, {'error': {'message':'rate limited'}},429,'AI_RATE_LIMIT'),
    (503, {'error': {'message':'busy'}},503,'AI_UNAVAILABLE'),
    (401, {'error': {'message':'invalid key'}},503,'AI_CONFIGURATION'),
])
def test_gemini_errors_are_safe_typed_and_not_retried(status,body,expected_status,code):
    async def exercise():
        calls=[]
        async def handler(request):
            calls.append(request)
            return httpx.Response(status,json=body)
        gateway=QwenGateway(api_key='private-test-secret',base_url='https://generativelanguage.googleapis.com/v1beta/openai/',model='gemini-3.6-flash',max_retries=4)
        await gateway.client.close()
        gateway.client=AsyncOpenAI(api_key='private-test-secret',base_url='https://generativelanguage.googleapis.com/v1beta/openai/',max_retries=4,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            with pytest.raises(AIServiceError) as failure:
                await gateway.generate_json('system','user')
            assert failure.value.status_code==expected_status
            assert failure.value.code==code
            assert 'private' not in str(failure.value)
            assert len(calls)==1
        finally: await gateway.client.close()
    asyncio.run(exercise())
