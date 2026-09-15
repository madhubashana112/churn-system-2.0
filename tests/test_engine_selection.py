"""Per-request choice between the hosted AI model and the local system model.

The engine used to be decided once, when ``dependencies`` was imported, which
meant comparing the two scorers needed an edited ``api_key.env`` and a restarted
server. Two things are pinned down here: that both engines are built from the
same rules the single gateway always was, and that asking for one that is not
configured is *refused* rather than quietly answered by the other — a user who
clicked "AI model" and got the deterministic scorer would have no way to tell.
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple

import pytest
from fastapi.testclient import TestClient

from churn_platform.config import Settings
from churn_platform.domain.models.sector import SECTOR_FINTECH, SECTOR_SAAS, SECTOR_TELECOM
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway
from churn_platform.infrastructure.ai.qwen_gateway import QwenGateway
from churn_platform.main import app
from churn_platform.presentation.api import dependencies
from churn_platform.presentation.api.dependencies import (
    ENGINE_AI,
    ENGINE_SYSTEM,
    ai_available,
    build_engine_registry,
    default_engine,
    get_analysis_batch_size,
    get_schema_resolver,
    get_sector_core,
    is_offline_engine,
    is_offline_gateway,
    resolve_engine,
)

from conftest import SECTOR_SCHEMAS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

KEY_FIELDS = (
    "DASHSCOPE_API_KEY",
    "ALIBABA_API_KEY",
    "GROQ_API_KEY",
    "HF_TOKEN",
    "OPENROUTER_API_KEY",
    "AI_API_KEY",
)

ALL_SECTORS = (SECTOR_SAAS, SECTOR_TELECOM, SECTOR_FINTECH)


def settings(**overrides: Any) -> Settings:
    """Settings with no env files and no ambient keys, so cases stay isolated."""
    return Settings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in KEY_FIELDS + ("QWEN_BASE_URL", "QWEN_MODEL", "QWEN_MODE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client() -> TestClient:
    from conftest import authenticated_client
    return authenticated_client()


@pytest.fixture
def ai_registry(monkeypatch):
    """Swap in a registry where a key exists, so the AI engine is selectable.

    The module-level registry is built at import time, when conftest has already
    stripped every credential, so the AI-available paths cannot be reached
    without replacing it. ``_REGISTRY`` is the seam the accessors read.
    """
    registry = build_engine_registry(settings(groq_api_key="gsk_test"))
    monkeypatch.setattr(dependencies, "_REGISTRY", registry)
    return registry


def uploads_for(sector: str) -> List[Tuple[str, Any]]:
    folder = os.path.join(DATA_DIR, sector.lower())
    return [
        ("files", (table.file_name, open(os.path.join(folder, table.file_name), "rb"), "text/csv"))
        for table in SECTOR_SCHEMAS[sector][1]
    ]


def register(client: TestClient, sector: str = "SaaS") -> str:
    response = client.post("/api/v1/tenants/", json={"name": f"{sector} Engine Co", "sector": sector})
    assert response.status_code == 200, response.text
    return response.json()["tenant_id"]


# -- what the registry builds -------------------------------------------------


def test_the_system_engine_needs_no_credentials():
    registry = build_engine_registry(settings())

    assert isinstance(registry.gateways[ENGINE_SYSTEM], MockQwenGateway)
    assert registry.gateways[ENGINE_AI] is None
    assert registry.default == ENGINE_SYSTEM


def test_a_provider_key_makes_the_ai_engine_the_default():
    registry = build_engine_registry(settings(groq_api_key="gsk_test"))

    assert isinstance(registry.gateways[ENGINE_AI], QwenGateway)
    assert registry.default == ENGINE_AI
    # The cores and the schema resolver must hold the *live* gateway: sharing
    # the offline one here is how a keyed deployment used to look configured
    # while still scoring locally.
    assert registry.cores[ENGINE_AI][SECTOR_SAAS].gateway is registry.gateways[ENGINE_AI]
    assert registry.resolvers[ENGINE_AI].gateway is registry.gateways[ENGINE_AI]
    assert registry.cores[ENGINE_SYSTEM][SECTOR_SAAS].gateway is registry.gateways[ENGINE_SYSTEM]


def test_mock_mode_turns_the_ai_engine_off_even_with_a_key():
    registry = build_engine_registry(settings(groq_api_key="gsk_test", qwen_mode="mock"))

    assert registry.gateways[ENGINE_AI] is None
    assert registry.default == ENGINE_SYSTEM


def test_forced_live_mode_without_a_key_still_fails_loudly():
    # QWEN_MODE=live is an operator insisting on model output. Degrading that to
    # the deterministic scorer would make a missing credential look like a
    # working deployment, so it has to raise exactly as build_ai_gateway does.
    with pytest.raises(EnvironmentError):
        build_engine_registry(settings(qwen_mode="live"))


def test_forced_live_mode_with_a_key_builds_the_ai_engine():
    registry = build_engine_registry(settings(groq_api_key="gsk_test", qwen_mode="live"))

    assert isinstance(registry.gateways[ENGINE_AI], QwenGateway)
    assert registry.default == ENGINE_AI


def test_every_available_engine_has_a_core_for_every_sector():
    for registry in (settings(), settings(groq_api_key="gsk_test")):
        built = build_engine_registry(registry)
        for engine, cores in built.cores.items():
            assert set(cores) == set(ALL_SECTORS), engine
            assert engine in built.resolvers


# -- resolving a request's choice ---------------------------------------------


def test_making_no_choice_means_the_deployment_default():
    assert resolve_engine(None) == default_engine() == ENGINE_SYSTEM
    assert resolve_engine("") == ENGINE_SYSTEM
    assert resolve_engine("auto") == ENGINE_SYSTEM


def test_the_system_engine_is_always_selectable(ai_registry):
    # Even with a key configured, a user must be able to ask for the local
    # scorer: it is the only engine that answers inside a serverless timeout.
    assert resolve_engine(ENGINE_SYSTEM) == ENGINE_SYSTEM
    assert is_offline_engine(ENGINE_SYSTEM) is True


def test_asking_for_an_unavailable_ai_engine_is_refused_not_substituted():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        resolve_engine(ENGINE_AI)


def test_an_unknown_engine_name_is_refused():
    with pytest.raises(ValueError, match="Unknown engine"):
        resolve_engine("turbo")


def test_the_chosen_engine_decides_which_gateway_scores(ai_registry):
    assert isinstance(get_sector_core("saas", ENGINE_AI).gateway, QwenGateway)
    assert isinstance(get_sector_core("saas", ENGINE_SYSTEM).gateway, MockQwenGateway)
    assert isinstance(get_schema_resolver(ENGINE_AI).gateway, QwenGateway)
    assert isinstance(get_schema_resolver(ENGINE_SYSTEM).gateway, MockQwenGateway)


def test_the_system_engine_scores_the_whole_population_in_one_call(ai_registry):
    # MockQwenGateway normalises each signal across the batch it is given, so
    # chunking it would rank a customer against an arbitrary slice of their
    # peers instead of against the tenant's whole base.
    assert get_analysis_batch_size(ENGINE_SYSTEM) == 0
    assert get_analysis_batch_size(ENGINE_AI) > 0


# -- what the deployment reports ----------------------------------------------


def test_status_says_the_ai_engine_cannot_be_chosen_without_a_key(client):
    status = client.get("/api/v1/analytics/status").json()

    assert status["ai_available"] is False
    assert status["default_engine"] == ENGINE_SYSTEM
    assert status["offline_mode"] is True


def test_status_says_the_ai_engine_is_the_default_with_a_key(client, ai_registry):
    status = client.get("/api/v1/analytics/status").json()

    assert status["ai_available"] is True
    assert status["default_engine"] == ENGINE_AI
    assert status["offline_mode"] is False
    assert is_offline_gateway() is False
    assert ai_available() is True


# -- the endpoint -------------------------------------------------------------


def test_an_explicit_system_engine_scores_offline(client):
    tenant_id = register(client)

    response = client.post(
        "/api/v1/upload/analyze",
        data={"tenant_id": tenant_id, "engine": ENGINE_SYSTEM},
        files=uploads_for("SaaS"),
    )

    assert response.status_code == 200, response.text
    assert response.json()["offline_mode"] is True


def test_omitting_the_engine_still_works(client):
    # Callers that predate the choice must keep working, and get the default.
    tenant_id = register(client)

    response = client.post(
        "/api/v1/upload/analyze",
        data={"tenant_id": tenant_id},
        files=uploads_for("SaaS"),
    )

    assert response.status_code == 200, response.text
    assert response.json()["offline_mode"] is True


def test_the_ai_engine_is_refused_when_no_key_is_configured(client):
    tenant_id = register(client)

    response = client.post(
        "/api/v1/upload/analyze",
        data={"tenant_id": tenant_id, "engine": ENGINE_AI},
        files=uploads_for("SaaS"),
    )

    assert response.status_code == 400
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_an_unknown_engine_is_refused(client):
    tenant_id = register(client)

    response = client.post(
        "/api/v1/upload/analyze",
        data={"tenant_id": tenant_id, "engine": "turbo"},
        files=uploads_for("SaaS"),
    )

    assert response.status_code == 400
    assert "Unknown engine" in response.json()["detail"]


def test_a_run_records_its_own_engine_not_the_deployment_default(client, ai_registry):
    # The deployment now defaults to the AI engine, so a run scored on the
    # system engine has to say so: offline_mode is what the dashboard reads
    # when it reloads a stored analysis, long after this request.
    tenant_id = register(client)

    response = client.post(
        "/api/v1/upload/analyze",
        data={"tenant_id": tenant_id, "engine": ENGINE_SYSTEM},
        files=uploads_for("SaaS"),
    )

    assert response.status_code == 200, response.text
    assert response.json()["offline_mode"] is True

    metrics = client.get(f"/api/v1/analytics/metrics?tenant_id={tenant_id}").json()
    assert metrics["offline_mode"] is True
