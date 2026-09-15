"""Shared fixtures: the three sector schema mappings and their mock datasets.

The schema definitions here are the same ones the rule-based resolver is
expected to rediscover, so they double as the ground truth for
`test_schema_resolver.py`.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

_test_state = tempfile.TemporaryDirectory(prefix="churn-tests-")
os.environ["CHURN_DB_PATH"] = str(Path(_test_state.name) / "state.sqlite3")
os.environ.pop("UPSTASH_REDIS_REST_URL", None)
os.environ.pop("UPSTASH_REDIS_REST_TOKEN", None)
os.environ.pop("VERCEL", None)
os.environ.pop("KV_REST_API_URL", None)
os.environ.pop("KV_REST_API_TOKEN", None)
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.schema_mapping import SchemaMapping, TableClassification

# This runs before pytest imports a single test module, which is the only point
# early enough: `dependencies.py` builds its gateway singleton at import time, so
# a real key left in `api_key.env` would already be baked into a live client by
# the time a fixture ran. The suite would then be slow, metered against the
# developer's free quota, and at the mercy of whatever a hosted model replied.
from churn_platform.config import PROVIDERS, Settings, get_settings

Settings.model_config["env_file"] = ()
for _provider in PROVIDERS.values():
    for _key_field in _provider.key_fields:
        os.environ.pop(_key_field.upper(), None)
for _knob in ("QWEN_MODE", "QWEN_BASE_URL", "QWEN_MODEL", "BATCH_SIZE", "MAX_ENTITIES"):
    os.environ.pop(_knob, None)
get_settings.cache_clear()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Entities are generated in cohort order: the first 25 are deliberately
# churning, the remaining 75 healthy.
CHURN_COHORT_SIZE = 25

SECTOR_SCHEMAS: Dict[str, Tuple[str, List[TableClassification]]] = {
    "SaaS": ("user_id", [
        TableClassification(
            file_name="users.csv", role="DIMENSION", primary_entity_key="user_id",
            timestamp_column="signup_date", noise_columns=["ip_address", "last_user_agent"]),
        TableClassification(
            file_name="events_log.csv", role="TIME_SERIES_EVENT", primary_entity_key="user_id",
            timestamp_column="timestamp", noise_columns=["session_hash"]),
        TableClassification(
            file_name="invoices.csv", role="TRANSACTIONAL", primary_entity_key="user_id",
            timestamp_column="due_date"),
        TableClassification(
            file_name="tickets.csv", role="UNSTRUCTURED_TEXT", primary_entity_key="user_id"),
    ]),
    "Telecom": ("subscriber_id", [
        TableClassification(
            file_name="subscribers.csv", role="DIMENSION", primary_entity_key="subscriber_id",
            noise_columns=["sim_imsi_hash"]),
        TableClassification(
            file_name="network_cdrs.csv", role="TIME_SERIES_EVENT", primary_entity_key="subscriber_id",
            timestamp_column="call_timestamp"),
        TableClassification(
            file_name="recharge_history.csv", role="TRANSACTIONAL", primary_entity_key="subscriber_id",
            timestamp_column="recharge_date"),
        TableClassification(
            file_name="complaints.csv", role="UNSTRUCTURED_TEXT", primary_entity_key="subscriber_id"),
    ]),
    "FinTech": ("account_id", [
        TableClassification(
            file_name="accounts.csv", role="DIMENSION", primary_entity_key="account_id",
            timestamp_column="created_at", noise_columns=["device_mac_hash"]),
        TableClassification(
            file_name="ledger_transactions.csv", role="TRANSACTIONAL", primary_entity_key="account_id",
            timestamp_column="timestamp"),
        TableClassification(
            file_name="card_swipes.csv", role="TIME_SERIES_EVENT", primary_entity_key="account_id",
            timestamp_column="swipe_timestamp"),
        TableClassification(
            file_name="disputes.csv", role="UNSTRUCTURED_TEXT", primary_entity_key="account_id",
            timestamp_column="open_date"),
    ]),
}

SECTORS = list(SECTOR_SCHEMAS)


@pytest.fixture(scope="session", autouse=True)
def ensure_mock_data() -> None:
    """Generate the mock CSVs once if a fresh clone has not run the generator."""
    expected = [
        os.path.join(DATA_DIR, sector.lower(), table.file_name)
        for sector, (_, tables) in SECTOR_SCHEMAS.items()
        for table in tables
    ]
    if all(os.path.exists(path) for path in expected):
        return
    import generate_mock_data

    generate_mock_data.create_dirs()
    generate_mock_data.generate_saas_data()
    generate_mock_data.generate_telecom_data()
    generate_mock_data.generate_fintech_data()


def sector_schema(sector: str) -> SchemaMapping:
    key, tables = SECTOR_SCHEMAS[sector]
    return SchemaMapping(primary_entity_key=key, tables=tables)


def sector_dataframes(sector: str) -> Dict[str, pd.DataFrame]:
    _, tables = SECTOR_SCHEMAS[sector]
    folder = os.path.join(DATA_DIR, sector.lower())
    return {t.file_name: pd.read_csv(os.path.join(folder, t.file_name)) for t in tables}


@pytest.fixture(params=SECTORS)
def sector(request) -> str:
    return request.param


@pytest.fixture
def schema(sector: str) -> SchemaMapping:
    return sector_schema(sector)


@pytest.fixture
def dataframes(sector: str) -> Dict[str, pd.DataFrame]:
    return sector_dataframes(sector)


def cohort_of(entity_id: str) -> str:
    """Map an entity id like ``usr_7`` onto its generated cohort label."""
    index = int(entity_id.rsplit("_", 1)[1]) - 1
    return "CHURN" if index < CHURN_COHORT_SIZE else "HEALTHY"


def cohort_frame(features: List[CustomerFeatures]) -> pd.DataFrame:
    """Flatten features into a frame with a cohort column for group comparisons."""
    rows = [
        {"entity_id": f.entity_id, "cohort": cohort_of(f.entity_id), **f.features}
        for f in features
    ]
    return pd.DataFrame(rows)


def sector_features(sector: str) -> List[CustomerFeatures]:
    """Run the full pipeline for a sector: synthesize primitives, then enrich.

    Goes through the use case so the dependency-injection wiring between the
    application layer and the infrastructure enricher is exercised too.
    """
    from churn_platform.application.use_cases.synthesize_features import SynthesizeFeaturesUseCase
    from churn_platform.infrastructure.parsers.feature_synthesizer import PandasFeatureSynthesizer
    from churn_platform.infrastructure.parsers.sector_feature_enrichers import enrich_features

    use_case = SynthesizeFeaturesUseCase(PandasFeatureSynthesizer(), enricher=enrich_features)
    return use_case.execute(sector_schema(sector), sector_dataframes(sector), sector=sector)


@pytest.fixture(scope="session")
def sector_feature_frames() -> Dict[str, pd.DataFrame]:
    """Cohort-labelled feature frames, computed once for the whole session."""
    return {sector: cohort_frame(sector_features(sector)) for sector in SECTORS}


@pytest.fixture
def feature_frame(sector: str, sector_feature_frames) -> pd.DataFrame:
    return sector_feature_frames[sector]


@pytest.fixture
def raw_csv():
    """Read a generated CSV straight off disk, bypassing the schema mapping."""

    def read(sector: str, file_name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(DATA_DIR, sector.lower(), file_name))

    return read


def run_async(coro):
    """Drive a coroutine to completion.

    pytest-asyncio is not a dependency, and adding one to test four thin
    `await` wrappers would be a poor trade.
    """
    return asyncio.run(coro)


def header_samples(dataframes: Dict[str, pd.DataFrame], rows: int = 3) -> Dict[str, str]:
    """Render frames into the header-plus-a-few-rows samples the resolver sees."""
    return {name: df.head(rows).to_csv(index=False) for name, df in dataframes.items()}


@pytest.fixture(scope="session")
def sector_predictions() -> Dict[str, List[Tuple[Any, Any]]]:
    """Each sector core run against the offline gateway, once for the session."""
    from churn_platform.infrastructure.ai.cores.fintech_core import FintechCore
    from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
    from churn_platform.infrastructure.ai.cores.telecom_core import TelecomCore
    from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway

    gateway = MockQwenGateway()
    cores = {"SaaS": SaasCore(gateway), "Telecom": TelecomCore(gateway), "FinTech": FintechCore(gateway)}
    return {
        sector: run_async(cores[sector].analyze(sector_features(sector)))
        for sector in SECTORS
    }


def authenticated_client():
    from uuid import uuid4
    from fastapi.testclient import TestClient
    from churn_platform.main import app
    client = TestClient(app)
    response = client.post('/api/auth/signup', json={
        'name': 'Integration tester', 'email': f'{uuid4().hex}@example.com', 'password': 'Test-only-password-42'})
    assert response.status_code == 201, response.text
    return client
