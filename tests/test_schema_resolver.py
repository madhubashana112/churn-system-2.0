"""The offline resolver must rediscover the schema the platform was designed for.

`conftest.SECTOR_SCHEMAS` is the ground truth: it is what the enrichers assume
and what a correct resolver should recover from three sample rows alone. These
tests assert an exact match, so a tweak to the classification rules that costs a
timestamp column or mislabels a role fails here instead of quietly degrading
every feature downstream.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from churn_platform.application.use_cases.resolve_multi_sheet_schema import (
    ResolveMultiSheetSchemaUseCase,
)
from churn_platform.domain.models.schema_mapping import TableClassification
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway
from churn_platform.infrastructure.parsers.schema_resolver import AISchemaResolver

from conftest import header_samples, run_async


def resolve(samples: Dict[str, str]):
    """Resolve through the use case, so the DTO parsing is exercised too."""
    resolver = AISchemaResolver(MockQwenGateway())
    return run_async(ResolveMultiSheetSchemaUseCase(resolver).execute(samples))


def by_name(tables: List[TableClassification]) -> Dict[str, TableClassification]:
    return {t.file_name: t for t in tables}


@pytest.fixture
def resolved(schema, dataframes) -> Dict[str, TableClassification]:
    return by_name(resolve(header_samples(dataframes)).tables)


class TestResolvedSchemaMatchesGroundTruth:
    def test_primary_entity_key(self, schema, dataframes):
        assert resolve(header_samples(dataframes)).primary_entity_key == schema.primary_entity_key

    def test_the_same_tables_come_back(self, schema, resolved):
        assert set(resolved) == {t.file_name for t in schema.tables}

    def test_roles_match(self, schema, resolved):
        assert {n: t.role for n, t in resolved.items()} == {t.file_name: t.role for t in schema.tables}

    def test_entity_keys_match(self, schema, resolved):
        assert {n: t.primary_entity_key for n, t in resolved.items()} == {
            t.file_name: t.primary_entity_key for t in schema.tables
        }

    def test_timestamp_columns_match(self, schema, resolved):
        assert {n: t.timestamp_column for n, t in resolved.items()} == {
            t.file_name: t.timestamp_column for t in schema.tables
        }

    def test_noise_columns_match(self, schema, resolved):
        assert {n: sorted(t.noise_columns) for n, t in resolved.items()} == {
            t.file_name: sorted(t.noise_columns) for t in schema.tables
        }


class TestClassificationRules:
    def test_a_sentiment_column_is_not_read_as_a_timestamp(self):
        """Regression guard: "time" is a substring of "senti-TIME-nt".

        A bare substring test picked `sentiment` as the timestamp column for
        `tickets.csv`, so the synthesizer tried to parse ticket sentiment as a
        date, got NaT for every row, and published no text features at all.
        """
        samples = {"tickets.csv": "ticket_id,user_id,subject,sentiment\nt1,u1,slow,Very Negative\n"}
        table = by_name(resolve(samples).tables)["tickets.csv"]

        assert table.timestamp_column is None
        assert table.role == "UNSTRUCTURED_TEXT"

    def test_a_domain_key_is_not_discarded_as_noise(self):
        """Regression guard: `tower_id` is signal, not a surrogate row key.

        Treating every ``*_id`` column as a discardable row key would silently
        disable the regional network fault flag, which the telecom enricher
        derives from call concentration on a single tower. A three-row sample
        cannot tell `event_id` from `tower_id` by cardinality, so neither is
        dropped.
        """
        samples = {
            "network_cdrs.csv": (
                "cdr_id,subscriber_id,call_timestamp,tower_id,outcome\n"
                "c1,s1,2025-05-30T10:00:00Z,TWR_7,COMPLETED\n"
            )
        }
        table = by_name(resolve(samples).tables)["network_cdrs.csv"]

        assert table.noise_columns == []
        assert table.timestamp_column == "call_timestamp"

    def test_hashes_and_device_fingerprints_are_discarded(self):
        samples = {
            "users.csv": (
                "user_id,tier,ip_address,last_user_agent,session_hash\n"
                "u1,pro,10.0.0.1,Mozilla,abc\n"
            )
        }
        table = by_name(resolve(samples).tables)["users.csv"]

        assert sorted(table.noise_columns) == ["ip_address", "last_user_agent", "session_hash"]
        assert table.role == "DIMENSION"

    def test_the_widely_shared_id_column_wins_the_primary_key(self):
        samples = {
            "orders.csv": "order_id,user_id,amount\no1,u1,5.0\n",
            "users.csv": "user_id,tier\nu1,pro\n",
            "refunds.csv": "refund_id,order_id,user_id,amount\nr1,o1,u1,5.0\n",
        }

        assert resolve(samples).primary_entity_key == "user_id"

    def test_a_table_without_the_primary_key_falls_back_to_its_own_id(self):
        samples = {
            "orders.csv": "order_id,user_id,amount\no1,u1,5.0\n",
            "notes.csv": "note_id,body\nn1,hello\n",
        }
        tables = by_name(resolve(samples).tables)

        assert tables["orders.csv"].primary_entity_key == "user_id"
        assert tables["notes.csv"].primary_entity_key == "note_id"


class TestResolverFailures:
    def test_no_id_column_requires_human_review(self):
        result = resolve({"a.csv": "amount,status\n10.0,PAID\n"})
        assert result.status == "REQUIRES_HUMAN_REVIEW"
        assert any("customer ID" in reason for reason in result.review_reasons)

    def test_an_empty_sample_set_is_an_error(self):
        with pytest.raises(ValueError, match="at least one file sample"):
            resolve({})

    def test_a_sample_that_is_not_parseable_csv_still_yields_a_header(self):
        samples = {"broken.csv": "user_id,amount,status"}
        table = by_name(resolve(samples).tables)["broken.csv"]

        assert table.primary_entity_key == "user_id"
        assert table.role == "TRANSACTIONAL"
