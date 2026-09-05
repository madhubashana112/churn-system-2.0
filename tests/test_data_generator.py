"""The generated datasets must be reproducible and must carry real signal.

Spec section 6 asked for twelve CSVs; section 2B asked for rolling velocity and
sentiment features. Those are only compatible if every activity stream carries a
timestamp, the free text varies, and a deliberate cohort churns. These tests pin
that contract down, including the entity ordering that `cohort_of` relies on.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

import generate_mock_data
from churn_platform.infrastructure.parsers.feature_synthesizer import to_naive_utc

EXPECTED_FILES = [
    "saas/users.csv",
    "saas/events_log.csv",
    "saas/invoices.csv",
    "saas/tickets.csv",
    "telecom/subscribers.csv",
    "telecom/network_cdrs.csv",
    "telecom/recharge_history.csv",
    "telecom/complaints.csv",
    "fintech/accounts.csv",
    "fintech/ledger_transactions.csv",
    "fintech/card_swipes.csv",
    "fintech/disputes.csv",
]

# Dimension table, entity key, and id prefix per sector folder.
DIMENSIONS = {
    "saas": ("users.csv", "user_id", "usr"),
    "telecom": ("subscribers.csv", "subscriber_id", "sub"),
    "fintech": ("accounts.csv", "account_id", "acc"),
}

TIMESTAMP_COLUMNS = {
    "saas/users.csv": "signup_date",
    "saas/events_log.csv": "timestamp",
    "saas/invoices.csv": "due_date",
    "telecom/network_cdrs.csv": "call_timestamp",
    "telecom/recharge_history.csv": "recharge_date",
    "fintech/accounts.csv": "created_at",
    "fintech/ledger_transactions.csv": "timestamp",
    "fintech/card_swipes.csv": "swipe_timestamp",
    "fintech/disputes.csv": "open_date",
}

# Thresholds sit at roughly half the observed cohort gap: tight enough to catch
# a generator that loses the injected pattern, loose enough to survive a rebalance.
STRONGER_WHEN_CHURNING = [
    ("SaaS", "failure_rate", 0.30),
    ("SaaS", "text_churn_score", 1.50),
    ("SaaS", "export_ratio", 0.15),
    ("SaaS", "invoices_failed_count", 1.0),
    ("Telecom", "failure_rate", 0.08),
    ("Telecom", "text_churn_score", 2.0),
    ("Telecom", "days_since_last_recharge", 12.0),
    ("Telecom", "max_recharge_gap_days", 6.0),
    ("FinTech", "failure_rate", 0.10),
    ("FinTech", "balance_drain_ratio", 0.20),
    ("FinTech", "p2p_failure_rate", 0.30),
    ("FinTech", "p2p_failure_streak", 1.0),
    ("FinTech", "outflow_recent", 300.0),
]

WEAKER_WHEN_CHURNING = [
    ("SaaS", "activity_velocity", 0.35),
    ("Telecom", "activity_velocity", 0.30),
    ("FinTech", "activity_velocity", 0.50),
]

BINARY_FLAGS = [
    ("SaaS", "has_negative_text", 0.60, 0.00),
    ("Telecom", "expanding_topup_intervals", 0.50, 0.10),
    ("Telecom", "regional_network_impact_flag", 0.25, 0.00),
    ("FinTech", "rapid_balance_drain", 0.50, 0.30),
]


def generate_into(monkeypatch, target: Path) -> Dict[str, str]:
    """Run all three generators against a scratch directory and hash the output."""
    monkeypatch.setattr(generate_mock_data, "DATA_DIR", str(target))
    generate_mock_data.create_dirs()
    generate_mock_data.generate_saas_data()
    generate_mock_data.generate_telecom_data()
    generate_mock_data.generate_fintech_data()
    return {
        rel: hashlib.sha256((target / rel).read_bytes()).hexdigest()
        for rel in EXPECTED_FILES
    }


def cohort_means(frame: pd.DataFrame, feature: str) -> tuple[float, float]:
    values = pd.to_numeric(frame[feature], errors="coerce")
    churn = values[frame["cohort"] == "CHURN"].mean()
    healthy = values[frame["cohort"] == "HEALTHY"].mean()
    return float(churn), float(healthy)


class TestReproducibility:
    def test_two_runs_produce_identical_bytes(self, tmp_path, monkeypatch):
        """A seeded generator must not drift between runs or between machines."""
        first = generate_into(monkeypatch, tmp_path / "run1")
        second = generate_into(monkeypatch, tmp_path / "run2")

        assert first == second

    def test_writes_exactly_the_twelve_specified_files(self, tmp_path, monkeypatch):
        generate_into(monkeypatch, tmp_path / "run")
        written = sorted(p.relative_to(tmp_path / "run").as_posix() for p in (tmp_path / "run").rglob("*.csv"))

        assert written == sorted(EXPECTED_FILES)

    def test_reference_date_is_frozen(self):
        """Wall-clock dates would make every rolling window a moving target."""
        assert generate_mock_data.REFERENCE_DATE == datetime(2025, 6, 1, 12, 0, 0)


class TestSpecGapsClosed:
    @pytest.mark.parametrize("rel,column", sorted(TIMESTAMP_COLUMNS.items()))
    def test_every_activity_stream_carries_a_parseable_timestamp(self, raw_csv, rel, column):
        """Spec section 6 left `network_cdrs` and `card_swipes` without one, which
        makes the section 2B rolling windows impossible to compute for them."""
        sector, name = rel.split("/")
        df = raw_csv(sector, name)

        assert column in df.columns, f"{rel} is missing {column}"
        assert to_naive_utc(df[column]).notna().all(), f"{rel}.{column} has unparseable values"

    def test_ticket_subjects_are_varied(self, raw_csv):
        """The original cycled through five fixed subjects."""
        assert raw_csv("saas", "tickets.csv")["subject"].nunique() > 5

    def test_complaint_notes_are_varied(self, raw_csv):
        """The original wrote the identical string 'Customer issue reported' 30 times."""
        notes = raw_csv("telecom", "complaints.csv")["notes"]

        assert notes.nunique() > 1
        assert notes.str.len().min() > 0

    def test_churning_complaints_carry_port_out_intent(self, raw_csv):
        """Text mining needs the churning cohort to actually say it is leaving."""
        complaints = raw_csv("telecom", "complaints.csv")
        churning = complaints[complaints["subscriber_id"].isin(
            [f"sub_{i}" for i in range(1, generate_mock_data.CHURN_COHORT_SIZE + 1)]
        )]
        haystack = churning["notes"].str.lower() + " " + churning["category"].str.lower()

        assert haystack.str.contains("port|mnp|cancel|switch").any()


class TestCohortStructure:
    @pytest.mark.parametrize("sector", sorted(DIMENSIONS))
    def test_entity_ids_keep_their_generated_cohort_order(self, raw_csv, sector):
        """`cohort_of` labels by numeric suffix, so ids must stay in generation order."""
        name, key, prefix = DIMENSIONS[sector]
        expected = [f"{prefix}_{i}" for i in range(1, generate_mock_data.N_ENTITIES + 1)]

        assert raw_csv(sector, name)[key].tolist() == expected

    def test_a_quarter_of_entities_are_seeded_to_churn(self):
        assert generate_mock_data.CHURN_COHORT_SIZE == generate_mock_data.N_ENTITIES // 4

    @pytest.mark.parametrize("sector", sorted(DIMENSIONS))
    def test_churning_entities_appear_in_the_event_streams(self, raw_csv, sector):
        """Every entity, churning or not, must have activity to synthesise from."""
        name, key, prefix = DIMENSIONS[sector]
        ids = set(raw_csv(sector, name)[key])
        stream = {
            "saas": "events_log.csv",
            "telecom": "network_cdrs.csv",
            "fintech": "ledger_transactions.csv",
        }[sector]
        seen = set(raw_csv(sector, stream)[key])

        assert ids <= seen, f"{len(ids - seen)} entities have no rows in {stream}"


class TestCohortSeparation:
    @pytest.mark.parametrize("sector,feature,min_gap", STRONGER_WHEN_CHURNING)
    def test_signal_is_stronger_for_the_churning_cohort(self, sector_feature_frames, sector, feature, min_gap):
        churn, healthy = cohort_means(sector_feature_frames[sector], feature)

        assert churn - healthy >= min_gap, f"{sector}.{feature}: churn {churn:.3f} vs healthy {healthy:.3f}"

    @pytest.mark.parametrize("sector,feature,min_gap", WEAKER_WHEN_CHURNING)
    def test_activity_decays_for_the_churning_cohort(self, sector_feature_frames, sector, feature, min_gap):
        """Churn here is modelled as decay, not as a last-minute usage surge."""
        churn, healthy = cohort_means(sector_feature_frames[sector], feature)

        assert healthy - churn >= min_gap, f"{sector}.{feature}: churn {churn:.3f} vs healthy {healthy:.3f}"

    @pytest.mark.parametrize("sector,feature,min_churn,max_healthy", BINARY_FLAGS)
    def test_binary_flags_fire_almost_exclusively_on_the_churning_cohort(
        self, sector_feature_frames, sector, feature, min_churn, max_healthy
    ):
        frame = sector_feature_frames[sector]
        assert frame[feature].notna().all(), f"{sector}.{feature} is missing for some entities"
        churn, healthy = cohort_means(frame, feature)

        assert churn >= min_churn, f"{sector}.{feature}: only {churn:.1%} of the churning cohort flagged"
        assert healthy <= max_healthy, f"{sector}.{feature}: {healthy:.1%} of healthy entities false-flagged"

    @pytest.mark.parametrize("sector", ["SaaS", "Telecom", "FinTech"])
    def test_recency_is_longer_for_the_churning_cohort(self, sector_feature_frames, sector):
        churn, healthy = cohort_means(sector_feature_frames[sector], "recency_days")

        assert churn > healthy
