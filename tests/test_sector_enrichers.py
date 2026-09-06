"""Sector enrichment tests using hand-built frames with hand-computable values.

Each test isolates one enricher by declaring only the table it reads, so the
temporal anchor is unambiguous and the expected numbers can be worked out by hand.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.schema_mapping import SchemaMapping, TableClassification
from churn_platform.infrastructure.parsers.sector_feature_enrichers import enrich_features

REFERENCE = pd.Timestamp("2025-06-01 12:00:00")


def ts(days_ago: float) -> str:
    return (REFERENCE - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def plain_date(days_ago: float) -> str:
    return (REFERENCE - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")


def blank_features(entity_ids: List[str], seed: Dict[str, Any] | None = None) -> List[CustomerFeatures]:
    return [CustomerFeatures(entity_id=eid, features=dict(seed or {})) for eid in entity_ids]


def by_entity(features: List[CustomerFeatures]) -> Dict[str, Dict[str, Any]]:
    return {f.entity_id: f.features for f in features}


def one_table_schema(file_name: str, role: str, key: str, timestamp_column: str | None) -> SchemaMapping:
    return SchemaMapping(primary_entity_key=key, tables=[
        TableClassification(
            file_name=file_name, role=role, primary_entity_key=key,
            timestamp_column=timestamp_column, noise_columns=[]),
    ])


class TestSaaSExportSpike:
    def schema(self) -> SchemaMapping:
        return one_table_schema("events_log.csv", "TIME_SERIES_EVENT", "user_id", "timestamp")

    def frame(self) -> pd.DataFrame:
        rows: list[tuple[str, float, str]] = []
        # u1 exports steadily and recently: 4 exports in the last 4 days, 6 logins.
        rows += [("u1", d, "export") for d in (0, 1, 2, 3)]
        rows += [("u1", d, "login") for d in (5, 6, 7, 8, 9, 10)]
        # u2 exported once recently and once 40 days ago.
        rows += [("u2", 0, "login"), ("u2", 2, "export"), ("u2", 40, "export")]
        return pd.DataFrame({
            "event_id": [f"evt_{i}" for i in range(len(rows))],
            "user_id": [r[0] for r in rows],
            "timestamp": [ts(r[1]) for r in rows],
            "event_type": [r[2] for r in rows],
        })

    def test_export_ratio_is_the_share_of_all_events(self):
        features = blank_features(["u1", "u2"])
        enrich_features("SaaS", features, self.schema(), {"events_log.csv": self.frame()})
        result = by_entity(features)

        assert result["u1"]["export_count"] == 4
        assert result["u1"]["export_ratio"] == pytest.approx(4 / 10)
        assert result["u2"]["export_count"] == 2
        assert result["u2"]["export_ratio"] == pytest.approx(2 / 3, abs=1e-4)

    def test_only_exports_inside_the_recent_window_are_counted_recent(self):
        features = blank_features(["u1", "u2"])
        enrich_features("SaaS", features, self.schema(), {"events_log.csv": self.frame()})
        result = by_entity(features)

        assert result["u1"]["export_count_recent"] == 4
        # u2's second export is 40 days old, outside the 14-day window.
        assert result["u2"]["export_count_recent"] == 1

    def test_sector_label_is_normalised(self):
        features = blank_features(["u1"])
        enrich_features("  Subscription ", features, self.schema(), {"events_log.csv": self.frame()})

        assert by_entity(features)["u1"]["export_count"] == 4


class TestTelecomTopUpIntervals:
    def schema(self) -> SchemaMapping:
        return one_table_schema("recharge_history.csv", "TRANSACTIONAL", "subscriber_id", "recharge_date")

    def frame(self, schedules: Dict[str, List[float]]) -> pd.DataFrame:
        rows = [
            (sid, days_ago)
            for sid, days in schedules.items()
            for days_ago in days
        ]
        return pd.DataFrame({
            "rec_id": [f"rec_{i}" for i in range(len(rows))],
            "subscriber_id": [r[0] for r in rows],
            "amount": [20] * len(rows),
            "recharge_date": [plain_date(r[1]) for r in rows],
        })

    def test_widening_gaps_are_flagged(self):
        # Chronological gaps of 5, 15 and 20 days: the last two average 17.5
        # against 5 for the rest, well past the 1.5x expansion ratio.
        features = blank_features(["s1"])
        frames = {"recharge_history.csv": self.frame({"s1": [40, 35, 20, 0]})}
        enrich_features("Telecom", features, self.schema(), frames)
        result = by_entity(features)["s1"]

        assert result["recharge_count"] == 4
        assert result["avg_recharge_gap_days"] == pytest.approx(13.33, abs=0.01)
        assert result["max_recharge_gap_days"] == pytest.approx(20.0)
        assert result["days_since_last_recharge"] == pytest.approx(0.0)
        assert result["expanding_topup_intervals"] is True

    def test_steady_gaps_are_not_flagged(self):
        features = blank_features(["s2"])
        frames = {"recharge_history.csv": self.frame({"s2": [30, 20, 10, 0]})}
        enrich_features("Telecom", features, self.schema(), frames)
        result = by_entity(features)["s2"]

        assert result["avg_recharge_gap_days"] == pytest.approx(10.0)
        assert result["expanding_topup_intervals"] is False

    def test_two_gaps_are_not_enough_to_establish_a_trend(self):
        """A jump from 1 day to 20 looks dramatic but is a single data point.

        Without the minimum-gap guard this subscriber would be flagged as
        drifting away on the strength of one missed top-up.
        """
        features = blank_features(["s3"])
        frames = {"recharge_history.csv": self.frame({"s3": [21, 20, 0]})}
        enrich_features("Telecom", features, self.schema(), frames)
        result = by_entity(features)["s3"]

        assert result["max_recharge_gap_days"] == pytest.approx(20.0)
        assert result["expanding_topup_intervals"] is False

    def test_postpaid_subscribers_with_no_top_ups_get_the_default(self):
        """Recharges only exist for prepaid, so the flag must still be present."""
        features = blank_features(["s4"])
        frames = {"recharge_history.csv": self.frame({"s1": [10, 0]})}
        enrich_features("Telecom", features, self.schema(), frames)
        result = by_entity(features)["s4"]

        assert result["expanding_topup_intervals"] is False
        assert "recharge_count" not in result


class TestTelecomRegionalNetworkFault:
    def schema(self) -> SchemaMapping:
        return one_table_schema("network_cdrs.csv", "TIME_SERIES_EVENT", "subscriber_id", "call_timestamp")

    def frame(self, calls: Dict[str, List[tuple[str, str]]]) -> pd.DataFrame:
        rows = [(sid, status, tower) for sid, log in calls.items() for status, tower in log]
        return pd.DataFrame({
            "call_id": [f"call_{i}" for i in range(len(rows))],
            "subscriber_id": [r[0] for r in rows],
            "call_timestamp": [ts(1)] * len(rows),
            "call_status": [r[1] for r in rows],
            "tower_id": [r[2] for r in rows],
        })

    def test_a_single_dropped_call_is_not_a_regional_fault(self):
        """One drop is trivially 100% concentrated on one tower.

        Regression guard: before the minimum-volume floor this subscriber
        scored a higher tower concentration than the genuinely faulty one.
        """
        calls = {"s1": [
            ("COMPLETED", "TWR_A"), ("COMPLETED", "TWR_B"), ("COMPLETED", "TWR_A"),
            ("COMPLETED", "TWR_B"), ("DROPPED", "TWR_A"),
        ]}
        features = blank_features(["s1"])
        enrich_features("Telecom", features, self.schema(), {"network_cdrs.csv": self.frame(calls)})
        result = by_entity(features)["s1"]

        assert result["dropped_call_count"] == 1
        assert result["dropped_call_rate"] == pytest.approx(0.2)
        assert result["dominant_tower_share"] == pytest.approx(1.0)
        assert result["regional_network_impact_flag"] is False

    def test_repeated_drops_on_one_tower_are_flagged(self):
        calls = {"s2": [
            ("DROPPED", "TWR_A"), ("DROPPED", "TWR_A"), ("DROPPED", "TWR_A"),
            ("DROPPED", "TWR_A"), ("COMPLETED", "TWR_B"),
        ]}
        features = blank_features(["s2"])
        enrich_features("Telecom", features, self.schema(), {"network_cdrs.csv": self.frame(calls)})
        result = by_entity(features)["s2"]

        assert result["dropped_call_count"] == 4
        assert result["dropped_call_rate"] == pytest.approx(0.8)
        assert result["dominant_tower"] == "TWR_A"
        assert result["regional_network_impact_flag"] is True

    def test_drops_spread_across_towers_are_not_a_local_fault(self):
        calls = {"s3": [
            ("DROPPED", "TWR_A"), ("DROPPED", "TWR_B"), ("DROPPED", "TWR_A"), ("DROPPED", "TWR_B"),
            ("COMPLETED", "TWR_A"), ("COMPLETED", "TWR_B"),
            ("COMPLETED", "TWR_A"), ("COMPLETED", "TWR_B"),
        ]}
        features = blank_features(["s3"])
        enrich_features("Telecom", features, self.schema(), {"network_cdrs.csv": self.frame(calls)})
        result = by_entity(features)["s3"]

        assert result["dropped_call_count"] == 4
        assert result["dominant_tower_share"] == pytest.approx(0.5)
        assert result["regional_network_impact_flag"] is False

    def test_subscribers_with_no_drops_get_the_defaults(self):
        calls = {
            "s4": [("COMPLETED", "TWR_A"), ("COMPLETED", "TWR_B")],
            # The outcome column is identified by failure vocabulary across the
            # whole table, so a second subscriber has to actually drop a call.
            "s5": [("DROPPED", "TWR_A"), ("COMPLETED", "TWR_B")],
        }
        features = blank_features(["s4", "s5"])
        enrich_features("Telecom", features, self.schema(), {"network_cdrs.csv": self.frame(calls)})
        result = by_entity(features)["s4"]

        assert result["dropped_call_count"] == 0
        assert result["dropped_call_rate"] == pytest.approx(0.0)
        assert result["dominant_tower_share"] == 0.0
        assert result["regional_network_impact_flag"] is False

    def test_a_cdr_table_with_no_failure_vocabulary_invents_nothing(self):
        """No dropped calls anywhere means no evidence, so no rate is asserted."""
        calls = {"s6": [("COMPLETED", "TWR_A"), ("COMPLETED", "TWR_B")]}
        features = blank_features(["s6"])
        enrich_features("Telecom", features, self.schema(), {"network_cdrs.csv": self.frame(calls)})
        result = by_entity(features)["s6"]

        assert "dropped_call_rate" not in result
        # The sector defaults still apply, so the feature shape stays stable.
        assert result["dominant_tower_share"] == 0.0
        assert result["regional_network_impact_flag"] is False


class TestFinTechBalanceDrain:
    def schema(self) -> SchemaMapping:
        return one_table_schema("ledger_transactions.csv", "TRANSACTIONAL", "account_id", "timestamp")

    def frame(self, ledgers: Dict[str, List[tuple[float, float, str, str]]]) -> pd.DataFrame:
        rows = [(aid, *entry) for aid, entries in ledgers.items() for entry in entries]
        return pd.DataFrame({
            "tx_id": [f"tx_{i}" for i in range(len(rows))],
            "account_id": [r[0] for r in rows],
            "timestamp": [ts(r[1]) for r in rows],
            "amount": [r[2] for r in rows],
            "tx_type": [r[3] for r in rows],
            "status": [r[4] for r in rows],
        })

    def test_drain_is_outflow_over_total_settled_flow(self):
        ledgers = {"a1": [(0, 100.0, "DEPOSIT", "SUCCESS"), (1, 300.0, "WITHDRAWAL", "SUCCESS")]}
        features = blank_features(["a1"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": self.frame(ledgers)})
        result = by_entity(features)["a1"]

        assert result["inflow_recent"] == pytest.approx(100.0)
        assert result["outflow_recent"] == pytest.approx(300.0)
        assert result["settled_count_recent"] == 2
        assert result["balance_drain_ratio"] == pytest.approx(0.75)
        assert result["rapid_balance_drain"] is True
        assert result["withdrawal_share_recent"] == pytest.approx(0.5)

    def test_a_lone_withdrawal_does_not_trip_the_flag(self):
        """A 1.0 drain ratio from one transaction is small-sample noise."""
        ledgers = {"a2": [(0, 500.0, "WITHDRAWAL", "SUCCESS")]}
        features = blank_features(["a2"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": self.frame(ledgers)})
        result = by_entity(features)["a2"]

        assert result["balance_drain_ratio"] == pytest.approx(1.0)
        assert result["settled_count_recent"] == 1
        assert result["rapid_balance_drain"] is False

    def test_peer_transfers_are_balance_neutral(self):
        """The ledger records no direction for P2P, so guessing would overstate drain.

        Treating the 900 transfer as outflow would give a drain of 0.9.
        """
        ledgers = {"a3": [(0, 100.0, "DEPOSIT", "SUCCESS"), (1, 900.0, "P2P", "SUCCESS")]}
        features = blank_features(["a3"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": self.frame(ledgers)})
        result = by_entity(features)["a3"]

        assert result["inflow_recent"] == pytest.approx(100.0)
        assert result["outflow_recent"] == pytest.approx(0.0)
        assert result["balance_drain_ratio"] == pytest.approx(0.0)
        assert result["rapid_balance_drain"] is False
        assert result["p2p_count"] == 1

    def test_failed_transactions_are_excluded_from_settled_flow(self):
        ledgers = {"a4": [(0, 100.0, "DEPOSIT", "SUCCESS"), (1, 300.0, "WITHDRAWAL", "FAILED")]}
        features = blank_features(["a4"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": self.frame(ledgers)})
        result = by_entity(features)["a4"]

        assert result["settled_count_recent"] == 1
        assert result["outflow_recent"] == pytest.approx(0.0)
        assert result["balance_drain_ratio"] == pytest.approx(0.0)
        # The withdrawal still counts towards the share of attempted activity.
        assert result["withdrawal_share_recent"] == pytest.approx(0.5)

    def test_transactions_outside_the_drain_window_are_ignored(self):
        ledgers = {"a5": [(0, 100.0, "DEPOSIT", "SUCCESS"), (40, 500.0, "WITHDRAWAL", "SUCCESS")]}
        features = blank_features(["a5"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": self.frame(ledgers)})
        result = by_entity(features)["a5"]

        assert result["settled_count_recent"] == 1
        assert result["outflow_recent"] == pytest.approx(0.0)
        assert result["balance_drain_ratio"] == pytest.approx(0.0)


class TestFinTechP2PFailureStreak:
    def schema(self) -> SchemaMapping:
        return one_table_schema("ledger_transactions.csv", "TRANSACTIONAL", "account_id", "timestamp")

    def test_streak_is_the_longest_run_not_the_total(self):
        """Ordered oldest-first: pass, fail, fail, pass, fail, fail, fail."""
        rows = [
            (7, 50.0, "P2P", "SUCCESS"),
            (6, 50.0, "P2P", "FAILED"),
            (5, 50.0, "P2P", "FAILED"),
            (4, 50.0, "P2P", "SUCCESS"),
            (3, 50.0, "P2P", "FAILED"),
            (2, 50.0, "P2P", "FAILED"),
            (1, 50.0, "P2P", "FAILED"),
        ]
        frame = pd.DataFrame({
            "tx_id": [f"tx_{i}" for i in range(len(rows))],
            "account_id": ["a1"] * len(rows),
            "timestamp": [ts(r[0]) for r in rows],
            "amount": [r[1] for r in rows],
            "tx_type": [r[2] for r in rows],
            "status": [r[3] for r in rows],
        })
        features = blank_features(["a1"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": frame})
        result = by_entity(features)["a1"]

        assert result["p2p_count"] == 7
        assert result["p2p_failure_streak"] == 3
        assert result["p2p_failure_rate"] == pytest.approx(5 / 7, abs=1e-4)

    def test_streak_counts_in_chronological_order_not_file_order(self):
        """The same seven rows written newest-first must give the same streak."""
        rows = [
            (1, 50.0, "P2P", "FAILED"),
            (2, 50.0, "P2P", "FAILED"),
            (3, 50.0, "P2P", "FAILED"),
            (4, 50.0, "P2P", "SUCCESS"),
            (5, 50.0, "P2P", "FAILED"),
            (6, 50.0, "P2P", "FAILED"),
            (7, 50.0, "P2P", "SUCCESS"),
        ]
        frame = pd.DataFrame({
            "tx_id": [f"tx_{i}" for i in range(len(rows))],
            "account_id": ["a1"] * len(rows),
            "timestamp": [ts(r[0]) for r in rows],
            "amount": [r[1] for r in rows],
            "tx_type": [r[2] for r in rows],
            "status": [r[3] for r in rows],
        })
        features = blank_features(["a1"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": frame})

        assert by_entity(features)["a1"]["p2p_failure_streak"] == 3

    def test_accounts_with_no_peer_transfers_get_no_p2p_keys(self):
        frame = pd.DataFrame({
            "tx_id": ["tx_0"],
            "account_id": ["a2"],
            "timestamp": [ts(0)],
            "amount": [10.0],
            "tx_type": ["DEPOSIT"],
            "status": ["SUCCESS"],
        })
        features = blank_features(["a2"])
        enrich_features("FinTech", features, self.schema(), {"ledger_transactions.csv": frame})

        assert "p2p_failure_streak" not in by_entity(features)["a2"]


class TestEnricherBoundaries:
    def test_unknown_sector_leaves_features_untouched(self):
        schema = one_table_schema("events_log.csv", "TIME_SERIES_EVENT", "user_id", "timestamp")
        features = blank_features(["u1"], seed={"event_count_7d": 3})

        returned = enrich_features("Retail", features, schema, {})

        assert returned is features
        assert by_entity(features)["u1"] == {"event_count_7d": 3}

    def test_empty_feature_list_is_not_an_error(self):
        schema = one_table_schema("events_log.csv", "TIME_SERIES_EVENT", "user_id", "timestamp")

        assert enrich_features("SaaS", [], schema, {}) == []

    def test_a_missing_column_degrades_to_the_generic_primitives(self):
        """Enrichment is additive; a ragged upload must not lose the whole run."""
        schema = one_table_schema("events_log.csv", "TIME_SERIES_EVENT", "user_id", "timestamp")
        frame = pd.DataFrame({
            "event_id": ["evt_0", "evt_1"],
            "user_id": ["u1", "u1"],
            "timestamp": [ts(0), ts(1)],
            # No event_type column at all, so there is nothing to mine.
        })
        features = blank_features(["u1"], seed={"event_count_7d": 2})
        enrich_features("SaaS", features, schema, {"events_log.csv": frame})
        result = by_entity(features)["u1"]

        assert result["event_count_7d"] == 2
        assert "export_ratio" not in result

    def test_a_table_named_in_the_schema_but_not_uploaded_is_skipped(self):
        schema = one_table_schema("events_log.csv", "TIME_SERIES_EVENT", "user_id", "timestamp")
        features = blank_features(["u1"], seed={"event_count_7d": 2})

        enrich_features("SaaS", features, schema, {})

        assert by_entity(features)["u1"] == {"event_count_7d": 2}
