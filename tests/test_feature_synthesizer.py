"""Feature synthesis tests using hand-built frames with hand-computable values.

These deliberately avoid the generated mock CSVs so a change to the data
generator cannot silently mask a regression in the feature math.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn_platform.domain.models.schema_mapping import SchemaMapping, TableClassification
from churn_platform.infrastructure.parsers.feature_synthesizer import (
    VELOCITY_EPSILON,
    WEEKS_IN_PRIOR_WINDOW,
    PandasFeatureSynthesizer,
    table_stem,
    to_naive_utc,
)

REFERENCE = pd.Timestamp("2025-06-01 12:00:00")


def ts(days_ago: float) -> str:
    return (REFERENCE - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def plain_date(days_ago: float) -> str:
    return (REFERENCE - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")


def by_entity(features) -> dict:
    return {f.entity_id: f.features for f in features}


def events_frame(rows: list[tuple[str, float]], key: str = "user_id") -> pd.DataFrame:
    """Build an event log from (entity_id, days_ago) pairs."""
    return pd.DataFrame({
        "event_id": [f"evt_{i}" for i in range(len(rows))],
        key: [entity for entity, _ in rows],
        "timestamp": [ts(days) for _, days in rows],
        "event_type": ["login"] * len(rows),
        "session_hash": [f"sess_{i}" for i in range(len(rows))],
    })


def events_schema(key: str = "user_id", noise=("session_hash",)) -> SchemaMapping:
    return SchemaMapping(primary_entity_key=key, tables=[
        TableClassification(
            file_name="events_log.csv", role="TIME_SERIES_EVENT", primary_entity_key=key,
            timestamp_column="timestamp", noise_columns=list(noise)),
    ])


def synthesize(schema, frames):
    return PandasFeatureSynthesizer().synthesize(schema, frames)


class TestActivityVelocity:
    def test_matches_the_specified_formula(self):
        """10 recent events against 20 in the prior window is exactly 2.1429."""
        rows = [("u1", d) for d in [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]]
        rows += [("u1", d) for d in range(8, 28)]
        features = by_entity(synthesize(events_schema(), {"events_log.csv": events_frame(rows)}))

        expected = 10 / (20 / WEEKS_IN_PRIOR_WINDOW)
        assert features["u1"]["event_count_7d"] == 10
        assert features["u1"]["event_count_prior_30d"] == 20
        assert features["u1"]["activity_velocity"] == pytest.approx(expected, abs=1e-4)
        assert features["u1"]["activity_velocity"] == pytest.approx(2.1429, abs=1e-4)

    def test_zero_prior_activity_floors_at_the_epsilon(self):
        """A dormant-then-active customer must surge, not divide by zero."""
        rows = [("u1", d) for d in range(0, 5)]
        features = by_entity(synthesize(events_schema(), {"events_log.csv": events_frame(rows)}))

        assert features["u1"]["event_count_prior_30d"] == 0
        assert features["u1"]["activity_velocity"] == pytest.approx(5 / VELOCITY_EPSILON)
        assert features["u1"]["activity_velocity"] == pytest.approx(10.0)

    def test_fully_dormant_entity_scores_zero(self):
        """An entity with no events at all is zero, not undefined."""
        users = pd.DataFrame({"user_id": ["u1", "u2"], "tier": ["Pro", "Basic"]})
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(file_name="users.csv", role="DIMENSION", primary_entity_key="user_id"),
            TableClassification(
                file_name="events_log.csv", role="TIME_SERIES_EVENT", primary_entity_key="user_id",
                timestamp_column="timestamp", noise_columns=["session_hash"]),
        ])
        frames = {
            "users.csv": users,
            # Fixed reference: offsets stay 1, 8, 9, 10, 11 and 12 days.
            # Dormancy spans the oldest event through the reference date.
            "events_log.csv": events_frame([("u1", d) for d in (1, 8, 9, 10, 11, 12)]),
        }
        features = by_entity(synthesize(schema, frames))

        assert features["u1"]["event_count_7d"] == 1
        assert features["u1"]["event_count_prior_30d"] == 5
        assert features["u1"]["activity_velocity"] == pytest.approx(1 / (5 / WEEKS_IN_PRIOR_WINDOW), abs=1e-4)
        assert features["u2"]["event_count_7d"] == 0
        assert features["u2"]["event_count_total"] == 0
        assert features["u2"]["activity_velocity"] == 0.0
        # No recorded activity means dormant for the whole observed span.
        assert features["u2"]["recency_days"] == pytest.approx(12.0)

    def test_events_outside_the_prior_window_are_excluded(self):
        """Activity older than 37 days must not inflate the baseline."""
        rows = [("u1", 0), ("u1", 40), ("u1", 55)]
        features = by_entity(synthesize(events_schema(), {"events_log.csv": events_frame(rows)}))

        assert features["u1"]["event_count_total"] == 3
        assert features["u1"]["event_count_prior_30d"] == 0
        assert features["u1"]["event_count_7d"] == 1


class TestFailureRatios:
    def invoice_frame(self, statuses: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "inv_id": [f"inv_{i}" for i in range(len(statuses))],
            "user_id": ["u1"] * len(statuses),
            "amount": [99.99] * len(statuses),
            "status": statuses,
            "due_date": [plain_date(5)] * len(statuses),
        })

    def invoice_schema(self) -> SchemaMapping:
        return SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(
                file_name="invoices.csv", role="TRANSACTIONAL", primary_entity_key="user_id",
                timestamp_column="due_date"),
        ])

    def test_failed_payment_rate(self):
        statuses = ["FAILED", "PAID", "FAILED", "PAID", "PAID", "FAILED", "PAID", "PAID", "PAID", "PAID"]
        features = by_entity(synthesize(
            self.invoice_schema(), {"invoices.csv": self.invoice_frame(statuses)}))

        assert features["u1"]["invoices_failed_count"] == 3
        assert features["u1"]["invoices_failure_rate"] == pytest.approx(0.3)
        assert features["u1"]["failure_rate"] == pytest.approx(0.3)

    def test_failure_vocabulary_beyond_failed(self):
        """DROPPED and DECLINED are failures even though they are not 'FAILED'."""
        statuses = ["DROPPED", "COMPLETED", "DECLINED", "COMPLETED"]
        features = by_entity(synthesize(
            self.invoice_schema(), {"invoices.csv": self.invoice_frame(statuses)}))

        assert features["u1"]["invoices_failure_rate"] == pytest.approx(0.5)

    def test_workflow_state_column_is_not_treated_as_failure(self):
        """OPEN/CLOSED is a status by name but not an outcome, so it must be ignored."""
        tickets = pd.DataFrame({
            "ticket_id": ["t1", "t2"],
            "user_id": ["u1", "u1"],
            "subject": ["How do I invite a team member", "Where are the API docs"],
            "status": ["OPEN", "CLOSED"],
        })
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(
                file_name="tickets.csv", role="UNSTRUCTURED_TEXT", primary_entity_key="user_id"),
        ])
        features = by_entity(synthesize(schema, {"tickets.csv": tickets}))

        assert "tickets_failure_rate" not in features["u1"]
        assert "failure_rate" not in features["u1"]

    def test_amount_column_is_summed(self):
        frame = self.invoice_frame(["PAID", "PAID", "FAILED"])
        features = by_entity(synthesize(self.invoice_schema(), {"invoices.csv": frame}))

        assert features["u1"]["invoices_amount_total"] == pytest.approx(299.97)
        assert features["u1"]["invoices_amount_column"] == "amount"


class TestTemporalAnchor:
    def test_future_dated_column_does_not_shift_the_window(self):
        """A due date next week must not become 'now' and flatten recent activity.

        Regression guard: an earlier anchor implementation name-sniffed every
        column containing 'date', picked up future-dated invoices, and reported
        zero recent events for every customer.
        """
        invoices = pd.DataFrame({
            "inv_id": ["inv_1", "inv_2"],
            "user_id": ["u1", "u1"],
            "amount": [99.99, 49.99],
            "status": ["PAID", "FAILED"],
            "due_date": [ts(-15), ts(3)],  # one 15 days out, one 3 days back
        })
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(
                file_name="events_log.csv", role="TIME_SERIES_EVENT", primary_entity_key="user_id",
                timestamp_column="timestamp"),
            TableClassification(
                file_name="invoices.csv", role="TRANSACTIONAL", primary_entity_key="user_id",
                timestamp_column="due_date"),
        ])
        frames = {
            "events_log.csv": events_frame([("u1", 0), ("u1", 1), ("u1", 2)]),
            "invoices.csv": invoices,
        }
        features = by_entity(synthesize(schema, frames))

        assert features["u1"]["event_count_7d"] == 3
        assert features["u1"]["recency_days"] == pytest.approx(0.0, abs=0.05)
        # The future invoice is a real record but not recent activity.
        assert features["u1"]["invoices_count_total"] == 2
        assert features["u1"]["invoices_count_7d"] == 1
        assert features["u1"]["invoices_recency_days"] == 0.0

    def test_mixed_tz_aware_and_naive_dates_are_comparable(self):
        """Exports interleave ISO-8601 Z instants with plain dates."""
        frame = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "user_id": ["u1", "u1"],
            "timestamp": ["2025-06-01T12:00:00Z", "2025-05-30"],
        })
        parsed = to_naive_utc(frame["timestamp"])

        assert parsed.notna().all()
        assert parsed.dt.tz is None
        assert (parsed.max() - parsed.min()).days == 2

    def test_unparseable_dates_coerce_to_nat_without_raising(self):
        frame = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "user_id": ["u1", "u1"],
            "timestamp": ["not a date", ts(1)],
        })
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(
                file_name="events_log.csv", role="TIME_SERIES_EVENT",
                primary_entity_key="user_id", timestamp_column="timestamp"),
        ])
        features = by_entity(synthesize(schema, {"events_log.csv": frame}))

        assert features["u1"]["event_count_total"] == 1

    def test_table_without_any_timestamp_still_yields_row_counts(self):
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(
                file_name="tickets.csv", role="UNSTRUCTURED_TEXT", primary_entity_key="user_id"),
        ])
        tickets = pd.DataFrame({
            "ticket_id": ["t1", "t2", "t3"],
            "user_id": ["u1", "u1", "u2"],
            "subject": ["Question about billing", "Thanks for the help", "Cancel my plan"],
        })
        features = by_entity(synthesize(schema, {"tickets.csv": tickets}))

        assert features["u1"]["tickets_row_count"] == 2
        assert features["u2"]["tickets_row_count"] == 1
        assert "activity_velocity" not in features["u1"]


class TestNoiseAndMutation:
    def test_noise_columns_never_reach_the_feature_vector(self):
        users = pd.DataFrame({
            "user_id": ["u1"],
            "tier": ["Pro"],
            "ip_address": ["10.0.0.1"],
            "last_user_agent": ["Mozilla/5.0"],
        })
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(
                file_name="users.csv", role="DIMENSION", primary_entity_key="user_id",
                noise_columns=["ip_address", "last_user_agent"]),
            TableClassification(
                file_name="events_log.csv", role="TIME_SERIES_EVENT", primary_entity_key="user_id",
                timestamp_column="timestamp", noise_columns=["session_hash"]),
        ])
        frames = {
            "users.csv": users,
            "events_log.csv": events_frame([("u1", 1), ("u1", 2)]),
        }
        features = by_entity(synthesize(schema, frames))["u1"]

        assert features["tier"] == "Pro"
        for leaked in ("ip_address", "last_user_agent", "session_hash", "event_id"):
            assert leaked not in features

    def test_caller_dataframes_are_not_mutated(self):
        """Regression guard: noise stripping used to reassign into the caller's dict."""
        frame = events_frame([("u1", 1), ("u1", 2)])
        frames = {"events_log.csv": frame}
        original_columns = list(frame.columns)

        synthesize(events_schema(), frames)

        assert list(frames["events_log.csv"].columns) == original_columns
        assert "session_hash" in frames["events_log.csv"].columns
        assert frames["events_log.csv"] is frame

    def test_schema_referencing_a_missing_file_is_skipped(self):
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(
                file_name="events_log.csv", role="TIME_SERIES_EVENT",
                primary_entity_key="user_id", timestamp_column="timestamp"),
            TableClassification(
                file_name="never_uploaded.csv", role="TRANSACTIONAL", primary_entity_key="user_id"),
        ])
        features = by_entity(synthesize(schema, {"events_log.csv": events_frame([("u1", 1)])}))

        assert features["u1"]["event_count_total"] == 1

    def test_entity_present_only_in_an_event_table_is_still_scored(self):
        users = pd.DataFrame({"user_id": ["u1"], "tier": ["Pro"]})
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(file_name="users.csv", role="DIMENSION", primary_entity_key="user_id"),
            TableClassification(
                file_name="events_log.csv", role="TIME_SERIES_EVENT", primary_entity_key="user_id",
                timestamp_column="timestamp"),
        ])
        frames = {"users.csv": users, "events_log.csv": events_frame([("u1", 1), ("ghost_9", 2)])}
        features = by_entity(synthesize(schema, frames))

        assert "ghost_9" in features
        assert features["ghost_9"]["event_count_total"] == 1
        assert "tier" not in features["ghost_9"]


class TestCanonicalAliases:
    """Canonical names must not depend on the order tables arrive in.

    Regression guard: the resolver sorts tables alphabetically and a browser
    submits files in whatever order the user selected them, so ``card_swipes.csv``
    used to claim ``activity_velocity`` ahead of ``ledger_transactions.csv`` — the
    table with ten times the rows and the real signal. Two tenants uploading the
    same exports got different features from nothing but filename spelling.
    """

    SWIPES = TableClassification(
        file_name="card_swipes.csv", role="TIME_SERIES_EVENT", primary_entity_key="user_id",
        timestamp_column="swipe_timestamp")
    LEDGER = TableClassification(
        file_name="ledger_transactions.csv", role="TRANSACTIONAL", primary_entity_key="user_id",
        timestamp_column="txn_timestamp")

    def frames(self) -> dict:
        return {
            # One sparse swipe at day 0 — which also makes it the temporal anchor.
            "card_swipes.csv": pd.DataFrame({
                "swipe_id": ["sw_1"], "user_id": ["u1"], "swipe_timestamp": [ts(0)],
            }),
            # Ten ledger rows: 7 in days 1-7 and 3 in days 8-10, so the velocity
            # is 7 / (3 / 4.2857) = exactly 10.0 against the swipes' 1 / 0.5 = 2.0.
            "ledger_transactions.csv": pd.DataFrame({
                "txn_id": [f"txn_{i}" for i in range(10)],
                "user_id": ["u1"] * 10,
                "txn_timestamp": [ts(day) for day in range(1, 11)],
            }),
        }

    def schema(self, reverse: bool = False) -> SchemaMapping:
        tables = [self.SWIPES, self.LEDGER]
        return SchemaMapping(
            primary_entity_key="user_id",
            tables=list(reversed(tables)) if reverse else tables,
        )

    def test_the_busier_table_owns_the_canonical_aliases(self):
        features = by_entity(synthesize(self.schema(), self.frames()))["u1"]

        assert features["activity_velocity"] == pytest.approx(10.0, abs=1e-4)
        assert features["card_swipes_velocity"] == pytest.approx(2.0, abs=1e-4)
        assert features["event_count_total"] == 10
        assert features["card_swipes_count_total"] == 1

    def test_reversing_the_schema_order_does_not_change_the_features(self):
        assert (
            by_entity(synthesize(self.schema(), self.frames()))
            == by_entity(synthesize(self.schema(reverse=True), self.frames()))
        )

    def test_a_dimension_table_never_claims_the_failure_alias(self):
        """Row volume alone would hand ``failure_rate`` to a subscriber table.

        Its ``account_status`` column carries lifecycle states like CANCELLED,
        which the failure vocabulary matches. But a dimension is not an activity
        stream, and if it claims the alias first then the dropped-call rate the
        telecom scorer actually reads goes unpublished for every customer.
        """
        accounts = pd.DataFrame({
            "user_id": ["u1"] + [f"sub_{i}" for i in range(19)],
            "account_status": ["ACTIVE"] + ["CANCELLED" if i % 4 == 0 else "ACTIVE" for i in range(19)],
        })
        cdrs = pd.DataFrame({
            "cdr_id": [f"cdr_{i}" for i in range(10)],
            "user_id": ["u1"] * 10,
            "call_timestamp": [ts(day) for day in range(0, 10)],
            "call_status": ["DROPPED"] * 3 + ["COMPLETED"] * 7,
        })
        schema = SchemaMapping(primary_entity_key="user_id", tables=[
            TableClassification(file_name="accounts.csv", role="DIMENSION", primary_entity_key="user_id"),
            TableClassification(
                file_name="network_cdrs.csv", role="TIME_SERIES_EVENT", primary_entity_key="user_id",
                timestamp_column="call_timestamp"),
        ])
        features = by_entity(synthesize(schema, {"accounts.csv": accounts, "network_cdrs.csv": cdrs}))

        # accounts.csv has twice the rows and sorts first; the CDR still wins.
        assert features["u1"]["failure_rate"] == pytest.approx(0.3)
        assert features["u1"]["failure_rate"] == features["u1"]["network_cdrs_failure_rate"]
        assert features["u1"]["accounts_failed_count"] == 0


class TestTableStem:
    @pytest.mark.parametrize("name,expected", [
        ("events_log.csv", "events_log"),
        ("users.csv", "users"),
        ("workbook.xlsx::Sheet One", "sheet_one"),
        ("nested/path/Network CDRs.csv", "network_cdrs"),
        ("...", "table"),
    ])
    def test_stems_are_stable_feature_prefixes(self, name, expected):
        assert table_stem(name) == expected
