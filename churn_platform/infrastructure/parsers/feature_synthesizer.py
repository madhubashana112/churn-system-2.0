"""Sector-agnostic feature synthesis from messy multi-sheet raw exports.

The synthesizer deliberately knows nothing about sectors. It receives a
`SchemaMapping` describing table roles and timestamp columns, and emits generic
behavioural primitives — activity velocity, failure rates, recency, text
sentiment. Sector-specific interpretation (what "velocity collapse" means for a
telco versus a bank) belongs to the cores and enrichers, not here.

Two rules keep this predictable:

* Every aggregate is published under a ``{table_stem}_`` prefixed key, so two
  tables of the same role never collide.
* The first table to actually earn an alias additionally publishes it under a
  canonical name (``activity_velocity``, ``failure_rate``, ``text_churn_score``,
  ...). Those are the names the sector scorers rely on, so which table earns them
  is decided by ``_canonical_table_order`` rather than by the order the files
  happened to arrive in.

All time windows are anchored to REFERENCE_DATE (2025-06-01),
never to the wall clock, so re-running on a frozen dataset is idempotent.
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

import pandas as pd

from churn_platform.domain.interfaces.i_feature_synthesizer import IFeatureSynthesizer
from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.schema_mapping import (
    REFERENCE_DATE,
    ROLE_DIMENSION,
    ROLE_TEXT,
    ROLE_TIME_SERIES,
    ROLE_TRANSACTIONAL,
    SchemaMapping,
    TableClassification,
)
from churn_platform.infrastructure.parsers.text_features import KeywordSentimentScorer

logger = logging.getLogger(__name__)

RECENT_WINDOW_DAYS = 7
PRIOR_WINDOW_DAYS = 37
# Days 8-37 is a 30 day span; dividing its count by this yields a weekly rate.
WEEKS_IN_PRIOR_WINDOW = (PRIOR_WINDOW_DAYS - RECENT_WINDOW_DAYS) / 7.0
# Floor on the weekly baseline. Without it a customer who was dormant last month
# and active this week divides by zero; with it they score as a sharp surge.
VELOCITY_EPSILON = 0.5
DEFAULT_SPAN_DAYS = 90.0

FAILURE_VALUES = frozenset({
    "FAILED", "FAILURE", "DECLINED", "DROPPED", "ERROR", "REJECTED",
    "CANCELLED", "CANCELED", "UNPAID", "OVERDUE", "REFUNDED", "CHARGEBACK",
})

STATUS_NAME_HINTS = ("status", "state", "result", "outcome")
AMOUNT_NAME_HINTS = ("amount", "total", "value", "price", "fee", "balance", "charge")


def to_naive_utc(series: pd.Series) -> pd.Series:
    """Parse a mixed timestamp/date column into comparable tz-naive datetimes.

    Raw exports interleave ISO-8601 ``...Z`` instants (parsed tz-aware) with
    plain ``YYYY-MM-DD`` dates (parsed tz-naive). Pandas refuses to compare the
    two, so everything is normalised onto a single naive-UTC basis.

    ISO-8601 is tried first because it is fast and unambiguous; if it parses
    nothing at all the column is genuinely messy, so fall back to permissive
    inference rather than collapsing every value to NaT.
    """
    source = series.dropna()
    parsed = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    if not source.empty and parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def table_stem(file_name: str) -> str:
    """Derive a stable feature-name prefix from an uploaded file or sheet name."""
    stem = PurePosixPath(file_name.replace("::", "/")).stem
    return re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower() or "table"


def detect_status_column(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    """Find the outcome column by name first, then by failure vocabulary."""
    for col in df.columns:
        if col in exclude or not any(hint in col.lower() for hint in STATUS_NAME_HINTS):
            continue
        return col
    for col in df.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = set(df[col].dropna().astype(str).str.upper().unique())
        # Low cardinality plus a known failure token is a strong signal.
        if 0 < len(values) <= 20 and values & FAILURE_VALUES:
            return col
    return None


def detect_amount_column(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    for col in df.columns:
        if col in exclude:
            continue
        if any(hint in col.lower() for hint in AMOUNT_NAME_HINTS) and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def resolve_temporal_anchor(
    schema: SchemaMapping,
    dataframes: Dict[str, pd.DataFrame],
) -> tuple[Optional[pd.Timestamp], float]:
    """Use the fixed 1 June 2025 reference, shared with sector enrichers.

    Activity tables determine the observed span first, then transactions.
    Future timestamps never move the reference date forward.
    """
    for role in (ROLE_TIME_SERIES, ROLE_TRANSACTIONAL):
        stamps = []
        for table in schema.tables:
            if table.role != role or not table.timestamp_column:
                continue
            df = dataframes.get(table.file_name)
            if df is None or table.timestamp_column not in df.columns:
                continue
            parsed = to_naive_utc(df[table.timestamp_column])
            if parsed.notna().any():
                stamps.append(parsed)
        if not stamps:
            continue
        combined = pd.concat(stamps)
        reference_ts = pd.Timestamp(REFERENCE_DATE).tz_localize(None)
        span = (reference_ts - combined.min()).total_seconds() / 86400
        return reference_ts, float(span) if span > 0 else DEFAULT_SPAN_DAYS
    return None, DEFAULT_SPAN_DAYS


# Tiebreak only: two tables with the same row count are separated by how
# activity-like their role is, then by name, so the result never depends on the
# order a dict or an upload happened to produce.
_ROLE_PRIORITY = {ROLE_TIME_SERIES: 0, ROLE_TRANSACTIONAL: 1, ROLE_TEXT: 2, ROLE_DIMENSION: 3}


def _canonical_table_order(
    tables: List[TableClassification],
    prepared: Dict[str, pd.DataFrame],
) -> List[TableClassification]:
    """Order tables so canonical aliases land on the primary activity stream.

    The order tables arrive in is an artefact — the resolver sorts by filename,
    and a browser submits files in whatever order the user selected them. Since
    canonical names go to the first table that earns them, that made
    ``activity_velocity`` mean something different for two tenants who uploaded
    the same four exports: ``card_swipes.csv`` sorts ahead of
    ``ledger_transactions.csv`` and quietly steals the alias.

    Row volume is the intrinsic measure of which table *is* the activity stream.
    A bank's ledger has far more rows per customer than its card swipes, and it
    is where the churn signal lives; an event log outranks an invoice table for
    the same reason.
    """
    return sorted(
        tables,
        key=lambda t: (
            t.role == ROLE_DIMENSION,
            -len(prepared.get(t.file_name, ())),
            _ROLE_PRIORITY.get(t.role, len(_ROLE_PRIORITY)),
            t.file_name,
        ),
    )


class _AliasTracker:
    """Hands out canonical feature names to the first table that earns them.

    Binding canonical names to table roles is too rigid: a FinTech ledger is
    classified TRANSACTIONAL yet is the real activity stream, so role-based
    aliasing would leave ``activity_velocity`` unpublished. Claiming on first
    actual use keeps the canonical names stable across sectors — and
    ``_canonical_table_order`` keeps "first" stable across uploads.
    """

    def __init__(self) -> None:
        self._claimed: set[str] = set()

    def claim(self, kind: str) -> bool:
        if kind in self._claimed:
            return False
        self._claimed.add(kind)
        return True


class PandasFeatureSynthesizer(IFeatureSynthesizer):
    """Joins raw multi-sheet exports on the resolved key and computes primitives."""

    def __init__(self, sentiment_scorer: Optional[KeywordSentimentScorer] = None) -> None:
        self.sentiment_scorer = sentiment_scorer or KeywordSentimentScorer()

    def synthesize(
        self,
        schema: SchemaMapping,
        dataframes: Dict[str, pd.DataFrame],
    ) -> List[CustomerFeatures]:
        schema, dataframes = self.prepare(schema, dataframes)
        prepared = self._prepare(schema, dataframes)
        if not prepared:
            logger.warning("No usable tables found in schema mapping; returning no features")
            return []

        reference_ts, span_days = resolve_temporal_anchor(schema, prepared)
        entity_ids = self._collect_entity_ids(schema, prepared)
        features: Dict[str, Dict[str, Any]] = {eid: {} for eid in entity_ids}
        aliases = _AliasTracker()

        for table in _canonical_table_order(schema.tables, prepared):
            if table.file_name not in prepared:
                continue
            df = prepared[table.file_name]
            prefix = f"{table_stem(table.file_name)}_"

            self._apply_dimension(table, df, features)
            self._apply_custom(table, df, features)
            self._apply_row_count(prefix, table, df, features)
            # A dimension table's date is a signup/creation date, not activity,
            # so deriving a velocity from it would be noise.
            if table.role != ROLE_DIMENSION and table.timestamp_column and reference_ts is not None:
                self._apply_temporal(prefix, aliases, table, df, features, reference_ts, span_days)
            self._apply_failures(prefix, aliases, table, df, features)
            self._apply_amounts(prefix, table, df, features)
            if table.role == ROLE_TEXT:
                self._apply_text(prefix, aliases, table, df, features)

        return [
            CustomerFeatures(entity_id=eid, features={k: _round(v) for k, v in payload.items()})
            for eid, payload in features.items()
        ]

    def prepare(self, schema, dataframes):
        """Apply confirmed names to copies, sharing the same schema with enrichers."""
        runtime = schema.model_copy(deep=True)
        frames = {name: df.copy() for name, df in dataframes.items()}
        for table in runtime.tables:
            if not table.columns or table.file_name not in frames:
                continue
            df = frames[table.file_name]
            drops = [c.source_column for c in table.columns if c.canonical_role == "NOISE_IGNORE"]
            kept = [c for c in table.columns if c.canonical_role != "NOISE_IGNORE"]
            renames = {c.source_column: c.target_name for c in kept}
            if len(set(renames.values())) != len(renames):
                raise ValueError("Mapped column names must be unique within each table")
            frames[table.file_name] = df.drop(columns=drops, errors="ignore").rename(columns=renames)
            table.primary_entity_key = renames.get(table.primary_entity_key, table.primary_entity_key)
            table.timestamp_column = renames.get(table.timestamp_column, table.timestamp_column)
            for column in kept:
                column.source_column = column.target_name
            table.columns = kept
            table.noise_columns = []
        runtime.primary_entity_key = next((t.primary_entity_key for t in runtime.tables), runtime.primary_entity_key)
        return runtime, frames

    def _apply_custom(self, table, df, features):
        key = table.primary_entity_key
        if key not in df.columns:
            return
        for column in table.columns:
            if column.canonical_role != "CUSTOM" or column.custom_label not in df.columns:
                continue
            for entity, group in df.groupby(key):
                payload = features.get(str(entity))
                if payload is None:
                    continue
                series = group[column.custom_label].dropna()
                if pd.api.types.is_numeric_dtype(series):
                    summary = {"mean": float(series.mean()), "sum": float(series.sum())} if len(series) else None
                else:
                    summary = list(dict.fromkeys(series.astype(str)))[:3]
                payload.setdefault("custom_metrics", {}).setdefault(table.file_name, {})[column.custom_label] = summary

    # -- preparation ---------------------------------------------------------

    def _prepare(
        self,
        schema: SchemaMapping,
        dataframes: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """Copy the caller's frames and strip noise columns.

        Copies matter: an earlier implementation reassigned into the caller's
        dict, silently deleting columns from a dataframe the caller still owned.
        """
        prepared: Dict[str, pd.DataFrame] = {}
        for table in schema.tables:
            df = dataframes.get(table.file_name)
            if df is None:
                logger.warning("Schema references %r but it was not uploaded; skipping", table.file_name)
                continue
            df = df.copy()
            drop = [c for c in table.noise_columns if c in df.columns]
            if drop:
                df = df.drop(columns=drop)
            prepared[table.file_name] = df
        return prepared

    def _collect_entity_ids(
        self,
        schema: SchemaMapping,
        prepared: Dict[str, pd.DataFrame],
    ) -> List[str]:
        """Union of every entity id seen in any table, in first-seen order.

        Seeding from the union rather than the dimension table means an entity
        that only appears in an event log is still scored.
        """
        ordered: Dict[str, None] = {}
        for table in schema.tables:
            df = prepared.get(table.file_name)
            if df is None:
                continue
            key = table.primary_entity_key or schema.primary_entity_key
            if key not in df.columns:
                logger.warning("%r has no join key %r; its entities will be skipped", table.file_name, key)
                continue
            for value in df[key].dropna().astype(str):
                ordered.setdefault(value, None)
        return list(ordered)

    # -- per-role aggregation ------------------------------------------------

    def _apply_dimension(
        self,
        table: TableClassification,
        df: pd.DataFrame,
        features: Dict[str, Dict[str, Any]],
    ) -> None:
        if table.role != ROLE_DIMENSION:
            return
        key = table.primary_entity_key
        if key not in df.columns:
            return
        custom_names = {c.custom_label for c in table.columns if c.canonical_role == "CUSTOM"}
        attribute_columns = [c for c in df.columns if c != key and c != table.timestamp_column
                             and c not in custom_names and c != "custom_metrics"]
        for row in df[[key, *attribute_columns]].to_dict("records"):
            payload = features.get(str(row[key]))
            if payload is None:
                continue
            for col in attribute_columns:
                value = row[col]
                payload[col] = None if pd.isna(value) else value

    def _apply_row_count(
        self,
        prefix: str,
        table: TableClassification,
        df: pd.DataFrame,
        features: Dict[str, Dict[str, Any]],
    ) -> None:
        key = table.primary_entity_key
        if key not in df.columns:
            return
        for entity_id, count in df.groupby(key).size().items():
            payload = features.get(str(entity_id))
            if payload is not None:
                payload[f"{prefix}row_count"] = int(count)

    def _apply_temporal(
        self,
        prefix: str,
        aliases: _AliasTracker,
        table: TableClassification,
        df: pd.DataFrame,
        features: Dict[str, Dict[str, Any]],
        reference_ts: pd.Timestamp,
        span_days: float,
    ) -> None:
        key = table.primary_entity_key
        ts_col = table.timestamp_column
        if key not in df.columns or ts_col not in df.columns:
            return

        work = df[[key]].copy()
        work["_ts"] = to_naive_utc(df[ts_col])
        work["_days_ago"] = (reference_ts - work["_ts"]).dt.total_seconds() / 86400
        # Rows dated after the anchor are real records (an unpaid invoice due
        # next week) but they are not recent activity, and counting them pushes
        # recency negative. The prior window is already bounded below by > 7.
        work["_recent"] = work["_days_ago"].between(0, RECENT_WINDOW_DAYS).fillna(False)
        work["_prior"] = (
            (work["_days_ago"] > RECENT_WINDOW_DAYS) & (work["_days_ago"] <= PRIOR_WINDOW_DAYS)
        ).fillna(False)
        work = work.dropna(subset=["_ts"])
        if work.empty:
            return
        canonical = aliases.claim("temporal")

        grouped = work.groupby(key).agg(
            count_7d=("_recent", "sum"),
            count_prior_30d=("_prior", "sum"),
            count_total=("_ts", "size"),
            last_ts=("_ts", "max"),
        )
        baseline = (grouped["count_prior_30d"] / WEEKS_IN_PRIOR_WINDOW).clip(lower=VELOCITY_EPSILON)
        grouped["velocity"] = grouped["count_7d"] / baseline
        grouped["recency_days"] = (
            (reference_ts - grouped["last_ts"]).dt.total_seconds() / 86400
        ).clip(lower=0.0)

        for entity_id, row in grouped.iterrows():
            payload = features.get(str(entity_id))
            if payload is None:
                continue
            payload[f"{prefix}count_7d"] = int(row["count_7d"])
            payload[f"{prefix}count_prior_30d"] = int(row["count_prior_30d"])
            payload[f"{prefix}count_total"] = int(row["count_total"])
            payload[f"{prefix}velocity"] = float(row["velocity"])
            payload[f"{prefix}recency_days"] = float(row["recency_days"])
            if not canonical:
                continue
            payload["event_count_7d"] = int(row["count_7d"])
            payload["event_count_prior_30d"] = int(row["count_prior_30d"])
            payload["event_count_total"] = int(row["count_total"])
            payload["activity_velocity"] = float(row["velocity"])
            payload["recency_days"] = float(row["recency_days"])

        # Entities absent from this table are dormant for the whole span.
        for payload in features.values():
            payload.setdefault(f"{prefix}count_7d", 0)
            payload.setdefault(f"{prefix}count_prior_30d", 0)
            payload.setdefault(f"{prefix}count_total", 0)
            payload.setdefault(f"{prefix}velocity", 0.0)
            payload.setdefault(f"{prefix}recency_days", span_days)
            if canonical:
                payload.setdefault("event_count_7d", 0)
                payload.setdefault("event_count_prior_30d", 0)
                payload.setdefault("event_count_total", 0)
                payload.setdefault("activity_velocity", 0.0)
                payload.setdefault("recency_days", span_days)

    def _apply_failures(
        self,
        prefix: str,
        aliases: _AliasTracker,
        table: TableClassification,
        df: pd.DataFrame,
        features: Dict[str, Dict[str, Any]],
    ) -> None:
        """Compute failure ratios for any table carrying an outcome column.

        Not restricted to TRANSACTIONAL: a telecom CDR is a TIME_SERIES_EVENT
        whose dropped-call rate is one of the strongest churn signals there is.
        """
        key = table.primary_entity_key
        if key not in df.columns:
            return
        exclude = {key, table.timestamp_column} | set(table.noise_columns)
        status_col = detect_status_column(df, {c for c in exclude if c})
        if status_col is None:
            return

        outcomes = df[status_col].astype(str).str.upper()
        # A column named "status" is not automatically an outcome column: support
        # tickets carry OPEN/CLOSED workflow states, and scoring those as failures
        # would emit a constant zero for every entity.
        if not outcomes.isin(FAILURE_VALUES).any():
            return
        canonical = aliases.claim("failure")

        work = df[[key]].copy()
        work["_failed"] = outcomes.isin(FAILURE_VALUES)
        grouped = work.groupby(key)["_failed"].agg(failed_count="sum", total="size")
        grouped["failure_rate"] = grouped["failed_count"] / grouped["total"]

        for entity_id, row in grouped.iterrows():
            payload = features.get(str(entity_id))
            if payload is None:
                continue
            payload[f"{prefix}failed_count"] = int(row["failed_count"])
            payload[f"{prefix}failure_rate"] = float(row["failure_rate"])
            payload[f"{prefix}status_column"] = status_col
            if canonical:
                payload["failed_transaction_count"] = int(row["failed_count"])
                payload["failure_rate"] = float(row["failure_rate"])

        # An entity with no rows here has no failures, which is a fact rather
        # than an unknown. The rate is left unset on purpose: zero failures over
        # zero records is undefined, and what a missing rate contributes to a
        # score is the scorer's decision, not the synthesizer's.
        for payload in features.values():
            payload.setdefault(f"{prefix}failed_count", 0)
            if canonical:
                payload.setdefault("failed_transaction_count", 0)

    def _apply_amounts(
        self,
        prefix: str,
        table: TableClassification,
        df: pd.DataFrame,
        features: Dict[str, Dict[str, Any]],
    ) -> None:
        key = table.primary_entity_key
        if key not in df.columns:
            return
        exclude = {key, table.timestamp_column} | set(table.noise_columns)
        amount_col = detect_amount_column(df, {c for c in exclude if c})
        if amount_col is None:
            return
        totals = df.groupby(key)[amount_col].sum()
        for entity_id, total in totals.items():
            payload = features.get(str(entity_id))
            if payload is not None:
                payload[f"{prefix}amount_total"] = float(total)
                payload[f"{prefix}amount_column"] = amount_col

    def _apply_text(
        self,
        prefix: str,
        aliases: _AliasTracker,
        table: TableClassification,
        df: pd.DataFrame,
        features: Dict[str, Dict[str, Any]],
    ) -> None:
        key = table.primary_entity_key
        if key not in df.columns:
            return
        exclude = {key, table.timestamp_column} | set(table.noise_columns)
        text_columns = [
            col for col in df.columns
            if col not in exclude
            and not pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        if not text_columns:
            return
        canonical = aliases.claim("text")

        # Concatenate every free-text column per record so a table carrying both
        # a subject and a notes body is mined as one utterance.
        combined = df[text_columns].astype(str).agg(" ".join, axis=1)
        grouped = combined.groupby(df[key]).apply(list, include_groups=False)

        for entity_id, texts in grouped.items():
            payload = features.get(str(entity_id))
            if payload is None:
                continue
            sentiment = self.sentiment_scorer.score_texts(texts)
            payload[f"{prefix}text_churn_score"] = sentiment.score
            payload[f"{prefix}text_record_count"] = sentiment.record_count
            payload[f"{prefix}has_negative_text"] = sentiment.is_negative
            payload[f"{prefix}text_matched_terms"] = sentiment.matched_terms
            if canonical:
                payload["text_churn_score"] = sentiment.score
                payload["text_record_count"] = sentiment.record_count
                payload["has_negative_text"] = sentiment.is_negative
                payload["text_matched_terms"] = sentiment.matched_terms

        # No rows in a text table means nothing was said, which the scorer
        # already treats as a zero score rather than an unknown.
        for payload in features.values():
            payload.setdefault(f"{prefix}text_churn_score", 0.0)
            payload.setdefault(f"{prefix}text_record_count", 0)
            payload.setdefault(f"{prefix}has_negative_text", False)
            if canonical:
                payload.setdefault("text_churn_score", 0.0)
                payload.setdefault("text_record_count", 0)
                payload.setdefault("has_negative_text", False)
