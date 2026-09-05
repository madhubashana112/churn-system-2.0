"""Sector-specific feature enrichment layered on the generic primitives.

`PandasFeatureSynthesizer` stays sector-agnostic on purpose. The signals that
only make sense for one industry — a telco's widening top-up intervals, a bank's
balance drain, a SaaS vendor's data-export spike — are computed here instead, so
adding a sector never means editing the synthesizer.

Every enricher degrades to a no-op when the columns it needs are absent, so a
tenant uploading an incomplete export still gets the generic primitives scored.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.schema_mapping import (
    ROLE_TIME_SERIES,
    ROLE_TRANSACTIONAL,
    SchemaMapping,
    TableClassification,
)
from churn_platform.domain.models.sector import (
    SECTOR_FINTECH,
    SECTOR_SAAS,
    SECTOR_TELECOM,
    normalize_sector,
)
from churn_platform.infrastructure.parsers.feature_synthesizer import (
    FAILURE_VALUES,
    detect_amount_column,
    detect_status_column,
    resolve_temporal_anchor,
    to_naive_utc,
)

logger = logging.getLogger(__name__)

DRAIN_WINDOW_DAYS = 14
RAPID_DRAIN_THRESHOLD = 0.6
# A drain ratio derived from a single withdrawal is noise, not a signal, so the
# binary flag needs a minimum volume before it is allowed to fire.
MIN_SETTLED_FOR_DRAIN = 2
RECENT_EXPORT_WINDOW_DAYS = 14
# A subscriber with one dropped call is trivially "100% concentrated" on a single
# tower, so the regional flag needs a minimum volume before it means anything.
MIN_DROPPED_FOR_REGIONAL_FLAG = 3
REGIONAL_TOWER_SHARE_THRESHOLD = 0.7
REGIONAL_DROPPED_RATE_THRESHOLD = 0.1
EXPANSION_RATIO = 1.5
MIN_GAPS_FOR_EXPANSION = 3

EXPORT_TOKENS = frozenset({"EXPORT", "DATA_EXPORT", "DOWNLOAD", "BULK_EXPORT"})
P2P_TOKENS = frozenset({"P2P", "PEER", "P2P_TRANSFER", "TRANSFER_P2P"})
INFLOW_TOKENS = frozenset({
    "DEPOSIT", "CREDIT", "IN", "INBOUND", "RECEIVED", "TOPUP", "TOP_UP", "PAYMENT_IN",
})
OUTFLOW_TOKENS = frozenset({
    "WITHDRAWAL", "WITHDRAW", "DEBIT", "OUT", "OUTBOUND", "SENT", "PAYMENT_OUT", "CASH_OUT",
})


def _categorical_column(
    df: pd.DataFrame,
    exclude: set[str],
    values: Optional[frozenset[str]] = None,
    name_hints: tuple[str, ...] = (),
    max_cardinality: int = 25,
) -> Optional[str]:
    """Find a low-cardinality string column, optionally matching known values."""
    for col in df.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(df[col]):
            continue
        if name_hints and not any(hint in col.lower() for hint in name_hints):
            continue
        observed = set(df[col].dropna().astype(str).str.upper().unique())
        if not observed or len(observed) > max_cardinality:
            continue
        if values is not None and not (observed & values):
            continue
        return col
    return None


def _outcome_column(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    """A status column only counts if it actually holds failure vocabulary."""
    col = detect_status_column(df, exclude)
    if col is None:
        return None
    if not df[col].astype(str).str.upper().isin(FAILURE_VALUES).any():
        return None
    return col


def _classify_flow(value: Any) -> str:
    token = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    if token in P2P_TOKENS or "P2P" in token:
        return "P2P"
    if token in INFLOW_TOKENS:
        return "INFLOW"
    if token in OUTFLOW_TOKENS:
        return "OUTFLOW"
    return "OTHER"


def within_recent(days_ago: pd.Series, window_days: float) -> pd.Series:
    """True for rows inside ``[0, window_days]`` of the temporal anchor.

    Bounded below because a future-dated row is not recent activity, and
    ``between`` already maps unparseable timestamps to False.
    """
    return days_ago.between(0, window_days)


def _max_failure_streak(flags: List[bool]) -> int:
    longest = run = 0
    for failed in flags:
        run = run + 1 if failed else 0
        longest = max(longest, run)
    return longest


class _Context:
    """Shared lookup surface for the enrichers."""

    def __init__(
        self,
        schema: SchemaMapping,
        dataframes: Dict[str, pd.DataFrame],
        features: List[CustomerFeatures],
    ) -> None:
        self.schema = schema
        self.dataframes = dataframes
        self.payloads: Dict[str, Dict[str, Any]] = {f.entity_id: f.features for f in features}
        self.reference_ts, self.span_days = resolve_temporal_anchor(schema, dataframes)

    def table(self, role: str) -> Optional[tuple[TableClassification, pd.DataFrame]]:
        for table in self.schema.tables:
            if table.role == role and table.file_name in self.dataframes:
                return table, self.dataframes[table.file_name]
        return None

    def put(self, entity_id: Any, key: str, value: Any) -> None:
        payload = self.payloads.get(str(entity_id))
        if payload is not None:
            payload[key] = value

    def days_ago(self, table: TableClassification, df: pd.DataFrame) -> Optional[pd.Series]:
        if self.reference_ts is None or not table.timestamp_column:
            return None
        if table.timestamp_column not in df.columns:
            return None
        parsed = to_naive_utc(df[table.timestamp_column])
        return (self.reference_ts - parsed).dt.total_seconds() / 86400


def _enrich_saas(ctx: _Context) -> None:
    """Data-export spikes precede a SaaS customer leaving with their data."""
    found = ctx.table(ROLE_TIME_SERIES)
    if found is None:
        return
    table, df = found
    key = table.primary_entity_key
    if key not in df.columns:
        return

    type_col = _categorical_column(
        df,
        exclude={key, table.timestamp_column} | set(table.noise_columns),
        values=EXPORT_TOKENS,
        name_hints=("event", "type", "action", "activity", "name"),
    )
    if type_col is None:
        return

    is_export = df[type_col].astype(str).str.upper().isin(EXPORT_TOKENS)
    work = pd.DataFrame({key: df[key].astype(str), "_export": is_export})
    days_ago = ctx.days_ago(table, df)
    if days_ago is not None:
        work["_recent_export"] = is_export & within_recent(days_ago, RECENT_EXPORT_WINDOW_DAYS)

    grouped = work.groupby(key)["_export"].agg(export_count="sum", event_total="size")
    grouped["export_ratio"] = grouped["export_count"] / grouped["event_total"]
    if "_recent_export" in work.columns:
        grouped["export_count_recent"] = work.groupby(key)["_recent_export"].sum()

    for entity_id, row in grouped.iterrows():
        ctx.put(entity_id, "export_count", int(row["export_count"]))
        ctx.put(entity_id, "export_ratio", round(float(row["export_ratio"]), 4))
        if "export_count_recent" in grouped.columns:
            ctx.put(entity_id, "export_count_recent", int(row["export_count_recent"]))


def _enrich_telecom_recharges(ctx: _Context) -> None:
    """Widening gaps between prepaid top-ups signal a subscriber drifting away."""
    found = ctx.table(ROLE_TRANSACTIONAL)
    if found is None or ctx.reference_ts is None:
        return
    table, df = found
    key = table.primary_entity_key
    if key not in df.columns or not table.timestamp_column:
        return
    if table.timestamp_column not in df.columns:
        return

    dates = to_naive_utc(df[table.timestamp_column])
    work = pd.DataFrame({key: df[key].astype(str), "_date": dates}).dropna()
    if work.empty:
        return

    for entity_id, group in work.groupby(key):
        ordered = group["_date"].sort_values()
        count = len(ordered)
        ctx.put(entity_id, "recharge_count", int(count))
        ctx.put(
            entity_id,
            "days_since_last_recharge",
            round(max(0.0, (ctx.reference_ts - ordered.max()).total_seconds() / 86400), 2),
        )
        if count < 2:
            continue
        gaps = ordered.diff().dropna().dt.total_seconds() / 86400
        if gaps.empty:
            continue
        ctx.put(entity_id, "avg_recharge_gap_days", round(float(gaps.mean()), 2))
        ctx.put(entity_id, "max_recharge_gap_days", round(float(gaps.max()), 2))
        if len(gaps) >= MIN_GAPS_FOR_EXPANSION:
            earliest = float(gaps.iloc[:-2].mean())
            latest = float(gaps.iloc[-2:].mean())
            expanding = earliest > 0 and latest > earliest * EXPANSION_RATIO
            ctx.put(entity_id, "expanding_topup_intervals", bool(expanding))


def _enrich_telecom_towers(ctx: _Context) -> None:
    """Dropped calls clustered on one tower point at a regional network fault."""
    found = ctx.table(ROLE_TIME_SERIES)
    if found is None:
        return
    table, df = found
    key = table.primary_entity_key
    if key not in df.columns:
        return
    exclude = {c for c in {key, table.timestamp_column} | set(table.noise_columns) if c}
    status_col = _outcome_column(df, exclude)
    if status_col is None:
        return
    tower_col = _categorical_column(
        df,
        exclude=exclude | {status_col},
        name_hints=("tower", "cell", "site", "node", "region", "area", "network"),
    )

    is_dropped = df[status_col].astype(str).str.upper().isin(FAILURE_VALUES)
    work = pd.DataFrame({key: df[key].astype(str), "_dropped": is_dropped})
    if tower_col is not None:
        work["_tower"] = df[tower_col].astype(str)

    for entity_id, group in work.groupby(key):
        dropped = group[group["_dropped"]]
        count = len(dropped)
        rate = count / len(group) if len(group) else 0.0
        ctx.put(entity_id, "dropped_call_count", int(count))
        ctx.put(entity_id, "dropped_call_rate", round(float(rate), 4))
        if tower_col is None or count == 0:
            continue
        shares = dropped["_tower"].value_counts(normalize=True)
        dominant_share = float(shares.iloc[0])
        ctx.put(entity_id, "dominant_tower_share", round(dominant_share, 4))
        ctx.put(entity_id, "dominant_tower", str(shares.index[0]))
        ctx.put(
            entity_id,
            "regional_network_impact_flag",
            bool(
                count >= MIN_DROPPED_FOR_REGIONAL_FLAG
                and dominant_share > REGIONAL_TOWER_SHARE_THRESHOLD
                and rate > REGIONAL_DROPPED_RATE_THRESHOLD
            ),
        )


def _enrich_telecom(ctx: _Context) -> None:
    _enrich_telecom_recharges(ctx)
    _enrich_telecom_towers(ctx)


def _enrich_fintech(ctx: _Context) -> None:
    """Balance drain and failed peer transfers are the strongest wallet-exit signs."""
    found = ctx.table(ROLE_TRANSACTIONAL)
    if found is None:
        return
    table, df = found
    key = table.primary_entity_key
    if key not in df.columns:
        return
    exclude = {c for c in {key, table.timestamp_column} | set(table.noise_columns) if c}
    amount_col = detect_amount_column(df, exclude)
    type_col = _categorical_column(
        df,
        exclude=exclude | ({amount_col} if amount_col else set()),
        values=P2P_TOKENS | INFLOW_TOKENS | OUTFLOW_TOKENS,
        name_hints=("type", "kind", "operation", "direction", "category"),
    )
    status_col = _outcome_column(df, exclude)
    if amount_col is None or type_col is None:
        return

    days_ago = ctx.days_ago(table, df)
    flows = df[type_col].map(_classify_flow)
    succeeded = (
        df[status_col].astype(str).str.upper().ne("FAILED")
        if status_col is not None
        else pd.Series(True, index=df.index)
    )
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    in_window = within_recent(days_ago, DRAIN_WINDOW_DAYS) if days_ago is not None else None

    work = pd.DataFrame({
        key: df[key].astype(str),
        "_flow": flows,
        "_amount": amounts,
        "_ok": succeeded,
        "_ts": to_naive_utc(df[table.timestamp_column]) if table.timestamp_column in df.columns else pd.NaT,
    })
    if in_window is not None:
        work["_window"] = in_window.to_numpy()

    for entity_id, group in work.groupby(key):
        # P2P direction is not recorded in a raw ledger, so it is treated as
        # balance-neutral: guessing would systematically overstate drain.
        if "_window" in group.columns:
            window = group[group["_window"]]
        else:
            window = group
        settled = window[window["_ok"]]
        inflow = float(settled.loc[settled["_flow"] == "INFLOW", "_amount"].sum())
        outflow = float(settled.loc[settled["_flow"] == "OUTFLOW", "_amount"].sum())
        drain = outflow / max(outflow + inflow, 1.0)
        ctx.put(entity_id, "inflow_recent", round(inflow, 2))
        ctx.put(entity_id, "outflow_recent", round(outflow, 2))
        ctx.put(entity_id, "settled_count_recent", int(len(settled)))
        ctx.put(entity_id, "balance_drain_ratio", round(drain, 4))
        ctx.put(
            entity_id,
            "rapid_balance_drain",
            bool(drain > RAPID_DRAIN_THRESHOLD and len(settled) >= MIN_SETTLED_FOR_DRAIN),
        )
        if len(window):
            ctx.put(
                entity_id,
                "withdrawal_share_recent",
                round(float((window["_flow"] == "OUTFLOW").mean()), 4),
            )

        p2p = group[group["_flow"] == "P2P"]
        if not p2p.empty:
            ordered = p2p.sort_values("_ts") if p2p["_ts"].notna().any() else p2p
            failed_flags = (~ordered["_ok"]).tolist()
            ctx.put(entity_id, "p2p_count", int(len(p2p)))
            ctx.put(entity_id, "p2p_failure_rate", round(float(sum(failed_flags) / len(p2p)), 4))
            ctx.put(entity_id, "p2p_failure_streak", int(_max_failure_streak(failed_flags)))


_ENRICHERS = {
    SECTOR_SAAS: _enrich_saas,
    SECTOR_TELECOM: _enrich_telecom,
    SECTOR_FINTECH: _enrich_fintech,
}

# The enrichers write these only when the qualifying condition is met, so a
# subscriber with no dropped calls ends up with no key at all. Absence and
# False are the same fact, and a missing key becomes NaN the moment features
# are batched for scoring.
_SECTOR_DEFAULTS: Dict[str, Dict[str, Any]] = {
    SECTOR_TELECOM: {
        "dominant_tower_share": 0.0,
        "regional_network_impact_flag": False,
        "expanding_topup_intervals": False,
    },
}


def enrich_features(
    sector: str,
    features: List[CustomerFeatures],
    schema: SchemaMapping,
    dataframes: Dict[str, pd.DataFrame],
) -> List[CustomerFeatures]:
    """Add sector-specific signals to already-synthesised features, in place.

    Returns the same list it was given. An unrecognised sector is not an error —
    the generic primitives are still perfectly usable on their own.
    """
    resolved = normalize_sector(sector) or ""
    enricher = _ENRICHERS.get(resolved)
    if enricher is None or not features:
        if sector and enricher is None:
            logger.info("No feature enricher for sector %r; keeping generic primitives", sector)
        return features
    try:
        enricher(_Context(schema, dataframes, features))
    except Exception:
        # Enrichment is additive; a missing column should never lose the whole
        # analysis for a tenant.
        logger.exception("Sector enrichment for %r failed; continuing with generic features", sector)
    defaults = _SECTOR_DEFAULTS.get(resolved, {})
    for feature in features:
        for name, value in defaults.items():
            feature.features.setdefault(name, value)
    return features
