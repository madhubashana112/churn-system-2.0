"""Turn one stored analysis run into the payload a sector dashboard renders.

Two halves, deliberately separated:

* Generic aggregation every sector needs — risk-tier mix, mean probability,
  at-risk share, probability histogram.
* Sector KPI builders, dispatched the same way the feature enrichers are. Each
  reads the features its own enricher produced, so "billing value at risk" is
  summed from actual invoice totals rather than estimated from a headcount.

Everything degrades on a missing feature key rather than raising: a tenant whose
export lacked a column should see a zero KPI and a thin chart, not a 500. The
one exception is a per-customer evidence column, which is dropped entirely —
showing "0" for a measurement that was never taken reads as a real result.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from churn_platform.application.dtos.metrics_dto import (
    CHART_BAR,
    CHART_DOUGHNUT,
    CHART_SCATTER,
    TONE_CRITICAL,
    TONE_GOOD,
    TONE_NEUTRAL,
    TONE_WARNING,
    ChartDataset,
    ChartSpec,
    EntityRow,
    Highlight,
    KpiCard,
    MetricsSummary,
    ScatterPoint,
    TierSlice,
)
from churn_platform.domain.interfaces.i_repository import IAnalysisRepository, ITenantRepository
from churn_platform.domain.models.analysis_run import RISK_TIERS, AnalysisRun, EntityOutcome
from churn_platform.domain.models.sector import (
    SECTOR_FINTECH,
    SECTOR_SAAS,
    SECTOR_TELECOM,
    SECTOR_LABELS,
    normalize_sector,
)

AT_RISK_TIERS = ("CRITICAL", "HIGH")
# Days 8-37 is the synthesizer's baseline window; converting its count to a
# weekly rate puts it on the same axis as the last-7-days count.
PRIOR_WINDOW_DAYS = 30
HISTOGRAM_BINS = 10
VELOCITY_COLLAPSE_THRESHOLD = 0.5
EXPORT_EXODUS_THRESHOLD = 0.3
P2P_STREAK_THRESHOLD = 3
PORT_OUT_TOKENS = ("mnp", "port out", "port_out")


def _num(features: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric feature, tolerating absent keys, text, booleans and NaN."""
    value = features.get(key, default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _flag(features: Mapping[str, Any], key: str) -> bool:
    return bool(features.get(key, False))


def _text(features: Mapping[str, Any], key: str) -> str:
    value = features.get(key)
    return "" if value is None else str(value)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _compact_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 10_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def _pct(fraction: float, digits: int = 1) -> str:
    return f"{fraction * 100:.{digits}f}%"


def _int(value: float) -> str:
    return f"{int(round(value)):,}"


def _ratio(value: float) -> str:
    return f"{value:,.2f}x"


def _is_at_risk(outcome: EntityOutcome) -> bool:
    return outcome.prediction.risk_tier in AT_RISK_TIERS


def _split(outcomes: Sequence[EntityOutcome]) -> Tuple[List[EntityOutcome], List[EntityOutcome]]:
    return [o for o in outcomes if _is_at_risk(o)], [o for o in outcomes if not _is_at_risk(o)]


def _share(part: int, total: int) -> float:
    return part / total if total else 0.0


def _tone_for_share(share: float) -> str:
    if share >= 0.3:
        return TONE_CRITICAL
    if share >= 0.15:
        return TONE_WARNING
    return TONE_GOOD


def reason_for(outcome: EntityOutcome) -> Optional[str]:
    prediction = outcome.prediction
    if prediction.root_cause:
        return prediction.root_cause
    if prediction.primary_drivers:
        return "; ".join(prediction.primary_drivers)
    if prediction.dormancy_type:
        return f"Dormancy: {prediction.dormancy_type}"
    return None


# -- evidence columns --------------------------------------------------------

# A column is a label plus a renderer that returns None to omit itself. Each
# sector declares its own, which is what makes the three prediction tables show
# genuinely different evidence rather than the same four generic numbers.
Column = Tuple[str, Callable[[EntityOutcome], Optional[str]]]


def _measure_column(label: str, key: str, format_value: Callable[[float], str]) -> Column:
    def render(outcome: EntityOutcome) -> Optional[str]:
        if key not in outcome.features:
            return None
        return format_value(_num(outcome.features, key))

    return label, render


def _text_column(label: str, key: str, fallback: str = "—") -> Column:
    def render(outcome: EntityOutcome) -> Optional[str]:
        if key not in outcome.features:
            return None
        return _text(outcome.features, key) or fallback

    return label, render


def _flag_column(label: str, key: str) -> Column:
    def render(outcome: EntityOutcome) -> Optional[str]:
        if key not in outcome.features:
            return None
        return "Yes" if _flag(outcome.features, key) else "No"

    return label, render


def _prediction_column(label: str, attribute: str, fallback: str = "—") -> Column:
    def render(outcome: EntityOutcome) -> Optional[str]:
        value = getattr(outcome.prediction, attribute, None)
        return str(value) if value else fallback

    return label, render


_SAAS_COLUMNS: Tuple[Column, ...] = (
    _text_column("Tier", "tier"),
    _measure_column("Invoiced", "invoices_amount_total", _money),
    _measure_column("Failed invoices", "invoices_failed_count", _int),
    _measure_column("Login velocity", "activity_velocity", _ratio),
    _measure_column("Export share", "export_ratio", _pct),
    _measure_column("Ticket sentiment", "text_churn_score", lambda v: f"{v:.2f}"),
)

_TELECOM_COLUMNS: Tuple[Column, ...] = (
    _text_column("Plan", "plan"),
    _text_column("Region", "region"),
    _text_column("Tower", "dominant_tower"),
    _measure_column("Dropped calls", "dropped_call_rate", _pct),
    _measure_column("Days since top-up", "days_since_last_recharge", lambda v: f"{v:,.0f}"),
    _flag_column("Regional fault", "regional_network_impact_flag"),
)

_FINTECH_COLUMNS: Tuple[Column, ...] = (
    _text_column("Tier", "tier"),
    _measure_column("Drain ratio", "balance_drain_ratio", lambda v: _pct(v, 0)),
    _measure_column("Withdrawal share", "withdrawal_share_recent", lambda v: _pct(v, 0)),
    _measure_column("Outflow (14d)", "outflow_recent", _money),
    _measure_column("P2P fail streak", "p2p_failure_streak", _int),
    _prediction_column("Dormancy", "dormancy_type"),
)


def _highlights(columns: Tuple[Column, ...], outcome: EntityOutcome) -> List[Highlight]:
    rendered = [
        Highlight(label=label, value=value)
        for label, render in columns
        if (value := render(outcome)) is not None
    ]
    return rendered


# -- generic charts ----------------------------------------------------------


def _tier_slices(run: AnalysisRun) -> List[TierSlice]:
    counts = Counter(o.prediction.risk_tier for o in run.outcomes)
    total = len(run.outcomes) or 1
    return [
        TierSlice(tier=tier, count=counts.get(tier, 0), share=counts.get(tier, 0) / total)
        for tier in RISK_TIERS
    ]


def _tier_mix_chart(slices: List[TierSlice]) -> ChartSpec:
    present = [s for s in slices if s.count]
    return ChartSpec(
        key="tier_mix",
        title="Customers by risk tier",
        kind=CHART_DOUGHNUT,
        subtitle="CRITICAL and HIGH are the intervention queue",
        labels=[s.tier for s in present],
        datasets=[ChartDataset(label="Customers", values=[float(s.count) for s in present])],
    )


def _probability_histogram(run: AnalysisRun) -> ChartSpec:
    counts = [0] * HISTOGRAM_BINS
    for outcome in run.outcomes:
        probability = min(max(outcome.prediction.churn_probability, 0.0), 1.0)
        counts[min(int(probability * HISTOGRAM_BINS), HISTOGRAM_BINS - 1)] += 1
    step = 100 // HISTOGRAM_BINS
    return ChartSpec(
        key="probability_histogram",
        title="Churn probability distribution",
        kind=CHART_BAR,
        subtitle="Every scored customer, binned by predicted churn likelihood",
        labels=[f"{i * step}-{(i + 1) * step}%" for i in range(HISTOGRAM_BINS)],
        datasets=[ChartDataset(label="Customers", values=[float(c) for c in counts])],
        x_label="Predicted churn probability",
        y_label="Customers",
    )


def _stacked_by_category(
    outcomes: Sequence[EntityOutcome],
    category_of: Callable[[EntityOutcome], str],
    *,
    key: str,
    title: str,
    subtitle: str,
    x_label: str,
) -> ChartSpec:
    """Group customers by a categorical feature and stack the risk tiers."""
    buckets: Dict[str, Counter] = {}
    for outcome in outcomes:
        buckets.setdefault(category_of(outcome) or "Unknown", Counter())[outcome.prediction.risk_tier] += 1
    labels = sorted(buckets)
    tiers_present = [t for t in RISK_TIERS if any(buckets[label].get(t) for label in labels)]
    return ChartSpec(
        key=key,
        title=title,
        kind=CHART_BAR,
        subtitle=subtitle,
        labels=labels,
        stacked=True,
        datasets=[
            ChartDataset(label=tier, values=[float(buckets[label].get(tier, 0)) for label in labels])
            for tier in tiers_present
        ],
        x_label=x_label,
        y_label="Customers",
    )


def _grouped_means(
    at_risk: Sequence[EntityOutcome],
    healthy: Sequence[EntityOutcome],
    measures: Sequence[Tuple[str, Callable[[Mapping[str, Any]], float]]],
    *,
    key: str,
    title: str,
    subtitle: str,
    y_label: str,
) -> ChartSpec:
    """Compare at-risk against healthy customers on same-unit measures."""
    return ChartSpec(
        key=key,
        title=title,
        kind=CHART_BAR,
        subtitle=subtitle,
        labels=[label for label, _ in measures],
        datasets=[
            ChartDataset(
                label=group_name,
                values=[round(_mean([compute(o.features) for o in group]), 4) for _, compute in measures],
            )
            for group_name, group in (("At risk", at_risk), ("Healthy", healthy))
        ],
        y_label=y_label,
    )


def _bucketed(
    groups: Mapping[str, Sequence[EntityOutcome]],
    buckets: Sequence[Tuple[float, float, str]],
    measure: str,
    *,
    key: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
) -> ChartSpec:
    """Histogram one numeric feature across named bands, one dataset per group."""

    def band_of(features: Mapping[str, Any]) -> str:
        value = _num(features, measure)
        for low, high, label in buckets:
            if low <= value < high:
                return label
        return buckets[-1][2]

    labels = [label for _, _, label in buckets]
    return ChartSpec(
        key=key,
        title=title,
        kind=CHART_BAR,
        subtitle=subtitle,
        labels=labels,
        datasets=[
            ChartDataset(
                label=group_name,
                values=[float(sum(1 for o in group if band_of(o.features) == label)) for label in labels],
            )
            for group_name, group in groups.items()
        ],
        x_label=x_label,
        y_label=y_label,
    )


# -- SaaS --------------------------------------------------------------------


def _saas_kpis(run: AnalysisRun, at_risk: Sequence[EntityOutcome]) -> Dict[str, KpiCard]:
    value_at_risk = sum(_num(o.features, "invoices_amount_total") for o in at_risk)
    total_billed = sum(_num(o.features, "invoices_amount_total") for o in run.outcomes)
    failed_payers = [o for o in at_risk if _num(o.features, "invoices_failed_count") > 0]
    collapsed = [o for o in run.outcomes if _num(o.features, "activity_velocity") < VELOCITY_COLLAPSE_THRESHOLD]
    exporters = [o for o in run.outcomes if _num(o.features, "export_ratio") >= EXPORT_EXODUS_THRESHOLD]
    negative = [o for o in run.outcomes if _flag(o.features, "has_negative_text")]
    share = _share(len(at_risk), len(run.outcomes))

    return {
        "mrr_at_risk": KpiCard(
            key="mrr_at_risk",
            label="Billing value at risk",
            value=_compact_money(value_at_risk),
            detail=f"of {_compact_money(total_billed)} invoiced across {len(run.outcomes)} accounts",
            tone=TONE_CRITICAL if share >= 0.3 else TONE_WARNING,
        ),
        "payment_failures": KpiCard(
            key="payment_failures",
            label="At-risk accounts with failed invoices",
            value=_int(len(failed_payers)),
            detail=(
                "mean failed-invoice rate "
                f"{_pct(_mean([_num(o.features, 'invoices_failure_rate') for o in failed_payers]))}"
                if failed_payers
                else "no failed invoices in the at-risk group"
            ),
            tone=TONE_WARNING if failed_payers else TONE_GOOD,
        ),
        "velocity_collapse": KpiCard(
            key="velocity_collapse",
            label="Login velocity collapse",
            value=_int(len(collapsed)),
            detail=f"recent logins below {VELOCITY_COLLAPSE_THRESHOLD:g}x the prior 30-day weekly rate",
            tone=TONE_WARNING if collapsed else TONE_GOOD,
        ),
        "data_export": KpiCard(
            key="data_export",
            label="Bulk-exporting their data",
            value=_int(len(exporters)),
            detail=f"{EXPORT_EXODUS_THRESHOLD:.0%} or more of their events are exports — an exit move",
            tone=TONE_CRITICAL if exporters else TONE_NEUTRAL,
        ),
        "negative_tickets": KpiCard(
            key="negative_tickets",
            label="Accounts with churn-intent tickets",
            value=_int(len(negative)),
            detail="support text matching cancel, refund or escalation language",
            tone=TONE_WARNING if negative else TONE_GOOD,
        ),
    }


def _saas_charts(
    run: AnalysisRun, at_risk: Sequence[EntityOutcome], healthy: Sequence[EntityOutcome]
) -> Dict[str, ChartSpec]:
    return {
        "segment_risk": _stacked_by_category(
            run.outcomes,
            lambda o: _text(o.features, "tier") or "Unspecified",
            key="segment_risk",
            title="Risk by subscription tier",
            subtitle="Which plan levels are leaving",
            x_label="Subscription tier",
        ),
        "usage_dropoff": _grouped_means(
            at_risk,
            healthy,
            [
                ("Logins, last 7 days", lambda f: _num(f, "event_count_7d")),
                ("Logins, days 8-37, per week", lambda f: _num(f, "event_count_prior_30d") * 7 / PRIOR_WINDOW_DAYS),
            ],
            key="usage_dropoff",
            title="Feature adoption drop-off",
            subtitle="Weekly-equivalent login rate, before and now",
            y_label="Logins per week",
        ),
        "velocity_spread": _bucketed(
            {"At risk": at_risk, "Healthy": healthy},
            [
                (0.0, 0.25, "<0.25x"),
                (0.25, 0.5, "0.25-0.5x"),
                (0.5, 1.0, "0.5-1x"),
                (1.0, 2.0, "1-2x"),
                (2.0, math.inf, ">2x"),
            ],
            "activity_velocity",
            key="velocity_spread",
            title="Login velocity spread",
            subtitle="Recent activity over the prior 30-day weekly rate; 1x means unchanged",
            x_label="Activity velocity",
            y_label="Customers",
        ),
    }


# -- Telecom -----------------------------------------------------------------


def _telecom_kpis(run: AnalysisRun, at_risk: Sequence[EntityOutcome]) -> Dict[str, KpiCard]:
    regional = [o for o in run.outcomes if _flag(o.features, "regional_network_impact_flag")]
    towers = Counter(_text(o.features, "dominant_tower") for o in at_risk if _text(o.features, "dominant_tower"))
    worst_tower, worst_count = towers.most_common(1)[0] if towers else ("", 0)
    worst_share = _mean(
        [
            _num(o.features, "dominant_tower_share")
            for o in at_risk
            if _text(o.features, "dominant_tower") == worst_tower
        ]
    )
    expanding = [o for o in run.outcomes if _flag(o.features, "expanding_topup_intervals")]
    port_out = [
        o for o in run.outcomes
        if any(token in str(o.features.get("complaints_text_matched_terms", "")).lower() for token in PORT_OUT_TOKENS)
    ]
    dropped_at_risk = _mean([_num(o.features, "dropped_call_rate") for o in at_risk])
    dropped_overall = _mean([_num(o.features, "dropped_call_rate") for o in run.outcomes])

    return {
        "regional_impact": KpiCard(
            key="regional_impact",
            label="Regional network fault flagged",
            value=_int(len(regional)),
            detail="dropped calls concentrated on a single tower, above the volume floor",
            tone=TONE_CRITICAL if regional else TONE_GOOD,
        ),
        "worst_tower": KpiCard(
            key="worst_tower",
            label="Worst tower among at-risk",
            value=worst_tower or "—",
            detail=(
                f"{worst_count} at-risk subscribers homed on it, {_pct(worst_share, 0)} of their calls"
                if worst_tower
                else "no tower concentration detected"
            ),
            tone=TONE_CRITICAL if worst_count else TONE_NEUTRAL,
        ),
        "dropped_calls": KpiCard(
            key="dropped_calls",
            label="Dropped-call rate",
            value=_pct(dropped_at_risk),
            detail=f"at risk, against {_pct(dropped_overall)} across the whole base",
            tone=TONE_CRITICAL if dropped_at_risk > 2 * dropped_overall else TONE_WARNING,
        ),
        "recharge_lapse": KpiCard(
            key="recharge_lapse",
            label="Days since last top-up",
            value=f"{_mean([_num(o.features, 'days_since_last_recharge') for o in at_risk]):,.1f}",
            detail=f"{len(expanding)} subscribers show widening intervals between recharges",
            tone=TONE_WARNING if expanding else TONE_NEUTRAL,
        ),
        "port_out_intent": KpiCard(
            key="port_out_intent",
            label="Stated port-out intent",
            value=_int(len(port_out)),
            detail="complaints mentioning MNP or porting to another carrier",
            tone=TONE_CRITICAL if port_out else TONE_GOOD,
        ),
    }


def _telecom_charts(
    run: AnalysisRun, at_risk: Sequence[EntityOutcome], healthy: Sequence[EntityOutcome]
) -> Dict[str, ChartSpec]:
    def rates_by_tower(group: Sequence[EntityOutcome]) -> Dict[str, List[float]]:
        buckets: Dict[str, List[float]] = {}
        for outcome in group:
            tower = _text(outcome.features, "dominant_tower") or "Unknown"
            buckets.setdefault(tower, []).append(_num(outcome.features, "dropped_call_rate"))
        return buckets

    at_risk_rates = rates_by_tower(at_risk)
    healthy_rates = rates_by_tower(healthy)
    towers = sorted(set(at_risk_rates) | set(healthy_rates))

    return {
        "region_risk": _stacked_by_category(
            run.outcomes,
            lambda o: _text(o.features, "region") or "Unspecified",
            key="region_risk",
            title="Risk by region",
            subtitle="A regional spike points at infrastructure rather than pricing",
            x_label="Region",
        ),
        "tower_health": ChartSpec(
            key="tower_health",
            title="Dropped-call rate by tower",
            kind=CHART_BAR,
            subtitle="Same tower, at-risk against healthy subscribers — a gap here is a local fault",
            labels=towers,
            datasets=[
                ChartDataset(
                    label=group_name,
                    values=[round(_mean(rates.get(tower, [])) * 100, 2) for tower in towers],
                )
                for group_name, rates in (("At risk", at_risk_rates), ("Healthy", healthy_rates))
            ],
            x_label="Tower",
            y_label="Dropped calls (%)",
        ),
        "recharge_gap": _grouped_means(
            at_risk,
            healthy,
            [
                ("Mean gap between top-ups", lambda f: _num(f, "avg_recharge_gap_days")),
                ("Longest gap between top-ups", lambda f: _num(f, "max_recharge_gap_days")),
                ("Days since last top-up", lambda f: _num(f, "days_since_last_recharge")),
            ],
            key="recharge_gap",
            title="Top-up cadence",
            subtitle="Widening gaps are the earliest observable signal of a subscriber leaving",
            y_label="Days",
        ),
    }


# -- FinTech -----------------------------------------------------------------


def _fintech_kpis(run: AnalysisRun, at_risk: Sequence[EntityOutcome]) -> Dict[str, KpiCard]:
    rapid = [o for o in run.outcomes if _flag(o.features, "rapid_balance_drain")]
    outflow = sum(_num(o.features, "outflow_recent") for o in at_risk)
    inflow = sum(_num(o.features, "inflow_recent") for o in at_risk)
    dormancy = Counter(o.prediction.dormancy_type or "UNKNOWN" for o in run.outcomes)
    hard = dormancy.get("HARD_DORMANCY", 0)
    soft = dormancy.get("SOFT_DORMANCY", 0)
    streaks = [o for o in run.outcomes if _num(o.features, "p2p_failure_streak") >= P2P_STREAK_THRESHOLD]
    disputed = [
        o for o in run.outcomes
        if _num(o.features, "disputes_row_count") > 0 and _flag(o.features, "disputes_has_negative_text")
    ]
    drain_at_risk = _mean([_num(o.features, "balance_drain_ratio") for o in at_risk])
    drain_overall = _mean([_num(o.features, "balance_drain_ratio") for o in run.outcomes])

    return {
        "liquidity_drain": KpiCard(
            key="liquidity_drain",
            label="Balance drain ratio",
            value=_pct(drain_at_risk, 0),
            detail=f"at risk, against {_pct(drain_overall, 0)} across the whole base",
            tone=TONE_CRITICAL if drain_at_risk > drain_overall else TONE_WARNING,
        ),
        "rapid_drain": KpiCard(
            key="rapid_drain",
            label="Accounts draining rapidly",
            value=_int(len(rapid)),
            detail="over 60% of settled flow leaving, above the minimum volume floor",
            tone=TONE_CRITICAL if rapid else TONE_GOOD,
        ),
        "net_outflow": KpiCard(
            key="net_outflow",
            label="Net outflow, at-risk accounts",
            value=_compact_money(outflow - inflow),
            detail=f"{_compact_money(outflow)} out against {_compact_money(inflow)} in over the last 14 days",
            tone=TONE_CRITICAL if outflow > inflow else TONE_GOOD,
        ),
        "dormant_accounts": KpiCard(
            key="dormant_accounts",
            label="Hard-dormant accounts",
            value=_int(hard),
            detail=(
                f"plus {soft} soft-dormant — transacting recently, but far less than their peers"
                if soft
                else "no recent activity at all; these need reactivation rather than a nudge"
            ),
            tone=TONE_WARNING if hard else TONE_NEUTRAL,
        ),
        "p2p_failures": KpiCard(
            key="p2p_failures",
            label="Repeated P2P transfer failures",
            value=_int(len(streaks)),
            detail=(
                f"mean streak {_mean([_num(o.features, 'p2p_failure_streak') for o in streaks]):.1f} consecutive failures"
                if streaks
                else f"no account has failed {P2P_STREAK_THRESHOLD} or more transfers in a row"
            ),
            tone=TONE_CRITICAL if streaks else TONE_GOOD,
        ),
        "fraud_disputes": KpiCard(
            key="fraud_disputes",
            label="Accounts with fraud disputes",
            value=_int(len(disputed)),
            detail="dispute text matching fraud or chargeback language",
            tone=TONE_WARNING if disputed else TONE_GOOD,
        ),
    }


def _fintech_charts(
    run: AnalysisRun, at_risk: Sequence[EntityOutcome], healthy: Sequence[EntityOutcome]
) -> Dict[str, ChartSpec]:
    dormancy = Counter(o.prediction.dormancy_type or "UNKNOWN" for o in run.outcomes)
    dormancy_labels = sorted(dormancy)

    return {
        "drain_vs_risk": ChartSpec(
            key="drain_vs_risk",
            title="Balance drain against predicted churn",
            kind=CHART_SCATTER,
            subtitle="Each point is one account — the rising band is where the drain signal earns its weight",
            datasets=[
                ChartDataset(
                    label=group_name,
                    points=[
                        ScatterPoint(
                            x=round(_num(o.features, "balance_drain_ratio"), 4),
                            y=round(o.prediction.churn_probability, 4),
                            label=o.prediction.entity_id,
                        )
                        for o in group
                    ],
                )
                for group_name, group in (("At risk", at_risk), ("Healthy", healthy))
            ],
            x_label="Balance drain ratio",
            y_label="Churn probability",
        ),
        "dormancy_mix": ChartSpec(
            key="dormancy_mix",
            title="Dormancy profile",
            kind=CHART_DOUGHNUT,
            subtitle="Hard dormancy needs reactivation; soft dormancy needs a nudge",
            labels=dormancy_labels,
            datasets=[ChartDataset(label="Accounts", values=[float(dormancy[label]) for label in dormancy_labels])],
        ),
        "withdrawal_share": _bucketed(
            {"At risk": at_risk, "Healthy": healthy},
            [
                (0.0, 0.2, "0-20%"),
                (0.2, 0.4, "20-40%"),
                (0.4, 0.6, "40-60%"),
                (0.6, 0.8, "60-80%"),
                (0.8, 1.01, "80-100%"),
            ],
            "withdrawal_share_recent",
            key="withdrawal_share",
            title="Recent withdrawal share of settled flow",
            subtitle="Last 14 days — a book of withdrawals with no deposits is a closing account",
            x_label="Withdrawal share",
            y_label="Accounts",
        ),
    }


_SECTOR_BUILDERS: Dict[
    str,
    Tuple[
        Callable[[AnalysisRun, Sequence[EntityOutcome]], Dict[str, KpiCard]],
        Callable[[AnalysisRun, Sequence[EntityOutcome], Sequence[EntityOutcome]], Dict[str, ChartSpec]],
    ],
] = {
    SECTOR_SAAS: (_saas_kpis, _saas_charts),
    SECTOR_TELECOM: (_telecom_kpis, _telecom_charts),
    SECTOR_FINTECH: (_fintech_kpis, _fintech_charts),
}

_SECTOR_COLUMNS = {
    SECTOR_SAAS: _SAAS_COLUMNS,
    SECTOR_TELECOM: _TELECOM_COLUMNS,
    SECTOR_FINTECH: _FINTECH_COLUMNS,
}


class SummarizeAnalysisUseCase:
    """Aggregate a stored run into the payload one sector dashboard renders."""

    def __init__(
        self,
        analysis_repo: IAnalysisRepository,
        tenant_repo: ITenantRepository,
    ) -> None:
        self.analysis_repo = analysis_repo
        self.tenant_repo = tenant_repo

    async def execute(self, tenant_id: str) -> Optional[MetricsSummary]:
        run = await self.analysis_repo.latest(tenant_id)
        if run is None:
            return None

        tenant = await self.tenant_repo.get(tenant_id)
        sector = normalize_sector(run.sector) or SECTOR_SAAS
        build_kpis, build_charts = _SECTOR_BUILDERS[sector]
        at_risk, healthy = _split(run.outcomes)

        kpis = build_kpis(run, at_risk)
        charts = build_charts(run, at_risk, healthy)

        total = len(run.outcomes)
        kpis["at_risk"] = KpiCard(
            key="at_risk",
            label="Customers needing intervention",
            value=_int(len(at_risk)),
            detail=f"{_pct(_share(len(at_risk), total), 0)} of {total} scored customers are HIGH or CRITICAL",
            tone=_tone_for_share(_share(len(at_risk), total)),
        )
        charts["tier_mix"] = _tier_mix_chart(_tier_slices(run))
        charts["probability_histogram"] = _probability_histogram(run)

        columns = _SECTOR_COLUMNS[sector]
        ordered = sorted(run.outcomes, key=lambda o: o.prediction.churn_probability, reverse=True)
        return MetricsSummary(
            tenant_id=run.tenant_id,
            tenant_name=tenant.name if tenant else "Unknown tenant",
            sector=sector,
            sector_label=SECTOR_LABELS[sector],
            entities_uploaded=run.entities_uploaded,
            entities_analyzed=run.entities_analyzed,
            mean_probability=round(_mean([o.prediction.churn_probability for o in run.outcomes]), 4),
            at_risk_count=len(at_risk),
            at_risk_share=round(_share(len(at_risk), total), 4),
            tiers=_tier_slices(run),
            kpis=kpis,
            charts=charts,
            rows=[
                EntityRow(
                    entity_id=o.prediction.entity_id,
                    churn_probability=o.prediction.churn_probability,
                    risk_tier=o.prediction.risk_tier,
                    reason=reason_for(o),
                    highlights=_highlights(columns, o),
                    playbook=o.playbook,
                )
                for o in ordered
            ],
            schema_mapping=run.schema_mapping,
            offline_mode=run.offline_mode,
            warnings=run.warnings,
            created_at=run.created_at,
        )
