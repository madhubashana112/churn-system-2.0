"""The risk vocabulary the platform shares, plus repair for a model that drifts from it.

Tiers are matched exactly everywhere downstream — `AT_RISK_TIERS`, the chart
ordering, the `pill--CRITICAL` styles — so a hosted open-weight model that
answers "High" instead of "HIGH" would quietly empty the at-risk KPI instead of
failing visibly. The offline oracle and the three sector cores both take their
bands from here, which is what keeps a repaired reply and an offline one
indistinguishable to the rest of the app.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

RISK_THRESHOLDS: Tuple[Tuple[float, str], ...] = ((0.30, "LOW"), (0.55, "MEDIUM"), (0.80, "HIGH"))
CRITICAL_TIER = "CRITICAL"
TIERS: Tuple[str, ...] = tuple(tier for _, tier in RISK_THRESHOLDS) + (CRITICAL_TIER,)


def risk_tier(probability: float) -> str:
    for threshold, tier in RISK_THRESHOLDS:
        if probability < threshold:
            return tier
    return CRITICAL_TIER


def normalise_probability(value: Any) -> float:
    """A probability in [0, 1].

    Out-of-range values are clamped rather than rejected: one customer scored
    "1.2" by a sloppy model should cost that customer's accuracy, not the whole
    batch of ten. Something that is not a number at all does fail the batch,
    because guessing at it would put a made-up score on the dashboard.
    """
    try:
        probability = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"churn_probability {value!r} is not a number") from None
    if probability != probability:
        raise ValueError("churn_probability is NaN")

    clamped = min(max(probability, 0.0), 1.0)
    if clamped != probability:
        logger.warning("churn_probability %r is outside [0, 1]; clamped to %r", probability, clamped)
    return clamped


def normalise_tier(value: Optional[Any], probability: float) -> str:
    """The tier the probability implies, honouring a model label that agrees with it.

    A tier here is a function of the score: ``RISK_THRESHOLDS`` is the only
    definition, and the dashboards colour, rank and count at-risk customers by it.
    Measured against a hosted model, a large share of replies named a tier that
    contradicted their own probability — gpt-oss-120b on Groq did it for 10 of 30
    customers — so a disagreeing label is treated as drift and re-derived rather
    than trusted.
    """
    derived = risk_tier(probability)
    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate == derived:
            return derived
        if candidate in TIERS:
            logger.warning(
                "risk_tier %r disagrees with probability %.3f, which falls in %s; used the band",
                value, probability, derived,
            )
            return derived
    if value is not None:
        logger.warning(
            "risk_tier %r is not one of %s; derived the tier from probability %.3f",
            value, "/".join(TIERS), probability,
        )
    return derived
