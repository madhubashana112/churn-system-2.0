"""Deterministic offline stand-in for the Qwen gateway.

Every sector core and the schema resolver funnel through `generate_json`, so
without an API key the platform cannot be run, demonstrated, or tested at all.
This module implements the same contract locally: rule-based schema resolution
and weighted feature scoring.

Two properties are load-bearing:

* **Scores come from the supplied feature values.** Nothing is canned. A batch
  where every customer looks identical produces identical, mid-range scores; a
  batch containing a customer with failed payments, decaying activity and an
  explicit cancellation request puts that customer in CRITICAL.
* **No randomness.** The output is a pure function of the input, so a test can
  assert an exact probability and a demo reproduces frame for frame.

Retention copy is a fixed template per dominant signal rather than bespoke
prose. Generating plausible-sounding personalised text would make the offline
mode look more capable than it is.
"""

from __future__ import annotations

import io
import json
import logging
import math
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence

import pandas as pd

from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway
from churn_platform.domain.models.schema_mapping import (
    ROLE_DIMENSION,
    ROLE_TEXT,
    ROLE_TIME_SERIES,
    ROLE_TRANSACTIONAL,
)
# The bands live in normalise_prediction; these two come along for re-export so
# there is one definition to read.
from churn_platform.infrastructure.ai.normalise_prediction import (
    CRITICAL_TIER,
    RISK_THRESHOLDS,
    risk_tier,
)

logger = logging.getLogger(__name__)

# Substrings that identify the caller. Matched rather than compared so prompt
# wording can drift without silently breaking offline mode.
SCHEMA_RESOLVER_MARKER = "expert Data Engineer AI"
SAAS_MARKER = "SaaS Retention AI Core"
TELECOM_MARKER = "Telecom & ISP Retention AI Core"
FINTECH_MARKER = "FinTech & Banking Retention AI Core"

# Scoring shape. MIDPOINT_SCORE is the raw score that maps to a 50% probability.
# It sits above zero so a customer with no net signal lands near a realistic
# churn base rate rather than at a coin flip.
SIGMOID_STEEPNESS = 3.0
MIDPOINT_SCORE = 0.25

# Continuous signals are winsorised before normalising: one customer with a
# tenfold usage spike would otherwise flatten the whole batch into the bottom
# of the range. Binary flags are already on a 0/1 scale and are left alone,
# because a flag held by 20% of the batch has a 90th percentile of zero and
# winsorising would erase it.
WINSORISE_QUANTILE = 0.10

# A feature is only named as a driver once it is at least this far along its
# risk direction, so a LOW-risk customer is not handed spurious explanations.
MIN_DRIVER_STRENGTH = 0.6
MAX_DRIVERS = 3
NO_SIGNAL_DRIVERS = ["no material churn signals detected"]

HARD_DORMANCY_RECENCY = 0.66
SOFT_DORMANCY_VELOCITY = 0.33


class Playbook(NamedTuple):
    action_type: str
    action_payload: str
    channel: str


class Signal(NamedTuple):
    """One scored feature: its weight, how to name it, and what to do about it.

    The weight is signed — positive raises churn risk, negative lowers it — so
    collapsing usage and rising usage can live in the same table.
    """

    weight: float
    label: str
    playbook: Playbook


SAAS_SIGNALS: Dict[str, Signal] = {
    "failure_rate": Signal(3.0, "repeated payment failures", Playbook(
        "DISCOUNT",
        "Your last invoice did not go through. Update your payment method and we will apply 20% off your next cycle.",
        "EMAIL",
    )),
    "text_churn_score": Signal(2.5, "cancellation intent in support tickets", Playbook(
        "CSM_CALL",
        "A customer success manager would like to walk through the issues you raised and agree a fix.",
        "PHONE",
    )),
    "activity_velocity": Signal(-2.5, "collapse in login velocity", Playbook(
        "IN_APP_TOUR",
        "You have been quiet lately. Here is a guided tour of the features your plan already pays for.",
        "IN_APP",
    )),
    "export_ratio": Signal(1.5, "unusual spike in data exports", Playbook(
        "CSM_CALL",
        "We noticed heavy export activity. Tell us what you are building and we will help you get more from the platform.",
        "IN_APP",
    )),
    "has_negative_text": Signal(1.0, "escalated support sentiment", Playbook(
        "CSM_CALL",
        "Your recent tickets were left unresolved. A customer success manager will call you within one business day.",
        "PHONE",
    )),
    "recency_days": Signal(1.5, "lengthening silence between sessions", Playbook(
        "WINBACK_EMAIL",
        "It has been a while. See what your team missed and pick up where you left off.",
        "EMAIL",
    )),
    "event_count_7d": Signal(-1.0, "falling weekly active usage", Playbook(
        "IN_APP_TOUR",
        "Your team's weekly usage is down. Here are the three workflows other teams your size rely on.",
        "IN_APP",
    )),
}

TELECOM_SIGNALS: Dict[str, Signal] = {
    "text_churn_score": Signal(2.5, "port-out or cancellation request logged", Playbook(
        "TARIFF_UPGRADE",
        "Before you port out: we can move you to a plan with more data at your current price. Reply YES to see options.",
        "SMS",
    )),
    "dropped_call_rate": Signal(2.5, "elevated dropped-call rate", Playbook(
        "FREE_DATA",
        "We know calls on your usual tower have been dropping. Here is 5GB free while our engineers work on it.",
        "SMS",
    )),
    "days_since_last_recharge": Signal(2.0, "lapsed prepaid top-up", Playbook(
        "TARIFF_UPGRADE",
        "Your balance has been idle for weeks. Reactivate now and we will double your first recharge.",
        "USSD",
    )),
    "expanding_topup_intervals": Signal(2.0, "widening gap between top-ups", Playbook(
        "TARIFF_UPGRADE",
        "Your recharge pattern has slowed. A monthly plan would cost less than your last three top-ups combined.",
        "USSD",
    )),
    "activity_velocity": Signal(-2.0, "collapse in network usage", Playbook(
        "FREE_DATA",
        "You have been using less lately. Here is 2GB on us to bring you back.",
        "SMS",
    )),
    "regional_network_impact_flag": Signal(1.5, "dropped calls concentrated on one tower", Playbook(
        "FREE_DATA",
        "A network fault in your area affected your calls. We are sorry — here is 5GB free while it is repaired.",
        "SMS",
    )),
    "max_recharge_gap_days": Signal(1.0, "longest ever gap between top-ups", Playbook(
        "TARIFF_UPGRADE",
        "Switch to auto-recharge and never run out mid-month again.",
        "USSD",
    )),
}

FINTECH_SIGNALS: Dict[str, Signal] = {
    "balance_drain_ratio": Signal(2.5, "rapid balance drain", Playbook(
        "FEE_WAIVER",
        "Your balance is moving out faster than it comes in. We have waived your transfer fees for the next 30 days.",
        "PUSH_NOTIFICATION",
    )),
    "p2p_failure_rate": Signal(2.0, "streak of failed peer transfers", Playbook(
        "CASHBACK",
        "Your recent transfers failed more than they should. Try again and we will add 2% cashback.",
        "PUSH_NOTIFICATION",
    )),
    "p2p_failure_streak": Signal(2.0, "repeated peer transfers failing back to back", Playbook(
        "CASHBACK",
        "Several of your transfers failed in a row. We are reviewing the cause — here is 2% cashback for the trouble.",
        "PUSH_NOTIFICATION",
    )),
    "activity_velocity": Signal(-2.0, "collapse in transaction volume", Playbook(
        "CASHBACK",
        "You have been transacting less lately. Spend this week and earn 1% cashback.",
        "PUSH_NOTIFICATION",
    )),
    "rapid_balance_drain": Signal(1.5, "outflows dominating recent activity", Playbook(
        "FEE_WAIVER",
        "Most of your recent movement was outbound. We have waived your fees for the next 30 days.",
        "PUSH_NOTIFICATION",
    )),
    "failure_rate": Signal(1.5, "settled transactions failing", Playbook(
        "FEE_WAIVER",
        "Some of your transactions were declined. We have waived the associated fees.",
        "EMAIL",
    )),
    "recency_days": Signal(1.5, "account going quiet", Playbook(
        "CASHBACK",
        "It has been a while since you last used your account. Come back and earn 1% cashback.",
        "EMAIL",
    )),
    "text_churn_score": Signal(1.5, "disputes citing fraud or non-delivery", Playbook(
        "CASHBACK",
        "We are sorry about the disputes on your account. Here is cashback while we make it right.",
        "EMAIL",
    )),
    "withdrawal_share_recent": Signal(1.0, "withdrawals dominating recent flow", Playbook(
        "FEE_WAIVER",
        "You have been withdrawing more than depositing. Free withdrawals for the next 30 days.",
        "PUSH_NOTIFICATION",
    )),
}

DEFAULT_PLAYBOOKS: Dict[str, Playbook] = {
    "saas": Playbook(
        "IN_APP_TOUR",
        "Thanks for being with us. Here is a tour of the features your plan already includes.",
        "IN_APP",
    ),
    "telecom": Playbook(
        "FREE_DATA",
        "Thanks for staying with us. Here is 1GB on us this month.",
        "SMS",
    ),
    "fintech": Playbook(
        "CASHBACK",
        "Thanks for banking with us. Earn 1% cashback on your next five transactions.",
        "PUSH_NOTIFICATION",
    ),
}

# -- schema resolution -------------------------------------------------------

# Matched on the exact column name: a substring test would let "plan_id" turn a
# transaction table into a dimension table.
DIMENSION_NAMES = frozenset({"tier", "plan", "region", "segment", "package"})
TEXT_HINTS = ("subject", "notes", "note", "comment", "description", "message", "body", "reason", "feedback", "remark", "detail")
AMOUNT_HINTS = ("amount", "total", "price", "fee", "balance", "charge", "cost", "revenue", "value")
# "time" has to be delimited: a bare substring test matches "sentiment".
TIMESTAMP_HINTS = ("timestamp", "date", "_at", "_ts", "_time", "time_")
TIMESTAMP_NAMES = frozenset({"time"})
NOISE_HINTS = ("hash", "ip_", "_ip", "mac", "agent", "session", "imsi", "jwt", "token", "cookie", "fingerprint")


class ScoredEntity(NamedTuple):
    entity_id: str
    probability: float
    tier: str
    drivers: List[str]
    dominant_feature: Optional[str]
    dominant: Optional[Signal]
    normalised: Dict[str, float]


def extract_payload(user_prompt: str) -> Any:
    """Recover the JSON the caller embedded in its prompt.

    Both prompt builders prefix the payload with prose that contains no
    brackets, so the first bracket position that parses is the payload.
    """
    for start, char in enumerate(user_prompt):
        if char not in "[{":
            continue
        try:
            return json.loads(user_prompt[start:])
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found in the user prompt")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalisation_bounds(values: pd.Series) -> tuple[float, float]:
    """Winsorise continuous signals; leave binary flags on their natural scale."""
    if values.nunique(dropna=True) <= 2:
        return float(values.min()), float(values.max())
    return (
        float(values.quantile(WINSORISE_QUANTILE)),
        float(values.quantile(1.0 - WINSORISE_QUANTILE)),
    )


def score_batch(payload: Sequence[Dict[str, Any]], signals: Dict[str, Signal]) -> List[ScoredEntity]:
    """Score a batch by normalising each signal across the batch.

    Normalising against the batch rather than against absolute thresholds keeps
    the output meaningful for a tenant whose whole customer base is healthy or
    wholly unhealthy, which is the same relative judgement an analyst makes.
    """
    entity_ids = [str(entry["entity_id"]) for entry in payload]
    frame = pd.DataFrame([entry.get("features") or {} for entry in payload], index=entity_ids)

    scores = pd.Series(0.0, index=frame.index)
    used_weight = 0.0
    normalised: Dict[str, pd.Series] = {}
    evidence: Dict[str, List[tuple[float, str, Signal]]] = {eid: [] for eid in entity_ids}

    for name, signal in signals.items():
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").astype("float64")
        if values.notna().sum() < 2:
            continue
        low, high = _normalisation_bounds(values)
        span = high - low
        if not math.isfinite(span) or span <= 0:
            # Constant across the batch: it cannot discriminate between customers.
            continue
        scaled = ((values - low) / span).clip(0.0, 1.0)
        # A customer with no row in that table sits at the batch average rather
        # than being pushed to one extreme by an absent value.
        observed = values.notna()
        scaled = scaled.mask(~observed, float(scaled[observed].mean()))
        normalised[name] = scaled
        scores = scores + signal.weight * scaled
        used_weight += abs(signal.weight)

        strength = scaled if signal.weight > 0 else 1.0 - scaled
        for entity_id, value in strength.items():
            if value >= MIN_DRIVER_STRENGTH:
                evidence[entity_id].append((abs(signal.weight) * float(value), name, signal))

    raw = scores / used_weight if used_weight > 0 else pd.Series(MIDPOINT_SCORE, index=frame.index)

    results: List[ScoredEntity] = []
    for entity_id in entity_ids:
        probability = _sigmoid(SIGMOID_STEEPNESS * (float(raw[entity_id]) - MIDPOINT_SCORE))
        ranked = sorted(evidence[entity_id], key=lambda item: item[0], reverse=True)
        drivers = [signal.label for _, _, signal in ranked[:MAX_DRIVERS]] or list(NO_SIGNAL_DRIVERS)
        top = ranked[0] if ranked else None
        results.append(ScoredEntity(
            entity_id=entity_id,
            probability=round(probability, 4),
            tier=risk_tier(probability),
            drivers=drivers,
            dominant_feature=top[1] if top else None,
            dominant=top[2] if top else None,
            normalised={name: round(float(series[entity_id]), 4) for name, series in normalised.items()},
        ))
    return results


def _playbook_for(scored: ScoredEntity, sector: str) -> Playbook:
    if scored.dominant is not None:
        return scored.dominant.playbook
    return DEFAULT_PLAYBOOKS[sector]


def _prediction_entry(scored: ScoredEntity, sector: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    playbook = _playbook_for(scored, sector)
    return {
        "entity_id": scored.entity_id,
        "churn_prediction": {
            "churn_probability": scored.probability,
            "risk_tier": scored.tier,
            **extra,
        },
        "retention_playbook": {
            "action_type": playbook.action_type,
            "action_payload": playbook.action_payload,
            "channel": playbook.channel,
        },
    }


class MockQwenGateway(IAIGateway):
    """Offline `IAIGateway` used when no Alibaba Cloud API key is configured."""

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        payload = extract_payload(user_prompt)
        if SCHEMA_RESOLVER_MARKER in system_prompt:
            return self.resolve_schema(payload)
        if SAAS_MARKER in system_prompt:
            return self.analyze_sector(payload, SAAS_SIGNALS, "saas", self._saas_extra)
        if TELECOM_MARKER in system_prompt:
            return self.analyze_sector(payload, TELECOM_SIGNALS, "telecom", self._telecom_extra)
        if FINTECH_MARKER in system_prompt:
            return self.analyze_sector(payload, FINTECH_SIGNALS, "fintech", self._fintech_extra)
        raise ValueError(
            "MockQwenGateway has no offline behaviour for a system prompt beginning "
            f"{system_prompt.strip()[:60]!r}"
        )

    # -- sector analysis -----------------------------------------------------

    def analyze_sector(
        self,
        payload: Sequence[Dict[str, Any]],
        signals: Dict[str, Signal],
        sector: str,
        extra,
    ) -> dict:
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of customer features for the {sector} core, got {type(payload).__name__}")
        features_by_id = {
            str(entry["entity_id"]): (entry.get("features") or {}) for entry in payload
        }
        scored = score_batch(payload, signals)
        return {
            "predictions": [
                _prediction_entry(entry, sector, extra(entry, features_by_id.get(entry.entity_id, {})))
                for entry in scored
            ]
        }

    def _saas_extra(self, scored: ScoredEntity, features: Dict[str, Any]) -> Dict[str, Any]:
        return {"primary_drivers": scored.drivers}

    def _telecom_extra(self, scored: ScoredEntity, features: Dict[str, Any]) -> Dict[str, Any]:
        # The enricher owns the volume and concentration thresholds behind this
        # flag; recomputing them here would duplicate that rule.
        regional = bool(features.get("regional_network_impact_flag", False))
        if regional:
            cause = "repeated dropped calls concentrated on a single tower point to a local network fault"
        elif scored.dominant is not None:
            cause = f"{scored.dominant.label} — the strongest churn signal in the last 30 days"
        else:
            cause = "no material churn signal in the last 30 days"
        return {"root_cause": cause, "regional_network_impact_flag": regional}

    def _fintech_extra(self, scored: ScoredEntity, features: Dict[str, Any]) -> Dict[str, Any]:
        recency = scored.normalised.get("recency_days")
        velocity = scored.normalised.get("activity_velocity")
        if recency is None or velocity is None:
            dormancy = "UNKNOWN"
        elif recency > HARD_DORMANCY_RECENCY:
            dormancy = "HARD_DORMANCY"
        elif velocity < SOFT_DORMANCY_VELOCITY:
            dormancy = "SOFT_DORMANCY"
        else:
            dormancy = "ACTIVE"
        return {"dormancy_type": dormancy}

    # -- schema resolution ---------------------------------------------------

    def resolve_schema(self, file_samples: Dict[str, str]) -> dict:
        """Classify uploaded tables by structure rather than by judgement.

        Column naming is a structural problem with a right answer, so resolving
        it with rules keeps offline mode exact and testable instead of
        approximating what an LLM might have said.
        """
        if not isinstance(file_samples, dict) or not file_samples:
            raise ValueError("Schema resolution needs at least one file sample")

        tables = [_Sample(name, _columns_of(sample)) for name, sample in sorted(file_samples.items())]
        primary_key = _primary_key(tables)
        return {
            "primary_entity_key": primary_key,
            "tables": [
                {
                    "file_name": table.file_name,
                    "role": _classify_role(table.columns),
                    "primary_entity_key": _entity_key_of(table.columns, primary_key),
                    "timestamp_column": _find_timestamp(table.columns),
                    "noise_columns": _find_noise(table.columns, _entity_key_of(table.columns, primary_key)),
                }
                for table in tables
            ],
        }


class _Sample(NamedTuple):
    file_name: str
    columns: List[str]


def _columns_of(sample: str) -> List[str]:
    """Read the header out of a sample CSV, falling back to a plain split."""
    try:
        return list(pd.read_csv(io.StringIO(sample), nrows=0).columns)
    except Exception:
        logger.warning("Could not parse a file sample as CSV; falling back to the header line")
        first_line = (sample or "").splitlines()[0] if sample else ""
        return [column.strip() for column in first_line.split(",") if column.strip()]


def _primary_key(tables: Sequence[_Sample]) -> str:
    """The ``*_id`` column shared by the most tables is the entity key.

    Frequency alone leaves ties: an upload where every table contributes one
    distinct id has nothing to separate them, and `max` would then return
    whichever name happened to sort first. The tiebreak is position — a table's
    own row key is conventionally its first column, so the candidate that most
    often appears somewhere *other* than first is the one other tables point at.
    """
    shared: Dict[str, int] = {}
    referenced: Dict[str, int] = {}
    for table in tables:
        id_columns = [c for c in table.columns if c.lower().endswith("_id")]
        for column in set(id_columns):
            shared[column] = shared.get(column, 0) + 1
        for column in id_columns[1:]:
            referenced[column] = referenced.get(column, 0) + 1
    if not shared:
        raise ValueError("No *_id column found in any uploaded sample; cannot infer a primary entity key")
    return max(sorted(shared), key=lambda name: (shared[name], referenced.get(name, 0)))


def _entity_key_of(columns: Iterable[str], primary_key: str) -> str:
    columns = list(columns)
    if primary_key in columns:
        return primary_key
    for column in columns:
        if column.lower().endswith("_id"):
            return column
    return columns[0] if columns else primary_key


def _has_hint(columns: Iterable[str], hints: Iterable[str]) -> bool:
    lowered = [c.lower() for c in columns]
    return any(hint in column for column in lowered for hint in hints)


def _classify_role(columns: List[str]) -> str:
    if DIMENSION_NAMES & {c.lower() for c in columns}:
        return ROLE_DIMENSION
    if _has_hint(columns, TEXT_HINTS):
        return ROLE_TEXT
    if _has_hint(columns, AMOUNT_HINTS):
        return ROLE_TRANSACTIONAL
    if _has_hint(columns, TIMESTAMP_HINTS):
        return ROLE_TIME_SERIES
    return ROLE_DIMENSION


def _find_timestamp(columns: List[str]) -> Optional[str]:
    for hint in TIMESTAMP_HINTS:
        for column in columns:
            if hint in column.lower():
                return column
    for column in columns:
        if column.lower() in TIMESTAMP_NAMES:
            return column
    return None


def _find_noise(columns: List[str], entity_key: str) -> List[str]:
    """Hashes, IPs, device fingerprints and session tokens carry no signal.

    Surrogate row keys are deliberately *not* treated as noise. A three-row
    sample cannot distinguish `event_id` from `tower_id` by cardinality, and
    dropping `tower_id` would silently disable the regional network fault flag
    that the telecom enricher derives from it.
    """
    return [
        column for column in columns
        if column != entity_key and any(hint in column.lower() for hint in NOISE_HINTS)
    ]
