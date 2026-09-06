"""Synthetic multi-sheet mock datasets for the three sector pipelines.

Each sector produces four CSVs matching the raw server exports a tenant would
upload: one DIMENSION table, one TIME_SERIES_EVENT table, one TRANSACTIONAL
table and one UNSTRUCTURED_TEXT table.

Entity indices 0-24 form a deliberate churning cohort and 25-99 a healthy one,
so downstream feature engineering has real signal to separate. Churn is modelled
as a *collapse in recent activity against a normal prior baseline*, plus
sector-specific distress (failed payments, dropped calls, balance drain) and
churn-intent language in free text.

Everything derives from SEED and REFERENCE_DATE, so repeated runs are
byte-identical. Feature windows must therefore be anchored to the maximum
timestamp present in the data rather than to the wall clock.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
REFERENCE_DATE = datetime(2025, 6, 1, 12, 0, 0)
N_ENTITIES = 100
CHURN_COHORT_SIZE = 25

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FMT = "%Y-%m-%d"

SAAS_TICKET_SUBJECTS_CHURNING = [
    "Need to cancel my Pro plan immediately",
    "Your competitor offers the same features at half the price",
    "Export took three hours and timed out twice",
    "We are switching to a cheaper provider next month",
    "Dashboard is far too slow to be usable, requesting a refund",
    "Cancel subscription and refund the last invoice please",
    "Billing error still unresolved after three separate tickets",
    "Seat downgrade request because most of the team is leaving",
    "Login has been broken for a week, escalating to our CTO",
    "Want to unsubscribe before the renewal charge hits",
    "Performance is unusable and support never resolved my case",
    "Chargeback filed with our bank for the duplicate invoice",
]

SAAS_TICKET_SUBJECTS_HEALTHY = [
    "How do I invite a new team member",
    "Question about the API rate limits",
    "Feature request: dark mode for the dashboard",
    "Thanks for the quick fix on the report export",
    "Need help understanding the invoice breakdown",
    "Minor typo on the settings page",
    "Can you add CSV import for contacts",
    "Rescheduling our onboarding call to next week",
    "Where do I configure single sign on",
    "Requesting a walkthrough of the analytics tab",
]

TELECOM_COMPLAINTS_CHURNING = [
    "Requesting MNP port out to another operator immediately",
    "Network coverage in my area has been terrible for three weeks",
    "Calls keep dropping, I want to cancel my connection",
    "Switching providers because data speeds are unusable",
    "Billing dispute unresolved after multiple calls to support",
    "Port out request, the competitor offers better prepaid rates",
    "Dropped every call today, this is unacceptable, refund my recharge",
    "Want to cancel the postpaid plan because of repeated overbilling",
    "No data service for two weeks, escalating this complaint",
    "MNP port out code requested, moving to a cheaper network",
]

TELECOM_COMPLAINTS_HEALTHY = [
    "Asked how to enable international roaming",
    "Requested a data plan upgrade for the family line",
    "Query about the last recharge receipt",
    "Reported a one time slow data experience during travel",
    "Asked to update the billing address on the account",
    "General enquiry about add on data packs",
    "Requested a duplicate SIM for a new handset",
    "Asked how the loyalty points redemption works",
]


def _entity_ids(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, N_ENTITIES + 1)]


def _is_churning(index: int) -> bool:
    return index < CHURN_COHORT_SIZE


def _sample_days(rng: np.random.Generator, buckets: list[tuple[int, int, float]], n: int) -> list[int]:
    """Sample n 'days ago' offsets from weighted inclusive (lo, hi, weight) buckets."""
    los = [b[0] for b in buckets]
    his = [b[1] for b in buckets]
    weights = np.array([b[2] for b in buckets], dtype=float)
    picks = rng.choice(len(buckets), size=n, p=weights / weights.sum())
    return [int(rng.integers(los[i], his[i] + 1)) for i in picks]


def _timestamp(rng: np.random.Generator, days_ago: int) -> str:
    dt = REFERENCE_DATE - timedelta(days=int(days_ago), seconds=int(rng.integers(0, 86400)))
    return dt.strftime(TS_FMT)


def _date(days_ago: int) -> str:
    return (REFERENCE_DATE - timedelta(days=int(days_ago))).strftime(DATE_FMT)


def _noise_hash(rng: np.random.Generator, prefix: str) -> str:
    return f"{prefix}_{int(rng.integers(100000, 999999))}"


def create_dirs() -> None:
    for sector in ("saas", "telecom", "fintech"):
        os.makedirs(os.path.join(DATA_DIR, sector), exist_ok=True)


def _write(sector: str, name: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, sector, name), index=False)
    return df


def generate_saas_data() -> None:
    rng = np.random.default_rng(SEED)
    user_ids = _entity_ids("usr")

    users = []
    for i, user_id in enumerate(user_ids):
        users.append({
            "user_id": user_id,
            "signup_date": _date(int(rng.integers(120, 730))),
            "tier": str(rng.choice(["Basic", "Pro", "Enterprise"], p=[0.5, 0.35, 0.15])),
            "ip_address": f"192.168.{int(rng.integers(0, 255))}.{int(rng.integers(1, 255))}",
            "last_user_agent": str(rng.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
                "Mozilla/5.0 (X11; Linux x86_64) Firefox/126.0",
            ])),
        })
    _write("saas", "users.csv", users)

    # Healthy users spread activity evenly across the 60 day window; churning
    # users were just as active but have gone quiet in the last week, which is
    # what drives activity_velocity below 1.0.
    events = []
    for i, user_id in enumerate(user_ids):
        churning = _is_churning(i)
        n = 15 if churning else 12
        buckets = [(0, 7, 0.04), (8, 37, 0.56), (38, 60, 0.40)] if churning \
            else [(0, 7, 0.12), (8, 37, 0.50), (38, 60, 0.38)]
        types, probs = (["login", "feature_a", "feature_b", "export", "logout"],
                        [0.20, 0.10, 0.05, 0.45, 0.20]) if churning \
            else (["login", "feature_a", "feature_b", "export", "logout"],
                  [0.40, 0.20, 0.20, 0.10, 0.10])
        for days_ago in _sample_days(rng, buckets, n):
            events.append({
                "event_id": f"evt_{len(events) + 1}",
                "user_id": user_id,
                "timestamp": _timestamp(rng, days_ago),
                "event_type": str(rng.choice(types, p=probs)),
                "session_hash": _noise_hash(rng, "sess"),
            })
    _write("saas", "events_log.csv", events)

    invoices = []
    for i, user_id in enumerate(user_ids):
        churning = _is_churning(i)
        n = 3 if churning else 2
        for _ in range(n):
            invoices.append({
                "inv_id": f"inv_{len(invoices) + 1}",
                "user_id": user_id,
                "amount": float(rng.choice([29.99, 99.99, 299.99])),
                "status": str(rng.choice(["PAID", "FAILED"], p=[0.35, 0.65] if churning else [0.95, 0.05])),
                "due_date": _date(int(rng.integers(-15, 60))),
            })
    _write("saas", "invoices.csv", invoices)

    tickets = []
    for i, user_id in enumerate(user_ids):
        churning = _is_churning(i)
        if churning:
            n = int(rng.integers(1, 4))
        elif rng.random() < 0.30:
            n = 1
        else:
            continue
        for _ in range(n):
            tickets.append({
                "ticket_id": f"tkt_{len(tickets) + 1}",
                "user_id": user_id,
                "subject": str(rng.choice(
                    SAAS_TICKET_SUBJECTS_CHURNING if churning else SAAS_TICKET_SUBJECTS_HEALTHY)),
                "sentiment": str(rng.choice(
                    ["Neutral", "Negative", "Very Negative"],
                    p=[0.05, 0.35, 0.60] if churning else [0.70, 0.25, 0.05])),
                "status": str(rng.choice(["OPEN", "CLOSED"], p=[0.75, 0.25] if churning else [0.35, 0.65])),
            })
    _write("saas", "tickets.csv", tickets)


def generate_telecom_data() -> None:
    rng = np.random.default_rng(SEED + 1)
    sub_ids = _entity_ids("sub")
    towers = ["TWR_A", "TWR_B", "TWR_C"]

    subscribers = []
    for i, sub_id in enumerate(sub_ids):
        subscribers.append({
            "subscriber_id": sub_id,
            "plan": str(rng.choice(["Prepaid", "Postpaid"], p=[0.65, 0.35])),
            "region": str(rng.choice(["North", "South", "East", "West"])),
            "sim_imsi_hash": _noise_hash(rng, "imsi"),
        })
    _write("telecom", "subscribers.csv", subscribers)

    plans = {s["subscriber_id"]: s["plan"] for s in subscribers}

    cdrs = []
    for i, sub_id in enumerate(sub_ids):
        churning = _is_churning(i)
        n = 12 if churning else 10
        buckets = [(0, 7, 0.05), (8, 37, 0.55), (38, 60, 0.40)] if churning \
            else [(0, 7, 0.12), (8, 37, 0.50), (38, 60, 0.38)]
        # Churning subscribers concentrate their dropped calls on a single
        # tower so the regional network impact flag has something to detect.
        home_tower = str(rng.choice(towers))
        for days_ago in _sample_days(rng, buckets, n):
            dropped = rng.random() < (0.20 if churning else 0.03)
            cdrs.append({
                "call_id": f"call_{len(cdrs) + 1}",
                "subscriber_id": sub_id,
                "call_timestamp": _timestamp(rng, days_ago),
                "duration_sec": int(rng.integers(5, 90) if dropped else rng.integers(30, 1200)),
                "call_status": "DROPPED" if dropped else "COMPLETED",
                "tower_id": home_tower if (churning and dropped and rng.random() < 0.8)
                else str(rng.choice(towers)),
            })
    _write("telecom", "network_cdrs.csv", cdrs)

    # Recharges walk backwards from the most recent top-up. Healthy subscribers
    # top up every week or so; churning ones let the gap widen each time and
    # have not recharged recently at all.
    recharges = []
    for i, sub_id in enumerate(sub_ids):
        if plans[sub_id] != "Prepaid":
            continue
        churning = _is_churning(i)
        if churning:
            gaps = [int(rng.integers(18, 28)), int(rng.integers(11, 17)),
                    int(rng.integers(6, 10)), int(rng.integers(4, 7))]
            days_ago = int(rng.integers(25, 45))
        else:
            gaps = [int(rng.integers(6, 11)) for _ in range(4)]
            days_ago = int(rng.integers(3, 9))
        for gap in [0] + gaps:
            days_ago += gap
            if days_ago > 90:
                break
            recharges.append({
                "rec_id": f"rec_{len(recharges) + 1}",
                "subscriber_id": sub_id,
                "amount": int(rng.choice([10, 20, 50, 100])),
                "recharge_date": _date(days_ago),
            })
    _write("telecom", "recharge_history.csv", recharges)

    complaints = []
    for i, sub_id in enumerate(sub_ids):
        churning = _is_churning(i)
        if churning:
            n = int(rng.integers(1, 3))
        elif rng.random() < 0.18:
            n = 1
        else:
            continue
        for _ in range(n):
            complaints.append({
                "comp_id": f"comp_{len(complaints) + 1}",
                "subscriber_id": sub_id,
                "category": str(rng.choice(
                    ["MNP_PORT_OUT", "BILLING", "NETWORK", "GENERAL"],
                    p=[0.45, 0.15, 0.35, 0.05] if churning else [0.05, 0.30, 0.25, 0.40])),
                "notes": str(rng.choice(
                    TELECOM_COMPLAINTS_CHURNING if churning else TELECOM_COMPLAINTS_HEALTHY)),
            })
    _write("telecom", "complaints.csv", complaints)


def generate_fintech_data() -> None:
    rng = np.random.default_rng(SEED + 2)
    account_ids = _entity_ids("acc")

    accounts = []
    for i, account_id in enumerate(account_ids):
        accounts.append({
            "account_id": account_id,
            "tier": str(rng.choice(["Standard", "Premium", "Metal"], p=[0.6, 0.3, 0.1])),
            "created_at": _date(int(rng.integers(120, 900))),
            "device_mac_hash": _noise_hash(rng, "mac"),
        })
    _write("fintech", "accounts.csv", accounts)

    # Churning accounts drain their balance in a burst during days 8-14 and then
    # go silent, so both balance_drain_ratio and activity_velocity fire together.
    # Healthy wallets accumulate, so deposits outweigh withdrawals and their
    # baseline drain ratio stays well clear of the 0.6 threshold.
    healthy_mix = [0.45, 0.25, 0.30]
    txns = []
    for i, account_id in enumerate(account_ids):
        churning = _is_churning(i)
        if churning:
            # The P2P volume sits in the older buckets so the 75% failure rate
            # produces a run of consecutive failed transfers rather than scatter.
            plan = [((8, 14), 4, [0.10, 0.80, 0.10], (180.0, 520.0)),
                    ((15, 37), 4, [0.15, 0.35, 0.50], (20.0, 300.0)),
                    ((38, 90), 4, [0.15, 0.35, 0.50], (20.0, 300.0))]
        else:
            plan = [((0, 7), 1, healthy_mix, (20.0, 300.0)),
                    ((8, 37), 4, healthy_mix, (20.0, 300.0)),
                    ((38, 90), 5, healthy_mix, (20.0, 300.0))]

        for (lo, hi), count, type_probs, amount_range in plan:
            tx_types = [str(t) for t in rng.choice(
                ["DEPOSIT", "WITHDRAWAL", "P2P"], size=count, p=type_probs)]
            for tx_type in tx_types:
                days_ago = int(rng.integers(lo, hi + 1))
                if churning and tx_type == "P2P":
                    status = str(rng.choice(["SUCCESS", "FAILED"], p=[0.25, 0.75]))
                else:
                    status = str(rng.choice(["SUCCESS", "FAILED"], p=[0.95, 0.05] if churning else [0.96, 0.04]))
                txns.append({
                    "tx_id": f"tx_{len(txns) + 1}",
                    "account_id": account_id,
                    "timestamp": _timestamp(rng, days_ago),
                    "amount": round(float(rng.uniform(*amount_range)), 2),
                    "tx_type": tx_type,
                    "status": status,
                })
    txns.sort(key=lambda t: t["tx_id"])
    _write("fintech", "ledger_transactions.csv", txns)

    swipes = []
    for i, account_id in enumerate(account_ids):
        churning = _is_churning(i)
        n = 2 if churning else 5
        buckets = [(8, 37, 0.60), (38, 90, 0.40)] if churning \
            else [(0, 7, 0.20), (8, 37, 0.45), (38, 90, 0.35)]
        for days_ago in _sample_days(rng, buckets, n):
            swipes.append({
                "swipe_id": f"swp_{len(swipes) + 1}",
                "account_id": account_id,
                "swipe_timestamp": _timestamp(rng, days_ago),
                "merchant_category": str(rng.choice(
                    ["GROCERY", "ENTERTAINMENT", "TRAVEL", "DINING"])),
                "status": str(rng.choice(
                    ["APPROVED", "DECLINED"], p=[0.82, 0.18] if churning else [0.97, 0.03])),
            })
    _write("fintech", "card_swipes.csv", swipes)

    disputes = []
    for i, account_id in enumerate(account_ids):
        churning = _is_churning(i)
        if churning:
            n = int(rng.integers(1, 4))
        elif rng.random() < 0.14:
            n = 1
        else:
            continue
        for _ in range(n):
            disputes.append({
                "dispute_id": f"dsp_{len(disputes) + 1}",
                "account_id": account_id,
                "reason": str(rng.choice(
                    ["Fraudulent", "Not Received", "Duplicate"],
                    p=[0.60, 0.20, 0.20] if churning else [0.20, 0.45, 0.35])),
                "open_date": _date(int(rng.integers(0, 45))),
            })
    _write("fintech", "disputes.csv", disputes)


if __name__ == "__main__":
    print(f"Generating mock data anchored to {REFERENCE_DATE:%Y-%m-%d} "
          f"(entities 1-{CHURN_COHORT_SIZE} are the churning cohort)...")
    create_dirs()
    generate_saas_data()
    generate_telecom_data()
    generate_fintech_data()
    print(f"Mock data generated successfully in {DATA_DIR}")
