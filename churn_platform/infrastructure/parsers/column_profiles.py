"""Profile actual uploaded values and make conservative offline column proposals."""
import csv
import io
import re
from churn_platform.domain.models.schema_mapping import ColumnMapping


def profiles(samples):
    result = {}
    for name, sample in samples.items():
        reader = csv.DictReader(io.StringIO(sample))
        values = {column: [] for column in (reader.fieldnames or [])}
        for row in reader:
            for column in values:
                value = str(row.get(column) or "").strip()[:200]
                if value and value not in values[column] and len(values[column]) < 3:
                    values[column].append(value)
        result[name] = values
    return result


def offline_column(source, values, table):
    lower = source.lower()
    role, confidence = "UNKNOWN", 0.35
    if source == table.primary_entity_key and lower in {"customer_id", "user_id", "subscriber_id", "account_id", "entity_id"}:
        role, confidence = "CUSTOMER_ID", .99
    elif source == table.timestamp_column:
        role, confidence = "TIMESTAMP", .95
    elif source in table.noise_columns:
        role, confidence = "NOISE_IGNORE", .95
    elif any(word in lower for word in ("amount", "total", "price", "fee", "balance", "charge", "cost", "revenue")):
        role, confidence = ("TRANSACTION_AMOUNT" if table.role == "TRANSACTIONAL" else "ATTRIBUTE"), .9
    elif lower in {"status", "state", "result", "outcome", "call_status"}:
        role, confidence = "STATUS", .95
    elif lower in {"event_type", "transaction_type", "tx_type", "type", "action", "activity"}:
        role, confidence = "EVENT_TYPE", .9
    elif lower in {"subject", "description", "message", "body", "notes", "reason", "feedback"}:
        role, confidence = "TEXT", .9
    elif lower in {"inv_id", "duration_sec", "rec_id", "comp_id", "tx_id", "tier", "plan", "plan_tier", "region", "segment", "package", "name", "email", "signup_date", "created_at", "open_date", "due_date", "age", "country", "tower_id", "duration_seconds", "duration", "merchant_category", "category", "ticket_id", "event_id", "invoice_id", "transaction_id", "call_id", "recharge_id", "complaint_id", "dispute_id", "swipe_id", "tenure_months", "sentiment", "sentiment_score", "priority", "resolution_status"}:
        role, confidence = "ATTRIBUTE", .9
    elif re.search(r"(date|timestamp|_at)$", lower) and source != table.timestamp_column:
        role, confidence = "ATTRIBUTE", .85
    return ColumnMapping(source_column=source, canonical_role=role, confidence=confidence,
        sample_values=values, reasoning="Recognized header and table context" if role != "UNKNOWN" else "Meaning cannot be established from these samples; please confirm")
