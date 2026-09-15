"""Schema vocabulary shared by the resolver, the synthesizer and the enrichers.

The role constants live here rather than in the parser layer because they are
part of the domain contract: an ``AISchemaResolver`` promises to emit one of
these four values and every consumer branches on them.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal

ROLE_DIMENSION = "DIMENSION"
ROLE_TIME_SERIES = "TIME_SERIES_EVENT"
ROLE_TRANSACTIONAL = "TRANSACTIONAL"
ROLE_TEXT = "UNSTRUCTURED_TEXT"

TABLE_ROLES = frozenset({ROLE_DIMENSION, ROLE_TIME_SERIES, ROLE_TRANSACTIONAL, ROLE_TEXT})


AUTO_RESOLVED = "AUTO_RESOLVED"
REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
CONFIDENCE_THRESHOLD = 0.80
CANONICAL_ROLES = ("CUSTOMER_ID", "TIMESTAMP", "TRANSACTION_AMOUNT", "STATUS", "EVENT_TYPE", "TEXT", "ATTRIBUTE", "NOISE_IGNORE", "CUSTOM", "UNKNOWN")
REQUIRED_SECTOR_ROLES = {
    "saas": {"CUSTOMER_ID", "TIMESTAMP"},
    "telecom": {"CUSTOMER_ID", "TIMESTAMP"},
    "fintech": {"CUSTOMER_ID", "TIMESTAMP", "TRANSACTION_AMOUNT"},
}
CANONICAL_NAMES = {"CUSTOMER_ID": "customer_id", "TIMESTAMP": "timestamp",
    "TRANSACTION_AMOUNT": "transaction_amount", "STATUS": "status", "EVENT_TYPE": "event_type", "TEXT": "text"}
REFERENCE_DATE = "2025-06-01T12:00:00Z"


class ColumnMapping(BaseModel):
    source_column: str
    canonical_role: str = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0, le=1)
    sample_values: list[str] = Field(default_factory=list, max_length=3)
    reasoning: str = ""
    status: Literal["AUTO_RESOLVED", "REQUIRES_HUMAN_REVIEW"] = REQUIRES_HUMAN_REVIEW
    custom_label: str | None = Field(default=None, max_length=100)

    @field_validator("canonical_role")
    @classmethod
    def validate_canonical_role(cls, value):
        role = value.strip().upper()
        if role not in CANONICAL_ROLES:
            raise ValueError("Unknown canonical role")
        return role

    @field_validator("custom_label")
    @classmethod
    def validate_label(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value or any(ord(c) < 32 for c in value):
            raise ValueError("Custom names must contain visible text")
        return value

    @model_validator(mode="after")
    def confidence_status(self):
        self.status = (REQUIRES_HUMAN_REVIEW if self.confidence < CONFIDENCE_THRESHOLD or
            self.canonical_role == "UNKNOWN" or (self.canonical_role == "CUSTOM" and not self.custom_label)
            else AUTO_RESOLVED)
        return self

    @property
    def target_name(self):
        return self.custom_label if self.canonical_role == "CUSTOM" else CANONICAL_NAMES.get(self.canonical_role, self.source_column)


class TableClassification(BaseModel):
    file_name: str
    role: str
    primary_entity_key: str
    timestamp_column: str | None = None
    noise_columns: List[str] = []
    columns: List[ColumnMapping] = Field(default_factory=list)

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        """Reject an unknown role instead of letting it silently match nothing.

        Kept as ``str`` rather than an enum so the resolved schema stays plain
        JSON all the way to the browser. An unrecognised role used to pass
        straight through and then skip every role-based branch downstream, so a
        table classified ``"DIMENSIONS"`` contributed no features at all with no
        error anywhere.
        """
        normalised = value.strip().upper()
        if normalised not in TABLE_ROLES:
            raise ValueError(
                f"Unknown table role {value!r}; expected one of {sorted(TABLE_ROLES)}"
            )
        return normalised


class SchemaMapping(BaseModel):
    primary_entity_key: str
    tables: List[TableClassification]
    status: Literal["AUTO_RESOLVED", "REQUIRES_HUMAN_REVIEW"] = AUTO_RESOLVED
    review_reasons: list[str] = Field(default_factory=list)

    sector: str = ""

    def assess(self, sector: str = ""):
        self.sector = sector or self.sector
        reasons = []
        critical = REQUIRED_SECTOR_ROLES.get(self.sector.strip().lower(), {"CUSTOMER_ID", "TIMESTAMP"})
        for table in self.tables:
            if not table.columns:
                continue  # Legacy stored analyses remain readable.
            roles = [c.canonical_role for c in table.columns]
            if roles.count("CUSTOMER_ID") != 1:
                reasons.append(f"{table.file_name}: select exactly one customer ID column")
            if table.role in (ROLE_TIME_SERIES, ROLE_TRANSACTIONAL) and roles.count("TIMESTAMP") != 1:
                reasons.append(f"{table.file_name}: select exactly one timestamp column")
            if table.role == ROLE_TRANSACTIONAL and "TRANSACTION_AMOUNT" in critical and "TRANSACTION_AMOUNT" not in roles:
                reasons.append(f"{table.file_name}: select a transaction amount column")
            targets = {}
            for column in table.columns:
                if column.canonical_role != "NOISE_IGNORE":
                    targets.setdefault(column.target_name, []).append(column.source_column)
            for target, sources in targets.items():
                if len(sources) > 1:
                    reasons.append(f"{table.file_name}: {', '.join(sources)} all map to '{target}'. "
                                   "Keep only one in this role; set the others to Attribute or give them unique custom names.")
            for column in table.columns:
                if column.status == REQUIRES_HUMAN_REVIEW:
                    reasons.append(f"{table.file_name}: confirm {column.source_column}")
        self.review_reasons = reasons
        self.status = REQUIRES_HUMAN_REVIEW if reasons else AUTO_RESOLVED
        return self

    @model_validator(mode="after")
    def assess_columns(self):
        return self.assess()
