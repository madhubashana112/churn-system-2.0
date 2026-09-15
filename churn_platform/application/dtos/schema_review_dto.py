from typing import Literal
from pydantic import BaseModel, Field
from churn_platform.domain.models.schema_mapping import SchemaMapping, CANONICAL_ROLES, CANONICAL_NAMES

class SchemaReviewResponse(BaseModel):
    requires_human_review: Literal[True] = True
    upload_session_id: str
    schema_mapping: SchemaMapping
    canonical_roles: list[str] = Field(default_factory=lambda: [r for r in CANONICAL_ROLES if r != "UNKNOWN"])
    canonical_names: dict[str, str] = Field(default_factory=lambda: dict(CANONICAL_NAMES))
    expires_in_seconds: int = 3600

class ConfirmedColumn(BaseModel):
    file_name: str
    source_column: str
    canonical_role: str
    custom_label: str | None = Field(default=None, max_length=100)

class ConfirmSchemaRequest(BaseModel):
    tenant_id: str
    upload_session_id: str
    mappings: list[ConfirmedColumn] = Field(min_length=1, max_length=500)
    table_roles: dict[str, str] = Field(default_factory=dict)
