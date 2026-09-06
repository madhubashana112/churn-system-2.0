"""Schema vocabulary shared by the resolver, the synthesizer and the enrichers.

The role constants live here rather than in the parser layer because they are
part of the domain contract: an ``AISchemaResolver`` promises to emit one of
these four values and every consumer branches on them.
"""

from pydantic import BaseModel, field_validator
from typing import List

ROLE_DIMENSION = "DIMENSION"
ROLE_TIME_SERIES = "TIME_SERIES_EVENT"
ROLE_TRANSACTIONAL = "TRANSACTIONAL"
ROLE_TEXT = "UNSTRUCTURED_TEXT"

TABLE_ROLES = frozenset({ROLE_DIMENSION, ROLE_TIME_SERIES, ROLE_TRANSACTIONAL, ROLE_TEXT})


class TableClassification(BaseModel):
    file_name: str
    role: str
    primary_entity_key: str
    timestamp_column: str | None = None
    noise_columns: List[str] = []

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
