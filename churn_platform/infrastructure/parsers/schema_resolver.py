from typing import Dict
import json
from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway
from churn_platform.domain.interfaces.i_schema_resolver import ISchemaResolver
from churn_platform.domain.models.schema_mapping import SchemaMapping, TableClassification, ColumnMapping, CANONICAL_NAMES, ROLE_DIMENSION
from churn_platform.infrastructure.ai.prompts.schema_resolver_prompts import SCHEMA_RESOLVER_SYSTEM_PROMPT
from churn_platform.infrastructure.parsers.column_profiles import profiles, offline_column
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway


class AISchemaResolver(ISchemaResolver):
    def __init__(self, gateway: IAIGateway):
        self.gateway = gateway

    async def resolve(self, file_samples: Dict[str, str], remembered: dict | None = None) -> SchemaMapping:
        if not file_samples:
            raise ValueError("Schema resolution needs at least one file sample")
        remembered = remembered or {}
        data = profiles(file_samples)
        known_tables, unresolved = {}, {}
        for filename, values in data.items():
            memory = remembered.get(filename, {})
            if memory and all(c in memory.get("columns", {}) for c in values):
                known_tables[filename] = {"file_name": filename, "role": memory["role"],
                    "primary_entity_key": "", "columns": []}
            else:
                unresolved[filename] = file_samples[filename]
        proposed = {}
        if unresolved:
            payload = unresolved if isinstance(self.gateway, MockQwenGateway) else {"samples": unresolved, "confirmed_columns": remembered, "column_profiles": {name: data[name] for name in unresolved}}
            try:
                response = await self.gateway.generate_json(SCHEMA_RESOLVER_SYSTEM_PROMPT,
                    "Analyze these file samples:\n" + json.dumps(payload, ensure_ascii=False))
            except ValueError:
                # No inferable key or malformed model JSON is precisely when
                # human review should take over, never silently skip a file.
                response = {}
            if not isinstance(response, dict) or not isinstance(response.get("tables", []), list):
                response = {}
            proposed = {t["file_name"]: t for t in response.get("tables", []) if isinstance(t, dict) and t.get("file_name") in unresolved}
        tables = []
        for filename, values in data.items():
            raw = known_tables.get(filename) or proposed.get(filename) or {"file_name":filename,"role":"DIMENSION","primary_entity_key":""}
            try:
                table = TableClassification(**{k:v for k,v in raw.items() if k != "columns"})
            except (ValueError, TypeError):
                table = TableClassification(file_name=filename,role="DIMENSION",primary_entity_key="")
            if filename in remembered:
                table.role = remembered[filename]["role"]
            ai_columns = {c.get("source_column"):c for c in raw.get("columns", []) if isinstance(c,dict)}
            memory = remembered.get(filename, {}).get("columns", {})
            for source, sample_values in values.items():
                if source in memory:
                    c = ColumnMapping(**{**memory[source], "source_column":source,"confidence":1.0,
                        "sample_values":sample_values,"reasoning":"Previously confirmed for this workspace"})
                elif source in ai_columns:
                    candidate = {**ai_columns[source], "sample_values":sample_values}
                    try:
                        c = ColumnMapping(**candidate)
                    except (ValueError, TypeError):
                        c = ColumnMapping(source_column=source, sample_values=sample_values, reasoning="Invalid model proposal; review required")
                elif isinstance(self.gateway, MockQwenGateway):
                    c = offline_column(source, sample_values, table)
                else:
                    c = ColumnMapping(source_column=source, sample_values=sample_values, reasoning="Model did not provide this column")
                table.columns.append(c)
            preserve_repeated_attributes(table, set(memory))
            keys = [c.source_column for c in table.columns if c.canonical_role == "CUSTOMER_ID"]
            stamps = [c.source_column for c in table.columns if c.canonical_role == "TIMESTAMP"]
            if len(keys) == 1: table.primary_entity_key = keys[0]
            if len(stamps) == 1: table.timestamp_column = stamps[0]
            tables.append(table)
        return SchemaMapping(primary_entity_key=next((t.primary_entity_key for t in tables if t.primary_entity_key), ""), tables=tables)


def preserve_repeated_attributes(table: TableClassification, remembered: set[str]) -> None:
    """Keep distinct metrics when an AI proposal reuses a non-key canonical role.

    Never change confirmed aliases, customer IDs or timestamps. Downgraded
    proposals remain below the review threshold so the owner sees the correction.
    """
    groups = {}
    for column in table.columns:
        if column.canonical_role in {"TRANSACTION_AMOUNT", "STATUS", "EVENT_TYPE", "TEXT"}:
            groups.setdefault(column.canonical_role, []).append(column)
    for role, columns in groups.items():
        if len(columns) < 2:
            continue
        protected = [c for c in columns if c.source_column in remembered]
        keeper = None
        if protected:
            keeper = protected[0]
        elif table.role != ROLE_DIMENSION:
            keeper = sorted(columns, key=lambda c: (
                c.source_column != CANONICAL_NAMES[role], -c.confidence, c.source_column))[0]
        for column in columns:
            if column is keeper or column.source_column in remembered:
                continue
            column.canonical_role = "ATTRIBUTE"
            column.confidence = min(column.confidence, .79)
            column.reasoning = (f"Multiple columns were proposed as {role}. "
                "Kept the original name to preserve this separate attribute; please confirm.")
            column.confidence_status()
