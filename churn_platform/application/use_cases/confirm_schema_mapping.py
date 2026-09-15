from churn_platform.domain.models.schema_mapping import ColumnMapping, TABLE_ROLES, REQUIRES_HUMAN_REVIEW
from churn_platform.domain.interfaces.i_schema_memory import ITenantSchemaMemoryRepository, IPendingUploadRepository

class ReviewSessionMissing(ValueError): pass
class ReviewSessionBusy(ValueError): pass

class ConfirmSchemaMappingUseCase:
    def __init__(self, memory: ITenantSchemaMemoryRepository, pending: IPendingUploadRepository, analyze, validate_schema=None):
        self.memory, self.pending, self.analyze = memory, pending, analyze
        self.validate_schema = validate_schema

    async def execute(self, request):
        tenant_id, session_id = request.tenant_id, request.upload_session_id
        cached = await self.pending.result(tenant_id, session_id)
        if cached is not None:
            return cached
        session = await self.pending.get(tenant_id, session_id)
        if session is None:
            raise ReviewSessionMissing("This upload session has expired or is unavailable. Upload the files again.")
        schema = session.schema_mapping.model_copy(deep=True)
        expected = {(t.file_name,c.source_column) for t in schema.tables for c in t.columns}
        provided = {(c.file_name,c.source_column) for c in request.mappings}
        if expected != provided or len(provided) != len(request.mappings):
            raise ValueError("Confirm every uploaded column exactly once; unknown or duplicate columns are not allowed")
        choices = {(c.file_name,c.source_column):c for c in request.mappings}
        if set(request.table_roles) - {t.file_name for t in schema.tables}:
            raise ValueError("Unknown table in confirmation")
        for table in schema.tables:
            role = request.table_roles.get(table.file_name, table.role)
            if role not in TABLE_ROLES:
                raise ValueError("Invalid table role")
            table.role = role
            mapped = []
            for original in table.columns:
                choice = choices[(table.file_name, original.source_column)]
                mapped.append(ColumnMapping(source_column=original.source_column, canonical_role=choice.canonical_role,
                    custom_label=choice.custom_label if choice.canonical_role == "CUSTOM" else None,
                    confidence=1.0, sample_values=original.sample_values, reasoning="Confirmed by workspace owner"))
            table.columns = mapped
            table.primary_entity_key = next((c.source_column for c in mapped if c.canonical_role == "CUSTOMER_ID"), "")
            table.timestamp_column = next((c.source_column for c in mapped if c.canonical_role == "TIMESTAMP"), None)
            table.noise_columns = [c.source_column for c in mapped if c.canonical_role == "NOISE_IGNORE"]
        if self.validate_schema:
            issues = self.validate_schema(schema, session.files)
            if issues:
                raise ValueError("; ".join(issues))
        schema.assess(session.sector)
        if schema.status == REQUIRES_HUMAN_REVIEW:
            raise ValueError("; ".join(schema.review_reasons))
        schema.primary_entity_key = schema.tables[0].primary_entity_key
        if not await self.pending.claim(tenant_id, session_id):
            raise ReviewSessionBusy("This upload is already being analyzed. Please wait before retrying.")
        try:
            await self.memory.save(tenant_id, schema)
            result = await self.analyze(session, schema)
            result = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
            await self.pending.finish(tenant_id, session_id, result)
            return result
        finally:
            await self.pending.release(tenant_id, session_id)
