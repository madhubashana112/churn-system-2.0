from uuid import uuid4
from churn_platform.domain.models.upload_session import PendingUploadSession
from churn_platform.domain.models.schema_mapping import REQUIRES_HUMAN_REVIEW
from churn_platform.domain.interfaces.i_schema_memory import ITenantSchemaMemoryRepository, IPendingUploadRepository
from churn_platform.domain.interfaces.i_schema_resolver import ISchemaResolver
from churn_platform.application.dtos.schema_review_dto import SchemaReviewResponse

class ReviewUploadSchemaUseCase:
    def __init__(self, resolver: ISchemaResolver, memory: ITenantSchemaMemoryRepository, pending: IPendingUploadRepository):
        self.resolver, self.memory, self.pending = resolver, memory, pending

    async def execute(self, tenant, files, samples, engine):
        remembered = await self.memory.get(tenant.tenant_id, list(samples))
        schema = await self.resolver.resolve(samples, remembered=remembered)
        schema.assess(tenant.sector)
        if schema.status != REQUIRES_HUMAN_REVIEW:
            return schema
        session = PendingUploadSession(upload_session_id=str(uuid4()), tenant_id=tenant.tenant_id,
            sector=tenant.sector, engine=engine, files=files, schema_mapping=schema)
        await self.pending.save(session)
        return SchemaReviewResponse(upload_session_id=session.upload_session_id, schema_mapping=schema)
