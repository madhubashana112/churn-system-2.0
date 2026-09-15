from pydantic import BaseModel
from churn_platform.domain.models.analysis_run import OriginalUpload
from churn_platform.domain.models.schema_mapping import SchemaMapping

class PendingUploadSession(BaseModel):
    upload_session_id: str
    tenant_id: str
    sector: str
    engine: str
    files: list[OriginalUpload]
    schema_mapping: SchemaMapping
