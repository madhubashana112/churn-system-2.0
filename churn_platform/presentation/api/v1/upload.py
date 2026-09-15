"""Upload endpoint: raw exports in, schema plus per-entity predictions out."""

from __future__ import annotations

import logging
from base64 import b64encode, b64decode
from typing import List

from churn_platform.presentation.api.auth import require_tenant
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from churn_platform.application.dtos.analysis_response_dto import (
    AnalysisResponse,
)
from churn_platform.application.use_cases.complete_upload_analysis import CompleteUploadAnalysisUseCase, AnalysisUnavailable
from churn_platform.application.use_cases.review_upload_schema import ReviewUploadSchemaUseCase
from churn_platform.application.use_cases.confirm_schema_mapping import ConfirmSchemaMappingUseCase, ReviewSessionMissing, ReviewSessionBusy
from churn_platform.application.dtos.schema_review_dto import SchemaReviewResponse, ConfirmSchemaRequest
from churn_platform.infrastructure.persistence.redis_repos import TenantSchemaMemoryRepository, PendingUploadRepository
from churn_platform.config import get_settings
from churn_platform.domain.models.analysis_run import OriginalUpload
from churn_platform.domain.models.sector import canonical_sector_label
from churn_platform.infrastructure.parsers.demo_data import demo_files
from churn_platform.infrastructure.parsers.file_ingestion import UnsupportedFileError, ingest
from churn_platform.presentation.api.dependencies import (
    get_analysis_repo,
    get_analysis_use_case,
    get_feature_enricher,
    get_feature_synthesizer,
    get_schema_resolver,
    get_sector_core,
    get_tenant_repo,
    is_offline_engine,
    resolve_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"], dependencies=[Depends(require_tenant)])


@router.post("/analyze", response_model=AnalysisResponse | SchemaReviewResponse)
async def upload_and_analyze(
    tenant_id: str = Form(...),
    files: List[UploadFile] = File(...),
    engine: str = Form("auto"),
) -> AnalysisResponse | SchemaReviewResponse:
    try:
        chosen_engine = resolve_engine(engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tenant = await get_tenant_repo().get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id!r} not found")

    try:
        get_sector_core(tenant.sector, chosen_engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uploads = []
    total_bytes = 0
    max_upload_bytes = 4 * 1024 * 1024
    for file in files:
        if not file.filename:
            continue
        content = await file.read(max_upload_bytes - total_bytes + 1)
        total_bytes += len(content)
        if total_bytes > max_upload_bytes:
            raise HTTPException(413, "Upload up to 4 MB of files per analysis.")
        uploads.append((file.filename, content))
    if not uploads:
        raise HTTPException(status_code=400, detail="No files were received")

    try:
        ingested = ingest(uploads)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if sum(len(frame.columns) for frame in ingested.dataframes.values()) > 500:
        raise HTTPException(400, "Upload at most 500 columns across all files and sheets.")

    originals = [OriginalUpload(filename=name, content_base64=b64encode(content).decode("ascii")) for name, content in uploads]
    memory, pending = review_repositories()
    schema = await ReviewUploadSchemaUseCase(
        get_schema_resolver(chosen_engine), memory, pending
    ).execute(tenant, originals, ingested.samples, chosen_engine)
    if isinstance(schema, SchemaReviewResponse):
        return schema

    return await complete_analysis(tenant, chosen_engine, schema, ingested, originals)


def review_repositories():
    return TenantSchemaMemoryRepository(), PendingUploadRepository()


async def complete_analysis(tenant, engine, schema, ingested, originals):
    use_case = CompleteUploadAnalysisUseCase(
        get_feature_synthesizer(), get_feature_enricher(), get_analysis_use_case(engine),
        get_analysis_repo(), get_settings().max_entities, is_offline_engine(engine))
    try:
        return await use_case.execute(tenant, schema, ingested.dataframes, originals,
                                     get_sector_core(tenant.sector, engine))
    except AnalysisUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/confirm-mapping", response_model=AnalysisResponse)
async def confirm_mapping(payload: ConfirmSchemaRequest):
    tenant = await get_tenant_repo().get(payload.tenant_id)
    if tenant is None:
        raise HTTPException(404, "Workspace not found")

    async def resume(session, schema):
        engine = resolve_engine(session.engine)
        parsed = ingest([(f.filename, b64decode(f.content_base64)) for f in session.files])
        return await complete_analysis(tenant, engine, schema, parsed, session.files)

    memory, pending = review_repositories()
    try:
        return await ConfirmSchemaMappingUseCase(memory, pending, resume).execute(payload)
    except ReviewSessionMissing as exc:
        raise HTTPException(410, str(exc)) from exc
    except ReviewSessionBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class DemoFile(BaseModel):
    name: str
    content_base64: str


class DemoDataResponse(BaseModel):
    """The sample exports for one sector, ready to drop into the upload panel."""

    sector_label: str
    files: List[DemoFile]


@router.get("/demo-data", response_model=DemoDataResponse)
async def read_demo_data(
    tenant_id: str = Query(..., description="Tenant whose sector decides which sample base to serve"),
) -> DemoDataResponse:
    """Serve the bundled sample exports for this tenant's sector.

    Keyed by tenant rather than by sector for the same reason ``/dashboard`` is:
    the tenant record is the only trustworthy source, so a hand-edited request
    cannot load a FinTech customer base into the Telecom dashboard.

    All the files come back in one response. On a serverless host that is a
    single invocation instead of one per CSV, which is worth more than the
    third again in size that base64 costs.
    """
    tenant = await get_tenant_repo().get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id!r} not found")

    try:
        files = demo_files(tenant.sector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.info("Serving %d sample exports for tenant %s", len(files), tenant.tenant_id)
    return DemoDataResponse(
        sector_label=canonical_sector_label(tenant.sector) or tenant.sector,
        files=[
            DemoFile(name=name, content_base64=b64encode(contents).decode("ascii"))
            for name, contents in files
        ],
    )
