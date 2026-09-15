"""Upload endpoint: raw exports in, schema plus per-entity predictions out."""

from __future__ import annotations

import logging
from base64 import b64encode
from typing import List

from churn_platform.presentation.api.auth import require_tenant
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from churn_platform.application.dtos.analysis_response_dto import (
    AnalysisResponse,
    PredictionResult,
)
from churn_platform.application.use_cases.resolve_multi_sheet_schema import (
    ResolveMultiSheetSchemaUseCase,
)
from churn_platform.application.use_cases.synthesize_features import SynthesizeFeaturesUseCase
from churn_platform.config import get_settings
from churn_platform.domain.models.analysis_run import AnalysisRun, EntityOutcome, OriginalUpload
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


@router.post("/analyze", response_model=AnalysisResponse)
async def upload_and_analyze(
    tenant_id: str = Form(...),
    files: List[UploadFile] = File(...),
    engine: str = Form("auto"),
) -> AnalysisResponse:
    try:
        chosen_engine = resolve_engine(engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tenant = await get_tenant_repo().get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id!r} not found")

    try:
        core = get_sector_core(tenant.sector, chosen_engine)
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

    schema = await ResolveMultiSheetSchemaUseCase(
        get_schema_resolver(chosen_engine)
    ).execute(ingested.samples)

    features = SynthesizeFeaturesUseCase(
        get_feature_synthesizer(),
        enricher=get_feature_enricher(),
    ).execute(schema, ingested.dataframes, sector=tenant.sector)

    warnings: List[str] = []
    entities_uploaded = len(features)
    cap = get_settings().max_entities
    if cap > 0 and entities_uploaded > cap:
        warnings.append(
            f"MAX_ENTITIES={cap} is set, so only the first {cap} of "
            f"{entities_uploaded} entities were analyzed"
        )
        features = features[:cap]

    results = await get_analysis_use_case(chosen_engine).execute(core, features)
    if len(results) < len(features):
        warnings.append(
            f"{len(features) - len(results)} of the {len(features)} submitted entities "
            "could not be scored; see the server log for the failed batch"
        )

    # offline_mode describes this run, not the deployment: the dashboard reloads
    # a stored analysis long after the request, and both engines can now be in
    # play in the same process.
    offline = is_offline_engine(chosen_engine)

    # The features travel with the run: the sector KPIs are recomputed from them
    # on every dashboard load, so a stored prediction without its evidence could
    # only ever be restated, not re-aggregated.
    features_by_id = {entry.entity_id: entry.features for entry in features}
    await get_analysis_repo().save(AnalysisRun(
        tenant_id=tenant.tenant_id,
        sector=tenant.sector,
        schema_mapping=schema,
        original_files=[OriginalUpload(filename=name, content_base64=b64encode(content).decode("ascii")) for name, content in uploads],
        outcomes=[
            EntityOutcome(prediction=pred, playbook=playbook, features=features_by_id.get(pred.entity_id, {}))
            for pred, playbook in results
        ],
        entities_uploaded=entities_uploaded,
        entities_analyzed=len(results),
        offline_mode=offline,
        warnings=warnings,
    ))

    return AnalysisResponse(
        schema_mapping=schema,
        predictions=[
            PredictionResult(prediction=pred, playbook=playbook)
            for pred, playbook in results
        ],
        entities_uploaded=entities_uploaded,
        entities_analyzed=len(results),
        warnings=warnings,
        offline_mode=offline,
    )


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
