"""Upload endpoint: raw exports in, schema plus per-entity predictions out."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from churn_platform.application.dtos.analysis_response_dto import (
    AnalysisResponse,
    PredictionResult,
)
from churn_platform.application.use_cases.resolve_multi_sheet_schema import (
    ResolveMultiSheetSchemaUseCase,
)
from churn_platform.application.use_cases.synthesize_features import SynthesizeFeaturesUseCase
from churn_platform.config import get_settings
from churn_platform.domain.models.analysis_run import AnalysisRun, EntityOutcome
from churn_platform.infrastructure.parsers.file_ingestion import UnsupportedFileError, ingest
from churn_platform.presentation.api.dependencies import (
    get_analysis_repo,
    get_analysis_use_case,
    get_feature_enricher,
    get_feature_synthesizer,
    get_schema_resolver,
    get_sector_core,
    get_tenant_repo,
    is_offline_gateway,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/analyze", response_model=AnalysisResponse)
async def upload_and_analyze(
    tenant_id: str = Form(...),
    files: List[UploadFile] = File(...),
) -> AnalysisResponse:
    tenant = await get_tenant_repo().get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id!r} not found")

    try:
        core = get_sector_core(tenant.sector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uploads = [(file.filename, await file.read()) for file in files if file.filename]
    if not uploads:
        raise HTTPException(status_code=400, detail="No files were received")

    try:
        ingested = ingest(uploads)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    schema = await ResolveMultiSheetSchemaUseCase(get_schema_resolver()).execute(ingested.samples)

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

    results = await get_analysis_use_case().execute(core, features)
    if len(results) < len(features):
        warnings.append(
            f"{len(features) - len(results)} of the {len(features)} submitted entities "
            "could not be scored; see the server log for the failed batch"
        )

    # The features travel with the run: the sector KPIs are recomputed from them
    # on every dashboard load, so a stored prediction without its evidence could
    # only ever be restated, not re-aggregated.
    features_by_id = {entry.entity_id: entry.features for entry in features}
    await get_analysis_repo().save(AnalysisRun(
        tenant_id=tenant.tenant_id,
        sector=tenant.sector,
        schema_mapping=schema,
        outcomes=[
            EntityOutcome(prediction=pred, playbook=playbook, features=features_by_id.get(pred.entity_id, {}))
            for pred, playbook in results
        ],
        entities_uploaded=entities_uploaded,
        entities_analyzed=len(results),
        offline_mode=is_offline_gateway(),
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
        offline_mode=is_offline_gateway(),
    )
