"""Dashboard metric feeds.

This used to be a stub returning ``{"status": "ok", "metrics": {}}``, which meant
the sector dashboards had nothing to draw. Everything here is aggregated from the
features stored with the last analysis run, so a KPI is a measurement of the
tenant's own upload rather than a restatement of a prediction count.
"""

from __future__ import annotations

from churn_platform.presentation.api.auth import require_tenant
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from churn_platform.application.dtos.entity_detail_dto import EntityDetail
from churn_platform.application.dtos.metrics_dto import MetricsSummary
from churn_platform.application.use_cases.describe_entity import DescribeEntityUseCase
from churn_platform.application.use_cases.summarize_analysis import SummarizeAnalysisUseCase
from churn_platform.config import get_settings
from churn_platform.presentation.api.dependencies import (
    ai_available,
    default_engine,
    get_analysis_batch_size,
    get_describe_entity_use_case,
    get_summarize_use_case,
    is_offline_gateway,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"], dependencies=[Depends(require_tenant)])


class PlatformStatus(BaseModel):
    """What the dashboard banner needs before any upload has happened."""

    offline_mode: bool
    qwen_mode: str
    model: str
    batch_size: int
    batch_size_note: str
    # The engine buttons need these before a run exists: ai_available decides
    # whether the AI one is clickable at all, and default_engine is what a
    # request that makes no choice would be scored on.
    ai_available: bool
    default_engine: str


@router.get("/metrics", response_model=MetricsSummary)
async def get_metrics(
    tenant_id: str = Query(..., description="Tenant whose latest analysis to summarize"),
    use_case: SummarizeAnalysisUseCase = Depends(get_summarize_use_case),
) -> MetricsSummary:
    summary = await use_case.execute(tenant_id)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No analysis has been run for tenant {tenant_id!r} yet. "
                "Upload that tenant's exports first."
            ),
        )
    return summary


@router.get("/customer", response_model=EntityDetail)
async def get_customer(
    tenant_id: str = Query(..., description="Tenant whose latest analysis to read"),
    entity_id: str = Query(..., description="Customer to describe"),
    use_case: DescribeEntityUseCase = Depends(get_describe_entity_use_case),
) -> EntityDetail:
    detail = await use_case.execute(tenant_id, entity_id)
    if detail is None:
        # One 404 covers both causes — the detail page shows this text verbatim,
        # so it has to tell the reader which of the two things to check.
        raise HTTPException(
            status_code=404,
            detail=(
                f"No customer {entity_id!r} in the latest analysis for tenant "
                f"{tenant_id!r}. Either that tenant has not uploaded exports "
                "yet, or this customer was not part of the upload."
            ),
        )
    return detail


@router.get("/status", response_model=PlatformStatus)
async def get_status() -> PlatformStatus:
    """The deployment's defaults.

    Nothing here describes a particular run: the engine is now chosen per
    request, so this reports what a request that made no choice would get and
    whether the AI engine can be chosen at all.
    """
    settings = get_settings()
    batch_size = get_analysis_batch_size()
    return PlatformStatus(
        offline_mode=is_offline_gateway(),
        qwen_mode=settings.qwen_mode,
        model=settings.resolved_model,
        batch_size=batch_size,
        batch_size_note=(
            "scoring locally in one pass; batch-relative normalisation needs the whole population"
            if batch_size == 0
            else f"live model calls are chunked into batches of {batch_size}"
        ),
        ai_available=ai_available(),
        default_engine=default_engine(),
    )
