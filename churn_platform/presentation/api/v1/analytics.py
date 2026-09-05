"""Dashboard metric feeds.

This used to be a stub returning ``{"status": "ok", "metrics": {}}``, which meant
the sector dashboards had nothing to draw. Everything here is aggregated from the
features stored with the last analysis run, so a KPI is a measurement of the
tenant's own upload rather than a restatement of a prediction count.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from churn_platform.application.dtos.metrics_dto import MetricsSummary
from churn_platform.application.use_cases.summarize_analysis import SummarizeAnalysisUseCase
from churn_platform.config import get_settings
from churn_platform.presentation.api.dependencies import (
    get_analysis_batch_size,
    get_summarize_use_case,
    is_offline_gateway,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class PlatformStatus(BaseModel):
    """What the dashboard banner needs before any upload has happened."""

    offline_mode: bool
    qwen_mode: str
    model: str
    batch_size: int
    batch_size_note: str


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


@router.get("/status", response_model=PlatformStatus)
async def get_status() -> PlatformStatus:
    settings = get_settings()
    batch_size = get_analysis_batch_size()
    return PlatformStatus(
        offline_mode=is_offline_gateway(),
        qwen_mode=settings.qwen_mode,
        model=settings.qwen_model,
        batch_size=batch_size,
        batch_size_note=(
            "scoring locally in one pass; batch-relative normalisation needs the whole population"
            if batch_size == 0
            else f"live Qwen calls are chunked into batches of {batch_size}"
        ),
    )
