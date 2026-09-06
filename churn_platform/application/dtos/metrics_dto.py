"""What a dashboard needs to render, in one payload.

The shapes are deliberately presentational — a KPI is a label and a pre-formatted
value, a chart is a kind plus its series — so the templates lay data out without
recomputing anything and without knowing which sector they belong to. Sector
identity shows up in *which* keys are present, and each dashboard template picks
out the ones it wants.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.domain.models.schema_mapping import SchemaMapping

CHART_BAR = "bar"
CHART_LINE = "line"
CHART_DOUGHNUT = "doughnut"
CHART_SCATTER = "scatter"

TONE_NEUTRAL = "neutral"
TONE_GOOD = "good"
TONE_WARNING = "warning"
TONE_CRITICAL = "critical"


class Highlight(BaseModel):
    """One sector-specific evidence column for a customer row."""

    label: str
    value: str


class EntityRow(BaseModel):
    entity_id: str
    churn_probability: float
    risk_tier: str
    reason: Optional[str] = None
    highlights: List[Highlight] = Field(default_factory=list)
    playbook: RetentionPlaybook


class KpiCard(BaseModel):
    key: str
    label: str
    value: str
    detail: str = ""
    tone: str = TONE_NEUTRAL


class ScatterPoint(BaseModel):
    x: float
    y: float
    label: Optional[str] = None


class ChartDataset(BaseModel):
    label: str
    # Categorical charts read `values` against `ChartSpec.labels`; scatter charts
    # read `points`. Both live here so one shape describes every chart the
    # dashboards draw.
    values: List[float] = Field(default_factory=list)
    points: List[ScatterPoint] = Field(default_factory=list)


class ChartSpec(BaseModel):
    key: str
    title: str
    kind: str = CHART_BAR
    subtitle: str = ""
    labels: List[str] = Field(default_factory=list)
    datasets: List[ChartDataset] = Field(default_factory=list)
    stacked: bool = False
    x_label: Optional[str] = None
    y_label: Optional[str] = None


class TierSlice(BaseModel):
    tier: str
    count: int
    share: float


class MetricsSummary(BaseModel):
    tenant_id: str
    tenant_name: str
    sector: str
    sector_label: str
    entities_uploaded: int
    entities_analyzed: int
    mean_probability: float
    at_risk_count: int
    at_risk_share: float
    tiers: List[TierSlice] = Field(default_factory=list)
    kpis: Dict[str, KpiCard] = Field(default_factory=dict)
    charts: Dict[str, ChartSpec] = Field(default_factory=dict)
    rows: List[EntityRow] = Field(default_factory=list)
    schema_mapping: SchemaMapping
    offline_mode: bool = False
    warnings: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
