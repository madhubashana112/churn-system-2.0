"""One completed analysis of one tenant's uploaded exports.

Predictions used to exist only inside the HTTP response that produced them, so
reloading the dashboard lost everything and the metrics endpoint had nothing to
aggregate. A run is what gets kept: the resolved schema, the features behind
each score, and the score itself, so the sector KPIs can be recomputed from
evidence rather than restated from a summary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.domain.models.schema_mapping import SchemaMapping

RISK_TIERS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
AT_RISK_TIERS = ("CRITICAL", "HIGH")


class EntityOutcome(BaseModel):
    """A single customer's score, the action it implies, and the evidence."""

    prediction: ChurnPrediction
    playbook: RetentionPlaybook
    features: Dict[str, Any] = Field(default_factory=dict)


class AnalysisRun(BaseModel):
    tenant_id: str
    sector: str
    schema_mapping: SchemaMapping
    outcomes: List[EntityOutcome] = Field(default_factory=list)
    entities_uploaded: int = 0
    entities_analyzed: int = 0
    offline_mode: bool = False
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def at_risk(self) -> List[EntityOutcome]:
        return [o for o in self.outcomes if o.prediction.risk_tier in AT_RISK_TIERS]
