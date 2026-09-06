"""What one customer's detail page needs, in one payload.

The dashboard summary answers "how is the base doing"; this answers "why did
*this* customer get that score". Every stored feature is positioned against the
tenant's own population — percentile and median — because a raw number like
"export_ratio 0.8" is only evidence once you know the base rarely exports.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from churn_platform.domain.models.retention_playbook import RetentionPlaybook


class FeatureEvidence(BaseModel):
    """One numeric feature of the entity, positioned against the whole base."""

    key: str
    value: float
    # Exclusive percentile rank/(n+1) with mid-ranks for ties; None when the
    # population is too small to compare against (n < 2).
    percentile: Optional[float] = None
    tenant_median: Optional[float] = None
    # abs(percentile - 0.5): how far from typical, in either direction. The
    # detail table sorts by this so the strangest signals lead.
    unusualness: Optional[float] = None


class EntityDetail(BaseModel):
    tenant_id: str
    entity_id: str
    churn_probability: float
    risk_tier: str
    reason: Optional[str] = None
    risk_rank: int
    population_size: int
    features: List[FeatureEvidence] = Field(default_factory=list)
    playbook: RetentionPlaybook
    created_at: Optional[datetime] = None
    # Which engine scored this customer. A deployment can run both, so the page
    # cannot infer it from the platform defaults.
    offline_mode: bool = False
