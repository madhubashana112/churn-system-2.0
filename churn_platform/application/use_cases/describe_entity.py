"""Describe one customer from the tenant's latest analysis run.

Everything is recomputed from the stored run rather than persisted alongside
it: the run is the single source of truth, and a peer comparison that was
frozen at upload time would silently drift the moment a new upload replaced
the run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from churn_platform.application.dtos.entity_detail_dto import EntityDetail, FeatureEvidence
from churn_platform.application.use_cases.summarize_analysis import reason_for
from churn_platform.domain.interfaces.i_repository import IAnalysisRepository
from churn_platform.domain.models.analysis_run import EntityOutcome


def _is_number(value: Any) -> bool:
    # bool is an int subclass; a flag column is not a magnitude and has no
    # meaningful percentile.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _risk_rank(outcomes: Sequence[EntityOutcome], entity_id: str) -> int:
    """1-based position by churn probability, highest first.

    Ties break on entity_id so the rank is stable across reloads — two customers
    with the same score must not swap places depending on dict ordering.
    """
    ordered = sorted(
        outcomes,
        key=lambda o: (-o.prediction.churn_probability, o.prediction.entity_id),
    )
    return next(
        i for i, o in enumerate(ordered, start=1) if o.prediction.entity_id == entity_id
    )


def _percentile(peers: Sequence[float], value: float) -> float:
    """Exclusive percentile rank/(n+1), mid-ranked when peers tie.

    The entity itself is one of the peers, so a customer alone at the top of a
    base of four lands at 0.8, never at 1.0 — nobody can be beyond their whole
    population.
    """
    n = len(peers)
    below = sum(1 for v in peers if v < value)
    equal = sum(1 for v in peers if v == value)
    return (below + (equal + 1) / 2) / (n + 1)


def _median(peers: Sequence[float]) -> float:
    ordered = sorted(peers)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _evidence(
    outcomes: Sequence[EntityOutcome], outcome: EntityOutcome
) -> List[FeatureEvidence]:
    evidence: List[FeatureEvidence] = []
    for key, value in outcome.features.items():
        if not _is_number(value):
            continue
        peers = [
            float(o.features[key]) for o in outcomes if _is_number(o.features.get(key))
        ]
        percentile = _percentile(peers, float(value)) if len(peers) >= 2 else None
        evidence.append(
            FeatureEvidence(
                key=key,
                value=float(value),
                percentile=percentile,
                tenant_median=_median(peers) if peers else None,
                unusualness=abs(percentile - 0.5) if percentile is not None else None,
            )
        )
    # Strangest signals first: a feature at the 2nd percentile is as interesting
    # as one at the 98th, so distance from 0.5 — not the percentile itself —
    # decides the order. Keys break remaining ties to keep the table stable.
    evidence.sort(key=lambda f: (f.unusualness is None, -(f.unusualness or 0.0), f.key))
    return evidence


class DescribeEntityUseCase:
    """Build the detail payload for one customer of one tenant."""

    def __init__(self, analysis_repo: IAnalysisRepository) -> None:
        self.analysis_repo = analysis_repo

    async def execute(self, tenant_id: str, entity_id: str) -> Optional[EntityDetail]:
        run = await self.analysis_repo.latest(tenant_id)
        if run is None:
            return None

        outcome = next(
            (o for o in run.outcomes if o.prediction.entity_id == entity_id), None
        )
        if outcome is None:
            return None

        prediction = outcome.prediction
        return EntityDetail(
            tenant_id=run.tenant_id,
            entity_id=prediction.entity_id,
            churn_probability=prediction.churn_probability,
            risk_tier=prediction.risk_tier,
            reason=reason_for(outcome),
            risk_rank=_risk_rank(run.outcomes, prediction.entity_id),
            population_size=len(run.outcomes),
            features=_evidence(run.outcomes, outcome),
            playbook=outcome.playbook,
            created_at=run.created_at,
            offline_mode=run.offline_mode,
        )
