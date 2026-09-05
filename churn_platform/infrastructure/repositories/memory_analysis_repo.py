"""In-memory analysis store.

Only the latest run per tenant is kept: the dashboard renders the current state
of a customer base, not its history, and an unbounded list of runs holding every
uploaded feature vector would grow for no reader.
"""

from typing import Dict, Optional

from churn_platform.domain.interfaces.i_repository import IAnalysisRepository
from churn_platform.domain.models.analysis_run import AnalysisRun


class MemoryAnalysisRepository(IAnalysisRepository):
    def __init__(self) -> None:
        self._runs: Dict[str, AnalysisRun] = {}

    async def save(self, run: AnalysisRun) -> None:
        self._runs[run.tenant_id] = run

    async def latest(self, tenant_id: str) -> Optional[AnalysisRun]:
        return self._runs.get(tenant_id)
