"""Persistence contracts. Both are async so a database-backed implementation
can be swapped in without touching a single caller."""

from abc import ABC, abstractmethod
from typing import Optional

from churn_platform.domain.models.analysis_run import AnalysisRun
from churn_platform.domain.models.tenant import Tenant


class ITenantRepository(ABC):
    @abstractmethod
    async def save(self, tenant: Tenant) -> None:
        pass

    @abstractmethod
    async def get(self, tenant_id: str) -> Optional[Tenant]:
        pass


class IAnalysisRepository(ABC):
    @abstractmethod
    async def save(self, run: AnalysisRun) -> None:
        pass

    @abstractmethod
    async def latest(self, tenant_id: str) -> Optional[AnalysisRun]:
        """The most recent run for a tenant, or None if it has never analyzed."""
