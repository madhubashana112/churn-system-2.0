"""Composition root.

One place decides which implementation of each interface gets wired up, so the
rest of the app only ever asks for an abstraction. Module-level singletons are
deliberate: the parsers are stateless and the repositories are in-memory, keyed
by tenant, so sharing them is what lets the dashboard reload into the last
analysis instead of an empty page.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from churn_platform.application.use_cases.execute_sector_analysis import ExecuteSectorAnalysisUseCase
from churn_platform.application.use_cases.summarize_analysis import SummarizeAnalysisUseCase
from churn_platform.config import Settings, get_settings
from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway
from churn_platform.domain.interfaces.i_churn_core import IChurnCore
from churn_platform.domain.models.sector import (
    SECTOR_FINTECH,
    SECTOR_SAAS,
    SECTOR_TELECOM,
    normalize_sector,
)
from churn_platform.infrastructure.ai.cores.fintech_core import FintechCore
from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
from churn_platform.infrastructure.ai.cores.telecom_core import TelecomCore
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway
from churn_platform.infrastructure.ai.qwen_gateway import QwenGateway
from churn_platform.infrastructure.parsers.feature_synthesizer import PandasFeatureSynthesizer
from churn_platform.infrastructure.parsers.schema_resolver import AISchemaResolver
from churn_platform.infrastructure.parsers.sector_feature_enrichers import enrich_features
from churn_platform.infrastructure.repositories.memory_analysis_repo import MemoryAnalysisRepository
from churn_platform.infrastructure.repositories.memory_tenant_repo import MemoryTenantRepository

logger = logging.getLogger(__name__)


def _live_gateway(settings: Settings) -> QwenGateway:
    # Settings are forwarded rather than re-read inside the gateway: a caller
    # that resolved them (or a test that supplied its own) must not be second
    # guessed by a second lookup against the process environment.
    return QwenGateway(
        api_key=settings.api_key,
        base_url=settings.qwen_base_url,
        model=settings.qwen_model,
    )


def build_ai_gateway(settings: Optional[Settings] = None) -> IAIGateway:
    """Pick the live Qwen client or the offline mock.

    "auto" is the default so a fresh clone runs end to end without credentials.
    The fallback is logged loudly rather than silently: a demo that quietly stops
    calling Qwen would otherwise be indistinguishable from one that does.
    """
    settings = settings if settings is not None else get_settings()

    if settings.qwen_mode == "live":
        return _live_gateway(settings)
    if settings.qwen_mode == "mock":
        logger.info("QWEN_MODE=mock: scoring locally, the Qwen API will not be called")
        return MockQwenGateway()

    if settings.api_key:
        return _live_gateway(settings)
    logger.warning(
        "No DASHSCOPE_API_KEY found; falling back to MockQwenGateway. Predictions "
        "are computed locally from the uploaded features and are not Qwen output."
    )
    return MockQwenGateway()


_tenant_repo = MemoryTenantRepository()
_analysis_repo = MemoryAnalysisRepository()
_ai_gateway = build_ai_gateway()
_schema_resolver = AISchemaResolver(_ai_gateway)
_feature_synthesizer = PandasFeatureSynthesizer()

_SECTOR_CORES: Dict[str, IChurnCore] = {
    SECTOR_SAAS: SaasCore(_ai_gateway),
    SECTOR_TELECOM: TelecomCore(_ai_gateway),
    SECTOR_FINTECH: FintechCore(_ai_gateway),
}


def get_tenant_repo() -> MemoryTenantRepository:
    return _tenant_repo


def get_analysis_repo() -> MemoryAnalysisRepository:
    return _analysis_repo


def get_ai_gateway() -> IAIGateway:
    return _ai_gateway


def get_schema_resolver() -> AISchemaResolver:
    return _schema_resolver


def get_feature_synthesizer() -> PandasFeatureSynthesizer:
    return _feature_synthesizer


def get_feature_enricher():
    return enrich_features


def get_analysis_batch_size() -> int:
    """Chunk live LLM calls; hand offline scoring the whole population at once.

    ``MockQwenGateway`` normalises every signal across the batch it is given, so
    chunking it would score a customer against an arbitrary slice of their peers
    instead of against the tenant's whole base. Offline there is also no prompt
    size to fit inside, so a single call is both more correct and faster.
    """
    return 0 if is_offline_gateway() else get_settings().batch_size


def get_analysis_use_case() -> ExecuteSectorAnalysisUseCase:
    return ExecuteSectorAnalysisUseCase(batch_size=get_analysis_batch_size())


def get_summarize_use_case() -> SummarizeAnalysisUseCase:
    return SummarizeAnalysisUseCase(analysis_repo=_analysis_repo, tenant_repo=_tenant_repo)


def get_sector_core(sector: str) -> IChurnCore:
    """Look a sector up tolerantly.

    Tenants register with free-text labels, so "saas", " SaaS " and
    "subscription" all reach the same core. An exact-match comparison used to
    make a lowercase registration fail at analysis time, long after the upload
    appeared to succeed.
    """
    resolved = normalize_sector(sector)
    if resolved is None:
        raise ValueError(
            f"Unknown sector {sector!r}; expected SaaS, Telecom or FinTech"
        )
    return _SECTOR_CORES[resolved]


def is_offline_gateway() -> bool:
    """True when predictions come from the local mock rather than from Qwen."""
    return isinstance(_ai_gateway, MockQwenGateway)
