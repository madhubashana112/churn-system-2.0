"""Composition root.

One place decides which implementation of each interface gets wired up, so the
rest of the app only ever asks for an abstraction. Module-level singletons are
deliberate: the parsers are stateless and the repositories are persistent, keyed
by tenant, so sharing them is what lets the dashboard reload into the last
analysis instead of an empty page.

Scoring is the exception. Both engines are built here once, but which one a
request uses is decided per request, so a user can compare the hosted model
against the deterministic scorer without restarting the server.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, NamedTuple, Optional

from churn_platform.application.use_cases.describe_entity import DescribeEntityUseCase
from churn_platform.application.use_cases.execute_sector_analysis import ExecuteSectorAnalysisUseCase
from churn_platform.application.use_cases.summarize_analysis import SummarizeAnalysisUseCase
from churn_platform.config import Settings, get_settings
from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway
from churn_platform.domain.interfaces.i_churn_core import IChurnCore
from churn_platform.domain.interfaces.i_repository import IAnalysisRepository, ITenantRepository
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
    logger.info(
        "Live AI: %s, model %r at %s",
        settings.api_provider,
        settings.resolved_model,
        settings.resolved_base_url,
    )
    return QwenGateway(
        api_key=settings.api_key,
        base_url=settings.resolved_base_url,
        model=settings.resolved_model,
        timeout=settings.qwen_timeout_seconds,
        max_retries=settings.qwen_max_retries,
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
        "No AI provider key found (GEMINI_API_KEY, GROQ_API_KEY, HF_TOKEN, OPENROUTER_API_KEY or "
        "DASHSCOPE_API_KEY); falling back to MockQwenGateway. Predictions are "
        "computed locally from the uploaded features and are not model output."
    )
    return MockQwenGateway()


def _build_repos():
    """Redis-backed state on Vercel, durable SQLite locally.

    Serverless functions are ephemeral and multiply instantiated, so process
    memory is not a store there; the presence of the Upstash REST credentials
    is what tells us we are in that world.
    """
    from churn_platform.infrastructure.repositories.state_store import store, SQLiteTenantRepository, SQLiteAnalysisRepository
    if store.redis is not None:
        from churn_platform.infrastructure.repositories.redis_repos import RedisTenantRepository, RedisAnalysisRepository
        logger.info("State store: Upstash Redis")
        return RedisTenantRepository(store.redis), RedisAnalysisRepository(store.redis)
    return SQLiteTenantRepository(), SQLiteAnalysisRepository()


_tenant_repo, _analysis_repo = _build_repos()

ENGINE_AI = "ai"
ENGINE_SYSTEM = "system"


class EngineRegistry(NamedTuple):
    """Both scorers, built once and chosen per request.

    An engine that is configured away or has no credentials stays in
    ``gateways`` mapped to ``None`` rather than being dropped, because a request
    for it then has to be refused with an explanation. Quietly scoring it on the
    other engine would make a missing key indistinguishable from a working one.
    """

    gateways: Dict[str, Optional[IAIGateway]]
    cores: Dict[str, Dict[str, IChurnCore]]
    resolvers: Dict[str, AISchemaResolver]
    default: str


def build_engine_registry(settings: Settings) -> EngineRegistry:
    """Both engines, from the same rules ``build_ai_gateway`` applies.

    ``qwen_mode=live`` is the one path that does not degrade: an operator who
    forced live scoring gets the ``EnvironmentError`` ``QwenGateway`` raises, at
    startup, exactly as before. Every other way of ending up without a usable key
    leaves the AI engine unavailable and the system engine as the default.
    """
    system_gateway = MockQwenGateway()

    ai_gateway: Optional[IAIGateway] = None
    if settings.qwen_mode == "live":
        ai_gateway = _live_gateway(settings)
    elif settings.qwen_mode == "mock":
        logger.info("QWEN_MODE=mock: the AI engine is off, scoring is local only")
    elif settings.api_key:
        ai_gateway = _live_gateway(settings)
    else:
        logger.warning(
            "No AI provider key found (GEMINI_API_KEY, GROQ_API_KEY, HF_TOKEN, OPENROUTER_API_KEY or "
            "DASHSCOPE_API_KEY); the AI engine is unavailable. Predictions from the "
            "system engine are computed locally from the uploaded features and are "
            "not model output."
        )

    gateways = {ENGINE_SYSTEM: system_gateway, ENGINE_AI: ai_gateway}
    cores: Dict[str, Dict[str, IChurnCore]] = {}
    resolvers: Dict[str, AISchemaResolver] = {}
    for engine, gateway in gateways.items():
        if gateway is None:
            continue
        cores[engine] = {
            SECTOR_SAAS: SaasCore(gateway),
            SECTOR_TELECOM: TelecomCore(gateway),
            SECTOR_FINTECH: FintechCore(gateway),
        }
        resolvers[engine] = AISchemaResolver(gateway)

    return EngineRegistry(
        gateways=gateways,
        cores=cores,
        resolvers=resolvers,
        default=ENGINE_AI if ai_gateway is not None else ENGINE_SYSTEM,
    )


_REGISTRY = build_engine_registry(get_settings())
_feature_synthesizer = PandasFeatureSynthesizer()


def get_tenant_repo() -> ITenantRepository:
    return _tenant_repo


def get_analysis_repo() -> IAnalysisRepository:
    return _analysis_repo


def resolve_engine(requested: Optional[str] = None) -> str:
    """The engine a request will actually be scored on.

    ``None``, ``""`` and ``"auto"`` all mean "whatever this deployment defaults
    to", so callers that predate the choice keep working. Asking for an engine
    that is configured away is refused rather than substituted: a user who
    clicked "AI model" and silently got the deterministic scorer would have no
    way to tell the two apart.
    """
    if not requested or requested == "auto":
        return _REGISTRY.default
    if requested not in _REGISTRY.gateways:
        raise ValueError(
            f"Unknown engine {requested!r}; expected {ENGINE_AI!r} or {ENGINE_SYSTEM!r}"
        )
    if _REGISTRY.gateways[requested] is None:
        raise ValueError(
            f"The {ENGINE_AI} engine is not available: no AI provider key is "
            "configured. Add GEMINI_API_KEY, GROQ_API_KEY, HF_TOKEN, OPENROUTER_API_KEY or "
            "DASHSCOPE_API_KEY to api_key.env, or run this analysis on the "
            f"{ENGINE_SYSTEM} engine."
        )
    return requested


def ai_available() -> bool:
    """Whether a provider key resolved to a live host the AI engine can use."""
    return _REGISTRY.gateways[ENGINE_AI] is not None


def default_engine() -> str:
    return _REGISTRY.default


def get_schema_resolver(engine: Optional[str] = None) -> AISchemaResolver:
    return _REGISTRY.resolvers[resolve_engine(engine)]


def get_feature_synthesizer() -> PandasFeatureSynthesizer:
    return _feature_synthesizer


def get_feature_enricher():
    return enrich_features


def get_analysis_batch_size(engine: Optional[str] = None) -> int:
    """Chunk live LLM calls; hand the system engine the whole population at once.

    ``MockQwenGateway`` normalises every signal across the batch it is given, so
    chunking it would score a customer against an arbitrary slice of their peers
    instead of against the tenant's whole base. There is also no prompt size to
    fit inside, so a single call is both more correct and faster.
    """
    return 0 if is_offline_engine(engine) else get_settings().resolved_batch_size


def get_analysis_use_case(engine: Optional[str] = None) -> ExecuteSectorAnalysisUseCase:
    return ExecuteSectorAnalysisUseCase(batch_size=get_analysis_batch_size(engine))


def get_summarize_use_case() -> SummarizeAnalysisUseCase:
    return SummarizeAnalysisUseCase(analysis_repo=_analysis_repo, tenant_repo=_tenant_repo)


def get_describe_entity_use_case() -> DescribeEntityUseCase:
    return DescribeEntityUseCase(analysis_repo=_analysis_repo)


def get_sector_core(sector: str, engine: Optional[str] = None) -> IChurnCore:
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
    return _REGISTRY.cores[resolve_engine(engine)][resolved]


def is_offline_engine(engine: Optional[str] = None) -> bool:
    """True when this request scores locally instead of calling a model."""
    return resolve_engine(engine) == ENGINE_SYSTEM


def is_offline_gateway() -> bool:
    """True when the deployment's *default* engine is the local scorer.

    For what a particular run used, ask ``is_offline_engine``; this answers the
    narrower question of what a caller that made no choice would get, which is
    the one ``/status`` can honestly report before any request arrives.
    """
    return _REGISTRY.default == ENGINE_SYSTEM
