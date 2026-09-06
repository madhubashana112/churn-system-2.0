"""Upstash Redis-backed repositories for serverless deployments.

Vercel functions are ephemeral and multiply instantiated, so the in-memory
repositories would lose every tenant on a cold start and never share state
between two concurrent instances. Upstash speaks REST, which needs no
persistent socket -- the one thing a serverless sandbox cannot guarantee.

Values are gzip-compressed, base64-encoded JSON: a single analysis run carries
a feature vector per customer and would otherwise sit close to the REST
payload limits for no reason.
"""

from __future__ import annotations

import base64
import zlib
from typing import Optional

from upstash_redis.asyncio import Redis

from churn_platform.domain.interfaces.i_repository import IAnalysisRepository, ITenantRepository
from churn_platform.domain.models.analysis_run import AnalysisRun
from churn_platform.domain.models.tenant import Tenant

# Runs are overwritten on every upload and only the latest is ever read, so
# the TTL is just hygiene against the free tier's memory cap, not a feature.
STATE_TTL_SECONDS = 7 * 24 * 3600


def _compress(text: str) -> str:
    return base64.b64encode(zlib.compress(text.encode("utf-8"))).decode("ascii")


def _decompress(blob: str) -> str:
    return zlib.decompress(base64.b64decode(blob)).decode("utf-8")


class RedisTenantRepository(ITenantRepository):
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def save(self, tenant: Tenant) -> None:
        await self._redis.set(
            f"tenant:{tenant.tenant_id}", tenant.model_dump_json(), ex=STATE_TTL_SECONDS
        )

    async def get(self, tenant_id: str) -> Optional[Tenant]:
        raw = await self._redis.get(f"tenant:{tenant_id}")
        return Tenant.model_validate_json(raw) if raw else None


class RedisAnalysisRepository(IAnalysisRepository):
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def save(self, run: AnalysisRun) -> None:
        await self._redis.set(
            f"run:{run.tenant_id}", _compress(run.model_dump_json()), ex=STATE_TTL_SECONDS
        )

    async def latest(self, tenant_id: str) -> Optional[AnalysisRun]:
        raw = await self._redis.get(f"run:{tenant_id}")
        return AnalysisRun.model_validate_json(_decompress(raw)) if raw else None
