"""Durable local state; shared Redis state when configured for serverless."""
from contextlib import contextmanager
import json
import os
import sqlite3
import time
from pathlib import Path


class StorageUnavailable(RuntimeError):
    """Account storage is not configured; never fall back to ephemeral accounts."""


class StateStore:
    def __init__(self):
        self.redis = None
        self.configuration_error = None
        # Vercel Marketplace uses KV_REST_API_*; standalone Upstash uses
        # UPSTASH_REDIS_REST_*. Use one complete credential pair, never mix them.
        url, token = os.getenv("UPSTASH_REDIS_REST_URL"), os.getenv("UPSTASH_REDIS_REST_TOKEN")
        if not url and not token:
            url, token = os.getenv("KV_REST_API_URL"), os.getenv("KV_REST_API_TOKEN")
        if url and token:
            from upstash_redis.asyncio import Redis
            self.redis = Redis(url=url, token=token)
        elif url or token or os.getenv("VERCEL"):
            self.configuration_error = "Account storage is not configured. Please contact the site owner."
        self.path = Path(os.getenv("CHURN_DB_PATH", str(Path(__file__).resolve().parents[3] / ".local" / "churn.sqlite3")))

    @contextmanager
    def connect(self):
        if self.configuration_error:
            raise StorageUnavailable(self.configuration_error)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL, expires REAL)")
        try:
            with db:
                yield db
        finally:
            db.close()

    async def get(self, key):
        if self.redis:
            raw = await self.redis.get("churn:v2:" + key)
            return json.loads(raw) if raw else None
        with self.connect() as db:
            row = db.execute("SELECT value FROM state WHERE key=? AND (expires IS NULL OR expires>?)", (key, time.time())).fetchone()
        return json.loads(row[0]) if row else None

    async def set(self, key, value, ttl=None, nx=False):
        raw = json.dumps(value)
        if self.redis:
            return bool(await self.redis.set("churn:v2:" + key, raw, ex=ttl, nx=nx))
        with self.connect() as db:
            db.execute("DELETE FROM state WHERE expires IS NOT NULL AND expires<=?", (time.time(),))
            query = "INSERT OR IGNORE" if nx else "INSERT OR REPLACE"
            result = db.execute(f"{query} INTO state VALUES (?,?,?)", (key, raw, time.time()+ttl if ttl else None))
            return result.rowcount == 1

    async def delete(self, key):
        if self.redis:
            await self.redis.delete("churn:v2:" + key)
        else:
            with self.connect() as db:
                db.execute("DELETE FROM state WHERE key=?", (key,))

    async def increment(self, key, ttl):
        if self.redis:
            key = "churn:v2:" + key
            # One atomic operation keeps the rate-limit expiry even across workers.
            return await self.redis.eval("local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],ARGV[1]) end; return n", keys=[key], args=[ttl])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value,expires FROM state WHERE key=?", (key,)).fetchone()
            current = int(row[0]) if row and row[1] > time.time() else 0
            expiry = row[1] if current else time.time()+ttl
            db.execute("INSERT OR REPLACE INTO state VALUES (?,?,?)", (key, str(current+1), expiry))
            return current+1


store = StateStore()


class SQLiteTenantRepository:
    async def save(self, tenant):
        await store.set("tenant:" + tenant.tenant_id, tenant.model_dump(mode="json"))

    async def get(self, tenant_id):
        from churn_platform.domain.models.tenant import Tenant
        data = await store.get("tenant:" + tenant_id)
        return Tenant.model_validate(data) if data else None


class SQLiteAnalysisRepository:
    async def save(self, run):
        await store.set("run:" + run.tenant_id, run.model_dump(mode="json"))

    async def latest(self, tenant_id):
        from churn_platform.domain.models.analysis_run import AnalysisRun
        data = await store.get("run:" + tenant_id)
        return AnalysisRun.model_validate(data) if data else None
