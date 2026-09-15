"""Gzipped review state in shared Redis, with the existing SQLite adapter locally."""
import asyncio
import base64
import gzip
import hashlib
import json
from churn_platform.domain.models.upload_session import PendingUploadSession
from churn_platform.infrastructure.repositories.state_store import store

UPLOAD_SESSION_TTL = 3600

def key_for(tenant_id, session_id):
    return "upload-review:" + tenant_id + ":" + session_id

def pack(value):
    return base64.b64encode(gzip.compress(json.dumps(value).encode())).decode()

def unpack(value):
    return json.loads(gzip.decompress(base64.b64decode(value)))

class PendingUploadRepository:
    async def save(self, session):
        await store.set(key_for(session.tenant_id, session.upload_session_id), pack(session.model_dump(mode="json")), ttl=UPLOAD_SESSION_TTL)

    async def get(self, tenant_id, session_id):
        value = await store.get(key_for(tenant_id, session_id))
        return PendingUploadSession.model_validate(unpack(value)) if value else None

    async def claim(self, tenant_id, session_id):
        return await store.set(key_for(tenant_id, session_id)+":lock", True, nx=True, ttl=900)

    async def release(self, tenant_id, session_id):
        await store.delete(key_for(tenant_id, session_id)+":lock")

    async def finish(self, tenant_id, session_id, result):
        await store.set(key_for(tenant_id, session_id)+":result", pack(result), ttl=UPLOAD_SESSION_TTL)
        await store.delete(key_for(tenant_id, session_id))

    async def result(self, tenant_id, session_id):
        value = await store.get(key_for(tenant_id, session_id)+":result")
        return unpack(value) if value else None

class TenantSchemaMemoryRepository:
    def key(self, tenant_id, name):
        return "schema-memory:" + tenant_id + ":" + hashlib.sha256(name.encode()).hexdigest()

    async def get(self, tenant_id, file_names):
        values = await asyncio.gather(*(store.get(self.key(tenant_id, name)) for name in file_names))
        return {name:value for name,value in zip(file_names,values) if value}

    async def save(self, tenant_id, schema):
        for table in schema.tables:
            key = self.key(tenant_id, table.file_name)
            saved = await store.get(key) or {"columns": {}}
            saved["role"] = table.role
            saved["columns"].update({c.source_column:{"canonical_role":c.canonical_role, "custom_label":c.custom_label} for c in table.columns})
            await store.set(key, saved)
