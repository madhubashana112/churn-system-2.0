"""Deployment configuration must not crash imports or create ephemeral accounts."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from churn_platform.infrastructure.repositories.state_store import StateStore, StorageUnavailable


def test_vercel_without_database_still_serves_account_pages(tmp_path):
    env = os.environ.copy()
    for key in ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_URL", "KV_REST_API_TOKEN"):
        env.pop(key, None)
    env.update(VERCEL="1", CHURN_DB_PATH=str(tmp_path / "must-not-exist.sqlite3"))
    code = '''
from fastapi.testclient import TestClient
from api.index import app
client = TestClient(app)
assert client.get('/').status_code == 200
page = client.get('/signup')
assert page.status_code == 200 and 'Account services are being configured' in page.text
assert client.get('/static/js/theme.js').status_code == 200
response = client.post('/api/auth/signup', json={'name':'Tester','email':'test@example.com','password':'long-enough-password'})
assert response.status_code == 503, response.text
assert response.headers['retry-after'] == '60'
assert client.get('/api/index.py?__vercel_path=/login').status_code == 200
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "must-not-exist.sqlite3").exists()


def test_marketplace_redis_names_are_supported(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("KV_REST_API_URL", "https://test-only.upstash.io")
    monkeypatch.setenv("KV_REST_API_TOKEN", "test-token-not-a-real-secret")
    configured = StateStore()
    assert configured.redis is not None
    assert configured.configuration_error is None


def test_partial_credentials_never_fall_back_to_local_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://test-only.upstash.io")
    monkeypatch.setenv("CHURN_DB_PATH", str(tmp_path / "must-not-exist.sqlite3"))
    configured = StateStore()
    with pytest.raises(StorageUnavailable):
        with configured.connect():
            pass
    assert not configured.path.exists()
