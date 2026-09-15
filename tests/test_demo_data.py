"""The bundled sample exports and the endpoint that serves them.

The point of this feature is that a visitor to a hosted deployment can try the
platform without owning any CSVs, so what matters is that the bytes the endpoint
hands back are the *same* bytes the upload pipeline already understands — not a
parallel sample dataset that only the demo path can read.
"""

from __future__ import annotations

from base64 import b64decode

import pytest
from fastapi.testclient import TestClient

from churn_platform.infrastructure.parsers import demo_data
from churn_platform.infrastructure.parsers.demo_data import demo_files
from churn_platform.infrastructure.parsers.file_ingestion import ingest
from churn_platform.main import app

from conftest import SECTOR_SCHEMAS

SECTOR_FILE_NAMES = {
    sector.lower(): sorted(table.file_name for table in tables)
    for sector, (_, tables) in SECTOR_SCHEMAS.items()
}


@pytest.fixture
def client() -> TestClient:
    from conftest import authenticated_client
    return authenticated_client()


def register(client: TestClient, sector: str) -> str:
    response = client.post("/api/v1/tenants/", json={"name": f"{sector} Demo Co", "sector": sector})
    assert response.status_code == 200, response.text
    return response.json()["tenant_id"]


# -- the loader ---------------------------------------------------------------


@pytest.mark.parametrize("sector", sorted(SECTOR_FILE_NAMES))
def test_every_sector_has_its_own_sample_exports(sector):
    files = demo_files(sector)

    assert sorted(name for name, _ in files) == SECTOR_FILE_NAMES[sector]
    assert all(contents for _, contents in files)


def test_a_free_text_sector_label_reaches_the_same_files():
    # Tenants register under labels like "subscription"; the folder is not named
    # that, so the sector has to be normalised before it is used as a path.
    assert [name for name, _ in demo_files("subscription")] == [name for name, _ in demo_files("saas")]


def test_an_unmodelled_sector_is_refused():
    with pytest.raises(ValueError, match="Unknown sector"):
        demo_files("logistics")


def test_missing_exports_are_reported_as_missing(monkeypatch, tmp_path):
    # A fresh clone that never ran the generator, or a deployment that did not
    # bundle data/, has to say so rather than serve an empty upload panel.
    monkeypatch.setattr(demo_data, "DEMO_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="generate_mock_data.py"):
        demo_files("saas")


def test_the_sample_bytes_drive_the_real_pipeline():
    # This is the whole claim of the feature: demo_files returns the (name,
    # bytes) shape ingest() already takes, so no separate demo code path exists.
    ingested = ingest(demo_files("saas"))

    assert sorted(ingested.dataframes) == SECTOR_FILE_NAMES["saas"]
    assert all(not frame.empty for frame in ingested.dataframes.values())
    assert set(ingested.samples) == set(ingested.dataframes)


# -- the endpoint -------------------------------------------------------------


def test_the_endpoint_serves_the_files_the_loader_read(client):
    tenant_id = register(client, "SaaS")

    response = client.get(f"/api/v1/upload/demo-data?tenant_id={tenant_id}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["sector_label"] == "SaaS"
    assert sorted(entry["name"] for entry in payload["files"]) == SECTOR_FILE_NAMES["saas"]

    decoded = {entry["name"]: b64decode(entry["content_base64"]) for entry in payload["files"]}
    assert decoded == dict(demo_files("saas"))


def test_the_sector_comes_from_the_tenant_record(client):
    # A Telecom tenant must not be able to ask for the SaaS base: the browser
    # does not get to choose which sector's dashboard it is looking at.
    tenant_id = register(client, "Telecom")

    payload = client.get(f"/api/v1/upload/demo-data?tenant_id={tenant_id}").json()

    assert payload["sector_label"] == "Telecom"
    assert sorted(entry["name"] for entry in payload["files"]) == SECTOR_FILE_NAMES["telecom"]


def test_an_unknown_tenant_is_not_found(client):
    response = client.get("/api/v1/upload/demo-data?tenant_id=nope")

    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found"


def test_the_served_files_can_be_analyzed_straight_back(client):
    # End to end without a file picker: fetch the sample exports, post them back
    # exactly as the browser would, and get a full population scored.
    tenant_id = register(client, "SaaS")
    payload = client.get(f"/api/v1/upload/demo-data?tenant_id={tenant_id}").json()

    uploads = [
        ("files", (entry["name"], b64decode(entry["content_base64"]), "text/csv"))
        for entry in payload["files"]
    ]
    analysis = client.post(
        "/api/v1/upload/analyze",
        data={"tenant_id": tenant_id, "engine": "system"},
        files=uploads,
    )

    assert analysis.status_code == 200, analysis.text
    assert analysis.json()["entities_analyzed"] == 100
