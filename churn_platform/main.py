"""FastAPI application entry point.

Every path is resolved from this file's location, so the app starts from any
working directory. Relative paths used to make `uvicorn churn_platform.main:app`
fail with a template or static-file lookup error unless it was run from the repo
root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from churn_platform.domain.models.sector import (
    SECTOR_FINTECH,
    SECTOR_SAAS,
    SECTOR_TELECOM,
    SECTOR_LABELS,
    normalize_sector,
)
from churn_platform.presentation.api.dependencies import get_tenant_repo
from churn_platform.presentation.api.v1 import analytics, tenants, upload

PRESENTATION_DIR = Path(__file__).resolve().parent / "presentation"

# Environment files are read by churn_platform.config, which resolves
# api_key.env and .env against the repo root itself.

SECTOR_TEMPLATES: Dict[str, str] = {
    SECTOR_SAAS: "dashboard_saas.html",
    SECTOR_TELECOM: "dashboard_telecom.html",
    SECTOR_FINTECH: "dashboard_fintech.html",
}

app = FastAPI(title="Domain-Adaptive Churn Prediction API")

app.mount(
    "/static",
    StaticFiles(directory=str(PRESENTATION_DIR / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(PRESENTATION_DIR / "templates"))

app.include_router(tenants.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request, tenant_id: Optional[str] = None):
    """Serve the dashboard belonging to this tenant's sector.

    The sector comes from the tenant record rather than from anything the browser
    claims, so a hand-edited localStorage cannot load a FinTech customer base
    into the Telecom dashboard.
    """
    if tenant_id is None:
        # A bootstrap page: it reads the stored tenant id and comes back here.
        return templates.TemplateResponse(request=request, name="dashboard.html")

    tenant = await get_tenant_repo().get(tenant_id)
    sector = normalize_sector(tenant.sector) if tenant else None
    if sector is None:
        # The in-memory store has no such tenant — a restart, usually. Rendering
        # the bootstrap page here would send the browser straight back to this
        # URL, so go to onboarding instead of looping.
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name=SECTOR_TEMPLATES[sector],
        context={
            "tenant_id": tenant.tenant_id,
            "tenant_name": tenant.name,
            "sector_key": sector,
            "sector_label": SECTOR_LABELS[sector],
        },
    )


@app.get("/customer", response_class=HTMLResponse)
async def read_customer(
    request: Request,
    tenant_id: Optional[str] = None,
    entity_id: Optional[str] = None,
):
    """One customer's detail page.

    Same contract as ``/dashboard``: the sector comes from the tenant record so
    the page inherits the right colourway, and anything the server cannot place
    goes to onboarding instead of rendering a page whose client-side fetch is
    guaranteed to 404.
    """
    tenant = await get_tenant_repo().get(tenant_id) if tenant_id and entity_id else None
    sector = normalize_sector(tenant.sector) if tenant else None
    if sector is None:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="customer_detail.html",
        context={
            "tenant_id": tenant.tenant_id,
            "tenant_name": tenant.name,
            "sector_key": sector,
            "sector_label": SECTOR_LABELS[sector],
            "entity_id": entity_id,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("churn_platform.main:app", host="0.0.0.0", port=8000, reload=True)
