"""FastAPI application entry point.

Every path is resolved from this file's location, so the app starts from any
working directory. Relative paths used to make `uvicorn churn_platform.main:app`
fail with a template or static-file lookup error unless it was run from the repo
root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
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
from churn_platform.presentation.api.v1 import analytics, tenants, upload, exports
from churn_platform.presentation.api import auth
from churn_platform.infrastructure.repositories.state_store import store, StorageUnavailable

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

app.include_router(auth.router)
app.include_router(exports.router, prefix="/api/v1")
app.include_router(tenants.router, prefix="/api/v1", dependencies=[Depends(auth.require_user)])
app.include_router(upload.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    user = await auth.session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request=request, name="index.html", context={"user": user})


@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request, tenant_id: Optional[str] = None):
    """Serve the dashboard belonging to this tenant's sector.

    The sector comes from the tenant record rather than from anything the browser
    claims, so a hand-edited localStorage cannot load a FinTech customer base
    into the Telecom dashboard.
    """
    user = await auth.session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if tenant_id is None:
        tenant_id = await store.get("workspace:" + user["id"])
        return RedirectResponse("/dashboard?tenant_id=" + tenant_id if tenant_id else "/", status_code=303)
    await auth.require_tenant(request, user)

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
            "user": user,
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
    user = await auth.session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    await auth.require_tenant(request, user)
    tenant = await get_tenant_repo().get(tenant_id) if tenant_id and entity_id else None
    sector = normalize_sector(tenant.sector) if tenant else None
    if sector is None:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="customer_detail.html",
        context={
            "user": user,
            "tenant_id": tenant.tenant_id,
            "tenant_name": tenant.name,
            "sector_key": sector,
            "sector_label": SECTOR_LABELS[sector],
            "entity_id": entity_id,
        },
    )




@app.middleware("http")
async def account_security(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        expected = str(request.base_url).rstrip("/")
        if (origin and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site requests are not allowed"}, status_code=403)
    response = await call_next(request)
    if not request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.get("/login", response_class=HTMLResponse)
@app.get("/signup", response_class=HTMLResponse)
async def account_page(request: Request):
    if not store.configuration_error and await auth.session_user(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request=request, name="auth.html", context={
        "signup": request.url.path == "/signup", "storage_error": store.configuration_error})


@app.exception_handler(StorageUnavailable)
async def storage_unavailable(request: Request, exc: StorageUnavailable):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": str(exc)}, status_code=503, headers={"Retry-After": "60"})
    return templates.TemplateResponse(request=request, name="auth.html", status_code=503,
        context={"signup": False, "storage_error": str(exc)}, headers={"Retry-After": "60"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("churn_platform.main:app", host="0.0.0.0", port=8000, reload=True)
