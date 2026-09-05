"""FastAPI application entry point.

Every path is resolved from this file's location, so the app starts from any
working directory. Relative paths used to make `uvicorn churn_platform.main:app`
fail with a template or static-file lookup error unless it was run from the repo
root.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from churn_platform.presentation.api.v1 import analytics, tenants, upload

PRESENTATION_DIR = Path(__file__).resolve().parent / "presentation"

# Environment files are read by churn_platform.config, which resolves
# api_key.env and .env against the repo root itself.

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
async def read_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("churn_platform.main:app", host="0.0.0.0", port=8000, reload=True)
