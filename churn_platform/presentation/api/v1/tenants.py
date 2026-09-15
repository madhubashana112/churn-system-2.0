"""Tenant registration and lookup."""

from fastapi import APIRouter, Depends, HTTPException
from churn_platform.presentation.api.auth import require_user, require_tenant
from churn_platform.infrastructure.repositories.state_store import store

from churn_platform.application.dtos.tenant_dto import RegisterTenantRequest, TenantResponse
from churn_platform.application.use_cases.register_tenant import RegisterTenantUseCase
from churn_platform.domain.models.sector import SECTOR_LABELS, canonical_sector_label

router = APIRouter(prefix="/tenants", tags=["Tenants"])


def get_register_use_case():
    from churn_platform.presentation.api.dependencies import get_tenant_repo
    return RegisterTenantUseCase(get_tenant_repo())


@router.post("/", response_model=TenantResponse)
async def register_tenant(
    request: RegisterTenantRequest,
    user=Depends(require_user),
    use_case: RegisterTenantUseCase = Depends(get_register_use_case),
):
    """Register a tenant, storing the sector under its canonical label.

    Normalising here rather than at every read means the stored sector is one of
    three known values, so the analysis core, the enricher and the dashboard
    template can all be selected from it without repeating the same tolerant
    matching. An exact-match check used to reject "saas" outright.
    """
    sector = canonical_sector_label(request.sector)
    if sector is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid sector {request.sector!r}. "
                f"Must be one of {', '.join(SECTOR_LABELS.values())}"
            ),
        )
    tenant = await use_case.execute(RegisterTenantRequest(name=request.name, sector=sector))
    await store.set("owner:" + tenant.tenant_id, user["id"])
    await store.set("workspace:" + user["id"], tenant.tenant_id)
    return tenant


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: str, user=Depends(require_tenant)):
    """Read a tenant back, which is how the dashboard learns its sector."""
    from churn_platform.presentation.api.dependencies import get_tenant_repo

    tenant = await get_tenant_repo().get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id!r} not found")
    return TenantResponse(tenant_id=tenant.tenant_id, name=tenant.name, sector=tenant.sector)
