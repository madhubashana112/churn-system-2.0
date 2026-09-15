"""Password authentication with revocable, opaque sessions and workspace ownership."""
import hashlib
import hmac
import os
import re
import secrets
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from churn_platform.infrastructure.repositories.state_store import store

router = APIRouter(prefix="/api/auth", tags=["Accounts"])
COOKIE = "churn_session"
SESSION_SECONDS = 7 * 24 * 3600


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def email_address(cls, value):
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address")
        return value


class Signup(Credentials):
    name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value):
        if not value.strip():
            raise ValueError("Enter your name")
        return value.strip()


async def session_user(request: Request):
    token = request.cookies.get(COOKIE, "")
    return await store.get("session:" + digest(token)) if token else None


async def require_user(request: Request):
    user = await session_user(request)
    if not user:
        raise HTTPException(401, "Please log in to continue")
    return user


async def require_tenant(request: Request, user=Depends(require_user)):
    tenant_id = request.path_params.get("tenant_id") or request.query_params.get("tenant_id")
    if not tenant_id and request.url.path.endswith("/analyze"):
        tenant_id = (await request.form()).get("tenant_id")
    if request.url.path.endswith("/confirm-mapping"):
        try:
            tenant_id = (await request.json()).get("tenant_id")
        except (ValueError, AttributeError):
            raise HTTPException(400, "Invalid confirmation request")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise HTTPException(400, "Workspace ID is required")
    if tenant_id and await store.get("owner:" + tenant_id) != user["id"]:
        raise HTTPException(404, "Workspace not found")
    return user


async def start_session(request, response, user):
    old = request.cookies.get(COOKIE)
    if old:
        await store.delete("session:" + digest(old))
    token = secrets.token_urlsafe(32)
    public = {key: user[key] for key in ("id", "name", "email")}
    await store.set("session:" + digest(token), public, ttl=SESSION_SECONDS)
    response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=request.url.scheme == "https" or os.getenv("COOKIE_SECURE") == "true",
                        samesite="strict", path="/")
    return {"user": public, "redirect": "/"}


async def throttle(request, email):
    ip = request.client.host if request.client else "unknown"
    for key, limit in (("ip:" + digest(ip), 60), ("email:" + digest(email), 15)):
        if await store.increment("auth-attempt:" + key, 900) > limit:
            raise HTTPException(429, "Too many attempts. Please try again in 15 minutes.")


@router.post("/signup", status_code=201)
async def signup(payload: Signup, request: Request, response: Response):
    await throttle(request, payload.email)
    salt = secrets.token_hex(16)
    user = {"id": str(uuid4()), "name": payload.name, "email": payload.email, "salt": salt,
            "hash": await run_in_threadpool(password_hash, payload.password, salt)}
    if not await store.set("account:" + digest(payload.email), user, nx=True):
        raise HTTPException(409, "An account with this email already exists. Log in instead.")
    return await start_session(request, response, user)


@router.post("/login")
async def login(payload: Credentials, request: Request, response: Response):
    await throttle(request, payload.email)
    user = await store.get("account:" + digest(payload.email))
    calculated = await run_in_threadpool(password_hash, payload.password, user["salt"] if user else "00"*16)
    if not user or not hmac.compare_digest(calculated, user["hash"]):
        raise HTTPException(401, "Email or password is incorrect")
    return await start_session(request, response, user)


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE)
    if token:
        await store.delete("session:" + digest(token))
    response.delete_cookie(COOKIE, path="/")
    return {"redirect": "/login"}


@router.get("/me")
async def me(user=Depends(require_user)):
    return user
