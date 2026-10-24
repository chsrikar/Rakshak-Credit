"""Small signed bearer sessions for the local borrower/lender MVP demo."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

_TOKEN_SECRET = os.environ.get("AUTH_SECRET_KEY", "")
_TOKEN_TTL_SECONDS = int(os.environ.get("AUTH_TOKEN_TTL_SECONDS", "28800"))
_DEMO_USERS = {
    "borrower": (
        os.environ.get("DEMO_BORROWER_USERNAME", ""),
        os.environ.get("DEMO_BORROWER_PASSWORD", ""),
    ),
    "lender": (
        os.environ.get("DEMO_LENDER_USERNAME", ""),
        os.environ.get("DEMO_LENDER_PASSWORD", ""),
    ),
}
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    username: str
    role: str


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def configuration_ready() -> bool:
    return bool(len(_TOKEN_SECRET) >= 32 and all(username and password for username, password in _DEMO_USERS.values()))


def authenticate(username: str, password: str) -> Principal | None:
    matched: Principal | None = None
    for role, (known_username, known_password) in _DEMO_USERS.items():
        username_ok = hmac.compare_digest(username.encode(), known_username.encode())
        password_ok = hmac.compare_digest(password.encode(), known_password.encode())
        if username_ok & password_ok:
            matched = Principal(username=known_username, role=role)
    return matched


def issue_access_token(principal: Principal) -> str:
    payload = _b64(json.dumps({
        "sub": principal.username,
        "role": principal.role,
        "exp": int(time.time()) + _TOKEN_TTL_SECONDS,
    }, separators=(",", ":")).encode("utf-8"))
    signature = _b64(hmac.new(_TOKEN_SECRET.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
    return payload + "." + signature


def _decode_token(token: str) -> Principal:
    if not _TOKEN_SECRET:
        raise HTTPException(status_code=503, detail="Demo sign-in is not configured.")
    try:
        payload, signature = token.split(".", 1)
        expected = _b64(hmac.new(_TOKEN_SECRET.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        data = json.loads(_unb64(payload))
        if int(data["exp"]) < int(time.time()):
            raise ValueError("expired")
        role = data["role"]
        username = data["sub"]
        if role not in _DEMO_USERS or username != _DEMO_USERS[role][0]:
            raise ValueError("user")
        return Principal(username=username, role=role)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Sign-in expired or invalid. Please sign in again.")


def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    token = request.cookies.get("rakshak_session")
    if not token and credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Sign in to continue.", headers={"WWW-Authenticate": "Bearer"})
    return _decode_token(token)


def require_roles(*roles: str):
    def dependency(principal: Principal = Depends(current_user)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status_code=403, detail="This account does not have access to that action.")
        return principal
    return dependency


def require_application_access(application_id: str, principal: Principal) -> None:
    if principal.role == "lender":
        return
    from .database import get_underwriting_application
    application = get_underwriting_application(application_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application was not found.")
    if application.get("owner_id") != principal.username:
        raise HTTPException(status_code=404, detail="Application was not found.")
