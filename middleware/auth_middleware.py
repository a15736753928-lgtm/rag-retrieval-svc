"""JWT 鉴权依赖项 —— 从 Authorization Header 提取并校验 Token。"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from service.auth_service import verify_token

# 不需要鉴权的路径前缀
EXCLUDE_PATHS = {
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/health",
    "/api/v1/auth/login",
}

_auth_scheme = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_auth_scheme),
) -> bool:
    """FastAPI 依赖注入：校验 JWT Token，失败抛 401。"""
    path = request.url.path

    # 排除路径
    if path in EXCLUDE_PATHS or path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi"):
        return True

    if credentials is None:
        raise HTTPException(status_code=401, detail="未提供认证凭证")

    token = credentials.credentials
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="认证凭证无效或已过期")

    return True
