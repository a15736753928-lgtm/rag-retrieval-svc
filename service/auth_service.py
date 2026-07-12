"""认证与密钥管理：bcrypt 密码校验 + JWT 令牌签发。"""

from __future__ import annotations

import json
import logging
import secrets
import string
import time
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from config import settings
from db.models import get_setting, set_setting

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_CONFIG = {
    "strategy": "recursive",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separator": "\n\n",
}


def _ensure_defaults():
    """初始化系统默认配置（幂等）。"""
    if get_setting("chunk_config") is None:
        set_setting("chunk_config", json.dumps(DEFAULT_CHUNK_CONFIG, ensure_ascii=False))
        logger.info("已初始化默认分片配置")

    if get_setting("access_key_hash") is None:
        _hash, _preview = _hash_access_key(settings.default_access_key)
        set_setting("access_key_hash", _hash)
        set_setting("access_key_preview", _preview)
        logger.info("已生成默认访问密钥: %s", _preview)


def _hash_access_key(raw: str) -> tuple[str, str]:
    """返回 (bcrypt_hash, 脱敏预览)。"""
    h = bcrypt.hashpw(raw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    if len(raw) > 10:
        preview = raw[:2] + "****" + raw[-4:]
    else:
        preview = raw[:2] + "****"
    return h, preview


def login(access_key: str) -> str | None:
    """校验访问密钥，成功返回 JWT token，失败返回 None。"""
    stored_hash = get_setting("access_key_hash")
    if not stored_hash:
        _ensure_defaults()
        stored_hash = get_setting("access_key_hash")
    if not bcrypt.checkpw(access_key.encode("utf-8"), stored_hash.encode("utf-8")):  # type: ignore[union-attr]
        return None

    now = datetime.now(timezone.utc)
    payload = {
        "sub": "admin",
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expire_hours),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token


def verify_token(token: str) -> bool:
    """校验 JWT Token，有效返回 True。"""
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return True
    except jwt.PyJWTError:
        return False


def regenerate_access_key() -> tuple[str, str]:
    """生成新访问密钥，返回 (raw_key, preview)。"""
    alphabet = string.ascii_lowercase + string.digits
    raw = "sk-rag-" + "".join(secrets.choice(alphabet) for _ in range(24))
    h, preview = _hash_access_key(raw)
    set_setting("access_key_hash", h)
    set_setting("access_key_preview", preview)
    return raw, preview


def get_access_key_preview() -> str | None:
    return get_setting("access_key_preview")
