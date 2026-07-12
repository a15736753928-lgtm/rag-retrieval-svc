"""系统设置服务 —— 模型信息、向量库状态、分片配置、访问密钥。"""

from __future__ import annotations

import json
import logging
import time

from config import settings
from db.models import get_setting, set_setting
from milvus_client import collection_crud as coll_crud
from service.auth_service import DEFAULT_CHUNK_CONFIG, regenerate_access_key as _regenerate_key

logger = logging.getLogger(__name__)


def get_all_settings() -> dict:
    return {
        "vector_model": _get_vector_model(),
        "vector_db": _get_vector_db(),
        "chunk_config": get_chunk_config(),
        "access_key_masked": _get_masked_key(),
    }


# ── vector_model ──────────────────────────────────────────────────────

def _get_vector_model() -> dict:
    from core.model_loader import _bge_m3, _reranker
    bge_ok = _bge_m3 is not None
    rerank_ok = _reranker is not None
    if bge_ok and rerank_ok:
        status = "running"
    elif not bge_ok and not rerank_ok:
        status = "stopped"
    else:
        status = "running"

    # 尝试获取模型版本
    version = ""
    try:
        if _bge_m3 is not None:
            # sentence-transformers 模型通常无显式版本，取模块版本
            import sentence_transformers
            version = getattr(sentence_transformers, "__version__", "")
    except Exception:
        pass

    return {
        "model_name": settings.bge_model_name,
        "model_version": version,
        "vector_dim": settings.dense_vector_dim,
        "status": status,
    }


# ── vector_db ─────────────────────────────────────────────────────────

def _get_vector_db(refresh: bool = False) -> dict:
    try:
        stats = coll_crud.get_stats()
    except Exception:
        stats = {}

    return {
        "db_type": "Milvus Lite",
        "connection_address": settings.milvus_data_dir,
        "status": "connected",
        "vector_count": stats.get("num_entities", 0),
        "storage_size": "--",
    }


def refresh_vector_db() -> dict:
    return _get_vector_db(refresh=True)


# ── chunk_config ──────────────────────────────────────────────────────

def get_chunk_config() -> dict:
    raw = get_setting("chunk_config")
    if raw:
        return json.loads(raw)
    return dict(DEFAULT_CHUNK_CONFIG)


def save_chunk_config(config: dict) -> dict:
    cleaned = {
        "strategy": config.get("strategy", "recursive"),
        "chunk_size": config.get("chunk_size", 500),
        "chunk_overlap": config.get("chunk_overlap", 100),
        "separator": config.get("separator", "\n\n"),
    }
    set_setting("chunk_config", json.dumps(cleaned, ensure_ascii=False))
    logger.info("分片配置已更新: %s", cleaned)
    return cleaned


def reset_chunk_config() -> dict:
    set_setting("chunk_config", json.dumps(DEFAULT_CHUNK_CONFIG, ensure_ascii=False))
    return dict(DEFAULT_CHUNK_CONFIG)


# ── access_key ────────────────────────────────────────────────────────

def _get_masked_key() -> str:
    return "••••••••••••••••"


def get_access_key_preview() -> str | None:
    from db.models import get_setting as gs
    return gs("access_key_preview")


def regenerate_access_key() -> dict:
    raw, _preview = _regenerate_key()
    return {"access_key": raw}
