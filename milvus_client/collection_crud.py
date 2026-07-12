"""集合操作 —— 初始化、统计、删除。milvus_lite 嵌入式。"""

from __future__ import annotations

from core.milvus_manager import drop_collection as _drop, ensure_collection, get_collection


def init():
    """服务启动时调用。"""
    ensure_collection()


def get_stats() -> dict:
    """集合统计。"""
    coll = get_collection()
    return {
        "name": coll.name,
        "num_entities": coll.num_entities,
    }


def get_collection_name() -> str:
    from config import settings
    return settings.milvus_collection_name


def drop():
    """删除整个 collection。"""
    _drop()
