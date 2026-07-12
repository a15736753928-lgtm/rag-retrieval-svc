"""向量 CRUD —— 插入、删除、多路召回。milvus_lite 嵌入式。"""

from __future__ import annotations

import logging

from config import settings
from core.milvus_manager import get_collection, get_lite

logger = logging.getLogger(__name__)

_FIELD_OUTPUT = ["id", "kb_id"]


def insert(entities: list[dict]) -> list[int]:
    """批量插入，返回 id 列表。"""
    coll = get_collection()
    ids = coll.insert(entities)
    logger.debug("插入 %d 条", len(ids))
    return ids


def search_dense(
    query_vector: list[float],
    top_k: int = 100,
    expr: str | None = None,
) -> list[dict]:
    """稠密向量召回（COSINE + HNSW）。"""
    coll = get_collection()
    results = coll.search(
        query_vectors=[query_vector],
        top_k=top_k,
        metric_type="COSINE",
        anns_field="dense_vector",
        expr=expr,
        output_fields=_FIELD_OUTPUT,
    )
    return _flatten(results)


def search_sparse(
    sparse_vector: dict[int, float] | None,
    top_k: int = 100,
    expr: str | None = None,
) -> list[dict]:
    """稀疏向量召回（IP）。"""
    if not sparse_vector:
        return []
    coll = get_collection()
    try:
        results = coll.search(
            query_vectors=[sparse_vector],
            top_k=top_k,
            metric_type="IP",
            anns_field="sparse_vector",
            expr=expr,
            output_fields=_FIELD_OUTPUT,
        )
        return _flatten(results)
    except Exception as e:
        logger.warning("稀疏检索失败: %s", e)
        return []


def get_by_id(doc_id: int) -> dict | None:
    """按主键查询。"""
    coll = get_collection()
    records = coll.get(pks=[doc_id], output_fields=_FIELD_OUTPUT)
    return records[0] if records else None


def delete(expr: str) -> int:
    """按表达式删除。先查询 id，再按 id 删除。"""
    coll = get_collection()
    rows = coll.query(expr=expr, output_fields=["id"], limit=10000)
    if not rows:
        return 0
    pks = [r["id"] for r in rows]
    return coll.delete(pks)


def delete_by_ids(pks: list[int]) -> int:
    """按主键列表删除。"""
    if not pks:
        return 0
    coll = get_collection()
    return coll.delete(pks)


def get_all_kb_ids() -> list[str]:
    """获取所有不重复的 kb_id。"""
    coll = get_collection()
    rows = coll.query(expr=None, output_fields=["kb_id"])
    seen: set[str] = set()
    for r in rows:
        kid = r.get("kb_id", "")
        if kid:
            seen.add(kid)
    return sorted(seen)


def total_count() -> int:
    """文档总数。"""
    coll = get_collection()
    return coll.num_entities


def count_by_expr(expr: str) -> int:
    """按表达式计数。"""
    coll = get_collection()
    rows = coll.query(expr=expr, output_fields=["count(id)"], limit=1)
    return rows[0].get("count(id)", 0) if rows else 0


# ── 内部 ──────────────────────────────────────────────────────

def _flatten(results: list) -> list[dict]:
    """search 返回 [[{...}, ...]] → [{...}, ...]"""
    out = []
    for hits in results:
        for hit in hits:
            entity = hit.get("entity", {})
            out.append({
                "id": hit["id"],
                "distance": float(hit["distance"]),
                "kb_id": entity.get("kb_id", ""),
            })
    return out
