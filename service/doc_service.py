"""文档管理 CRUD 服务 —— 元数据操作 + 分片查询。"""

from __future__ import annotations

import logging
import uuid

from db import models as dao
from milvus_client import vector_crud as vc
from storage import minio_client
from utils.text_processor import format_time

logger = logging.getLogger(__name__)


def _gen_doc_id() -> str:
    return "doc_" + uuid.uuid4().hex[:8]


def _enrich_doc(doc: dict) -> dict:
    """为文档记录补充 kb_name 和格式化时间。"""
    if doc and not doc.get("kb_name"):
        kb = dao.get_knowledge_base(doc.get("kb_id", ""))
        doc["kb_name"] = kb["name"] if kb else ""
    if doc:
        doc["ingested_at"] = format_time(doc.pop("uploaded_at", 0))
    return doc


def create_doc(kb_id: str, file_name: str, file_size: int = 0, file_type: str = "") -> dict:
    doc_id = _gen_doc_id()
    return dao.create_document(doc_id, kb_id, file_name, file_size, file_type)


def get_doc(doc_id: str) -> dict | None:
    doc = dao.get_document(doc_id)
    return _enrich_doc(doc) if doc else None


def list_docs(
    kb_id: str | None = None,
    page: int = 1,
    page_size: int = 6,
    status: str | None = None,
) -> tuple[list[dict], int]:
    items, total = dao.list_documents(kb_id, page, page_size, status)
    return [_enrich_doc(d) for d in items], total


def update_doc(doc_id: str, **fields) -> dict | None:
    doc = dao.update_document(doc_id, **fields)
    return _enrich_doc(doc) if doc else None


def delete_doc(doc_id: str) -> dict | None:
    doc = dao.get_document(doc_id)
    if not doc:
        return None

    kb_id = doc.get("kb_id", "")
    file_name = doc.get("file_name", "")

    try:
        expr = f'kb_id == "{kb_id}" and file_name == "{file_name}"'
        deleted_count = vc.delete(expr)
        logger.info("Milvus 删除完成: %d 条", deleted_count)
    except Exception as e:
        logger.warning("Milvus 删除失败: %s", e)

    chunk_count = doc.get("chunk_count", 0)
    dao.incr_kb_doc_count(kb_id, -1)
    dao.incr_kb_chunk_count(kb_id, -chunk_count)

    # 清理 PG chunks 表
    try:
        dao.delete_chunks_by_doc(doc_id)
    except Exception as e:
        logger.warning("PG chunks 删除失败: %s", e)

    # 清理 MinIO 原始文件
    object_key = doc.get("object_key", "")
    if object_key:
        try:
            minio_client.delete_file(object_key)
        except Exception as e:
            logger.warning("MinIO 删除失败: %s", e)

    return dao.delete_document(doc_id)


def get_doc_chunks(
    doc_id: str,
    page: int = 1,
    page_size: int = 100,
) -> tuple[list[dict], int]:
    """获取文档的分片列表（从 PG chunks 表查询）。index 从 1 开始。"""
    doc = dao.get_document(doc_id)
    if not doc:
        return [], 0

    items, total = dao.list_chunks(doc_id, page, page_size)

    result = []
    for it in items:
        content = it.get("chunk_text", "")
        result.append({
            "id": str(it.get("id", "")),
            "index": it.get("chunk_index", 0) + 1,  # 0-based → 1-based
            "content": content,
            "char_count": len(content),
        })
    return result, total
