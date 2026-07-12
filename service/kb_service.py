"""知识库管理 CRUD 服务。"""

from __future__ import annotations

import logging
import uuid

from db import models as dao
from milvus_client import vector_crud as vc
from utils.text_processor import format_time

logger = logging.getLogger(__name__)


def _gen_kb_id() -> str:
    return "kb_" + uuid.uuid4().hex[:8]


def _enrich_kb(kb: dict) -> dict:
    """格式化时间，统一字段名。"""
    if kb:
        kb["document_count"] = kb.pop("doc_count", 0)
        kb["created_at"] = format_time(kb.get("created_at", 0))
        kb["updated_at"] = format_time(kb.get("updated_at", 0))
    return kb


def create_kb(name: str, description: str = "") -> dict:
    kb_id = _gen_kb_id()
    kb = dao.create_knowledge_base(kb_id, name, description)
    logger.info("创建知识库: id=%s name=%s", kb_id, name)
    return _enrich_kb(kb)


def list_kbs(page: int = 1, page_size: int = 10, search: str | None = None) -> tuple[list[dict], int]:
    items, total = dao.list_knowledge_bases(page, page_size, search)
    return [_enrich_kb(d) for d in items], total


def get_kb(kb_id: str) -> dict | None:
    kb = dao.get_knowledge_base(kb_id)
    return _enrich_kb(kb) if kb else None


def update_kb(kb_id: str, name: str | None = None, description: str | None = None) -> dict | None:
    kb = dao.update_knowledge_base(kb_id, name, description)
    if kb:
        logger.info("更新知识库: id=%s", kb_id)
    return _enrich_kb(kb) if kb else None


def delete_kb(kb_id: str) -> bool:
    kb = dao.get_knowledge_base(kb_id)
    if not kb:
        return False

    docs, _ = dao.list_documents(kb_id=kb_id, page=1, page_size=10000)
    for doc in docs:
        _delete_milvus_by_doc(kb_id, doc.get("file_name", ""))
        dao.delete_document(doc["id"])

    dao.delete_knowledge_base(kb_id)
    logger.info("删除知识库: id=%s name=%s 级联删除 %d 个文档", kb_id, kb.get("name", ""), len(docs))
    return True


def _delete_milvus_by_doc(kb_id: str, file_name: str):
    try:
        expr = f'kb_id == "{kb_id}" and file_name == "{file_name}"'
        vc.delete(expr)
    except Exception as e:
        logger.warning("删除 Milvus 向量失败: kb_id=%s file=%s err=%s", kb_id, file_name, e)
