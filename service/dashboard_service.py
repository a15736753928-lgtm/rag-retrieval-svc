"""数据概览服务：核心统计、最近入库、知识库文档分布。"""

from __future__ import annotations

from db import models as dao
from utils.text_processor import format_time


def get_stats() -> dict:
    raw = dao.get_stats()
    return {
        "kb_count": raw["kb_count"],
        "document_count": raw["doc_count"],
        "chunk_count": raw["chunk_count"],
    }


def get_recent(limit: int = 5) -> list[dict]:
    rows = dao.get_recent_docs(limit)
    return [
        {
            "id": r["id"],
            "file_name": r["file_name"],
            "kb_id": r["kb_id"],
            "kb_name": r.get("kb_name", ""),
            "chunk_count": r["chunk_count"],
            "file_size": r.get("file_size", 0),
            "ingested_at": format_time(r.get("uploaded_at", 0)),
        }
        for r in rows
    ]


def get_distribution() -> list[dict]:
    rows = dao.get_distribution()
    return [
        {
            "kb_id": r["kb_id"],
            "kb_name": r["kb_name"],
            "document_count": r["doc_count"],
        }
        for r in rows
    ]
