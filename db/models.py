"""DAO 层 —— PostgreSQL 表操作的薄封装。"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from db.database import get_db_conn, put_db_conn


@contextmanager
def _get_db():
    """获取 PG 连接（autocommit），用后归还池。"""
    conn = get_db_conn()
    conn.autocommit = True
    try:
        yield conn
    finally:
        put_db_conn(conn)


def _row(cursor) -> dict | None:
    """取单行 → dict。"""
    r = cursor.fetchone()
    return dict(r) if r else None


def _rows(cursor) -> list[dict]:
    """取多行 → list[dict]。"""
    return [dict(r) for r in cursor.fetchall()]


# ═══════════════════════════════════════════════════════════════════════
#  knowledge_bases
# ═══════════════════════════════════════════════════════════════════════

def create_knowledge_base(kb_id: str, name: str, description: str = "") -> dict:
    now = int(time.time() * 1000)
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "INSERT INTO knowledge_bases (id, name, description, doc_count, chunk_count, created_at, updated_at) "
            "VALUES (%s, %s, %s, 0, 0, %s, %s)",
            (kb_id, name, description, now, now),
        )
        cur.execute("SELECT * FROM knowledge_bases WHERE id = %s", (kb_id,))
        return dict(cur.fetchone())


def get_knowledge_base(kb_id: str) -> dict | None:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM knowledge_bases WHERE id = %s", (kb_id,))
        return _row(cur)


def list_knowledge_bases(
    page: int = 1, page_size: int = 20, search: str | None = None,
) -> tuple[list[dict], int]:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        where = ""
        params: list = []
        if search:
            where = "WHERE name LIKE %s"
            params.append(f"%{search}%")

        cur.execute(f"SELECT COUNT(*) FROM knowledge_bases {where}", params)
        total = cur.fetchone()["count"]

        offset = (page - 1) * page_size
        cur.execute(
            f"SELECT * FROM knowledge_bases {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            params + [page_size, offset],
        )
        return _rows(cur), total


def update_knowledge_base(kb_id: str, name: str | None = None, description: str | None = None) -> dict | None:
    now = int(time.time() * 1000)
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if name is not None:
            cur.execute("UPDATE knowledge_bases SET name = %s, updated_at = %s WHERE id = %s", (name, now, kb_id))
        if description is not None:
            cur.execute("UPDATE knowledge_bases SET description = %s, updated_at = %s WHERE id = %s", (description, now, kb_id))
        cur.execute("SELECT * FROM knowledge_bases WHERE id = %s", (kb_id,))
        return _row(cur)


def delete_knowledge_base(kb_id: str) -> bool:
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM knowledge_bases WHERE id = %s", (kb_id,))
        return cur.rowcount > 0


def incr_kb_doc_count(kb_id: str, delta: int = 1):
    with _get_db() as conn:
        cur = conn.cursor()
        now = int(time.time() * 1000)
        cur.execute(
            "UPDATE knowledge_bases SET doc_count = GREATEST(0, doc_count + %s), updated_at = %s WHERE id = %s",
            (delta, now, kb_id),
        )


def incr_kb_chunk_count(kb_id: str, delta: int):
    with _get_db() as conn:
        cur = conn.cursor()
        now = int(time.time() * 1000)
        cur.execute(
            "UPDATE knowledge_bases SET chunk_count = GREATEST(0, chunk_count + %s), updated_at = %s WHERE id = %s",
            (delta, now, kb_id),
        )


# ═══════════════════════════════════════════════════════════════════════
#  documents
# ═══════════════════════════════════════════════════════════════════════

def create_document(
    doc_id: str,
    kb_id: str,
    file_name: str,
    file_size: int = 0,
    file_type: str = "",
    object_key: str = "",
    file_hash: str = "",
) -> dict:
    now = int(time.time() * 1000)
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "INSERT INTO documents (id, kb_id, file_name, file_size, file_type, chunk_count, object_key, file_hash, status, uploaded_at, created_at) "
            "VALUES (%s, %s, %s, %s, %s, 0, %s, %s, 'pending', 0, %s)",
            (doc_id, kb_id, file_name, file_size, file_type, object_key, file_hash, now),
        )
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        return dict(cur.fetchone())


def get_document(doc_id: str) -> dict | None:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        return _row(cur)


def check_duplicate(kb_id: str, file_hash: str) -> dict | None:
    """按 kb_id + file_hash 查重，返回已存在的文档或 None。"""
    if not file_hash:
        return None
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM documents WHERE kb_id = %s AND file_hash = %s LIMIT 1",
            (kb_id, file_hash),
        )
        return _row(cur)


def list_documents(
    kb_id: str | None = None,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
) -> tuple[list[dict], int]:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        clauses = []
        params: list = []
        if kb_id:
            clauses.append("kb_id = %s")
            params.append(kb_id)
        if status:
            clauses.append("status = %s")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        cur.execute(f"SELECT COUNT(*) FROM documents {where}", params)
        total = cur.fetchone()["count"]

        offset = (page - 1) * page_size
        cur.execute(
            f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            params + [page_size, offset],
        )
        return _rows(cur), total


def update_document(doc_id: str, **fields) -> dict | None:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        allowed = {"status", "chunk_count", "uploaded_at", "object_key"}
        for k, v in fields.items():
            if k in allowed:
                cur.execute(f"UPDATE documents SET {k} = %s WHERE id = %s", (v, doc_id))
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        return _row(cur)


def delete_document(doc_id: str) -> dict | None:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        doc = _row(cur)
        if doc:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        return doc


# ═══════════════════════════════════════════════════════════════════════
#  chunks
# ═══════════════════════════════════════════════════════════════════════

def insert_chunks(doc_id: str, kb_id: str, chunks: list[str], milvus_pks: list[int] | None = None) -> list[int]:
    """批量插入分片文本，返回自增 id 列表。"""
    now = int(time.time() * 1000)
    ids: list[int] = []
    with _get_db() as conn:
        cur = conn.cursor()
        for i, text in enumerate(chunks):
            pk = milvus_pks[i] if milvus_pks and i < len(milvus_pks) else 0
            cur.execute(
                "INSERT INTO chunks (doc_id, kb_id, chunk_index, chunk_text, milvus_pk, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (doc_id, kb_id, i, text, pk, now),
            )
            ids.append(cur.fetchone()[0])
    return ids


def list_chunks(doc_id: str, page: int = 1, page_size: int = 100) -> tuple[list[dict], int]:
    """分页查询文档的分片列表。index 从 0 开始。"""
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (doc_id,))
        total = cur.fetchone()["count"]

        offset = (page - 1) * page_size
        cur.execute(
            "SELECT id, chunk_index, chunk_text, milvus_pk FROM chunks "
            "WHERE doc_id = %s ORDER BY chunk_index LIMIT %s OFFSET %s",
            (doc_id, page_size, offset),
        )
        return _rows(cur), total


def get_chunks_by_milvus_pks(milvus_pks: list[int]) -> dict[int, dict]:
    """用 Milvus 主键批量查 PG chunks，返回 {milvus_pk: chunk_row}。"""
    if not milvus_pks:
        return {}
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        pks = tuple(milvus_pks)
        cur.execute(
            "SELECT c.id, c.doc_id, c.kb_id, c.chunk_index, c.chunk_text, c.milvus_pk, "
            "d.file_name FROM chunks c "
            "LEFT JOIN documents d ON c.doc_id = d.id "
            "WHERE c.milvus_pk IN %s",
            (pks,),
        )
        result: dict[int, dict] = {}
        for row in cur.fetchall():
            r = dict(row)
            result[r["milvus_pk"]] = r
        return result


def delete_chunks_by_doc(doc_id: str) -> int:
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        return cur.rowcount


# ═══════════════════════════════════════════════════════════════════════
#  upload_tasks
# ═══════════════════════════════════════════════════════════════════════

def create_upload_task(
    task_id: str, doc_id: str, kb_id: str,
    file_name: str = "", file_size: int = 0,
) -> dict:
    now = int(time.time() * 1000)
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "INSERT INTO upload_tasks (id, doc_id, kb_id, file_name, file_size, status, progress, message, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, 'waiting', 0, '', %s, %s)",
            (task_id, doc_id, kb_id, file_name, file_size, now, now),
        )
        cur.execute("SELECT * FROM upload_tasks WHERE id = %s", (task_id,))
        return dict(cur.fetchone())


def update_upload_task(
    task_id: str,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
) -> dict | None:
    now = int(time.time() * 1000)
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if status is not None:
            cur.execute("UPDATE upload_tasks SET status = %s, updated_at = %s WHERE id = %s", (status, now, task_id))
        if progress is not None:
            cur.execute("UPDATE upload_tasks SET progress = %s, updated_at = %s WHERE id = %s", (progress, now, task_id))
        if message is not None:
            cur.execute("UPDATE upload_tasks SET message = %s, updated_at = %s WHERE id = %s", (message, now, task_id))
        cur.execute("SELECT * FROM upload_tasks WHERE id = %s", (task_id,))
        return _row(cur)


def get_upload_task(task_id: str) -> dict | None:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM upload_tasks WHERE id = %s", (task_id,))
        return _row(cur)


# ═══════════════════════════════════════════════════════════════════════
#  system_settings
# ═══════════════════════════════════════════════════════════════════════

def get_setting(key: str) -> str | None:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT value FROM system_settings WHERE key = %s", (key,))
        row = cur.fetchone()
        return row["value"] if row else None


def set_setting(key: str, value: str):
    now = int(time.time() * 1000)
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO system_settings (key, value, updated_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
            (key, value, now),
        )


# ═══════════════════════════════════════════════════════════════════════
#  聚合统计
# ═══════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT COUNT(*) AS cnt FROM knowledge_bases")
        kb_count = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(*) AS cnt FROM documents")
        doc_count = cur.fetchone()["cnt"]
        cur.execute("SELECT COALESCE(SUM(chunk_count), 0) AS cnt FROM documents")
        chunk_total = cur.fetchone()["cnt"]
    return {"kb_count": kb_count, "doc_count": doc_count, "chunk_count": chunk_total}


def get_recent_docs(limit: int = 10) -> list[dict]:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT d.*, kb.name AS kb_name FROM documents d "
            "LEFT JOIN knowledge_bases kb ON d.kb_id = kb.id "
            "WHERE d.status = 'completed' "
            "ORDER BY d.uploaded_at DESC LIMIT %s",
            (limit,),
        )
        return _rows(cur)


def get_distribution() -> list[dict]:
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT kb.id AS kb_id, kb.name AS kb_name, COUNT(d.id) AS doc_count "
            "FROM knowledge_bases kb LEFT JOIN documents d ON kb.id = d.kb_id "
            "GROUP BY kb.id ORDER BY doc_count DESC"
        )
        return _rows(cur)


# ═══════════════════════════════════════════════════════════════════════
#  communities（图社区）
# ═══════════════════════════════════════════════════════════════════════

def insert_communities(communities: list[dict]):
    """批量插入社区记录。"""
    now = int(time.time() * 1000)
    with _get_db() as conn:
        cur = conn.cursor()
        for com in communities:
            cur.execute(
                "INSERT INTO communities (id, kb_id, name, summary, entity_ids, chunk_ids, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    com["id"],
                    com["kb_id"],
                    com.get("name", ""),
                    com.get("summary", ""),
                    json.dumps(com.get("entity_ids", []), ensure_ascii=False),
                    json.dumps(com.get("chunk_ids", []), ensure_ascii=False),
                    now,
                ),
            )


def clear_communities(kb_id: str):
    """删除知识库下所有社区记录。"""
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM communities WHERE kb_id = %s", (kb_id,))


def get_communities_by_chunk_pks(chunk_pks: list[int]) -> list[dict]:
    """根据 chunk 主键列表查找所属社区（用于检索时返回社区摘要）。"""
    if not chunk_pks:
        return []
    with _get_db() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        results: list[dict] = []
        seen: set[str] = set()
        for pk in chunk_pks:
            # PostgreSQL 的 JSON 包含查询
            cur.execute(
                "SELECT * FROM communities WHERE chunk_ids::jsonb @> %s::jsonb LIMIT 3",
                (json.dumps([pk]),),
            )
            for row in cur.fetchall():
                r = dict(row)
                if r["id"] not in seen:
                    seen.add(r["id"])
                    r["entity_ids"] = json.loads(r["entity_ids"]) if isinstance(r["entity_ids"], str) else r["entity_ids"]
                    r["chunk_ids"] = json.loads(r["chunk_ids"]) if isinstance(r["chunk_ids"], str) else r["chunk_ids"]
                    results.append(r)
        return results
