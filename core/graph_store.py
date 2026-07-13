"""Kuzu 嵌入式图数据库封装 —— 存储实体/关系知识图谱。

和 Milvus Lite 一样零运维，数据持久化到本地目录。
节点: Entity（概念）+ Chunk（分片引用）
边:   MENTIONS（实体→分片）+ RELATED（实体→实体）
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import kuzu

logger = logging.getLogger(__name__)

_DB: kuzu.Database | None = None
_CONN: kuzu.Connection | None = None


def init_graph(db_path: str = "storage/kuzu_data") -> kuzu.Connection:
    """初始化图数据库（幂等），创建表结构后返回连接。"""
    global _DB, _CONN
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    _DB = kuzu.Database(str(db_path))
    _CONN = kuzu.Connection(_DB)
    _create_schema(_CONN)
    logger.info("Kuzu 图数据库就绪: %s", db_path)
    return _CONN


def _get_conn() -> kuzu.Connection:
    """获取连接（惰性初始化）。"""
    global _CONN
    if _CONN is None:
        init_graph()
    return _CONN


def _create_schema(conn: kuzu.Connection):
    """幂等建表。"""
    try:
        conn.execute("CREATE NODE TABLE IF NOT EXISTS Entity("
                     "name STRING, type STRING, description STRING, "
                     "source_chunks STRING, kb_id STRING, "
                     "PRIMARY KEY(name))")
    except Exception:
        pass  # 表已存在
    try:
        conn.execute("CREATE NODE TABLE IF NOT EXISTS Chunk("
                     "chunk_id INT64, kb_id STRING, doc_id STRING, "
                     "PRIMARY KEY(chunk_id))")
    except Exception:
        pass
    try:
        conn.execute("CREATE REL TABLE IF NOT EXISTS MENTIONS("
                     "FROM Entity TO Chunk)")
    except Exception:
        pass
    try:
        conn.execute("CREATE REL TABLE IF NOT EXISTS RELATED("
                     "FROM Entity TO Entity, relation_type STRING)")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
#  写入
# ═══════════════════════════════════════════════════════════════════════

def upsert_entity(
    name: str,
    entity_type: str,
    description: str,
    source_chunk_ids: list[int],
    kb_id: str,
):
    """创建或更新实体节点。"""
    conn = _get_conn()
    chunks_json = json.dumps(source_chunk_ids, ensure_ascii=False)
    # 先删后建，实现 upsert 语义
    try:
        conn.execute(
            f"MATCH (e:Entity) WHERE e.name = '{_escape(name)}' DETACH DELETE e")
    except Exception:
        pass
    # Kuzu 不支持 CREATE 中 $param 属性值，用 f-string 内联
    # name/description 来自 LLM 输出，做基本转义
    safe_name = _escape(name)
    safe_type = _escape(entity_type)
    safe_desc = _escape(description)
    conn.execute(
        f"CREATE (e:Entity {{name: '{safe_name}', type: '{safe_type}', "
        f"description: '{safe_desc}', source_chunks: '{chunks_json}', "
        f"kb_id: '{kb_id}'}})"
    )


def upsert_chunk(chunk_id: int, kb_id: str, doc_id: str):
    """创建或更新 Chunk 节点。"""
    conn = _get_conn()
    try:
        conn.execute(
            f"MATCH (c:Chunk) WHERE c.chunk_id = {chunk_id} DETACH DELETE c")
    except Exception:
        pass
    conn.execute(
        f"CREATE (c:Chunk {{chunk_id: {chunk_id}, kb_id: '{kb_id}', doc_id: '{doc_id}'}})"
    )


def _escape(s: str) -> str:
    """转义 Kuzu 字符串中的单引号。"""
    return s.replace("'", "''").replace("\\", "\\\\")


def link_entity_to_chunk(entity_name: str, chunk_id: int):
    """创建 MENTIONS 边：实体 → 分片。"""
    conn = _get_conn()
    safe_name = _escape(entity_name)
    # 删除旧边
    try:
        conn.execute(
            f"MATCH (e:Entity)-[r:MENTIONS]->(c:Chunk) "
            f"WHERE e.name = '{safe_name}' AND c.chunk_id = {chunk_id} "
            "DELETE r"
        )
    except Exception:
        pass
    # 创建新边
    try:
        conn.execute(
            f"MATCH (e:Entity), (c:Chunk) "
            f"WHERE e.name = '{safe_name}' AND c.chunk_id = {chunk_id} "
            "CREATE (e)-[:MENTIONS]->(c)"
        )
    except Exception:
        logger.debug("MENTIONS 边创建失败: %s → chunk_%s", entity_name, chunk_id)


def upsert_relation(from_entity: str, to_entity: str, relation_type: str):
    """创建或更新实体间关系边。"""
    conn = _get_conn()
    safe_from = _escape(from_entity)
    safe_to = _escape(to_entity)
    safe_rel = _escape(relation_type)
    try:
        conn.execute(
            f"MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
            f"WHERE a.name = '{safe_from}' AND b.name = '{safe_to}' "
            "DELETE r"
        )
    except Exception:
        pass
    try:
        conn.execute(
            f"MATCH (a:Entity), (b:Entity) "
            f"WHERE a.name = '{safe_from}' AND b.name = '{safe_to}' "
            f"CREATE (a)-[:RELATED {{relation_type: '{safe_rel}'}}]->(b)"
        )
    except Exception:
        logger.debug("RELATED 边创建失败: %s → %s", from_entity, to_entity)


# ═══════════════════════════════════════════════════════════════════════
#  查询
# ═══════════════════════════════════════════════════════════════════════

def query_entities(query_terms: list[str], kb_id: str = "", top_k: int = 10) -> list[dict]:
    """按名称模糊匹配实体。"""
    conn = _get_conn()
    results: list[dict] = []
    seen: set[str] = set()

    for term in query_terms:
        term_clean = term.strip()
        if not term_clean or term_clean in seen:
            continue
        seen.add(term_clean)

        kb_filter = f"e.kb_id = '{kb_id}' AND " if kb_id else ""
        try:
            rows = conn.execute(
                f"MATCH (e:Entity) WHERE {kb_filter} "
                f"CONTAINS(LOWER(e.name), LOWER('{_escape(term_clean)}')) "
                "RETURN e.name, e.type, e.description, e.source_chunks "
                f"LIMIT {top_k}"
            )
            while rows.has_next():
                row = rows.get_next()
                results.append({
                    "name": row[0], "type": row[1],
                    "description": row[2], "source_chunks": json.loads(row[3]),
                })
        except Exception:
            logger.debug("实体查询失败: term=%s", term_clean)

    return results


def expand_from_entities(
    entity_names: list[str],
    hops: int = 1,
    kb_id: str = "",
) -> dict[str, list]:
    """从实体出发沿边扩展，返回关联的 chunk_id 列表和相关实体。

    Returns: {"chunk_ids": [...], "related_entities": [...]}
    """
    conn = _get_conn()
    chunk_ids: set[int] = set()
    related_entities: list[dict] = []

    for name in entity_names:
        safe_name = _escape(name)
        kb_filter = f"e.kb_id = '{kb_id}' AND " if kb_id else ""
        try:
            # 1-跳: Entity → MENTIONS → Chunk
            where_clause = f"e.kb_id = '{kb_id}' AND e.name = '{safe_name}'" if kb_id else f"e.name = '{safe_name}'"
            rows = conn.execute(
                f"MATCH (e:Entity)-[:MENTIONS]->(c:Chunk) "
                f"WHERE {where_clause} "
                "RETURN c.chunk_id"
            )
            while rows.has_next():
                chunk_ids.add(rows.get_next()[0])

            # N-跳: Entity → RELATED → Entity
            if hops > 1:
                related = conn.execute(
                    f"MATCH (e:Entity)-[:RELATED*1..{hops}]->(r:Entity) "
                    f"WHERE {where_clause} "
                    "RETURN r.name, r.type, r.description"
                )
                while related.has_next():
                    row = related.get_next()
                    related_entities.append({
                        "name": row[0], "type": row[1], "description": row[2],
                    })
        except Exception:
            logger.debug("图扩展失败: entity=%s", name)

    return {"chunk_ids": sorted(chunk_ids), "related_entities": related_entities}


def get_all_entities_for_kb(kb_id: str) -> list[dict]:
    """获取知识库下所有实体（用于社区发现）。"""
    conn = _get_conn()
    results: list[dict] = []
    try:
        rows = conn.execute(
            f"MATCH (e:Entity) WHERE e.kb_id = '{kb_id}' "
            "RETURN e.name, e.type, e.description, e.source_chunks"
        )
        while rows.has_next():
            row = rows.get_next()
            results.append({
                "name": row[0], "type": row[1],
                "description": row[2], "source_chunks": json.loads(row[3]),
            })
    except Exception as e:
        logger.warning("获取实体列表失败: %s", e)
    return results


def get_all_relations_for_kb(kb_id: str) -> list[dict]:
    """获取知识库下所有实体关系（用于社区发现）。"""
    conn = _get_conn()
    results: list[dict] = []
    try:
        # 找到该 kb 下的所有实体对之间的 RELATED 边
        rows = conn.execute(
            f"MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
            f"WHERE a.kb_id = '{kb_id}' AND b.kb_id = '{kb_id}' "
            "RETURN a.name, b.name, r.relation_type"
        )
        while rows.has_next():
            row = rows.get_next()
            results.append({"from": row[0], "to": row[1], "type": row[2]})
    except Exception as e:
        logger.warning("获取关系列表失败: %s", e)
    return results


def clear_kb(kb_id: str):
    """删除知识库下所有实体节点和 Chunk 节点。"""
    conn = _get_conn()
    try:
        conn.execute(f"MATCH (e:Entity) WHERE e.kb_id = '{kb_id}' DETACH DELETE e")
        conn.execute(f"MATCH (c:Chunk) WHERE c.kb_id = '{kb_id}' DELETE c")
    except Exception as e:
        logger.warning("清理图数据失败: %s", e)
