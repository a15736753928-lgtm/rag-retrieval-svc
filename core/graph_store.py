"""Kuzu 嵌入式图数据库封装 —— 存储实体/关系知识图谱。

和 Milvus Lite 一样零运维，数据持久化到本地目录。
节点: Entity（概念）+ Chunk（分片引用）
边:   MENTIONS（实体→分片）+ RELATED（实体→实体）

所有查询使用 $param 参数化，杜绝 Cypher 注入。
"""

from __future__ import annotations

import json
import logging
import os

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
        pass
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
    # 先删后建，实现 upsert
    try:
        conn.execute(
            "MATCH (e:Entity) WHERE e.name = $name DETACH DELETE e",
            {"name": name})
    except Exception:
        pass
    conn.execute(
        "CREATE (e:Entity {name: $name, type: $type, "
        "description: $dsc, source_chunks: $chunks, kb_id: $kb_id})",
        {"name": name, "type": entity_type, "dsc": description,
         "chunks": chunks_json, "kb_id": kb_id},
    )


def upsert_chunk(chunk_id: int, kb_id: str, doc_id: str):
    """创建或更新 Chunk 节点。"""
    conn = _get_conn()
    try:
        conn.execute(
            "MATCH (c:Chunk) WHERE c.chunk_id = $id DETACH DELETE c",
            {"id": chunk_id})
    except Exception:
        pass
    conn.execute(
        "CREATE (c:Chunk {chunk_id: $id, kb_id: $kb, doc_id: $doc})",
        {"id": chunk_id, "kb": kb_id, "doc": doc_id},
    )


def link_entity_to_chunk(entity_name: str, chunk_id: int):
    """创建 MENTIONS 边：实体 → 分片。"""
    conn = _get_conn()
    try:
        conn.execute(
            "MATCH (e:Entity)-[r:MENTIONS]->(c:Chunk) "
            "WHERE e.name = $name AND c.chunk_id = $cid DELETE r",
            {"name": entity_name, "cid": chunk_id},
        )
    except Exception:
        pass
    try:
        conn.execute(
            "MATCH (e:Entity), (c:Chunk) "
            "WHERE e.name = $name AND c.chunk_id = $cid "
            "CREATE (e)-[:MENTIONS]->(c)",
            {"name": entity_name, "cid": chunk_id},
        )
    except Exception:
        logger.debug("MENTIONS 边创建失败: %s → chunk_%s", entity_name, chunk_id)


def upsert_relation(from_entity: str, to_entity: str, relation_type: str):
    """创建或更新实体间关系边。"""
    conn = _get_conn()
    try:
        conn.execute(
            "MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
            "WHERE a.name = $from AND b.name = $to DELETE r",
            {"from": from_entity, "to": to_entity},
        )
    except Exception:
        pass
    try:
        conn.execute(
            "MATCH (a:Entity), (b:Entity) "
            "WHERE a.name = $from AND b.name = $to "
            "CREATE (a)-[:RELATED {relation_type: $rel}]->(b)",
            {"from": from_entity, "to": to_entity, "rel": relation_type},
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

        try:
            if kb_id:
                rows = conn.execute(
                    "MATCH (e:Entity) WHERE e.kb_id = $kb AND "
                    "CONTAINS(LOWER(e.name), LOWER($term)) "
                    "RETURN e.name, e.type, e.description, e.source_chunks "
                    f"LIMIT {top_k}",
                    {"kb": kb_id, "term": term_clean},
                )
            else:
                rows = conn.execute(
                    "MATCH (e:Entity) WHERE "
                    "CONTAINS(LOWER(e.name), LOWER($term)) "
                    "RETURN e.name, e.type, e.description, e.source_chunks "
                    f"LIMIT {top_k}",
                    {"term": term_clean},
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
        try:
            # 1-跳: Entity → MENTIONS → Chunk
            if kb_id:
                rows = conn.execute(
                    "MATCH (e:Entity)-[:MENTIONS]->(c:Chunk) "
                    "WHERE e.kb_id = $kb AND e.name = $name "
                    "RETURN c.chunk_id",
                    {"kb": kb_id, "name": name},
                )
            else:
                rows = conn.execute(
                    "MATCH (e:Entity)-[:MENTIONS]->(c:Chunk) "
                    "WHERE e.name = $name "
                    "RETURN c.chunk_id",
                    {"name": name},
                )
            while rows.has_next():
                chunk_ids.add(rows.get_next()[0])

            # N-跳: Entity → RELATED → Entity
            if hops > 1:
                if kb_id:
                    related = conn.execute(
                        f"MATCH (e:Entity)-[:RELATED*1..{hops}]->(r:Entity) "
                        "WHERE e.kb_id = $kb AND e.name = $name "
                        "RETURN r.name, r.type, r.description",
                        {"kb": kb_id, "name": name},
                    )
                else:
                    related = conn.execute(
                        f"MATCH (e:Entity)-[:RELATED*1..{hops}]->(r:Entity) "
                        "WHERE e.name = $name "
                        "RETURN r.name, r.type, r.description",
                        {"name": name},
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
            "MATCH (e:Entity) WHERE e.kb_id = $kb "
            "RETURN e.name, e.type, e.description, e.source_chunks",
            {"kb": kb_id},
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
        rows = conn.execute(
            "MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
            "WHERE a.kb_id = $kb AND b.kb_id = $kb "
            "RETURN a.name, b.name, r.relation_type",
            {"kb": kb_id},
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
        conn.execute("MATCH (e:Entity) WHERE e.kb_id = $kb DETACH DELETE e", {"kb": kb_id})
        conn.execute("MATCH (c:Chunk) WHERE c.kb_id = $kb DELETE c", {"kb": kb_id})
    except Exception as e:
        logger.warning("清理图数据失败: %s", e)
