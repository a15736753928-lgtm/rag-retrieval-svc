"""图检索服务 —— 查询时从知识图谱中召回关联的 chunk 和社区摘要。"""

from __future__ import annotations

import logging

from core.graph_store import expand_from_entities, query_entities
from core.llm_client import generate_json
from db.models import get_communities_by_chunk_pks

logger = logging.getLogger(__name__)

_EXTRACT_QUERY_ENTITIES_SYSTEM = """你是一个查询分析专家。从用户问题中提取关键概念/术语/实体名称。

要求：
1. 只提取问题中明确提到的具体概念（技术名词、工具名、方法名、人名等）
2. 每个提取结果是一个简短的实体名称（不超过 10 个字）
3. 不要提取过于宽泛的词（如"问题""方法""系统"）
4. 返回 JSON 格式：{"entities": ["实体1", "实体2", ...]}
5. 如果没有明确实体，返回 {"entities": []}"""


async def graph_search(
    query: str,
    kb_ids: list[str] | None = None,
    top_k: int = 20,
) -> dict:
    """从知识图谱中检索与查询相关的 chunk 和社区。

    Returns
    -------
    dict
        {
            "chunk_hits": [{"id": milvus_pk, ...}],   # 图召回的 chunk
            "community_summaries": [{"id": ..., "name": ..., "summary": ...}],
        }
    """
    result: dict = {"chunk_hits": [], "community_summaries": []}

    # ── 1. LLM 提取查询中的实体 ──────────────────────────────────
    query_entities = await _extract_entities_from_query(query)
    if not query_entities:
        logger.debug("查询中未提取到实体: query=%s", query[:50])
        return result

    logger.debug("查询实体: %s", query_entities)

    # ── 2. Kuzu 模糊匹配实体 + N-hop 扩展 ────────────────────────
    #  逐实体扩展，记录每个 chunk 被多少个查询实体命中 → 实体重叠率评分
    chunk_entity_hits: dict[int, int] = {}

    for name in query_entities:
        for kb_id in (kb_ids or []):
            expanded = expand_from_entities([name], hops=1, kb_id=kb_id)
            for cid in expanded.get("chunk_ids", []):
                chunk_entity_hits[cid] = chunk_entity_hits.get(cid, 0) + 1

        if not kb_ids:
            expanded = expand_from_entities([name], hops=1)
            for cid in expanded.get("chunk_ids", []):
                chunk_entity_hits[cid] = chunk_entity_hits.get(cid, 0) + 1

    # ── 3. 构建 chunk_hits（实体重叠率评分）──────────────────────
    total_query_entities = len(query_entities) or 1
    ranked_chunks = sorted(
        chunk_entity_hits.items(), key=lambda x: x[1], reverse=True
    )[:top_k]
    for pk, hit_count in ranked_chunks:
        # 实体重叠率 = 命中该 chunk 的查询实体数 / 查询实体总数
        score = round(min(hit_count / total_query_entities, 1.0), 4)
        result["chunk_hits"].append({
            "id": pk,
            "distance": score,
            "kb_id": "",
            "_source": "graph",
        })

    # ── 4. 查找相关社区摘要 ─────────────────────────────────────
    if chunk_entity_hits:
        try:
            chunk_ids_for_community = [pk for pk, _ in ranked_chunks]
            communities = get_communities_by_chunk_pks(chunk_ids_for_community)
            result["community_summaries"] = communities
        except Exception as e:
            logger.warning("社区摘要查询失败: %s", e)

    return result


async def _extract_entities_from_query(query: str) -> list[str]:
    """用 LLM 从查询中提取关键实体名称。"""
    try:
        resp = generate_json(
            query,
            system=_EXTRACT_QUERY_ENTITIES_SYSTEM,
            max_tokens=256,
        )
        entities = resp.get("entities", [])
        if isinstance(entities, list):
            return [e.strip() for e in entities if isinstance(e, str) and e.strip()]
    except Exception as e:
        logger.warning("查询实体提取失败: %s", e)
    return []
