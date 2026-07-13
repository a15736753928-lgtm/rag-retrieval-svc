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
    all_chunk_ids: set[int] = set()

    for kb_id in (kb_ids or []):
        # 模糊匹配实体
        matched = query_entities

        # 扩展获取关联 chunk
        expanded = expand_from_entities(matched, hops=1, kb_id=kb_id)
        all_chunk_ids.update(expanded.get("chunk_ids", []))

    if not kb_ids:
        # 无 kb 过滤时，全局搜索
        expanded = expand_from_entities(query_entities, hops=1)
        all_chunk_ids.update(expanded.get("chunk_ids", []))

    # ── 3. 构建 chunk_hits（转换为 dense/sparse 兼容格式）────────
    # 图召回的 chunk 没有相似度分数，给一个默认排名分
    for idx, pk in enumerate(sorted(all_chunk_ids)[:top_k]):
        result["chunk_hits"].append({
            "id": pk,
            "distance": 0.5 + (1.0 / (idx + 2)),  # 伪分数，排在向量结果中间
            "kb_id": "",
            "_source": "graph",
        })

    # ── 4. 查找相关社区摘要 ─────────────────────────────────────
    if all_chunk_ids:
        try:
            communities = get_communities_by_chunk_pks(list(all_chunk_ids)[:top_k])
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
