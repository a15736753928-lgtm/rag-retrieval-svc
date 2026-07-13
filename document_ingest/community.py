"""社区发现与摘要 —— Leiden/Louvain 算法 + LLM 社区总结。"""

from __future__ import annotations

import json
import logging
import uuid

import networkx as nx

from core.graph_store import get_all_entities_for_kb, get_all_relations_for_kb
from core.llm_client import generate

logger = logging.getLogger(__name__)

# ── 社区摘要 Prompt ────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """你是一个技术文档摘要专家。根据给定的实体列表及其描述，用一段 150-250 字的中文总结这个主题组群的共同主题和核心内容。

要求：
1. 概括这个组的主题是什么
2. 列举组内关键概念及其关系
3. 语言简洁专业，适合作为检索结果展示"""


def _build_summary_prompt(entities: list[dict]) -> str:
    """构建社区摘要 prompt。"""
    entity_texts = []
    for e in entities[:30]:  # 最多 30 个实体
        entity_texts.append(f"- {e['name']}（{e.get('type', '概念')}）: {e.get('description', '')}")
    return "请为以下概念组生成摘要：\n\n" + "\n".join(entity_texts)


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def build_communities(kb_id: str) -> list[dict]:
    """为指定的知识库执行完整的社区发现流程。

    返回社区列表，每个社区包含 id、name、summary、entity_ids、chunk_ids。
    同时写入 PG communities 表。
    """
    # ── 1. 加载图数据 ──────────────────────────────────────────────
    entities = get_all_entities_for_kb(kb_id)
    relations = get_all_relations_for_kb(kb_id)

    if not entities:
        logger.info("kb=%s 无实体，跳过社区发现", kb_id)
        return []

    logger.info("社区发现开始: kb=%s entities=%d relations=%d", kb_id, len(entities), len(relations))

    # ── 2. 构建 networkx 图 ───────────────────────────────────────
    G = nx.Graph()

    # 添加实体节点
    entity_names = {e["name"] for e in entities}
    for e in entities:
        G.add_node(e["name"], **e)

    # 添加关系边
    for r in relations:
        if r["from"] in entity_names and r["to"] in entity_names:
            G.add_edge(r["from"], r["to"], relation_type=r.get("type", "相关"))

    if G.number_of_edges() == 0:
        # 无关系时，每个实体自己就是一个社区
        communities = []
        for e in entities:
            communities.append({
                "id": f"com_{uuid.uuid4().hex[:8]}",
                "kb_id": kb_id,
                "entity_ids": [e["name"]],
                "chunk_ids": e.get("source_chunks", []),
            })
    else:
        # ── 3. Louvain 社区发现 ───────────────────────────────────
        try:
            raw_communities = nx.community.louvain_communities(G, seed=42)
        except Exception:
            logger.warning("Louvain 社区发现失败，回退到连通分量")
            raw_communities = list(nx.connected_components(G))

        communities = []
        for i, com_set in enumerate(raw_communities):
            chunk_ids: list[int] = []
            entity_ids: list[str] = []
            for name in com_set:
                entity_ids.append(name)
                node_data = G.nodes.get(name, {})
                source_chunks = node_data.get("source_chunks", [])
                if isinstance(source_chunks, str):
                    try:
                        source_chunks = json.loads(source_chunks)
                    except json.JSONDecodeError:
                        source_chunks = []
                chunk_ids.extend(source_chunks)

            communities.append({
                "id": f"com_{uuid.uuid4().hex[:8]}",
                "kb_id": kb_id,
                "entity_ids": entity_ids,
                "chunk_ids": sorted(set(chunk_ids)),
            })

    logger.info("社区划分完成: kb=%s communities=%d", kb_id, len(communities))

    # ── 4. 生成社区摘要 ───────────────────────────────────────────
    for com in communities:
        com_entities = [
            G.nodes.get(name, {"name": name, "type": "概念", "description": ""})
            for name in com["entity_ids"]
        ]
        try:
            summary = generate(
                _build_summary_prompt(com_entities),
                system=_SUMMARY_SYSTEM,
                max_tokens=500,
            )
            com["summary"] = summary.strip()
            # 生成一个简短名称
            com["name"] = _extract_community_name(summary) or (
                com_entities[0]["name"] if com_entities else "未命名组"
            )
        except Exception as e:
            logger.warning("社区摘要生成失败: com=%s → %s", com["id"], e)
            com["summary"] = ""
            com["name"] = com["entity_ids"][0] if com["entity_ids"] else "未命名组"

    # ── 5. 写入 PG ────────────────────────────────────────────────
    try:
        from db.models import clear_communities, insert_communities
        clear_communities(kb_id)
        insert_communities(communities)
        logger.info("社区摘要写入 PG 完成: kb=%s count=%d", kb_id, len(communities))
    except Exception as e:
        logger.warning("社区摘要写入 PG 失败: %s", e)

    return communities


def _extract_community_name(summary: str) -> str:
    """从摘要中提取社区名称（取第一句话的前 30 字）。"""
    if not summary:
        return ""
    # 取到第一个句号/逗号之前
    for sep in ["。", "，", "、", "\n"]:
        idx = summary.find(sep)
        if idx > 0:
            return summary[:idx][:30]
    return summary[:30]
