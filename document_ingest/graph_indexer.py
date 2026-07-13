"""图索引器 —— 使用 LLM 从 chunks 中抽取实体和关系，写入 Kuzu 图库。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from core.graph_store import (
    link_entity_to_chunk,
    upsert_chunk,
    upsert_entity,
    upsert_relation,
)
from core.llm_client import generate_json

logger = logging.getLogger(__name__)

# 每批处理的 chunk 数量
_BATCH_SIZE = 20

# ── Prompt 模板 ────────────────────────────────────────────────────────

_EXTRACT_ENTITIES_SYSTEM = """你是一个知识图谱实体抽取专家。从给定的文本片段中提取关键实体（概念、术语、技术、人物、组织、方法等）。

要求：
1. 每个实体必须包含：name（实体名称）、type（类型，如"技术""模型""算法""概念""工具""方法"）、description（一句话描述）
2. 只提取文本中明确提到的实体，不要推测
3. 实体名称要简洁规范（如"BGE-M3"而不是"BGE-M3 嵌入模型"）
4. 忽略过于宽泛或没有信息量的词（如"系统""结果""测试"）
5. 同一个实体在同一文本片段中出现多次只提取一次"""


def _build_extract_prompt(chunks: list[str]) -> str:
    """构建实体抽取 prompt。"""
    parts = []
    for i, text in enumerate(chunks):
        parts.append(f"[片段 {i}]\n{text[:800]}\n")  # 截断过长文本
    return "从以下文本片段中提取实体。对每个片段独立分析，返回每个片段各自的实体列表：\n\n" + "\n".join(parts)


_EXTRACT_RELATIONS_SYSTEM = """你是一个知识图谱关系抽取专家。给定一组实体及其出现的上下文，判断哪些实体之间存在语义关系。

要求：
1. 关系方向：from 是源实体，to 是目标实体
2. relation_type 使用简洁的动词短语（如"使用""属于""调用""依赖""实现""基于""部署在"）
3. 只输出存在明确关系且二者在同一上下文中被同时讨论的实体对
4. 不要输出过于间接或推测的关系"""


def _build_relation_prompt(entities: list[dict], chunks: list[str]) -> str:
    """构建关系抽取 prompt。"""
    entity_list = []
    for e in entities:
        entity_list.append(f"- {e['name']}（{e['type']}）: {e['description']}")
    return (
        f"已知实体列表：\n" + "\n".join(entity_list) +
        "\n\n文本上下文（片段）：\n" +
        "\n".join(f"[片段 {i}] {chunks[i][:300]}" for i in range(min(len(chunks), 10))) +
        "\n\n请判断上述实体中哪些之间存在关系，返回关系列表。"
    )


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def index_chunks_to_graph(
    chunks: list[str],
    chunk_pks: list[int],
    kb_id: str,
    doc_id: str,
):
    """从 chunks 中抽取实体和关系，写入 Kuzu 图库。

    步骤：
    1. 批量抽取实体 → 写入 Entity 节点 + MENTIONS 边
    2. 跨 chunk 抽取关系 → 写入 RELATED 边
    3. 记录 Chunk 节点（用于图查询时反向定位）

    设计为后台异步执行，失败只记日志，不影响主入库链路。
    """
    if not chunks:
        return

    start = time.perf_counter()
    logger.info("图索引开始: doc=%s kb=%s chunks=%d", doc_id, kb_id, len(chunks))

    try:
        # ── 1. 记录 Chunk 节点 ──────────────────────────────────────
        for pk in chunk_pks:
            try:
                upsert_chunk(chunk_id=pk, kb_id=kb_id, doc_id=doc_id)
            except Exception:
                pass

        # ── 2. 批量抽取实体 ────────────────────────────────────────
        all_entities: list[dict] = []
        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start:batch_start + _BATCH_SIZE]
            batch_pks = chunk_pks[batch_start:batch_start + _BATCH_SIZE]

            entities = _extract_entities_batch(batch)
            for ent in entities:
                ent.setdefault("source_chunks", [])
                # 将片段索引映射回 chunk_pk
                chunk_idx = ent.pop("_chunk_idx", -1)
                if 0 <= chunk_idx < len(batch_pks):
                    ent["source_chunks"].append(batch_pks[chunk_idx])

            all_entities.extend(entities)

            # 写入图库
            for ent in entities:
                try:
                    upsert_entity(
                        name=ent["name"],
                        entity_type=ent.get("type", "概念"),
                        description=ent.get("description", ""),
                        source_chunk_ids=ent.get("source_chunks", []),
                        kb_id=kb_id,
                    )
                    for pk in ent.get("source_chunks", []):
                        link_entity_to_chunk(ent["name"], pk)
                except Exception as e:
                    logger.debug("实体写入失败: %s → %s", ent.get("name"), e)

        logger.info("实体抽取完成: doc=%s 实体数=%d", doc_id, len(all_entities))

        # ── 3. 关系抽取 ────────────────────────────────────────────
        if len(all_entities) >= 2:
            relations = _extract_relations(all_entities, chunks)
            for rel in relations:
                try:
                    upsert_relation(
                        from_entity=rel["from"],
                        to_entity=rel["to"],
                        relation_type=rel.get("type", "相关"),
                    )
                except Exception:
                    pass
            logger.info("关系抽取完成: doc=%s 关系数=%d", doc_id, len(relations))

    except Exception as e:
        logger.exception("图索引异常: doc=%s", doc_id)

    took_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("图索引结束: doc=%s 耗时=%sms", doc_id, took_ms)


# ═══════════════════════════════════════════════════════════════════════
#  LLM 调用封装
# ═══════════════════════════════════════════════════════════════════════

def _extract_entities_batch(chunks: list[str]) -> list[dict]:
    """用 LLM 从一批 chunks 中抽取实体。"""
    try:
        result = generate_json(
            _build_extract_prompt(chunks),
            system=_EXTRACT_ENTITIES_SYSTEM,
            max_tokens=2048,
        )
        return _parse_entities_result(result, len(chunks))
    except Exception as e:
        logger.warning("实体抽取 LLM 调用失败: %s", e)
        return []


def _extract_relations(entities: list[dict], chunks: list[str]) -> list[dict]:
    """用 LLM 抽取实体间关系。"""
    try:
        result = generate_json(
            _build_relation_prompt(entities, chunks),
            system=_EXTRACT_RELATIONS_SYSTEM,
            max_tokens=2048,
        )
        return _parse_relations_result(result)
    except Exception as e:
        logger.warning("关系抽取 LLM 调用失败: %s", e)
        return []


def _parse_entities_result(result: dict, num_chunks: int) -> list[dict]:
    """解析 LLM 返回的实体列表。支持两种格式：
    - {"片段0": [{"name": ..., "type": ..., "description": ...}, ...], ...}
    - {"entities": [...]} 或直接 list
    """
    entities: list[dict] = []

    # 格式 1: 按片段分组
    for i in range(num_chunks):
        key = f"片段{i}" if f"片段{i}" in result else str(i)
        items = result.get(key, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "name" in item:
                    item["_chunk_idx"] = i
                    entities.append(item)

    if entities:
        return entities

    # 格式 2: entities 字段
    items = result.get("entities", [])
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "name" in item:
                item.setdefault("_chunk_idx", 0)
                entities.append(item)

    return entities


def _parse_relations_result(result: dict) -> list[dict]:
    """解析 LLM 返回的关系列表。"""
    items = result.get("relations", []) or result.get("relationships", [])
    if isinstance(result, list):
        items = result  # 直接是列表

    relations: list[dict] = []
    for item in items:
        if isinstance(item, dict) and "from" in item and "to" in item:
            relations.append({
                "from": item["from"],
                "to": item["to"],
                "type": item.get("relation_type", item.get("type", "相关")),
            })
    return relations
