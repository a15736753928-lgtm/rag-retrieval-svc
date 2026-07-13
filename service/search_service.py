"""完整检索链路：双向量召回 → RRF 融合 → PG 补全 → Rerank → 高亮。"""

from __future__ import annotations

import logging
import re
import time

from config import settings
from db import models as dao
from embedding.bge_encoder import encode_query_dense, encode_query_sparse
from milvus_client import vector_crud as vc
from rerank.bge_reranker import rerank as do_rerank
from schemas import HighlightSpan, SearchQueryRequest, SearchResultItem
from service import graph_search_service as graph_svc
from utils.text_processor import normalize_query

logger = logging.getLogger(__name__)

_TOKEN_SPLIT = re.compile(r"[^\w一-鿿]+")


async def search(req: SearchQueryRequest) -> list[SearchResultItem]:
    """检索主入口。返回结果列表（已按 similarity 降序）。"""
    start = time.perf_counter()

    if req.top_k == 0:
        final_limit = None
        rerank_top_k = settings.max_search_recall
    else:
        final_limit = req.top_k
        rerank_top_k = min(max(final_limit * 2, 20), settings.max_search_recall)
    dense_weight = 0.7
    sparse_weight = 0.3
    min_sim = req.min_similarity

    # ── 构建 kb_id 过滤表达式 ─────────────────────────────────
    kb_filter = ""
    if req.kb_ids:
        parts = [f'kb_id == "{k}"' for k in req.kb_ids]
        kb_filter = "(" + " or ".join(parts) + ")"

    # ── 为结果补 kb_name，预加载知识库名映射 ──────────────────
    kb_name_map: dict[str, str] = {}
    kbs, _ = dao.list_knowledge_bases(1, 10000)
    for kb in kbs:
        kb_name_map[kb["id"]] = kb["name"]

    # ── 0. 查询规范化 ───────────────────────────────────────
    query = normalize_query(req.query)

    # ── 1. 稠密召回 ─────────────────────────────────────────
    query_dense = encode_query_dense(query)
    dense_hits = vc.search_dense(query_dense, top_k=rerank_top_k, expr=kb_filter)

    # ── 2. 稀疏召回 ─────────────────────────────────────────
    query_sparse = encode_query_sparse(query)
    sparse_hits = vc.search_sparse(query_sparse, top_k=rerank_top_k, expr=kb_filter)

    # ── 3. 图召回 ─────────────────────────────────────────
    graph_result = await graph_svc.graph_search(
        query, kb_ids=req.kb_ids, top_k=rerank_top_k,
    )
    graph_hits = graph_result["chunk_hits"]
    community_map: dict[int, dict] = {}
    for com in graph_result.get("community_summaries", []):
        for pk in com.get("chunk_ids", []):
            community_map[pk] = {"id": com["id"], "name": com["name"]}

    # ── 4. RRF 三路融合 ────────────────────────────────────
    fused = _rrf_fuse(dense_hits, sparse_hits, graph_hits, rerank_top_k, dense_weight, sparse_weight)

    # ── 3.5 PG 补全：用 Milvus ID 批量查 chunks + documents ──
    if fused:
        milvus_ids = [r["id"] for r in fused]
        chunk_map = dao.get_chunks_by_milvus_pks(milvus_ids)
        # 补全每个融合结果
        for r in fused:
            ch = chunk_map.get(r["id"])
            if ch:
                r["chunk_text"] = ch.get("chunk_text", "")
                r["chunk_index"] = ch.get("chunk_index", 0)
                r["file_name"] = ch.get("file_name", "")
                r["kb_id"] = ch.get("kb_id", r.get("kb_id", ""))
            else:
                # PG 里没有对应记录（数据异常），用默认值
                r["chunk_text"] = ""
                r["chunk_index"] = 0
                r["file_name"] = ""
        # 去掉 PG 里也没文本的结果
        fused = [r for r in fused if r.get("chunk_text", "").strip()]

    # ── 3.6 回退策略：三路召回都为空时，放宽稠密召回 ─────────
    if not fused:
        logger.info("主检索返回空结果，尝试回退（稠密召回 ×2 放宽）: query=%r", query[:50])
        fallback_top_k = min(rerank_top_k * 2, settings.max_search_recall)
        dense_fallback = vc.search_dense(query_dense, top_k=fallback_top_k, expr=kb_filter)
        if dense_fallback:
            fb_ids = [r["id"] for r in dense_fallback]
            fb_chunk_map = dao.get_chunks_by_milvus_pks(fb_ids)
            for r in dense_fallback:
                ch = fb_chunk_map.get(r["id"])
                if ch and ch.get("chunk_text", "").strip():
                    r["chunk_text"] = ch["chunk_text"]
                    r["chunk_index"] = ch.get("chunk_index", 0)
                    r["file_name"] = ch.get("file_name", "")
                    r["kb_id"] = ch.get("kb_id", r.get("kb_id", ""))
                    r.setdefault("_source", "vector")
                    fused.append(r)
        if not fused:
            logger.warning("所有检索路径均无结果: query=%r", query[:50])

    # ── 4. Rerank ──────────────────────────────────────────
    if fused:
        texts = [r.get("chunk_text", "") for r in fused]
        reranked = do_rerank(query, texts, len(texts) if final_limit is None else final_limit)
        fused = [fused[idx] for idx, _ in reranked]
        for i, (_, score) in enumerate(reranked):
            if i < len(fused):
                fused[i]["_score"] = float(score)

    # ── 5. 构建结果 ────────────────────────────────────────
    results: list[SearchResultItem] = []
    candidates = fused if final_limit is None else fused[:final_limit]
    for r in candidates:
        similarity = round(r.get("_score", r.get("distance", 0.0)), 4)
        if similarity < min_sim:
            continue
        content = r.get("chunk_text", "")
        highlights = _extract_highlight_terms(query, content)
        kb_id = r.get("kb_id", "")

        milvus_id = r.get("id", 0)
        com_info = community_map.get(milvus_id, {})
        match_type = r.get("_source", "vector")

        results.append(SearchResultItem(
            id=str(milvus_id),
            content=content,
            file_name=r.get("file_name", ""),
            kb_id=kb_id,
            kb_name=kb_name_map.get(kb_id, ""),
            chunk_index=r.get("chunk_index", 0),
            similarity=similarity,
            highlights=highlights,
            community_id=com_info.get("id", ""),
            community_name=com_info.get("name", ""),
            match_type=match_type,
        ))

    took = (time.perf_counter() - start) * 1000
    kb_info = req.kb_ids if req.kb_ids else "(all)"
    logger.info("检索 kb=%s query=%r → %d results (%.1fms)", kb_info, query[:30], len(results), took)
    return results


async def get_chunk_content(chunk_id: int, query: str = "") -> dict | None:
    """获取分片原文 + 高亮位置（从 PG 查）。"""
    # chunk_id 对外是 milvus_pk，从 PG 查
    chunk_map = dao.get_chunks_by_milvus_pks([chunk_id])
    ch = chunk_map.get(chunk_id)
    if not ch:
        return None
    content = ch.get("chunk_text", "")
    highlights: list[HighlightSpan] = []
    if query:
        highlights = _compute_highlight_spans(query, content)
    return {
        "id": str(chunk_id),
        "kb_id": ch.get("kb_id", ""),
        "file_name": ch.get("file_name", ""),
        "chunk_index": ch.get("chunk_index", 0),
        "content": content,
        "highlights": highlights,
    }


# ═══════════════════════════════════════════════════════════════════════
#  RRF 融合
# ═══════════════════════════════════════════════════════════════════════

def _rrf_fuse(dense, sparse, graph, top_k, dense_w, sparse_w) -> list[dict]:
    k = settings.rrf_k
    graph_w = 0.25  # 图召回权重（基于实体重叠率评分）
    score_map: dict[int, dict] = {}
    for rank, r in enumerate(dense):
        score_map[r["id"]] = {"hit": r, "score": dense_w / (k + rank + 1)}
    for rank, r in enumerate(sparse):
        if r["id"] in score_map:
            score_map[r["id"]]["score"] += sparse_w / (k + rank + 1)
        else:
            score_map[r["id"]] = {"hit": r, "score": sparse_w / (k + rank + 1)}
    for rank, r in enumerate(graph):
        if r["id"] in score_map:
            score_map[r["id"]]["score"] += graph_w / (k + rank + 1)
            score_map[r["id"]]["hit"]["_source"] = "both"
        else:
            score_map[r["id"]] = {"hit": r, "score": graph_w / (k + rank + 1)}
    ranked = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    out = []
    for item in ranked[:top_k]:
        r = dict(item["hit"])
        r["_score"] = item["score"]
        out.append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  高亮
# ═══════════════════════════════════════════════════════════════════════

def _extract_highlight_terms(query: str, text: str) -> list[str]:
    """从 query 中提取在 text 中出现的词/短语作为高亮关键词。"""
    if not query or not text:
        return []
    text_lower = text.lower()
    terms = _TOKEN_SPLIT.split(query.lower())
    terms = [t for t in terms if len(t) >= 1]
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        if term in text_lower and term not in seen:
            seen.add(term)
            result.append(term)
    return result


def _compute_highlight_spans(query: str, text: str) -> list[HighlightSpan]:
    """计算检索词在文本中的 {start, end} 位置。"""
    if not query or not text:
        return []
    terms = _TOKEN_SPLIT.split(query.lower())
    terms = [t for t in terms if len(t) >= 1]
    if not terms:
        return []
    text_lower = text.lower()
    spans: list[HighlightSpan] = []
    seen: set[tuple[int, int]] = set()
    for term in terms:
        start = 0
        while True:
            pos = text_lower.find(term, start)
            if pos == -1:
                break
            key = (pos, pos + len(term))
            if key not in seen:
                seen.add(key)
                spans.append(HighlightSpan(start=pos, end=pos + len(term)))
            start = pos + len(term)
    spans.sort(key=lambda h: h.start)
    return spans
