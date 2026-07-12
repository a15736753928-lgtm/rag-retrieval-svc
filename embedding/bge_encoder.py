"""BGE-M3 ONNX INT8 稠密 / 稀疏向量批量编码（入库 + 检索共用）。"""

from __future__ import annotations

import logging

from config import settings
from core.model_loader import get_bge_m3, infer_lock

logger = logging.getLogger(__name__)


def encode_dense(texts: list[str]) -> list[list[float]]:
    """批量文本 → 稠密向量列表 (N × 1024)。"""
    if not texts:
        return []
    with infer_lock:
        model = get_bge_m3()
        emb = model.encode(
            texts,
            batch_size=settings.ingest_batch_size,
            normalize_embeddings=True,
        )
    return emb.tolist()


def encode_query_dense(text: str) -> list[float]:
    """单条查询 → 稠密向量。"""
    return encode_dense([text])[0]


def encode_sparse(texts: list[str]) -> list[dict[int, float]]:
    """批量文本 → 稀疏词权向量 [{token_id: weight}, ...]。

    ONNX INT8 模型原生输出稀疏向量，不再返回空字典。
    """
    if not texts:
        return []
    with infer_lock:
        model = get_bge_m3()
        return model.encode_sparse(texts, batch_size=settings.ingest_batch_size)


def encode_query_sparse(text: str) -> dict[int, float]:
    """单条查询 → 稀疏向量。"""
    results = encode_sparse([text])
    return results[0] if results else {}
