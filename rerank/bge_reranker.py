"""BGE-Reranker 交叉打分、结果去重、重排。"""

from __future__ import annotations

import logging

from core.model_loader import get_reranker, infer_lock

logger = logging.getLogger(__name__)


def rerank(query: str, texts: list[str], top_k: int = 10) -> list[tuple[int, float]]:
    """对候选集交叉打分重排。

    Args:
        query: 检索词
        texts: 候选文本列表
        top_k: 返回前 top_k 条

    Returns:
        [(原始索引, 得分)], 按得分从高到低
    """
    if not texts:
        return []
    with infer_lock:
        model = get_reranker()
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs, show_progress_bar=False)

    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        scores = [float(scores)]

    indexed = [(i, max(0.0, float(scores[i]))) for i in range(len(texts))]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]
