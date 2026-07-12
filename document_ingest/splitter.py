"""文本分片 —— LlamaIndex SentenceSplitter，按句子边界切割。"""

from __future__ import annotations

import time
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from config import settings


class _STEmbedAdapter(BaseEmbedding):
    """将 SentenceTransformer 适配为 LlamaIndex BaseEmbedding，复用已加载的模型实例。"""

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._st_model = model

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._st_model.encode([text], normalize_embeddings=True)[0].tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._st_model.encode(texts, normalize_embeddings=True).tolist()

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def _split_sentences(text: str) -> list[str]:
    """中英文分句 —— 过滤 PDF 噪声 + 段落合并 + 句末标点切分。"""
    import re

    # 1. 移除零宽字符（PDF 常见）
    text = re.sub(r"[​‌‍﻿]", "", text)

    # 2. 按段落边界（空行）分块
    paragraphs = re.split(r"\n\s*\n", text)

    sentences: list[str] = []
    for para in paragraphs:
        # 3. 合并被 PDF 硬换行切开的行
        lines = para.split("\n")
        merged_lines: list[str] = []
        buf = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # 过滤纯数字（页号 / 代码块行号残留）
            if re.match(r"^[\d\s\.\-\|]+$", stripped):
                continue
            # 过滤纯符号 / 零宽行
            if re.match(r"^[•\-\*\#\│\├\─\└\s​‌‍﻿]+$", stripped):
                continue
            # 短行（< 30 字符且不以标点结尾）→ 拼接下一行（修复 PDF 行内断句）
            if buf and len(stripped) < 30 and not re.search(r"[。！？.!?]$", stripped):
                buf += stripped
            elif buf:
                merged_lines.append(buf)
                buf = stripped
            else:
                buf = stripped
        if buf:
            merged_lines.append(buf)

        # 4. 按句末标点切分
        for line in merged_lines:
            parts = re.split(r"(?<=[。！？.!?])\s*", line)
            for p in parts:
                p = p.strip()
                # 再次过滤切分后的短碎片
                if p and len(p) >= 4:
                    sentences.append(p)

    return sentences


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
    strip_whitespace: bool = True,
) -> list[str]:
    """按句子边界分片，返回文本列表。

    separators 参数保留兼容，SentenceSplitter 不依赖分隔符列表，
    而是通过 nltk/punkt 做句子边界检测。
    """
    if strip_whitespace:
        text = text.strip()
    if not text:
        return []

    splitter = SentenceSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents([LlamaDocument(text=text)])
    return [node.get_content() for node in nodes]


def split_text_semantic(
    text: str,
    model,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
    min_chunk_size: int = 40,
    strip_whitespace: bool = True,
) -> list[str]:
    """按语义边界分片 —— 在 embedding 相似度低谷处切割，不破坏主题连贯性。

    model: SentenceTransformer 实例（如 BGE-M3），内部自动适配为 LlamaIndex 接口。
    适合长文档、跨段落主题切换频繁的内容。
    """
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    if strip_whitespace:
        text = text.strip()
    if not text:
        return []

    embed_model = _STEmbedAdapter(model)

    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        sentence_splitter=_split_sentences,
    )

    nodes = splitter.get_nodes_from_documents([LlamaDocument(text=text)])
    chunks = [node.get_content() for node in nodes]

    # 过短的 chunk 合并到相邻 chunk，避免碎片化
    chunks = _merge_short_chunks(chunks, min_chunk_size)

    # 如果 chunk 过长，用 SentenceSplitter 二次切割
    max_size = chunk_size or settings.chunk_size
    if max_size > 0:
        trimmed: list[str] = []
        fallback = SentenceSplitter(chunk_size=max_size, chunk_overlap=chunk_overlap or settings.chunk_overlap)
        for ch in chunks:
            if len(ch) <= max_size:
                trimmed.append(ch)
            else:
                sub_nodes = fallback.get_nodes_from_documents([LlamaDocument(text=ch)])
                trimmed.extend([n.get_content() for n in sub_nodes])
        return trimmed

    return chunks


def _merge_short_chunks(chunks: list[str], min_size: int) -> list[str]:
    """将过短 chunk 合并到相邻 chunk（优先向前合并）。"""
    if not chunks or min_size <= 0:
        return chunks

    merged: list[str] = []
    for ch in chunks:
        if merged and len(ch) < min_size:
            merged[-1] = merged[-1] + ch  # 向前合并
        else:
            merged.append(ch)

    # 如果第一个 chunk 过短，向后合并
    if len(merged) > 1 and len(merged[0]) < min_size:
        merged[1] = merged[0] + merged[1]
        merged = merged[1:]

    return merged


def make_entities(
    chunks: list[str],
    kb_id: str,
    file_path: str,
    file_name: str,
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[int, float]] | None = None,
) -> list[dict]:
    """已分片的文本列表 → Milvus 实体列表。"""
    now_ms = int(time.time() * 1000)

    entities = []
    for i, chunk_text in enumerate(chunks):
        entity = {
            "kb_id": kb_id,
            "file_path": file_path,
            "file_name": file_name,
            "chunk_index": i,
            "chunk_text": chunk_text,
            "dense_vector": dense_vectors[i] if i < len(dense_vectors) else [],
            "sparse_vector": sparse_vectors[i] if sparse_vectors and i < len(sparse_vectors) else {},
            "create_time": now_ms,
        }
        entities.append(entity)
    return entities
