"""完整入库链路：异步任务模式 —— MinIO 存原件 → 解析 → 分片 → 双向量编码 → Milvus 写入 → PG 写 chunks。"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import threading
import time
import uuid
from pathlib import Path

from config import settings
from db import models as dao
from document_ingest.parser import is_supported, parse_bytes
from core.model_loader import get_bge_m3
from document_ingest.splitter import make_entities, split_text_semantic
from embedding.bge_encoder import encode_dense, encode_sparse
from milvus_client import vector_crud as vc
from storage import minio_client

logger = logging.getLogger(__name__)


def _gen_task_id() -> str:
    return "task_" + uuid.uuid4().hex[:8]


def _get_chunk_config() -> dict:
    """从系统设置读取分片配置。"""
    from db.models import get_setting
    raw = get_setting("chunk_config")
    if raw:
        return json.loads(raw)
    return {"strategy": "recursive", "chunk_size": 500, "chunk_overlap": 100, "separator": "\n\n"}


def create_ingest_task(
    file_bytes: bytes,
    filename: str,
    kb_id: str,
    file_hash: str = "",
) -> dict:
    """创建入库任务：写入元数据，返回 task + doc 信息，启动后台线程处理。"""
    file_type = Path(filename).suffix.lower().lstrip(".")
    file_size = len(file_bytes)

    doc_id = "doc_" + uuid.uuid4().hex[:8]

    # 创建文档记录（含 file_hash 用于判重）
    doc = dao.create_document(
        doc_id=doc_id,
        kb_id=kb_id,
        file_name=filename,
        file_size=file_size,
        file_type=file_type,
        file_hash=file_hash,
    )

    # 创建上传任务
    task_id = _gen_task_id()
    task = dao.create_upload_task(
        task_id=task_id,
        doc_id=doc_id,
        kb_id=kb_id,
        file_name=filename,
        file_size=file_size,
    )

    # 更新知识库计数
    dao.incr_kb_doc_count(kb_id, 1)

    # 后台线程执行入库
    thread = threading.Thread(
        target=_run_ingest,
        args=(task_id, doc_id, kb_id, file_bytes, filename),
        daemon=True,
    )
    thread.start()

    logger.info("入库任务已创建: task_id=%s doc_id=%s file=%s", task_id, doc_id, filename)
    return {"task_id": task_id, "doc_id": doc_id, "status": "waiting"}


def _run_ingest(
    task_id: str,
    doc_id: str,
    kb_id: str,
    file_bytes: bytes,
    filename: str,
):
    """后台线程：执行完整入库链路。分片配置从系统设置读取。"""
    start = time.perf_counter()
    chunk_cfg = _get_chunk_config()
    chunk_size = chunk_cfg.get("chunk_size", 500)
    chunk_overlap = chunk_cfg.get("chunk_overlap", 100)

    try:
        # ── MinIO：存储原始文件 ────────────────────────────────
        dao.update_upload_task(task_id, progress=5, message="正在上传原始文件...")
        object_key = f"{kb_id}/{doc_id}/{filename}"
        dao.update_document(doc_id, object_key=object_key)
        try:
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            minio_client.upload_file(file_bytes, object_key, content_type)
        except Exception as e:
            logger.warning("MinIO 上传失败（不影响入库，MinIO恢复后重新上传即可）: %s", e)

        # ── 解析 ─────────────────────────────────────────────
        dao.update_upload_task(task_id, status="parsing", progress=10, message="正在解析文件...")
        if not is_supported(filename):
            _fail(task_id, doc_id, kb_id, "不支持的文件类型")
            return
        raw_text = parse_bytes(file_bytes, filename)
        if not raw_text.strip():
            _fail(task_id, doc_id, kb_id, "文件内容为空")
            return

        # ── 分片 ─────────────────────────────────────────────
        dao.update_upload_task(task_id, status="parsing", progress=30, message="正在文本分片...")
        chunks = split_text_semantic(raw_text, get_bge_m3(), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            _fail(task_id, doc_id, kb_id, "分片结果为空")
            return

        total_chunks = len(chunks)
        dao.update_upload_task(task_id, progress=40, message=f"分片完成，共 {total_chunks} 段")

        # ── 编码 ─────────────────────────────────────────────
        dao.update_upload_task(task_id, status="encoding", progress=50, message="正在向量编码...")
        try:
            dense_vecs = encode_dense(chunks)
            sparse_vecs = encode_sparse(chunks)
        except Exception as e:
            _fail(task_id, doc_id, kb_id, f"编码失败: {e}")
            return

        dao.update_upload_task(task_id, progress=75, message="编码完成，正在写入向量库...")

        # ── 入库 ─────────────────────────────────────────────
        dao.update_upload_task(task_id, status="indexing", progress=80)
        entities = make_entities(
            kb_id=kb_id,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
        )
        try:
            ids = vc.insert(entities)
        except Exception as e:
            _fail(task_id, doc_id, kb_id, f"向量入库失败: {e}")
            return

        # ── PG：写入 chunks 表 ─────────────────────────────────
        try:
            dao.insert_chunks(doc_id, kb_id, chunks, ids)
        except Exception as e:
            logger.warning("PG chunks 写入失败（不影响检索）: %s", e)

        # ── 完成 ────────────────────────────────────────────
        took_ms = round((time.perf_counter() - start) * 1000, 2)
        now_ms = int(time.time() * 1000)
        dao.update_document(doc_id, status="completed", chunk_count=len(ids), uploaded_at=now_ms)
        dao.update_upload_task(task_id, status="completed", progress=100, message=f"入库完成，{len(ids)}/{total_chunks} 分片，耗时 {took_ms}ms")
        dao.incr_kb_chunk_count(kb_id, len(ids))
        logger.info("入库完成: task=%s doc=%s file=%s chunks=%d took=%sms", task_id, doc_id, filename, len(ids), took_ms)

    except Exception as e:
        logger.exception("入库任务异常: task_id=%s", task_id)
        _fail(task_id, doc_id, kb_id, f"系统异常: {e}")


def _fail(task_id: str, doc_id: str, kb_id: str, message: str):
    dao.update_document(doc_id, status="failed")
    dao.update_upload_task(task_id, status="failed", message=message)
    dao.incr_kb_doc_count(kb_id, -1)
    logger.error("入库失败: task=%s doc=%s msg=%s", task_id, doc_id, message)
