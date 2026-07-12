"""Milvus Lite 嵌入式实例 —— 无外部服务依赖。"""

from __future__ import annotations

import logging
import threading

import milvus_lite
from milvus_lite import CollectionSchema, DataType, FieldSchema

from config import settings

logger = logging.getLogger(__name__)

_lite: milvus_lite.MilvusLite | None = None
_lite_lock = threading.Lock()
_collection_ready = False


def get_lite() -> milvus_lite.MilvusLite:
    """全局唯一 milvus_lite 实例。"""
    global _lite
    if _lite is not None:
        return _lite
    with _lite_lock:
        if _lite is not None:
            return _lite
        _lite = milvus_lite.MilvusLite(settings.milvus_data_dir)
        logger.info("Milvus Lite 就绪: %s", settings.milvus_data_dir)
        return _lite


def get_collection() -> milvus_lite.Collection:
    """获取已加载的 collection。"""
    ensure_collection()
    return get_lite().get_collection(settings.milvus_collection_name)


def ensure_collection():
    """确保 collection 存在且已加载（幂等）。"""
    global _collection_ready
    if _collection_ready:
        return

    lite = get_lite()
    name = settings.milvus_collection_name

    if lite.has_collection(name):
        coll = lite.get_collection(name)
        if coll.load_state != "Loaded":
            coll.load()
        _collection_ready = True
        return

    # 创建 schema —— 只存向量+ID，文本在 PG chunks 表，文件在 MinIO
    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("kb_id", DataType.VARCHAR, max_length=128),
            FieldSchema("dense_vector", DataType.FLOAT_VECTOR, dim=settings.dense_vector_dim),
            FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
        ],
        enable_dynamic_field=False,
    )

    lite.create_collection(name, schema)
    coll = lite.get_collection(name)

    # 索引
    try:
        coll.create_index("dense_vector", {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256},
        })
        coll.create_index("sparse_vector", {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2},
        })
        coll.create_index("kb_id", {
            "index_type": "INVERTED",
            "metric_type": "",
        })
        logger.info("索引创建完成")
    except Exception as e:
        logger.warning("索引创建失败（不影响读写）: %s", e)

    coll.load()
    _collection_ready = True
    logger.info("Collection %s 就绪", name)


def drop_collection():
    """删除 collection。"""
    global _collection_ready
    lite = get_lite()
    name = settings.milvus_collection_name
    if lite.has_collection(name):
        lite.drop_collection(name)
        _collection_ready = False
        logger.info("Collection %s 已删除", name)
