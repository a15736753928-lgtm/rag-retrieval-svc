"""Milvus 集合字段定义 —— 与代码强一致。

| 字段名           | 类型                    | 属性          | 描述                      |
|----------------|------------------------|---------------|---------------------------|
| id             | INT64                  | 主键、自动递增    | 分片唯一标识                   |
| kb_id          | VARCHAR(128)           | 索引字段        | 逻辑隔离标识                   |
| dense_vector   | FLOAT_VECTOR(1024)     | HNSW COSINE  | BGE-M3 稠密向量               |
| sparse_vector  | SPARSE_FLOAT_VECTOR    | IP           | BGE-M3 稀疏向量               |

索引：
  - dense_vector:  HNSW, M=16, efConstruction=256, metric=COSINE
  - sparse_vector: SPARSE_INVERTED_INDEX, drop_ratio_build=0.2, metric=IP
  - kb_id:         INVERTED
"""

from __future__ import annotations

PRIMARY_FIELD = "id"
KB_ID_FIELD = "kb_id"
DENSE_VECTOR_FIELD = "dense_vector"
SPARSE_VECTOR_FIELD = "sparse_vector"
