"""Milvus 集合字段定义 —— 与代码强一致。

| 字段名           | 类型                    | 属性          | 描述                      |
|----------------|------------------------|---------------|---------------------------|
| id             | INT64                  | 主键、自动递增    | 分片唯一标识                   |
| kb_id          | VARCHAR(128)           | 索引字段        | 逻辑隔离标识 project_domain     |
| file_path      | VARCHAR(1024)          | 索引字段        | 原始文件路径                   |
| file_name      | VARCHAR(256)           | 不索引         | 原始文件名称                   |
| chunk_index    | INT32                  | 不索引         | 分片序号                     |
| chunk_text     | VARCHAR(65535)         | 不索引         | 分片原文内容                   |
| dense_vector   | FLOAT_VECTOR(1024)     | HNSW COSINE  | BGE-M3 稠密向量               |
| sparse_vector  | SPARSE_FLOAT_VECTOR    | IP           | BGE-M3 稀疏向量               |
| create_time    | INT64                  | 不索引         | 创建时间戳(毫秒)                |

索引：
  - dense_vector:  HNSW, M=16, efConstruction=256, metric=COSINE
  - sparse_vector: SPARSE_INVERTED_INDEX, drop_ratio_build=0.2, metric=IP
  - kb_id:         INVERTED
"""

from __future__ import annotations

# 这些常量供程序内引用，实际 schema 在 core/milvus_manager.py 中构建

PRIMARY_FIELD = "id"
KB_ID_FIELD = "kb_id"
FILE_PATH_FIELD = "file_path"
FILE_NAME_FIELD = "file_name"
CHUNK_INDEX_FIELD = "chunk_index"
CHUNK_TEXT_FIELD = "chunk_text"
DENSE_VECTOR_FIELD = "dense_vector"
SPARSE_VECTOR_FIELD = "sparse_vector"
CREATE_TIME_FIELD = "create_time"
