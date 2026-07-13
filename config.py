"""全局常量配置 —— 所有参数通过 .env 或环境变量覆盖。"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── 服务配置 ─────────────────────────────────────────────────────
    service_name: str = "rag-retrieval-svc"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"

    # ── Milvus（嵌入式 Lite）──────────────────────────────────────
    milvus_data_dir: str = "storage/milvus_data"
    milvus_collection_name: str = "rag_collection"

    # ── 模型配置 ───────────────────────────────────────────────────────
    bge_model_name: str = "BAAI/bge-m3"
    bge_onnx_model_name: str = "gpahal/bge-m3-onnx-int8"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    embedding_cache_dir: str = "storage/cache"

    # ── 向量配置 ───────────────────────────────────────────────────────
    dense_vector_dim: int = 1024
    metric_type: str = "COSINE"

    # ── 检索配置 ───────────────────────────────────────────────────────
    default_top_k: int = 10
    search_ef: int = 64
    rrf_k: int = 60
    max_search_recall: int = 100000

    # ── 入库配置 ───────────────────────────────────────────────────────
    ingest_batch_size: int = 64
    chunk_size: int = 500
    chunk_overlap: int = 100
    chunk_separators: list[str] = ["\n\n", "\n", "。", ".", " ", ""]

    # ── 文件限制 ───────────────────────────────────────────────────────
    max_file_size: int = 20 * 1024 * 1024  # 20MB
    allowed_extents: str = "txt,md,docx,pdf"

    # ── PostgreSQL 元数据库 ───────────────────────────────────────────────
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "rag_retrieval"
    pg_user: str = "postgres"
    pg_password: str = "postgres"

    # ── MinIO 对象存储 ────────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    # ── JWT / 认证 ─────────────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production-use-env-var"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    default_access_key: str = "123456"

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


settings = Settings()
