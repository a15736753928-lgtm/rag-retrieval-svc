"""PostgreSQL 连接池管理与初始化。"""

from __future__ import annotations

import logging
import os

import psycopg2
from psycopg2 import pool

from config import settings

logger = logging.getLogger(__name__)

_pool: pool.ThreadedConnectionPool | None = None
_init_done: bool = False

_MIN_CONN = 1
_MAX_CONN = 5


def get_db_conn() -> psycopg2.extensions.connection:
    """从连接池获取 PostgreSQL 连接，首次调用自动建表。"""
    global _pool, _init_done
    if _pool is not None:
        if not _init_done:
            _init_tables()
        return _pool.getconn()

    _pool = pool.ThreadedConnectionPool(
        _MIN_CONN, _MAX_CONN,
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_db,
        user=settings.pg_user,
        password=settings.pg_password,
    )
    logger.info("PostgreSQL 连接池已创建: %s:%s/%s", settings.pg_host, settings.pg_port, settings.pg_db)
    _init_tables()
    return _pool.getconn()


def put_db_conn(conn: psycopg2.extensions.connection):
    """归还连接到池。"""
    if _pool is not None:
        _pool.putconn(conn)


def _init_tables():
    """幂等建表。"""
    global _init_done
    if _init_done:
        return

    conn = _pool.getconn()
    conn.autocommit = True
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id          TEXT PRIMARY KEY,
                name        VARCHAR(255) NOT NULL UNIQUE,
                description TEXT DEFAULT '',
                doc_count   INT DEFAULT 0,
                chunk_count INT DEFAULT 0,
                created_at  BIGINT NOT NULL,
                updated_at  BIGINT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                kb_id       TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                file_name   VARCHAR(500) NOT NULL,
                file_size   INT DEFAULT 0,
                file_type   VARCHAR(20) DEFAULT '',
                chunk_count INT DEFAULT 0,
                object_key  TEXT DEFAULT '',
                file_hash   VARCHAR(64) DEFAULT '',
                status      VARCHAR(20) DEFAULT 'pending',
                uploaded_at BIGINT DEFAULT 0,
                created_at  BIGINT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_kb_id ON documents(kb_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                kb_id       TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                chunk_text  TEXT NOT NULL,
                milvus_pk   BIGINT DEFAULT 0,
                created_at  BIGINT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS upload_tasks (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                kb_id       TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                file_name   VARCHAR(500) DEFAULT '',
                file_size   INT DEFAULT 0,
                status      VARCHAR(20) DEFAULT 'waiting',
                progress    INT DEFAULT 0,
                message     VARCHAR(500) DEFAULT '',
                created_at  BIGINT NOT NULL,
                updated_at  BIGINT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_upload_tasks_doc_id ON upload_tasks(doc_id)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key         VARCHAR(100) PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  BIGINT NOT NULL
            )
        """)
        # ── 存量兼容：为已存在的表追加 file_hash 列 + 唯一索引 ──
        # ALTER TABLE ADD COLUMN 先执行，确保列存在后再建索引
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64) DEFAULT ''")
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_kb_file_hash "
            "ON documents (kb_id, file_hash) WHERE file_hash != ''"
        )

        _init_done = True
        logger.info("PostgreSQL 表初始化完成")
    finally:
        _pool.putconn(conn)


def init_db():
    """显式初始化入口（兼容启动时调用）。幂等。"""
    get_db_conn()  # 首次调用自动建表
    # get_db_conn 的 _init_tables 已归还连接，这里不需要额外处理
