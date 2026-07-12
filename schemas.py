"""Pydantic 请求/响应模型 —— 与 API 文档 v2.0 严格一致。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
#  统一响应包裹
# ═══════════════════════════════════════════════════════════════════════

class Resp(BaseModel):
    code: int = 200
    message: str = "success"
    data: Any = None


# ═══════════════════════════════════════════════════════════════════════
#  健康检查
# ═══════════════════════════════════════════════════════════════════════

class MilvusStatus(BaseModel):
    status: str = "ok"
    message: str = ""


class ModelsStatus(BaseModel):
    bge_m3: str = "ok"
    bge_reranker: str = "ok"


class HealthCheck(BaseModel):
    status: str = "ok"
    milvus: MilvusStatus = Field(default_factory=MilvusStatus)
    models: ModelsStatus = Field(default_factory=ModelsStatus)
    timestamp: int = 0


# ═══════════════════════════════════════════════════════════════════════
#  认证
# ═══════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    access_key: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_in: int = 86400


class AuthStatusResponse(BaseModel):
    authenticated: bool = True


# ═══════════════════════════════════════════════════════════════════════
#  数据概览
# ═══════════════════════════════════════════════════════════════════════

class DashboardStats(BaseModel):
    kb_count: int = 0
    document_count: int = 0
    chunk_count: int = 0


class RecentDocItem(BaseModel):
    id: str = ""
    file_name: str = ""
    kb_id: str = ""
    kb_name: str = ""
    chunk_count: int = 0
    file_size: int = 0
    ingested_at: str = ""


class DistributionItem(BaseModel):
    kb_id: str = ""
    kb_name: str = ""
    document_count: int = 0


# ═══════════════════════════════════════════════════════════════════════
#  知识库管理
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class KnowledgeBaseItem(BaseModel):
    id: str = ""
    name: str = ""
    description: str = ""
    document_count: int = 0
    chunk_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class PaginatedKB(BaseModel):
    items: list[KnowledgeBaseItem] = []
    total: int = 0
    page: int = 1
    page_size: int = 10


# ═══════════════════════════════════════════════════════════════════════
#  文档管理
# ═══════════════════════════════════════════════════════════════════════

class DocumentItem(BaseModel):
    id: str = ""
    file_name: str = ""
    kb_id: str = ""
    kb_name: str = ""
    file_size: int = 0
    file_type: str = ""
    chunk_count: int = 0
    file_hash: str = ""
    status: str = "pending"
    ingested_at: str = ""


class PaginatedDocs(BaseModel):
    items: list[DocumentItem] = []
    total: int = 0
    page: int = 1
    page_size: int = 6


class DocumentDetail(BaseModel):
    id: str = ""
    file_name: str = ""
    kb_id: str = ""
    kb_name: str = ""
    file_size: int = 0
    file_type: str = ""
    chunk_count: int = 0
    file_hash: str = ""
    status: str = "pending"
    ingested_at: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  文档判重
# ═══════════════════════════════════════════════════════════════════════

class DuplicateCheckRequest(BaseModel):
    kb_id: str = Field(..., min_length=1)
    file_hash: str = Field(..., min_length=64, max_length=64)


class DuplicateCheckResponse(BaseModel):
    duplicate: bool = False
    existing: DocumentItem | None = None


class DuplicateConflictResponse(BaseModel):
    existing_doc_id: str = ""
    existing_file_name: str = ""
    existing_ingested_at: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  文档分片
# ═══════════════════════════════════════════════════════════════════════

class ChunkItem(BaseModel):
    id: str = ""
    index: int = 0
    content: str = ""
    char_count: int = 0


class PaginatedChunks(BaseModel):
    items: list[ChunkItem] = []
    total: int = 0
    page: int = 1
    page_size: int = 100


# ═══════════════════════════════════════════════════════════════════════
#  上传
# ═══════════════════════════════════════════════════════════════════════

class IngestUploadResponse(BaseModel):
    task_id: str = ""
    doc_id: str = ""
    status: str = "waiting"


class OriginalFileResponse(BaseModel):
    url: str = ""
    expires_in: int = 3600


class UploadTaskStatus(BaseModel):
    task_id: str = ""
    doc_id: str = ""
    kb_id: str = ""
    file_name: str = ""
    file_size: int = 0
    status: str = "waiting"
    progress: int = 0
    message: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  检索
# ═══════════════════════════════════════════════════════════════════════

class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    kb_ids: list[str] | None = Field(None)
    top_k: int = Field(..., ge=0, le=100000)
    min_similarity: float = Field(..., ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    id: str = ""
    content: str = ""
    file_name: str = ""
    kb_id: str = ""
    kb_name: str = ""
    chunk_index: int = 0
    similarity: float = 0.0
    highlights: list[str] = []


# ═══════════════════════════════════════════════════════════════════════
#  分片原文高亮
# ═══════════════════════════════════════════════════════════════════════

class HighlightSpan(BaseModel):
    start: int
    end: int


class ChunkContentResponse(BaseModel):
    id: str = ""
    kb_id: str = ""
    file_name: str = ""
    chunk_index: int = 0
    content: str = ""
    highlights: list[HighlightSpan] = []


# ═══════════════════════════════════════════════════════════════════════
#  系统设置
# ═══════════════════════════════════════════════════════════════════════

class VectorModelInfo(BaseModel):
    model_name: str = ""
    model_version: str = ""
    vector_dim: int = 0
    status: str = "running"


class VectorDbInfo(BaseModel):
    db_type: str = ""
    connection_address: str = ""
    status: str = "connected"
    vector_count: int = 0
    storage_size: str = ""


class ChunkConfigInfo(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 100
    separator: str = "\n\n"


class SettingsData(BaseModel):
    vector_model: VectorModelInfo = Field(default_factory=VectorModelInfo)
    vector_db: VectorDbInfo = Field(default_factory=VectorDbInfo)
    chunk_config: ChunkConfigInfo = Field(default_factory=ChunkConfigInfo)
    access_key_masked: str = ""


class ChunkConfigUpdate(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = Field(default=500, ge=200, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)
    separator: str = "\n\n"


class RegenerateKeyResponse(BaseModel):
    access_key: str = ""


class VectorDbRefreshResponse(BaseModel):
    vector_db: VectorDbInfo = Field(default_factory=VectorDbInfo)
