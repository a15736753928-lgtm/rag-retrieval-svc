"""RAG 检索微服务 —— FastAPI 入口，路由与 API 文档 v2.0 严格一致。"""

from __future__ import annotations

import hashlib
import logging
import sys
import time

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from middleware import cors_middleware, log_middleware
from middleware.auth_middleware import require_auth
from milvus_client import collection_crud as coll_crud
from schemas import (
    AuthStatusResponse,
    ChunkConfigUpdate,
    ChunkContentResponse,
    ChunkItem,
    DashboardStats,
    DistributionItem,
    DocumentDetail,
    DocumentItem,
    DuplicateCheckRequest,
    DuplicateCheckResponse,
    DuplicateConflictResponse,
    HealthCheck,
    IngestUploadResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseItem,
    KnowledgeBaseUpdate,
    LoginRequest,
    MilvusStatus,
    ModelsStatus,
    PaginatedChunks,
    PaginatedDocs,
    PaginatedKB,
    RecentDocItem,
    RegenerateKeyResponse,
    Resp,
    SearchQueryRequest,
    SettingsData,
    UploadTaskStatus,
    VectorDbInfo,
    VectorDbRefreshResponse,
)
from service import auth_service as auth_svc
from service import dashboard_service as dash_svc
from service import doc_service
from service import ingest_service as ingest_svc
from service import kb_service
from service import search_service as search_svc
from service import settings_service as settings_svc
from utils.file_validator import check_extension, check_size

# ── 日志 ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── FastAPI 应用 ────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

cors_middleware.register(app)
log_middleware.register(app)


# ═══════════════════════════════════════════════════════════════════════
#  全局异常
# ═══════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("未捕获异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": f"后端服务异常: {exc}", "data": None},
    )


# ═══════════════════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    logger.info("%s v%s 启动中 ...", settings.service_name, settings.service_version)
    from db.database import init_db
    init_db()
    auth_svc._ensure_defaults()
    coll_crud.init()

    # GPU 强制校验 —— 不通过则退出，拒绝 CPU 降级
    from core.model_loader import validate_gpu
    validate_gpu()

    # DeepSeek API 连通性校验
    from core.llm_client import generate
    try:
        test_resp = generate("ping", system="只回复 pong 一个词，不要其他内容", max_tokens=10)
        logger.info("DeepSeek API 连通性校验通过 → %s", test_resp.strip())
    except Exception as e:
        logger.warning("DeepSeek API 连通性校验失败（不影响启动）: %s", e)

    # 同步加载模型 —— 确保完成后才对外服务
    from core.graph_store import init_graph
    init_graph()
    from core.model_loader import get_bge_m3, get_reranker
    get_bge_m3()
    logger.info("BGE-M3 加载完成")
    get_reranker()
    logger.info("BGE-Reranker 加载完成")

    logger.info("========================================")
    logger.info("  全部配置加载完毕，服务就绪")
    logger.info("  http://%s:%s", settings.host, settings.port)
    logger.info("  默认密钥: %s", settings.default_access_key)
    logger.info("========================================")


# ═══════════════════════════════════════════════════════════════════════
#  3.3 健康检查（不鉴权）
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/health")
async def health():
    milvus_ok = "ok"
    milvus_msg = ""
    try:
        coll_crud.init()
    except Exception as e:
        milvus_ok = "error"
        milvus_msg = str(e)

    models_ok = {"bge_m3": "ok", "bge_reranker": "ok"}
    from core.model_loader import _bge_m3
    if _bge_m3 is None:
        models_ok["bge_m3"] = "loading"

    return Resp(data=HealthCheck(
        status="ok" if milvus_ok == "ok" else "error",
        milvus=MilvusStatus(status=milvus_ok, message=milvus_msg),
        models=ModelsStatus(**models_ok),
        timestamp=int(time.time() * 1000),
    ))


# ═══════════════════════════════════════════════════════════════════════
#  3.1 登录（不鉴权）
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/auth/login")
async def auth_login(req: LoginRequest):
    token = auth_svc.login(req.access_key)
    if token is None:
        return Resp(code=401, message="访问密钥无效", data=None)
    expires_in = settings.jwt_expire_hours * 3600
    return Resp(data={"token": token, "token_type": "bearer", "expires_in": expires_in})


# ═══════════════════════════════════════════════════════════════════════
#  3.2 校验登录态
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/auth/status")
async def auth_status(_auth: bool = Depends(require_auth)):
    return Resp(data=AuthStatusResponse(authenticated=True))


# ═══════════════════════════════════════════════════════════════════════
#  3.4 核心统计
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/dashboard/stats")
async def dashboard_stats(_auth: bool = Depends(require_auth)):
    s = dash_svc.get_stats()
    return Resp(data=DashboardStats(**s))


# ═══════════════════════════════════════════════════════════════════════
#  3.5 最近入库
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/dashboard/recent")
async def dashboard_recent(
    limit: int = Query(5, ge=1, le=20),
    _auth: bool = Depends(require_auth),
):
    items = dash_svc.get_recent(limit)
    return Resp(data=[RecentDocItem(**it) for it in items])


# ═══════════════════════════════════════════════════════════════════════
#  3.6 知识库文档分布
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/dashboard/distribution")
async def dashboard_distribution(_auth: bool = Depends(require_auth)):
    items = dash_svc.get_distribution()
    return Resp(data=[DistributionItem(**it) for it in items])


# ═══════════════════════════════════════════════════════════════════════
#  3.7 知识库列表
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/knowledge-bases")
async def list_knowledge_bases(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str | None = Query(None),
    _auth: bool = Depends(require_auth),
):
    items, total = kb_service.list_kbs(page, page_size, search)
    return Resp(data=PaginatedKB(
        items=[KnowledgeBaseItem(**it) for it in items],
        total=total,
        page=page,
        page_size=page_size,
    ))


# ═══════════════════════════════════════════════════════════════════════
#  3.8 新建知识库
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/knowledge-bases")
async def create_knowledge_base(
    req: KnowledgeBaseCreate,
    _auth: bool = Depends(require_auth),
):
    try:
        kb = kb_service.create_kb(req.name, req.description)
        return Resp(data=KnowledgeBaseItem(**kb))
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            return Resp(code=422, message="知识库名称已存在", data=None)
        raise


# ═══════════════════════════════════════════════════════════════════════
#  3.9 知识库详情
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/knowledge-bases/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    _auth: bool = Depends(require_auth),
):
    kb = kb_service.get_kb(kb_id)
    if not kb:
        return Resp(code=404, message="知识库不存在", data=None)
    return Resp(data=KnowledgeBaseItem(**kb))


# ═══════════════════════════════════════════════════════════════════════
#  3.10 编辑知识库
# ═══════════════════════════════════════════════════════════════════════

@app.put("/api/v1/knowledge-bases/{kb_id}")
async def update_knowledge_base(
    kb_id: str,
    req: KnowledgeBaseUpdate,
    _auth: bool = Depends(require_auth),
):
    kb = kb_service.update_kb(kb_id, req.name, req.description)
    if not kb:
        return Resp(code=404, message="知识库不存在", data=None)
    return Resp(data=KnowledgeBaseItem(**kb))


# ═══════════════════════════════════════════════════════════════════════
#  3.11 删除知识库
# ═══════════════════════════════════════════════════════════════════════

@app.delete("/api/v1/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    _auth: bool = Depends(require_auth),
):
    ok = kb_service.delete_kb(kb_id)
    if not ok:
        return Resp(code=404, message="知识库不存在", data=None)
    return Resp(data=None)


# ═══════════════════════════════════════════════════════════════════════
#  3.12 知识库下文档列表
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/knowledge-bases/{kb_id}/documents")
async def list_kb_documents(
    kb_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    _auth: bool = Depends(require_auth),
):
    kb = kb_service.get_kb(kb_id)
    if not kb:
        return Resp(code=404, message="知识库不存在", data=None)
    items, total = doc_service.list_docs(kb_id=kb_id, page=page, page_size=page_size)
    return Resp(data=PaginatedDocs(
        items=[DocumentItem(**it) for it in items],
        total=total,
        page=page,
        page_size=page_size,
    ))


# ═══════════════════════════════════════════════════════════════════════
#  3.13 全部文档列表
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/documents")
async def list_documents(
    kb_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(6, ge=1, le=100),
    _auth: bool = Depends(require_auth),
):
    items, total = doc_service.list_docs(kb_id=kb_id, page=page, page_size=page_size)
    return Resp(data=PaginatedDocs(
        items=[DocumentItem(**it) for it in items],
        total=total,
        page=page,
        page_size=page_size,
    ))


# ═══════════════════════════════════════════════════════════════════════
#  3.14 文档详情
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/documents/{doc_id}")
async def get_document(
    doc_id: str,
    _auth: bool = Depends(require_auth),
):
    doc = doc_service.get_doc(doc_id)
    if not doc:
        return Resp(code=404, message="文档不存在", data=None)
    return Resp(data=DocumentDetail(**doc))


# ═══════════════════════════════════════════════════════════════════════
#  3.14.1 获取原始文件
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/documents/{doc_id}/file")
async def get_document_file(
    doc_id: str,
    _auth: bool = Depends(require_auth),
):
    from io import BytesIO
    from storage.minio_client import get_file

    doc = doc_service.get_doc(doc_id)
    if not doc:
        return JSONResponse({"code": 404, "message": "文档不存在", "data": None}, status_code=404)

    object_key = doc.get("object_key", "")
    if not object_key:
        return JSONResponse({"code": 404, "message": "原始文件不存在", "data": None}, status_code=404)

    try:
        data, content_type = get_file(object_key)
    except Exception:
        return JSONResponse({"code": 404, "message": "文件读取失败", "data": None}, status_code=404)

    from urllib.parse import quote

    filename = doc.get("file_name", "file")
    # RFC 5987: 支持中文等非 ASCII 字符的 filename 编码
    encoded = quote(filename, safe="")
    return StreamingResponse(
        BytesIO(data),
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{encoded}",
            "Content-Type": content_type,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
#  3.15 删除文档
# ═══════════════════════════════════════════════════════════════════════

@app.delete("/api/v1/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    _auth: bool = Depends(require_auth),
):
    doc = doc_service.delete_doc(doc_id)
    if not doc:
        return Resp(code=404, message="文档不存在", data=None)
    return Resp(data=None)


# ═══════════════════════════════════════════════════════════════════════
#  3.16 文档分片预览
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    _auth: bool = Depends(require_auth),
):
    doc = doc_service.get_doc(doc_id)
    if not doc:
        return Resp(code=404, message="文档不存在", data=None)
    items, total = doc_service.get_doc_chunks(doc_id, page, page_size)
    return Resp(data=PaginatedChunks(
        items=[ChunkItem(**it) for it in items],
        total=total,
        page=page,
        page_size=page_size,
    ))


# ═══════════════════════════════════════════════════════════════════════
#  3.17 上传文档入库
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/documents/upload")
async def documents_upload(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    file_hash: str = Form(""),
    _auth: bool = Depends(require_auth),
):
    from db.models import check_duplicate
    from utils.text_processor import format_time

    # 检查知识库
    kb = kb_service.get_kb(kb_id)
    if not kb:
        return Resp(code=404, message="知识库不存在", data=None)

    # 文件校验
    if not check_extension(file.filename or ""):
        return Resp(code=415, message=f"不支持的文件类型: {file.filename}", data=None)
    content = await file.read()
    if not check_size(len(content)):
        return Resp(code=413, message=f"文件过大 ({len(content)/1024/1024:.1f}MB > {settings.max_file_size/1024/1024:.0f}MB)", data=None)

    # 后端计算 SHA-256 作为权威哈希值（不信任前端）
    file_hash_computed = hashlib.sha256(content).hexdigest()

    # 判重兜底（应对并发上传 or 前端绕过预校验）
    dup = check_duplicate(kb_id, file_hash_computed)
    if dup:
        return JSONResponse(
            status_code=409,
            content={
                "code": 409,
                "message": "该知识库下已存在相同文件",
                "data": {
                    "existing_doc_id": dup["id"],
                    "existing_file_name": dup["file_name"],
                    "existing_ingested_at": format_time(dup.get("uploaded_at", 0)),
                },
            },
        )

    result = ingest_svc.create_ingest_task(
        file_bytes=content,
        filename=file.filename or "unknown",
        kb_id=kb_id,
        file_hash=file_hash_computed,
    )
    return Resp(data=IngestUploadResponse(**result))


# ═══════════════════════════════════════════════════════════════════════
#  3.17.1 文件判重预校验
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/documents/check-duplicate")
async def documents_check_duplicate(
    kb_id: str = Query(...),
    file_hash: str = Query(..., min_length=64, max_length=64),
    _auth: bool = Depends(require_auth),
):
    from db.models import check_duplicate

    dup = check_duplicate(kb_id, file_hash.lower())
    if dup:
        # 组装已有的文档信息，方便前端展示
        existing = DocumentItem(
            id=dup["id"],
            file_name=dup["file_name"],
            kb_id=dup["kb_id"],
            file_size=dup.get("file_size", 0),
            file_type=dup.get("file_type", ""),
            chunk_count=dup.get("chunk_count", 0),
            file_hash=dup.get("file_hash", ""),
            status=dup.get("status", "completed"),
            ingested_at=dup.get("uploaded_at", ""),
        )
        return Resp(data=DuplicateCheckResponse(duplicate=True, existing=existing).model_dump())

    return Resp(data=DuplicateCheckResponse(duplicate=False, existing=None).model_dump())


# ═══════════════════════════════════════════════════════════════════════
#  3.18 查询上传进度
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/upload-tasks/{task_id}")
async def get_upload_task(
    task_id: str,
    _auth: bool = Depends(require_auth),
):
    from db.models import get_upload_task as dao_get_task
    task = dao_get_task(task_id)
    if not task:
        return Resp(code=404, message="任务不存在", data=None)
    return Resp(data=UploadTaskStatus(
        task_id=task["id"],
        doc_id=task["doc_id"],
        kb_id=task["kb_id"],
        file_name=task.get("file_name", ""),
        file_size=task.get("file_size", 0),
        status=task.get("status", "waiting"),
        progress=task.get("progress", 0),
        message=task.get("message", ""),
    ))


# ═══════════════════════════════════════════════════════════════════════
#  3.19 检索
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/search/query")
async def search_query(
    req: SearchQueryRequest,
    _auth: bool = Depends(require_auth),
):
    results = await search_svc.search(req)
    return Resp(data=[r.model_dump() for r in results])


# ═══════════════════════════════════════════════════════════════════════
#  3.20 分片原文高亮
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/search/chunk-content")
async def chunk_content(
    chunk_id: int = Query(...),
    query: str = Query(""),
    _auth: bool = Depends(require_auth),
):
    data = await search_svc.get_chunk_content(chunk_id, query)
    if data is None:
        return Resp(code=404, message="分片不存在", data=None)
    return Resp(data=ChunkContentResponse(**data))


# ═══════════════════════════════════════════════════════════════════════
#  3.21 获取全部设置
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/v1/settings")
async def get_settings(_auth: bool = Depends(require_auth)):
    data = settings_svc.get_all_settings()
    return Resp(data=SettingsData(**data))


# ═══════════════════════════════════════════════════════════════════════
#  3.22 保存分片配置
# ═══════════════════════════════════════════════════════════════════════

@app.put("/api/v1/settings/chunk")
async def save_chunk_config(
    req: ChunkConfigUpdate,
    _auth: bool = Depends(require_auth),
):
    config = settings_svc.save_chunk_config(req.model_dump())
    return Resp(data=config)


# ═══════════════════════════════════════════════════════════════════════
#  3.23 恢复默认分片配置
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/settings/chunk/reset")
async def reset_chunk_config(_auth: bool = Depends(require_auth)):
    config = settings_svc.reset_chunk_config()
    return Resp(data=config)


# ═══════════════════════════════════════════════════════════════════════
#  3.24 重新生成访问密钥
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/settings/access-key/regenerate")
async def regenerate_access_key(_auth: bool = Depends(require_auth)):
    key_info = settings_svc.regenerate_access_key()
    return Resp(data=RegenerateKeyResponse(**key_info))


# ═══════════════════════════════════════════════════════════════════════
#  3.25 刷新向量库状态
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/v1/settings/vector-db/refresh")
async def refresh_vector_db(_auth: bool = Depends(require_auth)):
    vdb = settings_svc.refresh_vector_db()
    return Resp(data=VectorDbRefreshResponse(vector_db=VectorDbInfo(**vdb)))


# ═══════════════════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
    )
