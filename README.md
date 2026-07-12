# RAG 本地知识检索系统 - 后端服务

## 技术栈

- **Python 3.12** / **FastAPI** / **Uvicorn**
- **Milvus Lite** 嵌入式向量存储（零外部依赖）
- **BGE-M3** 稠密 + 稀疏双向量编码
- **BGE-Reranker-V2-M3** 精排序
- **pdf-oxide** + **office-oxide** Rust 原生文档解析（极速）
- **RapidOCR** (PP-OCR v4) 扫描件 / 图片 OCR 兜底
- **Langchain-Text-Splitters** 文本分片

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 拷贝环境变量配置
cp .env.example .env

# 3. 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

# 4. 打开 Swagger 文档
# http://localhost:8000/docs
```

## 项目结构

```
rag-retrieval-svc/
├── main.py                     # 服务入口 & 路由
├── config.py                   # 全局配置
├── schemas.py                  # Pydantic 结构体
├── core/                       # 核心资源（Milvus Lite、模型）
├── milvus_client/              # 向量 CRUD 操作层
├── embedding/                  # BGE-M3 编码
├── rerank/                     # BGE-Reranker 重排
├── document_ingest/            # 解析（三层管线：fast→render→OCR）
├── service/                    # 业务逻辑层
├── middleware/                  # 中间件
├── utils/                     # 工具
├── storage/milvus_data/       # Milvus Lite 数据（嵌入式）
└── storage/cache/             # 模型缓存
```
