"""BGE-M3 ONNX INT8 模型单例 + 全局推理锁，服务常驻内存复用。

使用 gpahal/bge-m3-onnx-int8 —— ONNX Runtime INT8 量化版：
  - 推理速度约为 FP32 的 2×
  - 内存占用约为 FP32 的 1/4
  - 同时输出 dense / sparse / ColBERT 三种向量

所有模型强制 GPU 加载 —— 不提供 CPU 降级，GPU 不可用时直接报错退出。
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import defaultdict

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

infer_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════
#  GPU 校验 —— 启动时调用，失败即退出
# ═══════════════════════════════════════════════════════════════

_gpu_validated = False


def validate_gpu():
    """强制校验 GPU 可用性，不通过则立即退出进程。"""
    global _gpu_validated
    if _gpu_validated:
        return

    # 1. PyTorch CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            logger.critical("GPU 校验失败: PyTorch CUDA 不可用")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("GPU 校验通过 (PyTorch): %s", gpu_name)
    except ImportError:
        logger.critical("GPU 校验失败: PyTorch 未安装")
        sys.exit(1)

    # 2. ONNX Runtime CUDA
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            logger.critical("GPU 校验失败: ONNX Runtime 缺少 CUDAExecutionProvider (当前: %s)", providers)
            sys.exit(1)
        logger.info("GPU 校验通过 (ONNX Runtime): providers=%s", providers)
    except ImportError:
        logger.critical("GPU 校验失败: onnxruntime 未安装")
        sys.exit(1)

    _gpu_validated = True


# ═══════════════════════════════════════════════════════════════
#  BGE-M3 ONNX 封装
# ═══════════════════════════════════════════════════════════════

class _OnnxSession:
    """薄封装 onnxruntime.InferenceSession，暴露 ORTModelForCustomTasks 兼容的 __call__ 接口。"""

    def __init__(self, session):
        self._session = session
        self._valid_inputs = {inp.name for inp in session.get_inputs()}
        self._output_names = [out.name for out in session.get_outputs()]

    def __call__(self, **inputs):
        feed = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        return self._session.run(self._output_names, feed)


class BgeM3Onnx:
    """BGE-M3 ONNX INT8 封装，提供与 SentenceTransformer 兼容的 encode() 接口。"""

    def __init__(self, tokenizer, ort_model):
        self._tokenizer = tokenizer
        self._model = ort_model
        self._unused_tokens: set[int] = {
            tokenizer.cls_token_id or 0,
            tokenizer.eos_token_id or 0,
            tokenizer.pad_token_id or 0,
            tokenizer.unk_token_id or 0,
        }

    # ── 公开接口 ─────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """批量文本 → 稠密向量 (N × 1024)，与 SentenceTransformer.encode() 签名兼容。

        返回已 L2 归一化的 numpy 数组。
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        dense_all: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            outputs = self._model(**inputs)
            dense = outputs[0].astype(np.float32)  # shape: (batch, 1024)
            if normalize_embeddings:
                dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)
            dense_all.append(dense)

        return np.concatenate(dense_all, axis=0)

    def encode_sparse(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[dict[int, float]]:
        """批量文本 → 稀疏词权向量 [{token_id: weight}, ...]。

        当前版本返回空（sentence-transformers 限制），ONNX 版本返回真实稀疏向量。
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        results: list[dict[int, float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            outputs = self._model(**inputs)
            sparse_batch = outputs[1]  # shape: (batch, seq_len, 1)

            for j, input_ids in enumerate(inputs["input_ids"]):
                weights = sparse_batch[j].astype(float).squeeze(-1)
                sparse_dict: dict[int, float] = {}
                for w, tid in zip(weights, input_ids.tolist()):
                    if tid not in self._unused_tokens and w > 0:
                        key = int(tid)
                        if key not in sparse_dict or w > sparse_dict[key]:
                            sparse_dict[key] = float(w)
                results.append(sparse_dict)

        return results

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """LlamaIndex BaseEmbedding 接口兼容。"""
        return self.encode(texts, normalize_embeddings=True).tolist()


# ═══════════════════════════════════════════════════════════════
#  单例加载
# ═══════════════════════════════════════════════════════════════

_bge_m3: BgeM3Onnx | None = None
_bge_m3_lock = threading.Lock()


def get_bge_m3() -> BgeM3Onnx:
    global _bge_m3
    if _bge_m3 is not None:
        return _bge_m3
    with _bge_m3_lock:
        if _bge_m3 is not None:
            return _bge_m3

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        onnx_model_name = getattr(settings, "bge_onnx_model_name", "gpahal/bge-m3-onnx-int8")

        logger.info("正在加载 BGE-M3 ONNX INT8 (GPU) ... model=%s", onnx_model_name)

        model_path = hf_hub_download(
            repo_id=onnx_model_name,
            filename="model_quantized.onnx",
            cache_dir=settings.embedding_cache_dir,
        )

        session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"],
        )
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        _bge_m3 = BgeM3Onnx(tokenizer, _OnnxSession(session))
        logger.info("BGE-M3 ONNX INT8 加载完成 (GPU, dim=1024)")
        return _bge_m3


# ═══════════════════════════════════════════════════════════════
#  BGE-Reranker（不变）
# ═══════════════════════════════════════════════════════════════

_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        from sentence_transformers import CrossEncoder
        logger.info("正在加载 BGE-Reranker (GPU) ...")
        _reranker = CrossEncoder(
            settings.reranker_model_name,
            device="cuda",
            trust_remote_code=True,
        )
        logger.info("BGE-Reranker 加载完成 (GPU)")
        return _reranker


def unload_models():
    global _bge_m3, _reranker
    _bge_m3 = None
    _reranker = None
    import gc
    gc.collect()
