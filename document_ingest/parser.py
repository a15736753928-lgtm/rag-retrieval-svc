"""文件解析 —— 三层管线：快速文本提取 → 页面渲染 → OCR 兜底。

支持格式: .txt .md .docx .pdf .png .jpg .jpeg
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf", ".png", ".jpg", ".jpeg"}

# 单页文本低于此阈值时触发 OCR 兜底
_OCR_FALLBACK_THRESHOLD = 100

# RapidOCR 全局单例
_ocr = None


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def parse_bytes(data: bytes, filename: str) -> str:
    """解析二进制数据（上传文件） → 纯文本。"""
    ext = Path(filename).suffix.lower()

    # ── 纯文本：直接解码 ──────────────────────────────────
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")

    # ── 图片：直接走 OCR ──────────────────────────────────
    if ext in (".png", ".jpg", ".jpeg"):
        return _ocr_image(data)

    # ── PDF / DOCX：快速路径优先，不足时 OCR 兜底 ─────────
    fast_text = _extract_text_fast(data, filename, ext)
    if len(fast_text.strip()) >= _OCR_FALLBACK_THRESHOLD:
        return fast_text

    logger.info(
        "快速提取文本不足（%d 字符），触发 OCR 兜底: %s",
        len(fast_text.strip()),
        filename,
    )
    ocr_text = _ocr_document_pages(data, filename, ext)
    return _merge(fast_text, ocr_text)


# ═══════════════════════════════════════════════════════════════
#  第 1 层：快速原生文本提取（Rust FFI，毫秒级）
# ═══════════════════════════════════════════════════════════════

def _extract_text_fast(data: bytes, filename: str, ext: str) -> str:
    """pdf-oxide / office-oxide 快速提取所有文本。"""
    if ext == ".pdf":
        return _extract_pdf_text(data)
    elif ext == ".docx":
        return _extract_docx_text(data)
    return ""


def _extract_pdf_text(data: bytes) -> str:
    """pdf-oxide 逐页提取文本。"""
    from pdf_oxide import PdfDocument

    pdf = PdfDocument.from_bytes(data)
    texts = []
    for i in range(pdf.page_count()):
        try:
            page_text = pdf.extract_text(i)
            if page_text:
                texts.append(page_text.strip())
        except Exception:
            logger.debug("pdf-oxide 第 %d 页提取失败", i + 1)
    return "\n\n".join(texts)


def _extract_docx_text(data: bytes) -> str:
    """office-oxide 提取 DOCX 文本。"""
    from office_oxide import Document

    doc = Document.from_bytes(data, "docx")
    try:
        return doc.plain_text()
    finally:
        doc.close()


# ═══════════════════════════════════════════════════════════════
#  第 2 层：页面渲染为图片 → OCR
# ═══════════════════════════════════════════════════════════════

def _ocr_document_pages(data: bytes, filename: str, ext: str) -> str:
    """对文档中文本不足的页面/图片做 OCR。"""
    if ext == ".pdf":
        return _ocr_pdf_pages(data)
    elif ext == ".docx":
        return _ocr_docx_images(data)
    return ""


def _ocr_pdf_pages(data: bytes) -> str:
    """逐页判断 PDF：文本足的页直接取，不足的页渲染为图片 → OCR。"""
    from pdf_oxide import PdfDocument

    pdf = PdfDocument.from_bytes(data)
    texts: list[str] = []

    for i in range(pdf.page_count()):
        # 先取原生文本
        page_text = ""
        if pdf.has_text_layer(i):
            try:
                page_text = pdf.extract_text(i).strip()
            except Exception:
                pass

        if len(page_text) >= _OCR_FALLBACK_THRESHOLD:
            texts.append(page_text)
            continue

        # 文本不足 → 渲染为图片 → OCR
        logger.debug("PDF 第 %d 页文本不足（%d 字符），渲染 OCR", i + 1, len(page_text))
        try:
            pix = pdf.render_pixmap(i)
            img_bytes = pix.to_png()
            ocr_text = _ocr_image(img_bytes)
            if ocr_text.strip():
                texts.append(ocr_text)
            elif page_text:
                texts.append(page_text)
        except Exception:
            if page_text:
                texts.append(page_text)

    return "\n\n".join(texts)


def _ocr_docx_images(data: bytes) -> str:
    """提取 DOCX 中嵌入的图片 → OCR。"""
    from docx import Document

    doc = Document(io.BytesIO(data))
    texts: list[str] = []

    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            img_bytes = rel.target_part.blob
            ocr_text = _ocr_image(img_bytes)
            if ocr_text.strip():
                texts.append(ocr_text.strip())
        except Exception:
            logger.debug("DOCX 图片提取失败: %s", rel.reltype)

    return "\n\n".join(texts)


# ═══════════════════════════════════════════════════════════════
#  第 3 层：专业 OCR（RapidOCR → PP-OCR v4 模型）
# ═══════════════════════════════════════════════════════════════

def _ocr_image(img_data: bytes) -> str:
    """对单张图片做 OCR，返回识别文本。"""
    ocr = _get_ocr()
    result, _ = ocr(img_data)
    if not result:
        return ""
    lines = []
    for item in result:
        text = item[1]
        if text and text.strip():
            lines.append(text.strip())
    return "\n".join(lines)


def _get_ocr():
    """RapidOCR 全局单例，首次调用时加载模型。"""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
        logger.info("RapidOCR 模型加载完成")
    return _ocr


# ═══════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════

def _merge(fast_text: str, ocr_text: str) -> str:
    """合并快速提取文本和 OCR 文本。OCR 结果充足时优先采用。"""
    ocr = ocr_text.strip()
    if len(ocr) >= _OCR_FALLBACK_THRESHOLD:
        return ocr  # OCR 结果充足，直接使用
    fast = fast_text.strip()
    if fast:
        return fast
    return ocr  # 兜底：无论多少都返回 OCR 结果
