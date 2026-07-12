"""文件后缀、大小、MIME 类型校验。"""

from __future__ import annotations

from pathlib import Path

from config import settings
from document_ingest.parser import SUPPORTED_EXTENSIONS


def check_size(file_size: int) -> bool:
    return file_size <= settings.max_file_size


def check_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def get_file_size_mb(file_size: int) -> float:
    return file_size / (1024 * 1024)
