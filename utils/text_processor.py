"""通用文本格式化、清洗工具。"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """清洗文本：合并空白、去除控制字符。"""
    # 去除控制字符（保留换行和制表符）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 合并多个空白为一个空格（保留换行）
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(lines)


def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ═══════════════════════════════════════════════════════════════════════
#  时间格式化
# ═══════════════════════════════════════════════════════════════════════

from datetime import datetime, timezone


def format_time(ts_ms: int) -> str:
    """毫秒时间戳 → 'YYYY-MM-DD HH:mm' 字符串。0 或 None 返回空字符串。"""
    if not ts_ms:
        return ""
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        # 转为本地时间（UTC+8）
        from datetime import timedelta
        local = dt + timedelta(hours=8)
        return local.strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError):
        return ""
