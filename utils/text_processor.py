"""通用文本格式化、清洗工具。"""

from __future__ import annotations

import re
import unicodedata


# 全角 ASCII → 半角映射（与 cleaner.py 保持一致）
_FULLWIDTH_START = 0xFF01
_FULLWIDTH_END = 0xFF5E
_HALFWIDTH_START = 0x21
_FULLWIDTH_OFFSET = _FULLWIDTH_START - _HALFWIDTH_START

_FULLWIDTH_EXTRAS: dict[int, int] = {0x3000: 0x20}

_CJK_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'",   # ' '
    "“": '"', "”": '"',   # " "
    "（": "(", "）": ")",   # （ ）
    "，": ",", "；": ";",   # ， ；
    "：": ":",                   # ：
})


def normalize_query(text: str) -> str:
    """规范化查询文本：NFC + 全角→半角 + 中文标点→英文 + 空白折叠。

    确保查询与经过文档清洗管道处理后的文本使用相同的 Unicode 规范化，
    避免因全角/半角字符差异导致的检索失配。

    与 document_ingest/cleaner.py 中 normalize_unicode + normalize_fullwidth
    的逻辑完全一致。
    """
    if not text:
        return text

    # 1. NFC 规范化（与 normalize_unicode 一致）
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[‎‏­­]", "", text)
    text = re.sub(r"[​‌‍﻿⁠]", "", text)

    # 2. 全角 → 半角（与 normalize_fullwidth 一致）
    result: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _FULLWIDTH_EXTRAS:
            result.append(chr(_FULLWIDTH_EXTRAS[cp]))
        elif _FULLWIDTH_START <= cp <= _FULLWIDTH_END:
            result.append(chr(cp - _FULLWIDTH_OFFSET))
        else:
            result.append(ch)
    text = "".join(result)
    text = text.translate(_CJK_PUNCT_MAP)

    # 3. 空白折叠
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text(text: str, *, strip_control: bool = True, normalize_space: bool = True) -> str:
    """清洗文本：去除控制字符 + 合并多余空白。

    Parameters
    ----------
    text : str
        待清洗文本。
    strip_control : bool
        是否移除 ASCII 控制字符（保留换行和制表符）。默认 True。
    normalize_space : bool
        是否合并行内多余空白、去除行首尾空白。默认 True。

    Returns
    -------
    str
        清洗后的文本。
    """
    if not text:
        return text

    # Unicode NFC 规范化
    text = unicodedata.normalize("NFC", text)

    # 零宽字符 & 不可见字符（PDF / OCR 常见噪声）
    text = re.sub(r"[​‌‍﻿⁠‎‏]", "", text)

    if strip_control:
        # 只保留 \n \t 两个控制字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    if normalize_space:
        lines = text.split("\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
        text = "\n".join(lines)

    return text


def truncate(text: str, max_len: int = 200) -> str:
    """超长文本截断，超出部分用 "..." 替代。"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


SAFE_HIGHLIGHT_TAGS = frozenset({"em", "strong", "b", "i", "u", "mark", "span", "code"})


def highlighter(text: str, query: str, tag: str = "em") -> str:
    """对文本中的查询词做简单高亮标记。

    Parameters
    ----------
    text : str
        原始文本。
    query : str
        要高亮的查询词。
    tag : str
        包裹标签名，默认 ``"em"``。仅允许白名单中的安全标签。

    Returns
    -------
    str
        带 HTML 高亮标签的文本。

    Raises
    ------
    ValueError
        当 ``tag`` 不在白名单中时抛出。
    """
    if tag not in SAFE_HIGHLIGHT_TAGS:
        raise ValueError(f"不允许的高亮标签: {tag!r}（白名单: {sorted(SAFE_HIGHLIGHT_TAGS)}）")
    if not query or not text:
        return text
    escaped = re.escape(query)
    return re.sub(
        f"({escaped})",
        f"<{tag}>\\1</{tag}>",
        text,
        flags=re.IGNORECASE,
    )


def count_tokens_approx(text: str) -> int:
    """粗略估计 token 数：中文按字符数，英文/数字按 4 字符≈1 token。

    仅用于分片大小的辅助判断，非精确计数。
    """
    if not text:
        return 0
    # 中文字符 ≈ 1 token
    cjk = len(re.findall(r"[一-鿿㐀-䶿]", text))
    # 其他字符：约 4 字符 / token
    other = len(text) - cjk
    return cjk + max(1, other // 4)


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
