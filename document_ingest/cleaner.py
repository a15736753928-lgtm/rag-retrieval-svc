"""文档清洗模块 —— 在解析之后、分片之前对原始文本做规范化处理。

每条规则都是独立函数，通过 ``clean_document()`` 组合为管道，
按序执行。函数签名统一为 (text, file_type) → text，方便增删和
单元测试。
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser


# ═══════════════════════════════════════════════════════════════════════
#  HTML 标签剥离
# ═══════════════════════════════════════════════════════════════════════

class _HTMLStripper(HTMLParser):
    """收集 HTML 标签外的纯文本，替换常见块级标签为换行。"""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._block_tags = {"br", "p", "div", "li", "tr", "h1", "h2", "h3",
                            "h4", "h5", "h6", "section", "article", "table"}

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(text: str, file_type: str) -> str:
    """去除 HTML 标签，保留纯文本，块级标签转换为换行。"""
    _ = file_type  # 与文件类型无关，通用处理
    if "<" not in text or ">" not in text:
        return text  # 快速跳过
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


# ═══════════════════════════════════════════════════════════════════════
#  Unicode 规范化 & 全角半角
# ═══════════════════════════════════════════════════════════════════════

# 全角字符 → 半角映射（仅对 ASCII 有对应关系的做转换，避免误伤中文标点）
_FULLWIDTH_START = 0xFF01
_FULLWIDTH_END = 0xFF5E
_HALFWIDTH_START = 0x21
_FULLWIDTH_OFFSET = _FULLWIDTH_START - _HALFWIDTH_START

# 额外的全角 → 半角映射（不在连续区间的）
_FULLWIDTH_EXTRAS: dict[int, int] = {
    0x3000: 0x20,     # 全角空格 → 半角空格
}

# 中文标点 → 英文标点（适合技术文档检索）
_CJK_PUNCT_MAP = str.maketrans({
    "‘": "'",  # ‘ → '
    "’": "'",  # ' → '
    "“": '"',  # " → "
    "”": '"',  # " → "
    "（": "(",  # （ → (
    "）": ")",  # ） → )
    "，": ",",  # ， → ,
    "；": ";",  # ； → ;
    "：": ":",  # ： → :
})


def normalize_unicode(text: str, file_type: str) -> str:
    """Unicode NFC 规范化 + 不可见字符清洗。"""
    _ = file_type
    # NFC 规范化（组合字符统一为预组合形式）
    text = unicodedata.normalize("NFC", text)
    # 移除 Unicode 双向控制字符和软连字符
    text = re.sub(r"[‎‏­﻿]", "", text)
    # 移除零宽字符（PDF 常见）
    text = re.sub(r"[​‌‍﻿⁠]", "", text)
    return text


def normalize_fullwidth(text: str, file_type: str) -> str:
    """全角字母/数字/符号 → 半角，中文标点 → 英文标点。"""
    _ = file_type
    result: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _FULLWIDTH_EXTRAS:
            result.append(chr(_FULLWIDTH_EXTRAS[cp]))
        elif _FULLWIDTH_START <= cp <= _FULLWIDTH_END:
            result.append(chr(cp - _FULLWIDTH_OFFSET))
        elif ch in _CJK_PUNCT_MAP:
            result.append(ch.translate(_CJK_PUNCT_MAP) if isinstance(_CJK_PUNCT_MAP, dict) else ch)
        else:
            result.append(ch)
    text = "".join(result)
    # str.translate 版本更高效
    text = text.translate(_CJK_PUNCT_MAP)
    return text


# ═══════════════════════════════════════════════════════════════════════
#  控制字符 & 空白规范化
# ═══════════════════════════════════════════════════════════════════════

def remove_control_chars(text: str, file_type: str) -> str:
    """移除 ASCII 控制字符（保留 \n 和 \t，保留高位 Unicode）。"""
    _ = file_type
    # 移除 0x00-0x08, 0x0b, 0x0c, 0x0e-0x1f, 0x7f
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def normalize_whitespace(text: str, file_type: str) -> str:
    """合并多余空白：行内多个空格→1 个，去除行首行尾空白。"""
    _ = file_type
    lines = text.split("\n")
    cleaned = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned)


def normalize_empty_lines(text: str, file_type: str) -> str:
    """折叠多余空行：≥3 个连续空行压缩为 2 个。"""
    _ = file_type
    return re.sub(r"\n{3,}", "\n\n", text)


# ═══════════════════════════════════════════════════════════════════════
#  页眉 / 页脚 / 页码 过滤
# ═══════════════════════════════════════════════════════════════════════

# 常见页眉页脚模式
_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$"),            # 纯页码: "42", "- 5 -"
    re.compile(r"^\s*第\s*\d{1,4}\s*页\s*$"),            # 中文页码: "第 5 页"
    re.compile(r"^\s*page\s*\d{1,4}(\s*of\s*\d{1,4})?\s*$", re.IGNORECASE),  # "Page 5", "Page 1 of 20"
    re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$"),              # "5/20"
    re.compile(r"^[\(\[\{]*\s*\d{1,4}\s*[\)\]\}]*$"),    # "(5)", "[42]"
]


def filter_header_footer(text: str, file_type: str) -> str:
    """去除疑似页眉页脚行（独立成行的页码、重复标题等）。

    每行独立判断：如果是纯页码/章节标题模式且与内容上下文不连贯，
    则移除。用启发式规则，宁可漏杀不可误杀。
    """
    _ = file_type
    lines = text.split("\n")
    kept: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        # 纯页码/编号行 —— 独立成段时移除
        if _looks_like_page_number(stripped):
            # 只有当前后都是空行（独立成段）时才移除
            prev_empty = i == 0 or not lines[i - 1].strip()
            next_empty = i == len(lines) - 1 or not lines[i + 1].strip()
            if prev_empty and next_empty:
                continue

        # 短纯符号分隔线
        if re.match(r"^[\-\*_=~#]{3,}$", stripped):
            continue

        kept.append(line)

    return "\n".join(kept)


def _looks_like_page_number(line: str) -> bool:
    """判断一行是否像孤立页码。"""
    if len(line) > 20:
        return False
    for pat in _PAGE_NUMBER_PATTERNS:
        if pat.match(line):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
#  文档尾部噪声过滤（引用 / 版权声明 / 广告）
# ═══════════════════════════════════════════════════════════════════════

# 尾部噪声起始标记
_FOOTER_NOISE_STARTS = [
    re.compile(r"^参考\s*文献\s*$", re.IGNORECASE),
    re.compile(r"^references?\s*$", re.IGNORECASE),
    re.compile(r"^bibliography\s*$", re.IGNORECASE),
    re.compile(r"^免责声明\s*$", re.IGNORECASE),
    re.compile(r"^disclaimer\s*$", re.IGNORECASE),
    re.compile(r"^版权\s*(所有|声明|信息)?\s*$", re.IGNORECASE),
    re.compile(r"^copyright\s*", re.IGNORECASE),
    re.compile(r"^all rights reserved", re.IGNORECASE),
    re.compile(r"^本(文|报告|文档).*(仅|不|未)", re.IGNORECASE),
    re.compile(r"^the\s+information\s+contained", re.IGNORECASE),
    re.compile(r"^confidential", re.IGNORECASE),
]


def trim_trailing_noise(text: str, file_type: str) -> str:
    """裁剪文档尾部的引用列表、版权声明等噪声。

    从后往前扫描：当连续 3 行都匹配噪声模式时，认为从此行开始都是
    尾部噪声。只在文档后 30% 范围检测，避免误删正文。
    """
    _ = file_type
    lines = text.split("\n")
    total = len(lines)
    if total < 10:
        return text

    # 只在文档尾部检测
    scan_start = max(total * 7 // 10, 0)  # 后 30%

    noise_start = total
    consecutive_noise = 0

    for i in range(total - 1, scan_start - 1, -1):
        stripped = lines[i].strip()
        if not stripped:
            consecutive_noise = 0
            continue

        if _looks_like_footer_noise(stripped):
            consecutive_noise += 1
            if consecutive_noise >= 3:
                noise_start = i
        else:
            consecutive_noise = 0

    if noise_start < total:
        return "\n".join(lines[:noise_start])
    return text


def _looks_like_footer_noise(line: str) -> bool:
    """判断一行是否为尾部噪声（引用条目、版权声明等）。"""
    # 参考文献条目：[1] Author. Title. ...
    if re.match(r"^\s*\[\d+\]", line):
        return True
    for pat in _FOOTER_NOISE_STARTS:
        if pat.search(line):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
#  Markdown 语法清理（可选 —— 默认不启用，保留语义信息）
# ═══════════════════════════════════════════════════════════════════════

def strip_markdown(text: str, file_type: str) -> str:
    """剥离常见 Markdown 语法标记，保留纯文本结构。

    仅在 file_type 为 'md' 时启用。
    """
    if file_type != "md":
        return text

    # 标题标记 # → 保留文本
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 粗体/斜体
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # 行内代码 `code` → code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 链接 [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # 图片 ![](url) → 移除
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # 列表标记 -/* /+ /1. → 保留文本
    text = re.sub(r"^[\s]*[-*+]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # 引用 >
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    # 水平线 --- / *** / ___
    text = re.sub(r"^[\s]*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    return text


# ═══════════════════════════════════════════════════════════════════════
#  管道：组合所有规则
# ═══════════════════════════════════════════════════════════════════════

# 默认清洗管道 —— 按顺序执行的规则列表
# 每条规则: (函数, 名称), 名称用于日志/调试
DEFAULT_PIPELINE: list[tuple[callable, str]] = [
    (strip_html,              "HTML 标签剥离"),
    (normalize_unicode,       "Unicode 规范化"),
    (normalize_fullwidth,     "全角 → 半角"),
    (remove_control_chars,    "控制字符移除"),
    (strip_markdown,          "Markdown 语法剥离"),
    (filter_header_footer,    "页眉/页脚/页码过滤"),
    (trim_trailing_noise,     "尾部噪声裁剪"),
    (normalize_whitespace,    "空白规范化"),
    (normalize_empty_lines,   "空行压缩"),
]


def clean_document(
    text: str,
    file_type: str = "",
    pipeline: list[tuple[callable, str]] | None = None,
) -> str:
    """执行完整的文档清洗管道。

    Parameters
    ----------
    text : str
        原始文本（由 parser 提取）。
    file_type : str
        文件扩展名（不含点），如 "pdf"、"docx"、"md"。
        部分规则会据此调整行为。
    pipeline : list[tuple[callable, str]] | None
        自定义管道。传 ``None`` 使用默认管道。

    Returns
    -------
    str
        清洗后的文本。
    """
    if not text:
        return text

    steps = pipeline if pipeline is not None else DEFAULT_PIPELINE

    current = text
    for func, step_name in steps:
        try:
            before = len(current)
            current = func(current, file_type)
            after = len(current)
            if before != after:
                import logging
                logging.getLogger(__name__).debug(
                    "清洗步骤 [%s]: %d → %d 字符 (差值 %+d)",
                    step_name, before, after, after - before,
                )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "清洗步骤 [%s] 执行异常，跳过", step_name, exc_info=True,
            )
    return current


def clean_document_light(text: str, file_type: str = "") -> str:
    """轻量清洗：仅规范化 Unicode + 控制字符 + 空白。

    适用于 txt/md 等格式相对干净的文件，保留更多原始结构。
    """
    return clean_document(text, file_type, pipeline=[
        (normalize_unicode,       "Unicode 规范化"),
        (remove_control_chars,    "控制字符移除"),
        (normalize_whitespace,    "空白规范化"),
        (normalize_empty_lines,   "空行压缩"),
    ])
