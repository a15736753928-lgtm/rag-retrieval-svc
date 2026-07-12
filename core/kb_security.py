"""kb_id 格式校验工具。

kb_id 由应用层生成（`kb_{uuid8}`），不再由 project_code + domain_code 拼接。
本模块仅保留格式校验功能。
"""

from __future__ import annotations

import re

# kb_id 格式：kb_ 前缀 + 8位十六进制字符
_KB_ID_PATTERN = re.compile(r"^kb_[a-f0-9]{8}$")


def validate_kb_id(kb_id: str) -> bool:
    """校验 kb_id 格式是否合法。"""
    return bool(_KB_ID_PATTERN.match(kb_id))


# ── 以下为废弃函数，保留以避免 import 报错，新代码不应使用 ──

def make_kb_id(project_code: str, domain_code: str) -> str:
    """[已废弃] 拼接 kb_id。新代码应使用应用层生成的 kb_id（`kb_{uuid8}`）。"""
    return f"{project_code}_{domain_code}"


def split_kb_id(kb_id: str) -> tuple[str, str]:
    """[已废弃] 拆分 kb_id → (project_code, domain_code)。"""
    if "_" not in kb_id:
        return (kb_id, "")
    idx = kb_id.index("_")
    return (kb_id[:idx], kb_id[idx + 1:])


def extract_project(kb_id: str) -> str:
    return split_kb_id(kb_id)[0]


def extract_domain(kb_id: str) -> str:
    return split_kb_id(kb_id)[1]
