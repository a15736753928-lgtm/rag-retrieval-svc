"""DeepSeek API 客户端 —— OpenAI 兼容接口封装。

全局单例模式，和 model_loader.py 的 BGE-M3 实例同一设计。
实体抽取、关系判断、社区摘要等 GraphRAG 任务共用此客户端。
"""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """获取全局单例 OpenAI 客户端（惰性初始化）。"""
    global _client
    if _client is None:
        if not settings.llm_api_key:
            raise RuntimeError(
                "LLM_API_KEY 未配置，请在 .env 中添加 LLM_API_KEY=sk-xxxx"
            )
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
        )
        logger.info(
            "DeepSeek API 客户端就绪: model=%s base=%s",
            settings.llm_model,
            settings.llm_api_base,
        )
    return _client


def generate(
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 0,
    temperature: float | None = None,
    model: str = "",
) -> str:
    """调用 LLM 生成文本回复。

    Parameters
    ----------
    prompt : str
        用户消息内容。
    system : str
        系统提示词（角色设定）。
    max_tokens : int
        最大输出 token 数，0 表示使用配置默认值。
    temperature : float | None
        采样温度，None 表示使用配置默认值。
    model : str
        模型名，空字符串表示使用配置默认值。

    Returns
    -------
    str
        LLM 生成的文本回复。
    """
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        max_tokens=max_tokens or settings.llm_max_tokens,
        temperature=temperature if temperature is not None else settings.llm_temperature,
    )
    content = response.choices[0].message.content or ""
    return content


def generate_json(
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 0,
    temperature: float | None = None,
    max_retries: int = 2,
) -> dict:
    """调用 LLM 生成 JSON 回复。

    会追加 JSON 格式约束到 system prompt，自动处理 ```json 代码块包裹
    和尾部逗号等常见问题。解析失败时自动重试（最多 max_retries 次）。

    Returns
    -------
    dict
        解析后的 JSON 对象。重试耗尽仍失败则返回空 dict 并记录警告日志。
    """
    json_system = (system + "\n\n" if system else "") + (
        "请以纯 JSON 格式回复，不要包含 markdown 代码块标记（```），"
        "不要添加任何解释文字，只输出 JSON 对象。"
    )

    for attempt in range(max_retries + 1):
        raw = generate(
            prompt,
            system=json_system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # 尝试提取 JSON
        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed

        logger.warning(
            "LLM JSON 解析失败 (第 %d/%d 次)，重试中...\n原始输出: %s",
            attempt + 1, max_retries + 1, raw[:200],
        )

    logger.error("LLM JSON 解析重试耗尽，返回空 dict")
    return {}


def _extract_json(raw: str) -> dict | None:
    """从 LLM 原始输出中提取 JSON 对象。"""
    if not raw:
        return None

    text = raw.strip()

    # 去掉 markdown 代码块包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 去掉尾部逗号（LLM 常见问题）
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 尝试直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 尝试提取首个 JSON 对象 {...}
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None
