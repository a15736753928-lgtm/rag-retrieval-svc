"""全局请求日志 & 耗时统计中间件。"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request

logger = logging.getLogger("api")


def register(app: FastAPI):
    @app.middleware("http")
    async def log_request(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method, request.url.path,
            response.status_code, elapsed,
        )
        return response
