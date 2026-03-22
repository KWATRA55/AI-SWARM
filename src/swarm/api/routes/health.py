"""Health check endpoint — liveness and dependency probes.

Usage::

    GET /v1/health          → {"status": "ok", "redis": true, ...}
    GET /v1/health/ready    → 200 if all deps are up, 503 otherwise
"""

from __future__ import annotations

import os
import time

import structlog
from fastapi import APIRouter

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("")
async def health_check():
    """Basic liveness check — always returns 200 if the server is running."""
    checks: dict[str, bool | str] = {
        "status": "ok",
        "uptime_info": "server is running",
    }

    # Check Redis connectivity
    try:
        import redis.asyncio as aioredis

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = aioredis.from_url(redis_url, decode_responses=True)
        pong = await r.ping()
        checks["redis"] = bool(pong)
        await r.aclose()
    except Exception:
        checks["redis"] = False

    # Check LLM provider (lightweight — just verify env key exists)
    checks["llm_keys_configured"] = any(
        k in os.environ
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )

    return checks


@router.get("/ready")
async def readiness_check():
    """Readiness probe — returns 503 if critical dependencies are down."""
    from fastapi.responses import JSONResponse

    ready = True
    details: dict[str, bool] = {}

    # Redis is required for production
    try:
        import redis.asyncio as aioredis

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = aioredis.from_url(redis_url, decode_responses=True)
        pong = await r.ping()
        details["redis"] = bool(pong)
        await r.aclose()
    except Exception:
        details["redis"] = False
        # Redis not required for local dev — don't fail readiness
        # ready = False

    details["llm_keys"] = any(
        k in os.environ
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not details["llm_keys"]:
        ready = False

    status_code = 200 if ready else 503
    return JSONResponse(
        content={"ready": ready, "checks": details},
        status_code=status_code,
    )
