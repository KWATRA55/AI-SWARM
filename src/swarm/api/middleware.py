"""Aegis API Middleware — rate limiting and API key enforcement for FastAPI.

Provides ASGI middleware layers that protect the Swarm API endpoints:

1. **RateLimitMiddleware** — Token-bucket rate limiting per IP and per API key.
   Prevents DDoS and runaway clients from overwhelming the swarm.

2. **APIKeyMiddleware** — Enforces X-API-Key header on all /api/ routes
   except health checks. Uses the existing APIKeyManager from auth.py.

Usage::

    from swarm.api.middleware import RateLimitMiddleware, APIKeyMiddleware

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    app.add_middleware(APIKeyMiddleware, exclude_paths=["/api/health", "/ws"])
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Rate Limit Middleware
# ══════════════════════════════════════════════════════════════════════════════

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter per client IP.

    Returns HTTP 429 when rate limit is exceeded, with Retry-After header.

    Parameters
    ----------
    requests_per_minute
        Maximum requests per minute per client IP.
    burst
        Maximum burst size (bucket capacity). Defaults to 2x rate.
    """

    def __init__(
        self,
        app: Any,
        requests_per_minute: int = 60,
        burst: int | None = None,
    ) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst or requests_per_minute * 2
        # IP → (tokens, last_refill_time)
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Check rate limit before processing request."""
        # Skip rate limiting for WebSocket upgrades and health checks
        if request.url.path in ("/ws", "/api/health", "/health"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()

        # Prune timestamps older than 60s
        timestamps = self._buckets[client_ip]
        timestamps[:] = [t for t in timestamps if now - t < 60.0]

        if len(timestamps) >= self._rpm:
            # Rate limit exceeded
            wait = 60.0 - (now - timestamps[0])
            await logger.warning(
                "aegis.rate_limited",
                client_ip=client_ip,
                path=request.url.path,
                wait_seconds=round(wait, 1),
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": round(wait, 1),
                    "limit": f"{self._rpm} requests/minute",
                },
                headers={"Retry-After": str(int(wait) + 1)},
            )

        timestamps.append(now)

        # Clean up old IPs periodically (every 100 requests)
        if sum(len(v) for v in self._buckets.values()) > 1000:
            self._cleanup_buckets(now)

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_buckets(self, now: float) -> None:
        """Remove stale IP entries."""
        stale = [ip for ip, ts in self._buckets.items() if not ts or now - ts[-1] > 120]
        for ip in stale:
            del self._buckets[ip]


# ══════════════════════════════════════════════════════════════════════════════
# API Key Enforcement Middleware
# ══════════════════════════════════════════════════════════════════════════════

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforces API key authentication on all API routes.

    Routes in `exclude_paths` are exempt (e.g., health checks, WebSocket).

    Parameters
    ----------
    exclude_paths
        List of path prefixes to skip authentication for.
    """

    def __init__(
        self,
        app: Any,
        exclude_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._exclude = set(exclude_paths or [
            "/api/health",
            "/health",
            "/ws",
            "/",              # Dashboard static files
        ])

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Validate API key for protected routes."""
        path = request.url.path

        # Skip excluded paths
        if any(path.startswith(ex) for ex in self._exclude):
            return await call_next(request)

        # Skip non-API routes (static files, etc.)
        if not path.startswith("/api/"):
            return await call_next(request)

        # Check for API key
        api_key = request.headers.get("x-api-key")
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing X-API-Key header"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Validate key
        from swarm.api.auth import get_key_manager
        manager = get_key_manager()
        record = manager.validate(api_key)

        if record is None:
            await logger.warning(
                "aegis.invalid_api_key",
                client_ip=self._get_client_ip(request),
                path=path,
            )
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Attach key record to request state for downstream use
        request.state.api_key_record = record
        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Request logging middleware
# ══════════════════════════════════════════════════════════════════════════════

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing for observability."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        if request.url.path.startswith("/api/"):
            await logger.info(
                "aegis.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration_ms, 1),
            )

        return response
