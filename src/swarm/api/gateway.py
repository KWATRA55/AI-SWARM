"""Swarm API Gateway — the unified FastAPI application.

Mounts all route groups and the existing dashboard into a single server.

Architecture::

    /v1/health                    → Health checks (no auth)
    /v1/swarm/dispatch            → Start swarm runs (auth: dispatch)
    /v1/swarm/sessions            → Session management (auth: read)
    /dashboard/                   → Existing dashboard UI (optional)

Usage::

    from swarm.api.gateway import create_app

    app = create_app()
    # Run with: uvicorn swarm.api.gateway:create_app --factory
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


def create_app(
    *,
    enable_dashboard: bool = True,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build and return the full Swarm API gateway.

    Parameters
    ----------
    enable_dashboard:
        Mount the existing dashboard at ``/dashboard``.
    cors_origins:
        Allowed CORS origins.  Defaults to ``["*"]`` for local dev.
    """
    app = FastAPI(
        title="🐝 Swarm API Gateway",
        version="1.0.0",
        description=(
            "Enterprise API for the Swarm OS — dispatch tasks, "
            "manage sessions, and monitor telemetry."
        ),
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Global exception handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        await logger.error(
            "api.unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # --- Mount route groups ---
    from swarm.api.routes.health import router as health_router
    from swarm.api.routes.dispatch import router as dispatch_router
    from swarm.api.routes.sessions import router as sessions_router
    from swarm.api.routes.telemetry import router as telemetry_router
    from swarm.api.routes.analytics import router as analytics_router
    from swarm.api.routes.hitl import router as hitl_router

    app.include_router(health_router)
    app.include_router(dispatch_router)
    app.include_router(sessions_router)
    app.include_router(telemetry_router)
    app.include_router(analytics_router)
    app.include_router(hitl_router)

    # --- Root endpoint ---
    @app.get("/")
    async def root():
        return {
            "name": "Swarm API Gateway",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/v1/health",
        }

    # --- Mount existing dashboard (optional) ---
    if enable_dashboard:
        try:
            from swarm.dashboard.dashboard import SwarmDashboard

            dashboard = SwarmDashboard()
            app.mount("/dashboard", dashboard.app)
            logger.info("api.dashboard_mounted", path="/dashboard")
        except Exception as exc:
            logger.warning(
                "api.dashboard_mount_failed",
                error=str(exc),
                msg="Dashboard will not be available.",
            )

    # --- Startup / shutdown events ---
    @app.on_event("startup")
    async def on_startup():
        from swarm.api.auth import get_key_manager

        manager = get_key_manager()
        await logger.info(
            "api.gateway.started",
            api_keys_loaded=manager.key_count,
        )

    return app
