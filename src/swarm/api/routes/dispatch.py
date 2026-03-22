"""Swarm dispatch routes — trigger and control swarm runs.

Endpoints::

    POST /v1/swarm/dispatch           → Start a full swarm run from a YAML config
    POST /v1/swarm/dispatch/agent     → Dispatch a single task to an agent in a running session
    POST /v1/swarm/dispatch/cancel    → Cancel a running session
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from swarm.api.auth import APIKeyRecord, verify_api_key

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/swarm", tags=["dispatch"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class DispatchRequest(BaseModel):
    """Request body for starting a swarm run."""

    config_path: str = Field(
        description="Path to the YAML config file (relative to workspace or absolute).",
    )
    task: str = Field(
        default="",
        description="Optional task prompt to override the default in config.",
    )
    workspace: str = Field(
        default="",
        description="Optional workspace path override.",
    )


class DispatchResponse(BaseModel):
    """Response after dispatching a swarm run."""

    session_id: str
    status: str = "running"
    message: str = ""


class AgentDispatchRequest(BaseModel):
    """Request body for dispatching a task to a single agent."""

    session_id: str = Field(description="Target session ID.")
    agent_name: str = Field(description="Agent to dispatch the task to.")
    task: str = Field(description="Task prompt for the agent.")


class CancelRequest(BaseModel):
    """Request body for cancelling a session."""

    session_id: str


# ---------------------------------------------------------------------------
# Session registry (in-memory for single-process; Redis-backed for production)
# ---------------------------------------------------------------------------

# Active orchestrator instances, keyed by session_id
_active_sessions: dict[str, Any] = {}
_session_tasks: dict[str, asyncio.Task] = {}


def get_active_sessions() -> dict[str, Any]:
    """Return the active sessions dict (for use by other routes)."""
    return _active_sessions


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch_swarm(
    req: DispatchRequest,
    key: APIKeyRecord = Depends(verify_api_key("dispatch")),
):
    """Start a full swarm run from a YAML configuration.

    This is the primary entry point for external apps. The swarm runs
    asynchronously in the background — use the sessions API to monitor.
    """
    from swarm.config.loader import load_config
    from swarm.core.orchestrator import Orchestrator

    # Resolve config path
    config_path = Path(req.config_path)
    if not config_path.is_absolute():
        # Try relative to cwd
        config_path = Path.cwd() / config_path

    if not config_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Config file not found: {config_path}",
        )

    try:
        config = load_config(str(config_path))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid config: {exc}",
        )

    # Override workspace if provided
    if req.workspace:
        workspace_path = Path(req.workspace)
        if not workspace_path.is_absolute():
            raise HTTPException(400, "Workspace path must be absolute.")
        config = config.model_copy(update={"workspace": workspace_path})

    orchestrator = Orchestrator(config)
    session_id = orchestrator._state.session_id

    # Store reference
    _active_sessions[session_id] = orchestrator

    # Run asynchronously
    async def _run_session():
        try:
            await orchestrator.run(task=req.task or None)
        except Exception as exc:
            await logger.error(
                "api.dispatch.session_failed",
                session_id=session_id,
                error=str(exc),
            )
        finally:
            # Keep in registry for status checks, but mark as done
            pass

    task = asyncio.create_task(_run_session())
    _session_tasks[session_id] = task

    await logger.info(
        "api.dispatch.started",
        session_id=session_id,
        config=str(config_path),
        tenant=key.tenant_id,
    )

    return DispatchResponse(
        session_id=session_id,
        status="running",
        message=f"Swarm session started from {config_path.name}",
    )


@router.post("/dispatch/agent", response_model=DispatchResponse)
async def dispatch_to_agent(
    req: AgentDispatchRequest,
    key: APIKeyRecord = Depends(verify_api_key("dispatch")),
):
    """Dispatch a task to a specific agent in a running session."""
    orchestrator = _active_sessions.get(req.session_id)
    if orchestrator is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found.")

    try:
        await orchestrator.dispatch_dynamic_task(req.agent_name, req.task)
    except Exception as exc:
        raise HTTPException(400, f"Dispatch failed: {exc}")

    return DispatchResponse(
        session_id=req.session_id,
        status="dispatched",
        message=f"Task dispatched to '{req.agent_name}'",
    )


@router.post("/dispatch/cancel")
async def cancel_session(
    req: CancelRequest,
    key: APIKeyRecord = Depends(verify_api_key("admin")),
):
    """Cancel a running swarm session (kill switch)."""
    orchestrator = _active_sessions.get(req.session_id)
    if orchestrator is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found.")

    orchestrator.graceful_stop()

    # Cancel the asyncio task
    task = _session_tasks.get(req.session_id)
    if task and not task.done():
        task.cancel()

    await logger.info(
        "api.dispatch.cancelled",
        session_id=req.session_id,
        tenant=key.tenant_id,
    )

    return {"session_id": req.session_id, "status": "cancelled"}
