"""HITL API routes — approve and reject checkpoints via REST.

Endpoints::

    GET  /v1/hitl/pending              → List pending checkpoints
    GET  /v1/hitl/{checkpoint_id}      → Checkpoint details
    POST /v1/hitl/{checkpoint_id}/approve → Approve
    POST /v1/hitl/{checkpoint_id}/reject  → Reject
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from swarm.api.auth import APIKeyRecord, verify_api_key
from swarm.core.hitl import get_checkpoint_manager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/hitl", tags=["hitl"])


class ApprovalRequest(BaseModel):
    """Request body for approving a checkpoint."""
    message: str = Field(default="", description="Optional message for the agent.")


class RejectionRequest(BaseModel):
    """Request body for rejecting a checkpoint."""
    reason: str = Field(default="", description="Rejection reason.")


@router.get("/pending")
async def list_pending(
    session_id: str | None = None,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """List all pending HITL checkpoints."""
    manager = get_checkpoint_manager()
    pending = await manager.list_pending(session_id=session_id)
    return {"checkpoints": pending, "total": len(pending)}


@router.get("/{checkpoint_id}")
async def get_checkpoint(
    checkpoint_id: str,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """Get details of a specific checkpoint."""
    manager = get_checkpoint_manager()
    cp = await manager.get_checkpoint(checkpoint_id)
    if cp is None:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found.")
    return cp


@router.post("/{checkpoint_id}/approve")
async def approve_checkpoint(
    checkpoint_id: str,
    req: ApprovalRequest = ApprovalRequest(),
    key: APIKeyRecord = Depends(verify_api_key("dispatch")),
):
    """Approve a pending checkpoint, unblocking the waiting agent."""
    manager = get_checkpoint_manager()
    ok = await manager.approve(
        checkpoint_id,
        message=req.message,
        approved_by=f"api:{key.tenant_id}",
    )
    if not ok:
        raise HTTPException(
            400,
            f"Cannot approve checkpoint '{checkpoint_id}'. "
            "It may not exist or is no longer pending.",
        )
    return {"checkpoint_id": checkpoint_id, "status": "approved"}


@router.post("/{checkpoint_id}/reject")
async def reject_checkpoint(
    checkpoint_id: str,
    req: RejectionRequest = RejectionRequest(),
    key: APIKeyRecord = Depends(verify_api_key("dispatch")),
):
    """Reject a pending checkpoint."""
    manager = get_checkpoint_manager()
    ok = await manager.reject(
        checkpoint_id,
        reason=req.reason,
        rejected_by=f"api:{key.tenant_id}",
    )
    if not ok:
        raise HTTPException(
            400,
            f"Cannot reject checkpoint '{checkpoint_id}'. "
            "It may not exist or is no longer pending.",
        )
    return {"checkpoint_id": checkpoint_id, "status": "rejected"}
