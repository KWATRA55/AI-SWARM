"""Human-in-the-Loop checkpoint manager.

Manages checkpoints where agent execution pauses and waits for human
approval before continuing.

Architecture::

    Agent calls request_approval() tool
        → CheckpointManager.create_checkpoint()
            → asyncio.Event is created (blocking the agent)
            → Event published to dashboard/API

    Human approves/rejects via API or dashboard
        → CheckpointManager.approve() / .reject()
            → asyncio.Event is set (unblocking the agent)
            → Agent receives approval/rejection message

Usage::

    from swarm.core.hitl import CheckpointManager, CheckpointStatus

    manager = CheckpointManager()

    # Agent side (in worker loop):
    cp_id = await manager.create_checkpoint(
        agent="backend-dev",
        session_id="abc123",
        description="About to delete all test files",
        context={"files": ["test1.py", "test2.py"]},
    )
    result = await manager.await_approval(cp_id, timeout_seconds=300)
    if result.status == CheckpointStatus.APPROVED:
        # proceed
    elif result.status == CheckpointStatus.REJECTED:
        # handle rejection

    # Human side (via API):
    await manager.approve(cp_id, message="Looks good, proceed")
    # or
    await manager.reject(cp_id, reason="Don't delete those files")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CheckpointStatus(str, Enum):
    """Status of a HITL checkpoint."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    EXPIRED = "expired"


@dataclass
class Checkpoint:
    """A single HITL checkpoint."""

    checkpoint_id: str
    agent: str
    session_id: str
    description: str
    context: dict[str, Any] = field(default_factory=dict)
    status: CheckpointStatus = CheckpointStatus.PENDING
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None
    resolved_by: str | None = None
    message: str = ""
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


@dataclass
class CheckpointResult:
    """Result returned to the agent after checkpoint resolution."""

    status: CheckpointStatus
    message: str = ""
    resolved_by: str = ""


class CheckpointManager:
    """Manages HITL checkpoints across all sessions.

    Thread-safe via asyncio locks. Checkpoints are stored in-memory
    for single-process deployment; for multi-process, back with Redis.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}
        self._lock = asyncio.Lock()

    async def create_checkpoint(
        self,
        agent: str,
        session_id: str,
        description: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Create a new checkpoint and return its ID.

        The agent's execution will be paused until the checkpoint is
        resolved (approved or rejected).
        """
        checkpoint_id = f"cp-{uuid.uuid4().hex[:12]}"

        cp = Checkpoint(
            checkpoint_id=checkpoint_id,
            agent=agent,
            session_id=session_id,
            description=description,
            context=context or {},
        )

        async with self._lock:
            self._checkpoints[checkpoint_id] = cp

        logger.info(
            "hitl.checkpoint_created",
            checkpoint_id=checkpoint_id,
            agent=agent,
            session_id=session_id,
            description=description[:100],
        )

        return checkpoint_id

    async def await_approval(
        self,
        checkpoint_id: str,
        timeout_seconds: int = 1800,
    ) -> CheckpointResult:
        """Block until the checkpoint is resolved or times out.

        Returns a CheckpointResult with the status and optional message.
        """
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return CheckpointResult(
                status=CheckpointStatus.EXPIRED,
                message="Checkpoint not found.",
            )

        try:
            await asyncio.wait_for(cp._event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            async with self._lock:
                cp.status = CheckpointStatus.TIMED_OUT
                cp.resolved_at = time.time()

            logger.warning(
                "hitl.checkpoint_timed_out",
                checkpoint_id=checkpoint_id,
                timeout=timeout_seconds,
            )

            return CheckpointResult(
                status=CheckpointStatus.TIMED_OUT,
                message=f"Checkpoint timed out after {timeout_seconds}s.",
            )

        return CheckpointResult(
            status=cp.status,
            message=cp.message,
            resolved_by=cp.resolved_by or "",
        )

    async def approve(
        self,
        checkpoint_id: str,
        message: str = "",
        approved_by: str = "human",
    ) -> bool:
        """Approve a pending checkpoint, unblocking the waiting agent."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None or cp.status != CheckpointStatus.PENDING:
            return False

        async with self._lock:
            cp.status = CheckpointStatus.APPROVED
            cp.message = message
            cp.resolved_at = time.time()
            cp.resolved_by = approved_by
            cp._event.set()

        logger.info(
            "hitl.checkpoint_approved",
            checkpoint_id=checkpoint_id,
            approved_by=approved_by,
        )
        return True

    async def reject(
        self,
        checkpoint_id: str,
        reason: str = "",
        rejected_by: str = "human",
    ) -> bool:
        """Reject a pending checkpoint."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None or cp.status != CheckpointStatus.PENDING:
            return False

        async with self._lock:
            cp.status = CheckpointStatus.REJECTED
            cp.message = reason
            cp.resolved_at = time.time()
            cp.resolved_by = rejected_by
            cp._event.set()

        logger.info(
            "hitl.checkpoint_rejected",
            checkpoint_id=checkpoint_id,
            reason=reason[:100],
            rejected_by=rejected_by,
        )
        return True

    async def list_pending(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all pending checkpoints, optionally filtered by session."""
        results = []
        for cp in self._checkpoints.values():
            if cp.status != CheckpointStatus.PENDING:
                continue
            if session_id and cp.session_id != session_id:
                continue
            results.append({
                "checkpoint_id": cp.checkpoint_id,
                "agent": cp.agent,
                "session_id": cp.session_id,
                "description": cp.description,
                "context": cp.context,
                "created_at": cp.created_at,
                "waiting_seconds": round(time.time() - cp.created_at, 1),
            })
        return results

    async def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Get full checkpoint details."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return None
        return {
            "checkpoint_id": cp.checkpoint_id,
            "agent": cp.agent,
            "session_id": cp.session_id,
            "description": cp.description,
            "context": cp.context,
            "status": cp.status.value,
            "created_at": cp.created_at,
            "resolved_at": cp.resolved_at,
            "resolved_by": cp.resolved_by,
            "message": cp.message,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_manager: CheckpointManager | None = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get or create the global checkpoint manager."""
    global _manager
    if _manager is None:
        _manager = CheckpointManager()
    return _manager
