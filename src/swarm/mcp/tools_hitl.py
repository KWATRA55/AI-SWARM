"""Request Approval MCP tool — Human-in-the-Loop.

Allows agents to pause execution and request human approval
before proceeding with sensitive operations.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "request_approval",
        "description": (
            "Pause execution and request human approval before proceeding. "
            "Use this before destructive operations, production deployments, "
            "or any action that requires human oversight. "
            "Returns 'approved' or 'rejected' with an optional message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What you're about to do and why you need approval.",
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for the reviewer (files affected, commands to run, etc.).",
                    "default": {},
                },
                "blocking": {
                    "type": "boolean",
                    "description": "If true (default), wait for approval. If false, continue but log the request.",
                    "default": True,
                },
            },
            "required": ["description"],
        },
    },
}


async def execute(
    arguments: dict[str, Any],
    *,
    agent_name: str = "unknown",
    session_id: str = "",
    config: Any = None,
) -> dict[str, Any]:
    """Execute the request_approval tool."""
    from swarm.core.hitl import get_checkpoint_manager

    description = arguments.get("description", "")
    context = arguments.get("context", {})
    blocking = arguments.get("blocking", True)

    if not description:
        return {"success": False, "error": "Description is required.", "output": ""}

    # Get timeout from config
    timeout_seconds = 1800  # 30 min default
    if config and hasattr(config, "default_timeout_minutes"):
        timeout_seconds = config.default_timeout_minutes * 60

    manager = get_checkpoint_manager()

    # Create the checkpoint
    checkpoint_id = await manager.create_checkpoint(
        agent=agent_name,
        session_id=session_id,
        description=description,
        context=context,
    )

    if not blocking:
        return {
            "success": True,
            "output": f"Approval requested (non-blocking). Checkpoint ID: {checkpoint_id}",
            "checkpoint_id": checkpoint_id,
            "status": "pending",
        }

    # Wait for human response
    result = await manager.await_approval(
        checkpoint_id,
        timeout_seconds=timeout_seconds,
    )

    return {
        "success": result.status.value in ("approved",),
        "output": (
            f"Checkpoint {result.status.value}. "
            f"{result.message or 'No additional message.'}"
        ),
        "checkpoint_id": checkpoint_id,
        "status": result.status.value,
        "message": result.message,
        "resolved_by": result.resolved_by,
    }
