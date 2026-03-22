"""Post-run analytics API routes.

Endpoints::

    GET /v1/analytics/{session_id}          → Full analytics summary
    GET /v1/analytics/{session_id}/compare  → Compare two sessions
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from swarm.api.auth import APIKeyRecord, verify_api_key
from swarm.api.routes.dispatch import get_active_sessions

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/analytics", tags=["analytics"])


def _compute_analytics(orchestrator: Any) -> dict[str, Any]:
    """Compute analytics from an orchestrator's state and metrics."""
    import asyncio

    try:
        snapshot = asyncio.get_event_loop().run_until_complete(
            orchestrator._state.snapshot()
        )
    except RuntimeError:
        # We're already in an async context
        snapshot = None

    # Agent performance
    agents_data: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    total_iterations = 0

    if snapshot:
        for name, agent in snapshot.agents.items():
            tokens = agent.token_usage.total_tokens
            total_tokens += tokens
            total_iterations += agent.iterations
            agents_data[name] = {
                "status": agent.status.value,
                "tokens": tokens,
                "iterations": agent.iterations,
                "tokens_per_iteration": round(tokens / max(agent.iterations, 1), 1),
            }

    # Task summary
    tasks_completed = 0
    tasks_failed = 0
    if snapshot:
        for task in snapshot.tasks.values():
            if task.status.value == "completed":
                tasks_completed += 1
            elif task.status.value == "failed":
                tasks_failed += 1

    return {
        "total_tokens": total_tokens,
        "total_iterations": total_iterations,
        "agent_count": len(agents_data),
        "agents": agents_data,
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
        "avg_tokens_per_iteration": round(total_tokens / max(total_iterations, 1), 1),
    }


@router.get("/{session_id}")
async def get_analytics(
    session_id: str,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """Get computed analytics summary for a completed or running session."""
    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)
    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    try:
        snapshot = await orchestrator._state.snapshot()

        # Build analytics
        agents_data = {}
        total_tokens = 0
        total_iterations = 0

        for name, agent in snapshot.agents.items():
            tokens = agent.token_usage.total_tokens
            total_tokens += tokens
            total_iterations += agent.iterations
            budget = 50_000
            for a in orchestrator._config.agents:
                if a.name == name:
                    budget = a.token_budget
                    break

            agents_data[name] = {
                "status": agent.status.value,
                "tokens": tokens,
                "iterations": agent.iterations,
                "budget": budget,
                "efficiency": round(tokens / max(budget, 1), 3),
                "tokens_per_iteration": round(tokens / max(agent.iterations, 1), 1),
            }

        # Existing metrics if available
        metrics_data = {}
        metrics = getattr(orchestrator, "_metrics", None)
        if metrics:
            try:
                metrics_data = metrics.to_dict(session_id)
            except Exception:
                pass

        return JSONResponse(content={
            "session_id": session_id,
            "total_tokens": total_tokens,
            "total_iterations": total_iterations,
            "agent_count": len(agents_data),
            "avg_tokens_per_iteration": round(total_tokens / max(total_iterations, 1), 1),
            "agents": agents_data,
            "tasks_completed": sum(1 for t in snapshot.tasks.values() if t.status.value == "completed"),
            "tasks_failed": sum(1 for t in snapshot.tasks.values() if t.status.value == "failed"),
            "metrics": metrics_data,
        })
    except Exception as exc:
        raise HTTPException(500, f"Failed to compute analytics: {exc}")


@router.get("/{session_id}/compare")
async def compare_sessions(
    session_id: str,
    compare_to: str = Query(..., description="Session ID to compare against."),
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """Compare two sessions side by side."""
    sessions = get_active_sessions()

    orch_a = sessions.get(session_id)
    orch_b = sessions.get(compare_to)

    if orch_a is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    if orch_b is None:
        raise HTTPException(404, f"Session '{compare_to}' not found.")

    try:
        snap_a = await orch_a._state.snapshot()
        snap_b = await orch_b._state.snapshot()

        tokens_a = sum(a.token_usage.total_tokens for a in snap_a.agents.values())
        tokens_b = sum(a.token_usage.total_tokens for a in snap_b.agents.values())

        return JSONResponse(content={
            "sessions": [session_id, compare_to],
            "comparison": {
                "total_tokens": [tokens_a, tokens_b],
                "delta_tokens": tokens_a - tokens_b,
                "agent_counts": [len(snap_a.agents), len(snap_b.agents)],
                "task_counts": [len(snap_a.tasks), len(snap_b.tasks)],
            },
        })
    except Exception as exc:
        raise HTTPException(500, f"Comparison failed: {exc}")
