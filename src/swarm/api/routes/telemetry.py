"""Telemetry API routes — live streaming and polling endpoints.

Endpoints::

    GET /v1/telemetry/{session_id}/stream  → SSE event stream
    GET /v1/telemetry/{session_id}/tokens  → Current token usage per agent
    GET /v1/telemetry/{session_id}/files   → Files created/modified
    GET /v1/telemetry/{session_id}/timeline → Ordered event timeline
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from swarm.api.auth import APIKeyRecord, verify_api_key
from swarm.api.routes.dispatch import get_active_sessions

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/telemetry", tags=["telemetry"])


@router.get("/{session_id}/tokens")
async def get_token_usage(
    session_id: str,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """Get current token usage per agent (polling endpoint)."""
    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)
    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    try:
        snapshot = await orchestrator._state.snapshot()
        usage = {}
        for name, agent in snapshot.agents.items():
            config = None
            for a in orchestrator._config.agents:
                if a.name == name:
                    config = a
                    break
            budget = config.token_budget if config else 50_000
            usage[name] = {
                "prompt_tokens": agent.token_usage.prompt_tokens,
                "completion_tokens": agent.token_usage.completion_tokens,
                "total_tokens": agent.token_usage.total_tokens,
                "budget": budget,
                "utilization": round(agent.token_usage.total_tokens / max(budget, 1), 3),
            }
        return JSONResponse(content={"session_id": session_id, "tokens": usage})
    except Exception as exc:
        raise HTTPException(500, f"Failed to read token data: {exc}")


@router.get("/{session_id}/files")
async def get_file_changes(
    session_id: str,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """List files created/modified during the session."""
    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)
    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    try:
        snapshot = await orchestrator._state.snapshot()
        artifacts = [
            {"name": name, "path": path}
            for name, path in snapshot.artifacts.items()
        ]
        return JSONResponse(content={
            "session_id": session_id,
            "files": artifacts,
            "total": len(artifacts),
        })
    except Exception as exc:
        raise HTTPException(500, f"Failed to read file data: {exc}")


@router.get("/{session_id}/timeline")
async def get_timeline(
    session_id: str,
    limit: int = 100,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """Get ordered event timeline for the session."""
    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)
    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    try:
        # Read from session ledger if available
        ledger = getattr(orchestrator, "_ledger", None)
        if ledger and hasattr(ledger, "_events"):
            events = ledger._events[-limit:]
            return JSONResponse(content={
                "session_id": session_id,
                "events": [e if isinstance(e, dict) else {} for e in events],
                "total": len(events),
            })

        # Fallback: build timeline from state
        snapshot = await orchestrator._state.snapshot()
        timeline = []
        for msg in snapshot.messages[-limit:]:
            timeline.append({
                "type": "message",
                "timestamp": msg.timestamp.isoformat(),
                "sender": msg.sender,
                "recipient": msg.recipient,
                "content_preview": msg.content[:100],
            })
        return JSONResponse(content={
            "session_id": session_id,
            "events": timeline,
            "total": len(timeline),
        })
    except Exception as exc:
        raise HTTPException(500, f"Failed to read timeline: {exc}")


@router.get("/{session_id}/stream")
async def stream_events(
    session_id: str,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """SSE endpoint for streaming live telemetry events.

    Clients connect and receive real-time events as they happen:
    token updates, tool calls, file writes, agent status changes.
    """
    from starlette.responses import StreamingResponse

    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)
    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    async def event_generator():
        """Yield SSE events by polling state changes."""
        last_event_count = 0
        last_token_snapshot: dict[str, int] = {}

        while True:
            try:
                snapshot = await orchestrator._state.snapshot()

                # Check for token changes
                current_tokens = {
                    name: a.token_usage.total_tokens
                    for name, a in snapshot.agents.items()
                }

                if current_tokens != last_token_snapshot:
                    data = json.dumps({
                        "type": "token_update",
                        "agents": {
                            name: {
                                "total": a.token_usage.total_tokens,
                                "status": a.status.value,
                                "iterations": a.iterations,
                            }
                            for name, a in snapshot.agents.items()
                        },
                    })
                    yield f"data: {data}\n\n"
                    last_token_snapshot = current_tokens

                # Check if session is done
                all_done = all(
                    a.status.value in ("completed", "failed")
                    for a in snapshot.agents.values()
                )
                if all_done and snapshot.agents:
                    data = json.dumps({"type": "session_complete"})
                    yield f"data: {data}\n\n"
                    break

                await asyncio.sleep(1.0)  # Poll interval

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(2.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
