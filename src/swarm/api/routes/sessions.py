"""Session management routes — list, inspect, and query swarm sessions.

Endpoints::

    GET  /v1/swarm/sessions              → List active/recent sessions
    GET  /v1/swarm/sessions/{id}/status  → Full state snapshot
    GET  /v1/swarm/sessions/{id}/replay  → Download session replay JSONL
    POST /v1/swarm/sessions/{id}/kill    → Kill a running session
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from swarm.api.auth import APIKeyRecord, verify_api_key
from swarm.api.routes.dispatch import get_active_sessions

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/swarm/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """List all active and recently completed sessions."""
    sessions = get_active_sessions()
    result = []

    for session_id, orchestrator in sessions.items():
        try:
            state = orchestrator._state
            agents_info = {}
            for name, agent in state._agents.items():
                agents_info[name] = {
                    "status": agent.status.value,
                    "iterations": agent.iterations,
                    "tokens": agent.token_usage.total_tokens,
                }

            result.append({
                "session_id": session_id,
                "swarm_name": state._swarm_name,
                "agents": agents_info,
                "task_count": len(state._tasks),
            })
        except Exception:
            result.append({
                "session_id": session_id,
                "error": "Could not read state",
            })

    return {"sessions": result, "total": len(result)}


@router.get("/{session_id}/status")
async def session_status(
    session_id: str,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """Get the full state snapshot of a session."""
    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)

    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    try:
        snapshot = await orchestrator._state.snapshot()
        return JSONResponse(content={
            "session_id": snapshot.session_id,
            "swarm_name": snapshot.swarm_name,
            "agents": {
                name: {
                    "status": a.status.value,
                    "iterations": a.iterations,
                    "tokens": {
                        "prompt": a.token_usage.prompt_tokens,
                        "completion": a.token_usage.completion_tokens,
                        "total": a.token_usage.total_tokens,
                    },
                    "started_at": a.started_at.isoformat() if a.started_at else None,
                    "finished_at": a.finished_at.isoformat() if a.finished_at else None,
                }
                for name, a in snapshot.agents.items()
            },
            "tasks": {
                tid: {
                    "name": t.name,
                    "status": t.status.value,
                    "agent": t.assigned_agent,
                    "result": t.result,
                    "error": t.error,
                }
                for tid, t in snapshot.tasks.items()
            },
            "message_count": len(snapshot.messages),
            "artifact_count": len(snapshot.artifacts),
        })
    except Exception as exc:
        raise HTTPException(500, f"Failed to read session state: {exc}")


@router.get("/{session_id}/replay")
async def download_replay(
    session_id: str,
    key: APIKeyRecord = Depends(verify_api_key("read")),
):
    """Download the session replay JSONL file."""
    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)

    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    # Find the replay file from the ledger
    ledger = getattr(orchestrator, "_ledger", None)
    if ledger is None or not hasattr(ledger, "output_path"):
        raise HTTPException(404, "Session replay not available (telemetry disabled).")

    replay_path = ledger.output_path
    if replay_path is None or not Path(replay_path).exists():
        raise HTTPException(404, "Session replay file not found on disk.")

    return FileResponse(
        path=str(replay_path),
        filename=Path(replay_path).name,
        media_type="application/x-ndjson",
    )


@router.post("/{session_id}/kill")
async def kill_session(
    session_id: str,
    key: APIKeyRecord = Depends(verify_api_key("admin")),
):
    """Trigger the kill switch on a running session."""
    sessions = get_active_sessions()
    orchestrator = sessions.get(session_id)

    if orchestrator is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    orchestrator.graceful_stop()

    await logger.info(
        "api.sessions.killed",
        session_id=session_id,
        tenant=key.tenant_id,
    )

    return {"session_id": session_id, "status": "kill_switch_activated"}
