"""Real-time Swarm Dashboard — monitor, control, and chat with the AI swarm.

Provides a FastAPI server with:
- WebSocket endpoint for real-time event streaming
- REST API for swarm control (kill, pause, resume)
- Agent configuration editing (prompt, model, temperature)
- Manager chat (human ↔ pro model conversation)
- Post-run analytics and metrics

Architecture::

    ┌─────────────────────────┐
    │  Redis EventBus         │  (same bus the swarm uses)
    └───────────┬─────────────┘
                │ subscribe
    ┌───────────▼─────────────┐
    │  Dashboard Server       │
    │  ├─ /ws                 │ ──► browser (events)
    │  ├─ /api/status         │
    │  ├─ /api/kill           │
    │  ├─ /api/agent/*/config │ ──► edit prompts/model/temp
    │  ├─ /api/chat           │ ──► manager chat
    │  ├─ /api/metrics        │ ──► post-run analytics
    │  └─ /api/agent/*/pause  │
    └───────────┬─────────────┘
                │
    ┌───────────▼─────────────┐
    │  Browser Dashboard      │
    │  (HTML/JS/CSS)          │
    └─────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = structlog.get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class SwarmDashboard:
    """Real-time swarm monitoring dashboard with chat and config editing."""

    def __init__(
        self,
        *,
        orchestrator: Any = None,
        redis_url: str = "redis://localhost:6379/0",
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self._orchestrator = orchestrator
        self._redis_url = redis_url
        self._host = host
        self._port = port

        # Connected WebSocket clients
        self._clients: list[WebSocket] = []
        self._event_buffer: list[dict[str, Any]] = []
        self._max_buffer = 500

        # Stats
        self._start_time = time.time()
        self._event_count = 0

        # Manager chat (lazy-initialized)
        self._manager_chat: Any = None

        # Build the app
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        """Build the FastAPI application."""
        app = FastAPI(title="🐝 Swarm Dashboard", version="2.0.0")

        # ── Core endpoints ──

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return self._get_dashboard_html()

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._clients.append(ws)
            try:
                for event in self._event_buffer[-100:]:
                    await ws.send_json(event)
                while True:
                    data = await ws.receive_text()
                    try:
                        cmd = json.loads(data)
                        await self._handle_command(cmd)
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                if ws in self._clients:
                    self._clients.remove(ws)

        @app.get("/api/status")
        async def get_status():
            return JSONResponse(await self._get_status())

        # ── Kill / Pause / Resume ──

        @app.post("/api/kill")
        async def kill_swarm():
            if self._orchestrator:
                self._orchestrator.graceful_stop()
                await self.broadcast_event("system", "kill_switch_triggered",
                    {"message": "Kill switch activated from dashboard"})
            return JSONResponse({"status": "kill_switch_activated"})

        @app.post("/api/agent/{name}/pause")
        async def pause_agent(name: str):
            if self._orchestrator:
                await self._orchestrator.pause_agent(name, "Paused from dashboard")
                await self.broadcast_event("system", "agent_paused",
                    {"agent": name})
            return JSONResponse({"status": "paused", "agent": name})

        @app.post("/api/agent/{name}/resume")
        async def resume_agent(name: str):
            if self._orchestrator:
                await self._orchestrator.resume_agent(name)
                await self.broadcast_event("system", "agent_resumed",
                    {"agent": name})
            return JSONResponse({"status": "resumed", "agent": name})

        # ── Agent Config ──

        @app.get("/api/agents")
        async def list_agents():
            """List all agents with their configs."""
            if not self._orchestrator:
                return JSONResponse({"agents": []})
            configs = []
            for a in self._orchestrator._config.agents:
                configs.append({
                    "name": a.name,
                    "role": a.role.value,
                    "model": a.model,
                    "temperature": a.temperature,
                    "max_iterations": a.max_iterations,
                    "system_prompt": a.system_prompt,
                    "tools": a.tools,
                    "depends_on": a.depends_on,
                })
            return JSONResponse({"agents": configs})

        @app.get("/api/agent/{name}/config")
        async def get_agent_config(name: str):
            """Get a single agent's config."""
            if not self._orchestrator:
                return JSONResponse({"error": "no orchestrator"}, status_code=503)
            for a in self._orchestrator._config.agents:
                if a.name == name:
                    return JSONResponse({
                        "name": a.name,
                        "role": a.role.value,
                        "model": a.model,
                        "temperature": a.temperature,
                        "max_iterations": a.max_iterations,
                        "system_prompt": a.system_prompt,
                        "tools": a.tools,
                    })
            return JSONResponse({"error": f"Agent '{name}' not found"}, status_code=404)

        @app.put("/api/agent/{name}/config")
        async def update_agent_config(name: str):
            """Update an agent's config (prompt, model, temperature, etc)."""
            from starlette.requests import Request
            # Get request body manually since FastAPI needs the request object
            # This is fine for an internal API
            pass

        @app.api_route("/api/agent/{name}/config", methods=["PUT"])
        async def update_agent_config_impl(name: str):
            """Update agent config — accepts JSON body."""
            if not self._orchestrator:
                return JSONResponse({"error": "no orchestrator"}, status_code=503)

            # Read body from the client
            from starlette.requests import Request
            import inspect
            # We'll handle this from the websocket command instead
            return JSONResponse({"error": "Use WebSocket command for config updates"}, status_code=400)

        # ── Manager Chat ──

        @app.post("/api/chat")
        async def chat_message():
            """This is handled via WebSocket for real-time streaming."""
            return JSONResponse({"error": "Use WebSocket for chat"}, status_code=400)

        # ── Metrics & Analytics ──

        @app.get("/api/metrics")
        async def get_metrics():
            """Get collected metrics from the run."""
            if not self._orchestrator:
                return JSONResponse({"error": "no orchestrator"}, status_code=503)
            metrics = getattr(self._orchestrator, '_metrics', None)
            if metrics is None:
                return JSONResponse({"metrics": {}})
            session_id = self._orchestrator._state.session_id
            return JSONResponse(metrics.to_dict(session_id))

        @app.get("/api/analytics")
        async def get_analytics():
            """Get computed analytics summary."""
            if not self._orchestrator:
                return JSONResponse({"error": "no orchestrator"}, status_code=503)
            metrics = getattr(self._orchestrator, '_metrics', None)
            if metrics is None:
                return JSONResponse({"analytics": {}})
            session_id = self._orchestrator._state.session_id
            analytics = metrics.build_analytics(session_id)
            return JSONResponse(analytics.model_dump())

        return app

    async def broadcast_event(
        self,
        source: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast an event to all connected dashboard clients."""
        event = {
            "source": source,
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
            "id": self._event_count,
        }
        self._event_count += 1

        self._event_buffer.append(event)
        if len(self._event_buffer) > self._max_buffer:
            self._event_buffer = self._event_buffer[-self._max_buffer:]

        disconnected = []
        for ws in self._clients:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._clients.remove(ws)

    async def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Handle a command from the dashboard UI via WebSocket."""
        action = cmd.get("action")

        if action == "kill":
            if self._orchestrator:
                self._orchestrator.graceful_stop()

        elif action == "pause":
            agent = cmd.get("agent", "")
            if self._orchestrator and agent:
                await self._orchestrator.pause_agent(agent, "Dashboard pause")

        elif action == "resume":
            agent = cmd.get("agent", "")
            if self._orchestrator and agent:
                await self._orchestrator.resume_agent(agent)

        elif action == "update_config":
            await self._handle_config_update(cmd)

        elif action == "chat":
            await self._handle_chat_message(cmd)

        elif action == "dispatch_task":
            await self._handle_dispatch(cmd)

    async def _handle_config_update(self, cmd: dict[str, Any]) -> None:
        """Handle agent config update from dashboard."""
        if not self._orchestrator:
            return

        name = cmd.get("agent", "")
        updates = cmd.get("config", {})
        if not name or not updates:
            return

        # Find and update the agent config
        for i, a in enumerate(self._orchestrator._config.agents):
            if a.name == name:
                update_dict: dict[str, Any] = {}
                if "system_prompt" in updates:
                    update_dict["system_prompt"] = updates["system_prompt"]
                if "model" in updates:
                    update_dict["model"] = updates["model"]
                if "temperature" in updates:
                    update_dict["temperature"] = float(updates["temperature"])
                if "max_iterations" in updates:
                    update_dict["max_iterations"] = int(updates["max_iterations"])

                if update_dict:
                    self._orchestrator._config.agents[i] = a.model_copy(
                        update=update_dict,
                    )
                    await self.broadcast_event("system", "config_updated", {
                        "agent": name,
                        "updates": list(update_dict.keys()),
                    })
                    await logger.info(
                        "dashboard.config_updated",
                        agent=name, fields=list(update_dict.keys()),
                    )
                break

    async def _handle_chat_message(self, cmd: dict[str, Any]) -> None:
        """Handle a chat message from the user to the manager."""
        message = cmd.get("message", "").strip()
        if not message:
            return

        # Broadcast user message
        await self.broadcast_event("user", "chat_message", {
            "role": "user",
            "message": message,
        })

        # Initialize manager chat if needed
        if self._manager_chat is None:
            from swarm.core.manager_chat import ManagerChat

            model = "gemini/gemini-2.5-pro"
            workspace = Path("/tmp/screener-mvp")
            if self._orchestrator:
                model = self._orchestrator._config.agents[0].model
                workspace = Path(str(self._orchestrator._config.workspace))

            dispatch_fn = None
            if self._orchestrator:
                dispatch_fn = self._orchestrator.dispatch_dynamic_task

            state = None
            if self._orchestrator:
                state = self._orchestrator._state

            self._manager_chat = ManagerChat(
                model=model,
                workspace=workspace,
                dispatch_fn=dispatch_fn,
                dashboard_cb=self.broadcast_event,
                state=state,
            )

        # Send typing indicator
        await self.broadcast_event("manager", "chat_typing", {})

        # Get response with timeout to prevent silent hangs
        try:
            response = await asyncio.wait_for(
                self._manager_chat.send_message(message),
                timeout=120.0,  # 2 minute timeout for manager response
            )
        except asyncio.TimeoutError:
            response = "⏰ Manager response timed out (exceeded 2 minutes). The workspace may be too large for a full scan. Try a more specific request."
            await logger.error("dashboard.chat_timeout", message=message[:100])
        except Exception as exc:
            response = f"❌ Error: {exc}"
            await logger.error("dashboard.chat_error", error=str(exc), exc_info=True)

        # Broadcast manager response
        await self.broadcast_event("manager", "chat_message", {
            "role": "manager",
            "message": response,
        })
        await logger.info(
            "dashboard.chat_response_broadcast",
            response_length=len(response),
        )

    async def _handle_dispatch(self, cmd: dict[str, Any]) -> None:
        """Handle manual task dispatch from dashboard."""
        if not self._orchestrator:
            return
        agent = cmd.get("agent", "")
        task = cmd.get("task", "")
        if agent and task:
            try:
                await self._orchestrator.dispatch_dynamic_task(agent, task)
                await self.broadcast_event("system", "task_dispatched", {
                    "agent": agent,
                    "task": task[:200],
                })
            except Exception as exc:
                await self.broadcast_event("system", "dispatch_error", {
                    "agent": agent,
                    "error": str(exc),
                })

    async def _get_status(self) -> dict[str, Any]:
        """Get current swarm status."""
        status: dict[str, Any] = {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "connected_clients": len(self._clients),
            "total_events": self._event_count,
        }
        if self._orchestrator:
            try:
                snapshot = await self._orchestrator._state.snapshot()
                status["session_id"] = snapshot.session_id
                status["agents"] = {}
                for name, agent in snapshot.agents.items():
                    status["agents"][name] = {
                        "status": agent.status.value,
                        "iterations": agent.iterations,
                        "tokens": agent.token_usage.total_tokens,
                    }
                status["tasks"] = {}
                for tid, task in snapshot.tasks.items():
                    status["tasks"][tid] = {
                        "name": task.name,
                        "status": task.status.value,
                        "agent": task.assigned_agent,
                    }
            except Exception:
                status["error"] = "Could not read orchestrator state"
        return status

    async def start(self) -> asyncio.Task:
        """Start the dashboard server as a background task."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        await logger.info(
            "dashboard.started",
            url=f"http://localhost:{self._port}",
        )
        return task

    def _get_dashboard_html(self) -> str:
        """Return the dashboard HTML."""
        html_path = STATIC_DIR / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<html><body><h1>Dashboard HTML not found</h1></body></html>"
