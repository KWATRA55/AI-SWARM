"""MCP server factory — exposes tools to sandboxed agents.

This is the **host-side** server that Docker containers connect to in order
to invoke tools (read_file, write_file, run_command, etc.).

Transport modes
---------------

.. code-block:: text

    ┌──────────────────────────────────────────────────────┐
    │                    HOST SIDE                         │
    │                                                      │
    │  MCPServerFactory.create(agent_config)               │
    │       │                                              │
    │       ├─── SSE transport ────────────────────────┐   │
    │       │    uvicorn / Starlette SSE endpoint      │   │
    │       │    http://0.0.0.0:<port>/sse             │   │
    │       │    http://0.0.0.0:<port>/messages        │   │
    │       │                                          │   │
    │       └─── STDIO transport ──────────────────┐   │   │
    │            docker exec -i <container>         │   │   │
    │            JSON-RPC over stdin/stdout         │   │   │
    │                                              │   │   │
    └──────────────────────────────────────┬───────┼───┘   │
                                           │       │       │
    ┌──────────────────────────────────────┼───────┼───────┘
    │              CONTAINER               │       │
    │                                      ▼       ▼
    │  Agent process ──► MCP Client ──► (SSE|STDIO)
    │       │                              │
    │       └── tool_call("read_file") ────┘
    │                                      │
    └──────────────────────────────────────┘

The server validates every tool call against the agent's ``allowed_tools``
whitelist from ``MCPConfig`` before dispatching to the ``ToolExecutor``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import structlog

from swarm.config.models import AgentConfig, MCPTransport
from swarm.events.bus import EventBus
from swarm.mcp.tools import ToolExecutor, ToolResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON-RPC protocol models
# ---------------------------------------------------------------------------


class JsonRpcRequest:
    """A minimal JSON-RPC 2.0 request container."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.id = raw.get("id")
        self.method = raw.get("method", "")
        self.params = raw.get("params", {})

    @classmethod
    def from_json(cls, data: str) -> "JsonRpcRequest":
        return cls(json.loads(data))


class JsonRpcResponse:
    """A minimal JSON-RPC 2.0 response builder."""

    @staticmethod
    def success(request_id: Any, result: Any) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    @staticmethod
    def error(
        request_id: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> dict[str, Any]:
        resp: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        if data is not None:
            resp["error"]["data"] = data
        return resp


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


class MCPServer:
    """Host-side MCP server that dispatches tool calls to the ToolExecutor.

    This is protocol-agnostic — transport adapters (SSE, STDIO) call into
    ``handle_request()`` and get a JSON-RPC response back.

    Usage::

        server = MCPServer(
            agent_name="backend-agent",
            tool_executor=executor,
            allowed_tools=["read_file", "write_file"],
        )

        # Handle a raw JSON-RPC request
        response = await server.handle_request(json_rpc_dict)

        # Or handle a raw JSON string
        response = await server.handle_json(json_string)
    """

    # MCP protocol methods
    METHOD_INITIALIZE = "initialize"
    METHOD_LIST_TOOLS = "tools/list"
    METHOD_CALL_TOOL = "tools/call"
    METHOD_PING = "ping"

    def __init__(
        self,
        agent_name: str,
        tool_executor: ToolExecutor,
        allowed_tools: list[str],
    ) -> None:
        self._agent_name = agent_name
        self._executor = tool_executor
        self._allowed_tools = set(allowed_tools)
        self._session_id = uuid.uuid4().hex[:12]
        self._initialized = False

    async def handle_json(self, raw: str) -> str:
        """Handle a raw JSON-RPC string, return a JSON response string."""
        try:
            request = JsonRpcRequest.from_json(raw)
            response = await self.handle_request(request)
            return json.dumps(response)
        except json.JSONDecodeError as exc:
            return json.dumps(
                JsonRpcResponse.error(None, -32700, f"Parse error: {exc}")
            )

    async def handle_request(self, request: JsonRpcRequest) -> dict[str, Any]:
        """Route a JSON-RPC request to the appropriate handler."""
        method = request.method

        await logger.info(
            "mcp.request",
            agent=self._agent_name,
            method=method,
            request_id=request.id,
        )

        if method == self.METHOD_INITIALIZE:
            return await self._handle_initialize(request)
        elif method == self.METHOD_LIST_TOOLS:
            return await self._handle_list_tools(request)
        elif method == self.METHOD_CALL_TOOL:
            return await self._handle_call_tool(request)
        elif method == self.METHOD_PING:
            return JsonRpcResponse.success(request.id, {"status": "pong"})
        else:
            return JsonRpcResponse.error(
                request.id, -32601, f"Method not found: {method}",
            )

    async def _handle_initialize(
        self, request: JsonRpcRequest,
    ) -> dict[str, Any]:
        """Handle MCP initialize handshake."""
        self._initialized = True
        return JsonRpcResponse.success(request.id, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": f"swarm-mcp-{self._agent_name}",
                "version": "0.1.0",
            },
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "sessionId": self._session_id,
        })

    async def _handle_list_tools(
        self, request: JsonRpcRequest,
    ) -> dict[str, Any]:
        """Return the list of available tools."""
        descriptors = self._executor.get_tool_descriptors()
        tools = [
            {
                "name": d.name,
                "description": d.description,
                "inputSchema": d.parameters,
            }
            for d in descriptors
            if d.name in self._allowed_tools
        ]
        return JsonRpcResponse.success(request.id, {"tools": tools})

    async def _handle_call_tool(
        self, request: JsonRpcRequest,
    ) -> dict[str, Any]:
        """Execute a tool call and return the result."""
        tool_name = request.params.get("name", "")
        arguments = request.params.get("arguments", {})

        # Enforce whitelist
        if tool_name not in self._allowed_tools:
            return JsonRpcResponse.error(
                request.id,
                -32602,
                f"Tool '{tool_name}' is not permitted for agent '{self._agent_name}'.",
            )

        # Execute
        result = await self._executor.execute(tool_name, arguments)

        # Format as MCP content block
        content = [
            {
                "type": "text",
                "text": result.output if result.success else result.error,
            }
        ]

        return JsonRpcResponse.success(request.id, {
            "content": content,
            "isError": not result.success,
        })


# ---------------------------------------------------------------------------
# Transport adapters
# ---------------------------------------------------------------------------


class StdioTransport:
    """STDIO transport — reads JSON-RPC from stdin, writes to stdout.

    Used with ``docker exec -i <container> <cmd>`` to pipe messages
    between the host MCP server and the containerised agent.
    """

    def __init__(self, server: MCPServer) -> None:
        self._server = server
        self._running = False

    async def run(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Run the STDIO message loop."""
        self._running = True

        while self._running:
            try:
                # Read a Content-Length header + body (LSP-style framing)
                header = await reader.readline()
                if not header:
                    break

                header_str = header.decode("utf-8").strip()

                # Handle Content-Length framing
                if header_str.startswith("Content-Length:"):
                    length = int(header_str.split(":")[1].strip())
                    await reader.readline()  # empty separator line
                    body = await reader.readexactly(length)
                    raw = body.decode("utf-8")
                else:
                    # Simple newline-delimited JSON fallback
                    raw = header_str

                if not raw:
                    continue

                response = await self._server.handle_json(raw)

                # Write with Content-Length framing
                response_bytes = response.encode("utf-8")
                frame = (
                    f"Content-Length: {len(response_bytes)}\r\n"
                    f"\r\n"
                ).encode("utf-8") + response_bytes

                writer.write(frame)
                await writer.drain()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                await logger.error("mcp.stdio_error", error=str(exc))
                break

        self._running = False

    def stop(self) -> None:
        self._running = False


class SSETransport:
    """SSE transport — HTTP server with Server-Sent Events.

    Exposes two endpoints:
    * ``GET /sse``       — SSE stream for server→client messages
    * ``POST /messages`` — Client→server JSON-RPC requests

    Uses the lightweight ``starlette`` ASGI framework (already a
    dependency of ``uvicorn``).
    """

    def __init__(
        self,
        server: MCPServer,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self._server = server
        self._host = host
        self._port = port
        self._message_queues: dict[str, asyncio.Queue[str]] = {}
        self._app: Any = None
        self._uvicorn_server: Any = None

    def build_app(self) -> Any:
        """Build the Starlette ASGI app."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from sse_starlette.sse import EventSourceResponse

        async def sse_endpoint(request: Request) -> EventSourceResponse:
            """SSE stream endpoint — sends server events to the client."""
            client_id = uuid.uuid4().hex[:8]
            queue: asyncio.Queue[str] = asyncio.Queue()
            self._message_queues[client_id] = queue

            async def event_generator():
                # Send initial connection event with message endpoint
                yield {
                    "event": "endpoint",
                    "data": f"/messages?session_id={client_id}",
                }
                try:
                    while True:
                        msg = await queue.get()
                        yield {"event": "message", "data": msg}
                except asyncio.CancelledError:
                    pass
                finally:
                    self._message_queues.pop(client_id, None)

            return EventSourceResponse(event_generator())

        async def messages_endpoint(request: Request) -> JSONResponse:
            """POST endpoint for incoming JSON-RPC requests."""
            session_id = request.query_params.get("session_id", "")
            body = await request.body()
            raw = body.decode("utf-8")

            response_str = await self._server.handle_json(raw)

            # Push response to SSE stream
            queue = self._message_queues.get(session_id)
            if queue:
                await queue.put(response_str)

            return JSONResponse(
                content=json.loads(response_str),
                status_code=200,
            )

        async def health_endpoint(_: Request) -> JSONResponse:
            return JSONResponse({"status": "ok", "agent": self._server._agent_name})

        routes = [
            Route("/sse", sse_endpoint),
            Route("/messages", messages_endpoint, methods=["POST"]),
            Route("/health", health_endpoint),
        ]

        self._app = Starlette(routes=routes)
        return self._app

    async def start(self) -> None:
        """Start the SSE server (non-blocking)."""
        import uvicorn

        app = self.build_app()
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)

        await logger.info(
            "mcp.sse_starting",
            host=self._host,
            port=self._port,
        )

        # Run in background task
        asyncio.create_task(self._uvicorn_server.serve())

    async def stop(self) -> None:
        """Stop the SSE server."""
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            await logger.info("mcp.sse_stopped", port=self._port)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


class MCPServerFactory:
    """Factory for creating per-agent MCP servers with the right transport.

    Usage::

        factory = MCPServerFactory(
            workspace=config.workspace,
            event_bus=bus,
            compressor=compressor,
        )

        # Create a server for an agent
        server, transport = await factory.create(agent_config, sandbox_info)

        # Start the transport (SSE or STDIO)
        await transport.start()  # or transport.run() for STDIO
    """

    def __init__(
        self,
        *,
        workspace: Path,
        event_bus: EventBus | None = None,
        compressor: Any | None = None,
    ) -> None:
        self._workspace = workspace
        self._event_bus = event_bus
        self._compressor = compressor
        self._servers: dict[str, MCPServer] = {}

    async def create(
        self,
        agent_config: AgentConfig,
        *,
        sandbox_exec: Any | None = None,
        host_port: int | None = None,
    ) -> tuple[MCPServer, SSETransport | StdioTransport]:
        """Create an MCP server and transport for an agent.

        Parameters
        ----------
        agent_config:
            The agent's configuration.
        sandbox_exec:
            Optional callable for executing commands in the container.
        host_port:
            Host port for SSE transport (from SandboxManager port allocation).

        Returns
        -------
        (server, transport)
            The MCPServer and its transport adapter.
        """
        # Create tool executor
        executor = ToolExecutor(
            workspace=self._workspace,
            event_bus=self._event_bus,
            compressor=self._compressor,
            sandbox_exec=sandbox_exec,
            allowed_tools=agent_config.mcp.allowed_tools,
            blocked_commands=agent_config.mcp.blocked_commands,
            agent_name=agent_config.name,
        )

        # Create server
        server = MCPServer(
            agent_name=agent_config.name,
            tool_executor=executor,
            allowed_tools=agent_config.mcp.allowed_tools,
        )
        self._servers[agent_config.name] = server

        # Create transport
        if agent_config.mcp.transport == MCPTransport.SSE:
            port = host_port or agent_config.mcp.port
            transport = SSETransport(server, port=port)
        else:
            transport = StdioTransport(server)

        await logger.info(
            "mcp.server_created",
            agent=agent_config.name,
            transport=agent_config.mcp.transport.value,
            allowed_tools=agent_config.mcp.allowed_tools,
        )

        return server, transport

    def get_server(self, agent_name: str) -> MCPServer | None:
        """Get an existing server by agent name."""
        return self._servers.get(agent_name)

    async def shutdown_all(self) -> None:
        """Clean up all servers."""
        self._servers.clear()
        await logger.info("mcp.all_servers_shutdown")
