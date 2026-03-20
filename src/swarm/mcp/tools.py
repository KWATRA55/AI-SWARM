"""MCP tool definitions — the capabilities exposed to sandboxed agents.

Each tool is defined as a Pydantic schema (for input validation and
OpenAI-compatible function calling) paired with an async executor function.

Architecture
------------

.. code-block:: text

    Agent (inside Docker) ──► MCP Client
                                  │
                          (SSE or STDIO)
                                  │
                                  ▼
    Host ──► MCP Server ──► ToolRegistry
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                 ▼
            read_file        write_file       run_command
          (direct I/O)    (Event Bus lock)   (docker exec)
                                  │
                                  ▼
                          FILE_WRITE_REQUEST
                          (Redis Stream)
                                  │
                                  ▼
                          FileWriteHandler
                          (acquires lock → writes → releases)

CRITICAL: ``write_file`` does NOT write directly to disk.
It publishes a ``FILE_WRITE_REQUEST`` event to the Redis Event Bus so that
concurrent agent writes are safely serialised through the distributed lock.

Tool list
---------
* ``read_file``      — Read file contents from the shared workspace.
* ``write_file``     — Write/create a file (via Event Bus lock).
* ``list_directory``  — List contents of a directory.
* ``run_command``    — Execute a shell command in the agent's sandbox.
* ``git_diff``       — Show unstaged changes in the workspace.
* ``git_commit``     — Stage and commit changes.
* ``search_memory``  — Search long-term memory for relevant knowledge.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from swarm.events.bus import EventBus, SwarmEvent
from swarm.events.channels import EventType, StreamChannel

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool input/output schemas (OpenAI function-calling compatible)
# ---------------------------------------------------------------------------


class ReadFileInput(BaseModel):
    """Input schema for the read_file tool."""
    path: str = Field(description="Relative path to the file within the workspace.")
    start_line: int | None = Field(default=None, description="Optional 1-based start line.")
    end_line: int | None = Field(default=None, description="Optional 1-based end line.")
    structure_only: bool = Field(default=False, description="Return only class/function signatures, not full content. Use for understanding code structure.")


class WriteFileInput(BaseModel):
    """Input schema for the write_file tool."""
    path: str = Field(description="Relative path to the file within the workspace.")
    content: str = Field(description="Full file content to write.")
    create_dirs: bool = Field(default=True, description="Create parent directories if needed.")


class ListDirectoryInput(BaseModel):
    """Input schema for the list_directory tool."""
    path: str = Field(default=".", description="Relative directory path within the workspace.")
    recursive: bool = Field(default=False, description="Whether to list recursively.")
    max_depth: int = Field(default=3, ge=1, le=10, description="Max depth for recursive listing.")


class RunCommandInput(BaseModel):
    """Input schema for the run_command tool."""
    command: str = Field(description="Shell command to execute.")
    timeout: float = Field(default=60.0, ge=1.0, le=600.0, description="Timeout in seconds.")
    workdir: str | None = Field(default=None, description="Optional working directory override.")


class GitDiffInput(BaseModel):
    """Input schema for the git_diff tool."""
    path: str | None = Field(default=None, description="Optional specific file/directory to diff.")
    staged: bool = Field(default=False, description="Show staged changes instead of unstaged.")


class GitCommitInput(BaseModel):
    """Input schema for the git_commit tool."""
    message: str = Field(description="Commit message.")
    paths: list[str] = Field(default_factory=lambda: ["."], description="Paths to stage before committing.")


class SearchMemoryInput(BaseModel):
    """Input schema for the search_memory tool."""
    query: str = Field(description="Natural language search query.")
    category: str | None = Field(default=None, description="Optional category filter: skills, bugs, rules, patterns, preferences.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum results to return.")


class ToolResult(BaseModel):
    """Standardised tool execution result."""
    success: bool
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool descriptor (for OpenAI function-calling format)
# ---------------------------------------------------------------------------


class ToolDescriptor(BaseModel):
    """Describes a single MCP tool in OpenAI function-calling format."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(description="JSON Schema for the tool's input.")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# Map of tool name → (descriptor, input_model_class)
def _minimize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip Pydantic bloat from JSON schemas to save ~50% tokens per LLM call.

    Removes $defs, title, default values, and nested title fields that
    Pydantic's model_json_schema() adds but LLMs don't need.
    """
    clean: dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("$defs", "title", "default"):
            continue
        if k == "properties" and isinstance(v, dict):
            clean[k] = {
                pk: {fk: fv for fk, fv in pv.items() if fk not in ("title", "default")}
                for pk, pv in v.items()
            }
        else:
            clean[k] = v
    return clean


TOOL_DEFINITIONS: dict[str, ToolDescriptor] = {
    "read_file": ToolDescriptor(
        name="read_file",
        description="Read a file. Use start_line/end_line for large files.",
        parameters=_minimize_schema(ReadFileInput.model_json_schema()),
    ),
    "write_file": ToolDescriptor(
        name="write_file",
        description="Write/create a file (auto-locked for concurrent safety).",
        parameters=_minimize_schema(WriteFileInput.model_json_schema()),
    ),
    "list_directory": ToolDescriptor(
        name="list_directory",
        description="List directory contents (max 50 entries).",
        parameters=_minimize_schema(ListDirectoryInput.model_json_schema()),
    ),
    "run_command": ToolDescriptor(
        name="run_command",
        description="Run a shell command. Returns stdout and exit code.",
        parameters=_minimize_schema(RunCommandInput.model_json_schema()),
    ),
    "search_memory": ToolDescriptor(
        name="search_memory",
        description="Search team's long-term memory for past knowledge.",
        parameters=_minimize_schema(SearchMemoryInput.model_json_schema()),
    ),
}


# ---------------------------------------------------------------------------
# Tool phases — only send tools the agent needs per iteration
# ---------------------------------------------------------------------------

TOOL_PHASES: dict[str, list[str]] = {
    "planning":  ["list_directory", "read_file", "search_memory", "write_file"],
    "execution": ["read_file", "write_file", "run_command", "search_memory"],
    "all":       list(TOOL_DEFINITIONS.keys()),
}


# ---------------------------------------------------------------------------
# AST-based code summarizer (Phase 3)
# ---------------------------------------------------------------------------


def _ast_summarize(file_path: Path, content: str) -> str | None:
    """Return a compact structural summary of a code file.

    For Python files, uses ``ast.parse`` to extract class names, function
    signatures, and docstrings. For JS/TS, uses regex.
    Falls back to ``None`` if parsing fails (caller reads full content).
    """
    suffix = file_path.suffix.lower()

    if suffix == ".py":
        return _ast_summarize_python(content)
    elif suffix in (".js", ".ts", ".tsx", ".jsx"):
        return _ast_summarize_js(content)
    # For other languages, return first 40 lines preview
    lines = content.splitlines()
    if len(lines) > 40:
        return "\n".join(lines[:40]) + f"\n\n... [{len(lines)} lines total — use start_line/end_line for full content]"
    return None  # File is small, full read is fine


def _ast_summarize_python(content: str) -> str | None:
    """Extract Python signatures via AST."""
    import ast as _ast

    try:
        tree = _ast.parse(content)
    except SyntaxError:
        return None

    lines: list[str] = []
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, _ast.ClassDef):
            bases = ", ".join(_ast.unparse(b) for b in node.bases) if node.bases else ""
            lines.append(f"class {node.name}({bases}):")
            doc = _ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc[:100]}"""')
            for item in node.body:
                if isinstance(item, _ast.FunctionDef | _ast.AsyncFunctionDef):
                    sig = _ast.unparse(item.args) if hasattr(_ast, "unparse") else "..."
                    prefix = "async " if isinstance(item, _ast.AsyncFunctionDef) else ""
                    lines.append(f"    {prefix}def {item.name}({sig})")
        elif isinstance(node, _ast.FunctionDef | _ast.AsyncFunctionDef):
            sig = _ast.unparse(node.args) if hasattr(_ast, "unparse") else "..."
            prefix = "async " if isinstance(node, _ast.AsyncFunctionDef) else ""
            lines.append(f"{prefix}def {node.name}({sig})")
            doc = _ast.get_docstring(node)
            if doc:
                lines.append(f'    """{doc[:80]}"""')
        elif isinstance(node, _ast.Import):
            for alias in node.names:
                lines.append(f"import {alias.name}")
        elif isinstance(node, _ast.ImportFrom):
            names = ", ".join(a.name for a in node.names[:5])
            lines.append(f"from {node.module} import {names}")

    return "\n".join(lines) if lines else None


def _ast_summarize_js(content: str) -> str | None:
    """Extract JS/TS structure via regex."""
    import re

    patterns = [
        r"^export\s+(default\s+)?(async\s+)?(?:function|class|const|interface|type)\s+\w+.*",
        r"^(?:async\s+)?function\s+\w+.*",
        r"^class\s+\w+.*",
        r"^interface\s+\w+.*",
        r"^type\s+\w+\s*=.*",
        r"^import\s+.*from\s+['\"].*['\"]",
    ]
    combined = "|".join(f"({p})" for p in patterns)
    matches = re.findall(combined, content, re.MULTILINE)
    if not matches:
        return None

    lines = [next(g for g in m if g) for m in matches]
    return "\n".join(lines[:30])  # Cap at 30 structural lines


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Executes MCP tools with safety checks and event bus integration.

    The executor is the bridge between LLM tool calls and actual operations.
    It validates inputs, enforces the tool whitelist and blocked commands,
    and routes writes through the event bus for lock-based serialisation.

    Usage::

        executor = ToolExecutor(
            workspace=Path("/workspace"),
            event_bus=bus,
            allowed_tools=["read_file", "write_file", "run_command"],
            blocked_commands=["rm -rf /"],
        )

        result = await executor.execute("read_file", {"path": "src/main.py"})
    """

    def __init__(
        self,
        *,
        workspace: Path,
        event_bus: EventBus | None = None,
        compressor: Any | None = None,
        sandbox_exec: Any | None = None,
        allowed_tools: list[str] | None = None,
        blocked_commands: list[str] | None = None,
        agent_name: str = "unknown",
        secret_manager: Any | None = None,
        rate_limits: dict[str, int] | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        self._event_bus = event_bus
        self._compressor = compressor
        self._sandbox_exec = sandbox_exec  # Callable: async (cmd, timeout) -> (exit_code, output)
        self._allowed = set(allowed_tools or TOOL_DEFINITIONS.keys())
        self._blocked = blocked_commands or []
        self._agent_name = agent_name
        self._secret_manager = secret_manager

        # Rate limiting: tool_name → max calls per minute
        self._rate_limits: dict[str, int] = rate_limits or {}
        # Token bucket internals: tool_name → list of timestamps
        self._call_timestamps: dict[str, list[float]] = {}

        # Dispatch table
        self._handlers: dict[str, Any] = {
            "read_file": self._exec_read_file,
            "write_file": self._exec_write_file,
            "list_directory": self._exec_list_directory,
            "run_command": self._exec_run_command,
            "git_diff": self._exec_git_diff,
            "git_commit": self._exec_git_commit,
            "search_memory": self._exec_search_memory,
        }

    def get_tool_descriptors(self) -> list[ToolDescriptor]:
        """Return descriptors for all allowed tools (for LLM function calling)."""
        return [
            TOOL_DEFINITIONS[name]
            for name in self._allowed
            if name in TOOL_DEFINITIONS
        ]

    def get_openai_tools(self, phase: str = "all") -> list[dict[str, Any]]:
        """Return tools in OpenAI function-calling format for litellm.

        Parameters
        ----------
        phase
            Tool phase: 'planning' (iter 1), 'execution' (iter 2+), or 'all'.
        """
        phase_tools = TOOL_PHASES.get(phase, TOOL_PHASES["all"])
        # Intersect with allowed tools for this agent
        active_tools = [t for t in phase_tools if t in self._allowed and t in TOOL_DEFINITIONS]
        tools = []
        for name in active_tools:
            desc = TOOL_DEFINITIONS[name]
            tools.append({
                "type": "function",
                "function": {
                    "name": desc.name,
                    "description": desc.description,
                    "parameters": desc.parameters,
                },
            })
        return tools

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Includes rate limiting (token bucket per-minute) and PII masking
        of tool output to prevent credential leakage.
        """
        import time as _time

        if tool_name not in self._allowed:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is not in the allowed list for this agent.",
            )

        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: '{tool_name}'.",
            )

        # --- Rate limiting (token bucket) ---
        if tool_name in self._rate_limits:
            max_per_min = self._rate_limits[tool_name]
            now = _time.time()
            stamps = self._call_timestamps.setdefault(tool_name, [])
            # Prune timestamps older than 60s
            stamps[:] = [t for t in stamps if now - t < 60.0]
            if len(stamps) >= max_per_min:
                wait = 60.0 - (now - stamps[0])
                await logger.warning(
                    "tool.rate_limited",
                    tool=tool_name,
                    agent=self._agent_name,
                    wait_seconds=round(wait, 1),
                )
                await asyncio.sleep(max(wait, 0.5))
            stamps.append(_time.time())

        try:
            result = await handler(arguments)
            await logger.info(
                "tool.executed",
                tool=tool_name,
                agent=self._agent_name,
                success=result.success,
            )

            # --- PII masking ---
            if self._secret_manager is not None and result.output:
                result = ToolResult(
                    success=result.success,
                    output=self._secret_manager.mask(result.output),
                    error=result.error,
                    metadata=result.metadata,
                )

            return result
        except Exception as exc:
            await logger.error(
                "tool.execution_failed",
                tool=tool_name,
                agent=self._agent_name,
                error=str(exc),
            )
            return ToolResult(success=False, error=f"Tool execution failed: {exc}")

    # -------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------

    async def _exec_read_file(self, args: dict[str, Any]) -> ToolResult:
        """Read file contents from the workspace.

        When the EventBus is connected, acquires a MRSW read lock so
        that writers are blocked while reading, preventing partial reads.
        """
        params = ReadFileInput.model_validate(args)
        target = self._resolve_path(params.path)

        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {params.path}")
        if not target.is_file():
            return ToolResult(success=False, error=f"Not a file: {params.path}")

        # Acquire read lock (MRSW) if event bus is available
        read_lock = None
        if self._event_bus is not None and self._event_bus.connected:
            try:
                read_lock = self._event_bus.file_read_lock(params.path)
                await read_lock.acquire()
            except Exception:
                read_lock = None  # fall through to unlocked read

        try:
            content = target.read_text(encoding="utf-8")
            total_lines = content.count("\n") + 1

            # --- AST-based structure summary (Phase 3) ---
            if params.structure_only:
                summary = _ast_summarize(target, content)
                if summary:
                    return ToolResult(
                        success=True,
                        output=summary,
                        metadata={"path": str(target), "mode": "structure", "lines": total_lines},
                    )
                # Fall through to normal read if AST fails

            # Apply line range if specified
            if params.start_line or params.end_line:
                lines = content.splitlines(keepends=True)
                start = (params.start_line or 1) - 1  # 0-indexed
                end = params.end_line or len(lines)
                content = "".join(lines[start:end])

            # Cap at 2K chars (V2: down from 4K)
            _MAX = 2_000
            truncated = len(content) > _MAX
            if truncated:
                content = content[:_MAX] + f"\n... [TRUNCATED — {len(content):,} chars total, {total_lines} lines. Use start_line/end_line or structure_only=true.]"

            return ToolResult(
                success=True,
                output=content,
                metadata={"path": str(target), "size": target.stat().st_size, "lines": total_lines},
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error=f"Cannot read binary file: {params.path}",
            )
        finally:
            if read_lock is not None:
                await read_lock.release()

    async def _exec_write_file(self, args: dict[str, Any]) -> ToolResult:
        """Write a file via the Event Bus distributed lock.

        CRITICAL: This does NOT write directly to disk.
        The write request is published to the Redis Event Bus, which
        serialises it through a distributed lock to prevent corruption.

        If no event bus is available (local dev mode), falls back to a
        local asyncio lock for process-level serialisation.
        """
        params = WriteFileInput.model_validate(args)
        target = self._resolve_path(params.path)

        # --- Route through Event Bus (production) ---
        if self._event_bus is not None and self._event_bus.connected:
            request_id = uuid.uuid4().hex[:12]

            # Publish the write request
            event = SwarmEvent(
                channel=StreamChannel.FILE_WRITE_REQUEST.value,
                event_type=EventType.WRITE_REQUEST,
                sender=self._agent_name,
                payload={
                    "file_path": str(target),
                    "content": params.content,
                    "agent": self._agent_name,
                    "request_id": request_id,
                    "create_dirs": params.create_dirs,
                },
            )
            entry_id = await self._event_bus.publish(
                StreamChannel.FILE_WRITE_REQUEST.value, event,
            )

            if entry_id:
                await logger.info(
                    "tool.write_file_queued",
                    agent=self._agent_name,
                    path=params.path,
                    request_id=request_id,
                    entry_id=entry_id,
                )
                return ToolResult(
                    success=True,
                    output=f"Write request queued for '{params.path}' (request_id={request_id}).",
                    metadata={"request_id": request_id, "entry_id": entry_id},
                )
            else:
                # Publish failed — fall through to local write
                await logger.warning(
                    "tool.write_file_publish_failed",
                    agent=self._agent_name,
                    path=params.path,
                    msg="Falling back to direct write with local lock.",
                )

        # --- Fallback: direct write with local lock ---
        return await self._direct_write(target, params.content, params.create_dirs)

    async def _direct_write(
        self, target: Path, content: str, create_dirs: bool,
    ) -> ToolResult:
        """Direct file write with local asyncio lock (fallback mode)."""
        try:
            if create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"File written: {target} ({len(content)} bytes)",
                metadata={"path": str(target), "bytes_written": len(content)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Write failed: {exc}")

    async def _exec_list_directory(self, args: dict[str, Any]) -> ToolResult:
        """List directory contents.

        V2: Strips metadata (byte sizes, inodes) — returns only [DIR]/[FILE]
        prefixed relative paths.  Saves ~40% tokens per listing.
        """
        params = ListDirectoryInput.model_validate(args)
        target = self._resolve_path(params.path)

        if not target.exists():
            return ToolResult(success=False, error=f"Directory not found: {params.path}")
        if not target.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {params.path}")

        entries: list[str] = []

        def _walk(dir_path: Path, depth: int = 0, prefix: str = "") -> None:
            if depth > params.max_depth:
                return
            try:
                items = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
                for item in items:
                    # Skip hidden files and common ignore patterns
                    if item.name.startswith(".") or item.name in {
                        "node_modules", "__pycache__", ".git", ".venv", "venv",
                    }:
                        continue

                    rel = item.resolve().relative_to(self._workspace)
                    if item.is_dir():
                        entries.append(f"[DIR] {rel}")
                        if params.recursive:
                            _walk(item, depth + 1, prefix)
                    else:
                        entries.append(f"[FILE] {rel}")

                    if len(entries) > 50:  # Token-saving cap
                        entries.append("... (truncated at 50 entries)")
                        return
            except PermissionError:
                entries.append(f"{prefix}[permission denied]")

        _walk(target)
        return ToolResult(
            success=True,
            output="\n".join(entries) if entries else "(empty directory)",
            metadata={"count": len(entries)},
        )

    async def _exec_run_command(self, args: dict[str, Any]) -> ToolResult:
        """Execute a shell command."""
        params = RunCommandInput.model_validate(args)

        # Security: check blocked commands
        cmd_lower = params.command.lower()
        for blocked in self._blocked:
            if blocked.lower() in cmd_lower:
                return ToolResult(
                    success=False,
                    error=f"Command blocked by security policy: contains '{blocked}'.",
                )

        # Route through sandbox if available
        if self._sandbox_exec is not None:
            try:
                # Prepend cd if workdir is specified
                cmd = params.command
                if params.workdir:
                    cmd = f"cd {params.workdir} && {cmd}"
                exit_code, output = await self._sandbox_exec(
                    cmd, params.timeout,
                )
                return ToolResult(
                    success=exit_code == 0,
                    output=output,
                    error="" if exit_code == 0 else f"Exit code: {exit_code}",
                    metadata={"exit_code": exit_code},
                )
            except Exception as exc:
                return ToolResult(success=False, error=f"Sandbox exec failed: {exc}")

        # Fallback: local subprocess
        try:
            workdir = params.workdir or str(self._workspace)
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workdir,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=params.timeout,
            )
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            exit_code = proc.returncode or 0

            return ToolResult(
                success=exit_code == 0,
                output=output[-2_000:],  # Cap output (was 10K — major token waste)
                error="" if exit_code == 0 else f"Exit code: {exit_code}",
                metadata={"exit_code": exit_code},
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Command timed out after {params.timeout}s.",
            )

    async def _exec_git_diff(self, args: dict[str, Any]) -> ToolResult:
        """Show git diff."""
        params = GitDiffInput.model_validate(args)
        cmd = "git diff"
        if params.staged:
            cmd += " --staged"
        if params.path:
            cmd += f" -- {params.path}"

        return await self._exec_run_command({"command": cmd, "timeout": 30.0})

    async def _exec_git_commit(self, args: dict[str, Any]) -> ToolResult:
        """Stage and commit files."""
        params = GitCommitInput.model_validate(args)

        # Stage files
        paths_str = " ".join(params.paths)
        stage_result = await self._exec_run_command({
            "command": f"git add {paths_str}",
            "timeout": 30.0,
        })
        if not stage_result.success:
            return stage_result

        # Commit
        safe_msg = params.message.replace('"', '\\"')
        return await self._exec_run_command({
            "command": f'git commit -m "{safe_msg}"',
            "timeout": 30.0,
        })

    async def _exec_search_memory(self, args: dict[str, Any]) -> ToolResult:
        """Search long-term memory."""
        params = SearchMemoryInput.model_validate(args)

        if self._compressor is None:
            return ToolResult(
                success=False,
                error="Long-term memory is not available.",
            )

        try:
            memories = await self._compressor.search_memories(
                params.query,
                category=params.category,
                limit=params.limit,
            )

            if not memories:
                return ToolResult(
                    success=True,
                    output="No relevant memories found.",
                )

            lines: list[str] = []
            for i, mem in enumerate(memories, 1):
                lines.append(f"[{i}] [{mem.category.upper()}] {mem.title}")
                lines.append(f"    {mem.content}")
                if mem.tags:
                    lines.append(f"    Tags: {', '.join(mem.tags)}")
                lines.append(f"    Source: {mem.source_agent} / {mem.source_task}")
                lines.append("")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"count": len(memories)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Memory search failed: {exc}")

    # -------------------------------------------------------------------
    # Path resolution & security
    # -------------------------------------------------------------------

    def _resolve_path(self, relative: str) -> Path:
        """Resolve a relative path against the workspace, preventing traversal."""
        clean = Path(relative).as_posix()
        # Prevent directory traversal
        if ".." in clean.split("/"):
            raise ValueError(f"Path traversal not allowed: {relative}")

        resolved = (self._workspace / clean).resolve()

        # Ensure the resolved path is within the workspace
        try:
            resolved.relative_to(self._workspace)
        except ValueError:
            raise ValueError(
                f"Path '{relative}' resolves outside the workspace: {resolved}"
            )

        return resolved
