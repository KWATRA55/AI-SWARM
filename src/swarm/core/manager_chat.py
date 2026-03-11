"""ManagerChat — human ↔ pro model chat interface for the dashboard.

Provides a conversational interface where the user talks to the Manager
(architect/pro model) on the dashboard. The Manager can:

1. Scan the existing codebase and suggest features/bugs/improvements
2. Assign tasks to sub-agents dynamically via the orchestrator
3. Ask the user clarifying questions
4. Report progress from sub-agents

Architecture::

    ┌──────────────────────┐
    │  Dashboard Chat UI   │
    │  (WebSocket)         │
    └──────────┬───────────┘
               │  POST /api/chat
    ┌──────────▼───────────┐
    │  ManagerChat         │
    │  ├─ litellm (Pro)    │
    │  ├─ codebase tools   │
    │  └─ task dispatch    │
    └──────────┬───────────┘
               │  dispatch_dynamic_task()
    ┌──────────▼───────────┐
    │  Orchestrator        │
    │  (runs sub-agents)   │
    └──────────────────────┘
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)


class ManagerChat:
    """Conversational interface to the Manager (pro model).

    The Manager has access to the workspace, can read files, and can
    dispatch tasks to sub-agents via the orchestrator.

    Usage::

        chat = ManagerChat(
            model="gemini/gemini-2.5-pro",
            workspace=Path("/tmp/screener-mvp"),
        )

        response = await chat.send_message("What needs improvement?")
    """

    SYSTEM_PROMPT = """You are the **Manager** of an AI engineering team. You communicate with the human team lead through this chat.

**Your capabilities:**
1. **Codebase Analysis**: You can scan and read any file in the workspace to understand the current state of the project.
2. **Task Assignment**: You can assign tasks to your sub-agents (backend-dev, frontend-dev, db-engineer, qa-engineer) by calling the dispatch function.
3. **Progress Monitoring**: You can track what each agent has done and report progress.
4. **Suggestions**: Proactively suggest features, optimizations, bug fixes, and architectural improvements.

**Communication style:**
- Be concise and direct — you're a principal engineer
- Use markdown formatting for readability
- When suggesting work, break it into actionable tasks
- When the user asks you to do something, confirm the plan first, then execute

**Available tools:**
- scan_codebase() — get a full directory tree with file sizes
- read_file(path) — read a specific file
- dispatch_task(agent, task_description) — assign work to a sub-agent
- get_agent_status() — check what agents are doing

**Current workspace:** {workspace}

Always be proactive. If the user says "go ahead", start dispatching tasks immediately."""

    def __init__(
        self,
        *,
        model: str = "gemini/gemini-2.5-pro",
        workspace: Path,
        dispatch_fn: Callable[..., Coroutine] | None = None,
        dashboard_cb: Callable[..., Coroutine] | None = None,
    ) -> None:
        self._model = model
        self._workspace = workspace
        self._dispatch_fn = dispatch_fn
        self._dashboard_cb = dashboard_cb

        # Conversation history
        self._messages: list[dict[str, str]] = [{
            "role": "system",
            "content": self.SYSTEM_PROMPT.format(workspace=workspace),
        }]
        self._message_count = 0

    def set_dispatch_fn(self, fn: Callable[..., Coroutine]) -> None:
        """Set the function to dispatch tasks to sub-agents."""
        self._dispatch_fn = fn

    async def send_message(self, user_message: str) -> str:
        """Send a user message and get the manager's response.

        The manager can use tools (codebase scan, file reads) to
        ground its response in the actual project state.
        """
        import litellm

        self._message_count += 1
        self._messages.append({"role": "user", "content": user_message})

        # Include codebase context if this is the first message or user asks for scan
        scan_keywords = {"scan", "look", "analyze", "check", "review", "what", "suggest", "improve", "feature"}
        should_scan = (
            self._message_count <= 2
            or any(k in user_message.lower() for k in scan_keywords)
        )

        if should_scan:
            codebase_ctx = await self._scan_codebase()
            self._messages.append({
                "role": "system",
                "content": f"--- CODEBASE SNAPSHOT ---\n{codebase_ctx}\n--- END SNAPSHOT ---",
            })

        # Define tools the manager can call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file from workspace root",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dispatch_task",
                    "description": "Assign a task to a sub-agent. Available agents: backend-dev, frontend-dev, db-engineer, qa-engineer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "Name of the agent to assign the task to",
                            },
                            "task_description": {
                                "type": "string",
                                "description": "Detailed task description for the agent",
                            },
                        },
                        "required": ["agent", "task_description"],
                    },
                },
            },
        ]

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=self._messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as exc:
            error_msg = f"❌ Manager LLM error: {exc}"
            await logger.error("manager_chat.llm_failed", error=str(exc))
            return error_msg

        choice = response.choices[0]
        message = choice.message

        # Handle tool calls
        if message.tool_calls:
            self._messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                import json
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                result = await self._execute_tool(fn_name, args)
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Get the final response after tool use
            try:
                response2 = await litellm.acompletion(
                    model=self._model,
                    messages=self._messages,
                    temperature=0.3,
                )
                content = response2.choices[0].message.content or ""
            except Exception as exc:
                content = f"Manager processed tools but follow-up failed: {exc}"
        else:
            content = message.content or ""

        self._messages.append({"role": "assistant", "content": content})

        # Broadcast to dashboard
        if self._dashboard_cb:
            try:
                await self._dashboard_cb("manager_chat", "response", {
                    "message": content[:500],
                })
            except Exception:
                pass

        return content

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a manager tool call."""
        if name == "read_file":
            return await self._read_file(args.get("path", ""))
        elif name == "dispatch_task":
            return await self._dispatch_task(
                args.get("agent", ""),
                args.get("task_description", ""),
            )
        return f"Unknown tool: {name}"

    async def _scan_codebase(self) -> str:
        """Scan the workspace and return a directory tree."""
        lines: list[str] = []
        file_count = 0

        for root, dirs, files in os.walk(self._workspace):
            dirs[:] = [
                d for d in dirs
                if d not in {'.git', '.venv', 'node_modules', '__pycache__',
                             '.swarm_memory', '.next'}
            ]
            level = root.replace(str(self._workspace), '').count(os.sep)
            indent = '  ' * level
            basename = os.path.basename(root) or str(self._workspace)
            lines.append(f"{indent}{basename}/")

            sub_indent = '  ' * (level + 1)
            for f in sorted(files):
                if f.startswith('.'):
                    continue
                filepath = os.path.join(root, f)
                try:
                    size = os.path.getsize(filepath)
                    lines.append(f"{sub_indent}{f}  ({size:,} bytes)")
                    file_count += 1
                except OSError:
                    lines.append(f"{sub_indent}{f}")
                    file_count += 1

                if file_count > 150:
                    break
            if file_count > 150:
                lines.append("  ... (truncated at 150 files)")
                break

        return '\n'.join(lines) if lines else '(empty workspace)'

    async def _read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        full_path = self._workspace / path
        try:
            if not full_path.exists():
                return f"File not found: {path}"
            content = full_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n... (truncated)"
            return content
        except Exception as exc:
            return f"Error reading {path}: {exc}"

    async def _dispatch_task(self, agent: str, task: str) -> str:
        """Dispatch a task to a sub-agent."""
        if not self._dispatch_fn:
            return "Task dispatch not available — orchestrator not connected"

        await logger.info(
            "manager_chat.dispatch_task",
            agent=agent,
            task=task[:100],
        )

        try:
            asyncio.create_task(self._dispatch_fn(agent, task))
            return f"✅ Task dispatched to {agent}: {task[:200]}"
        except Exception as exc:
            return f"❌ Failed to dispatch to {agent}: {exc}"

    @property
    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return self._message_count
