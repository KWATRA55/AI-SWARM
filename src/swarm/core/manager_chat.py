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

    SYSTEM_PROMPT = """You are the **Manager / Architect** of an AI engineering team.

## INTENT CLASSIFICATION — MANDATORY
Before responding, classify the user's message into ONE of these categories:

### STATUS (query / report / explanation)
Trigger: user asks "what's happening", "status", "update", "how's it going", "what did X do", "show me", or any question.
Allowed tools: get_agent_status(), read_file()
FORBIDDEN: dispatch_task() — NEVER dispatch on a status query.
Response: conversational text with data from tools.

### ACTION (explicit build / fix / create command)
Trigger: user says "build", "create", "fix", "implement", "deploy", "run", "start", "add", "remove", "refactor" + a specific target.
Allowed tools: dispatch_task(), read_file()
Response: confirm which agents got which tasks.

### AMBIGUOUS (greeting, vague, unclear)
Trigger: "hello", "hey", "go ahead", "yes", "ok", or anything without a clear action target.
Allowed tools: get_agent_status() (optional)
FORBIDDEN: dispatch_task()
Response: ask what they'd like to work on, or summarize current state.

## RULES
1. You MUST classify intent BEFORE choosing tools.
2. NEVER dispatch agents unless classification is ACTION with an explicit target.
3. After agents complete, WAIT for human "go" / "next" before dispatching more.
4. When dispatching, batch ALL agents in one response.
5. Be concise — no walls of text.

Workspace: {workspace}"""


    def __init__(
        self,
        *,
        model: str = "gemini/gemini-2.5-pro",
        workspace: Path,
        dispatch_fn: Callable[..., Coroutine] | None = None,
        dashboard_cb: Callable[..., Coroutine] | None = None,
        state: Any | None = None,
        ledger: Any | None = None,
    ) -> None:
        self._model = model
        self._workspace = workspace
        self._dispatch_fn = dispatch_fn
        self._dashboard_cb = dashboard_cb
        self._state = state  # SwarmState — for agent status lookups
        self._ledger = ledger  # SessionLedger (optional)

        # Conversation history
        self._messages: list[dict[str, str]] = [{
            "role": "system",
            "content": self.SYSTEM_PROMPT.format(workspace=workspace),
        }]
        self._message_count = 0
        self._cached_scan: str | None = None  # Cache codebase scan
        self._dispatch_blocked = False  # V2.3: Hard dispatch guard flag

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

        # Only scan on first message or explicit scan keywords (not every msg)
        scan_keywords = {"scan", "rescan", "analyze", "review"}
        should_scan = (
            self._message_count == 1  # Only first msg, not first 2
            or any(k in user_message.lower() for k in scan_keywords)
        )

        if should_scan:
            if self._cached_scan is None or any(k in user_message.lower() for k in {"rescan"}):
                await logger.info("manager_chat.scanning_codebase")
                self._cached_scan = await self._scan_codebase()
                await logger.info("manager_chat.scan_complete", ctx_length=len(self._cached_scan))
            self._messages.append({
                "role": "system",
                "content": f"--- CODEBASE SNAPSHOT ---\n{self._cached_scan}\n--- END SNAPSHOT ---",
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
                    "description": (
                        "Assign a task to a sub-agent. ONLY use when user explicitly requests building/creating/fixing. "
                        "NEVER for status queries. Agent roles: "
                        "architect (docs, architecture, project structure, HEALTH_CHECK, README), "
                        "backend-dev (backend code, APIs, services), "
                        "frontend-dev (UI, frontend code, CSS), "
                        "db-engineer (database, schemas, migrations), "
                        "qa-engineer (tests, testing, quality). "
                        "For documentation/file-structure tasks, always use architect."
                    ),
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
            {
                "type": "function",
                "function": {
                    "name": "get_agent_status",
                    "description": "Get the current status of all agents — running, completed, failed, idle",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]

        # V2.3: HARD DISPATCH GUARD — programmatic intent classification.
        # If the message looks like a status query or greeting, physically
        # remove dispatch_task from tools. The LLM cannot hallucinate a
        # tool call if the tool doesn't exist in the API payload.
        _lower = user_message.lower().strip()
        _STATUS_PATTERNS = {
            "status", "update", "done", "progress", "how", "what",
            "show", "check", "report", "?",
        }
        _GREET_PATTERNS = {"hello", "hi", "hey", "yo", "sup", "ok", "yes", "no", "thanks"}
        _ACTION_VERBS = {
            "build", "create", "fix", "implement", "deploy", "run",
            "start", "add", "remove", "refactor", "write", "make",
            "generate", "setup", "install", "configure", "test",
        }
        _is_action = any(verb in _lower for verb in _ACTION_VERBS)
        _is_status = any(pat in _lower for pat in _STATUS_PATTERNS)
        _is_greet = any(_lower.startswith(g) for g in _GREET_PATTERNS)

        # V2.5: Action verbs ALWAYS take priority.
        # Previously, "create a FastAPI app returning {\"status\":\"ok\"}" was
        # blocked because "status" matched _STATUS_PATTERNS even though
        # "create" matched _ACTION_VERBS.
        if _is_action:
            # Has an action verb → allow dispatch, period.
            self._dispatch_blocked = False
        elif _is_status or _is_greet or not _is_action:
            # No action verbs AND looks like a question/greeting → block
            tools = [t for t in tools if t["function"]["name"] != "dispatch_task"]
            self._dispatch_blocked = True
            await logger.info(
                "manager_chat.dispatch_guard_active",
                message_preview=_lower[:80],
                reason="non-action intent detected",
            )
        else:
            self._dispatch_blocked = False

        # Multi-round tool calling loop
        max_rounds = 5  # V2: was 10 — manager shouldn't need more than 5
        content = ""
        for round_num in range(max_rounds):
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
                await logger.error("manager_chat.llm_failed", error=str(exc), round=round_num)
                return error_msg

            choice = response.choices[0]
            message = choice.message

            await logger.info(
                "manager_chat.llm_response",
                round=round_num,
                has_tool_calls=bool(message.tool_calls),
                content_length=len(message.content or ""),
            )

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
                    await logger.info(
                        "manager_chat.tool_executed",
                        tool=fn_name,
                        result_length=len(result),
                    )
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                # Continue loop — LLM needs to process tool results
                continue
            else:
                content = message.content or ""
                break

        # If all rounds were tool calls, force a final text response
        if not content:
            await logger.warning(
                "manager_chat.forcing_text_response",
                rounds_exhausted=max_rounds,
            )
            try:
                final_response = await litellm.acompletion(
                    model=self._model,
                    messages=self._messages,
                    tool_choice="none",  # Force text, no tools
                    temperature=0.3,
                )
                content = final_response.choices[0].message.content or ""
            except Exception as exc:
                content = f"I analyzed the codebase but ran out of processing rounds. Error: {exc}"
                await logger.error("manager_chat.force_text_failed", error=str(exc))

        self._messages.append({"role": "assistant", "content": content})

        # NOTE: Do NOT broadcast here — dashboard.py handles broadcasting
        # the manager response after calling send_message(). Broadcasting
        # here too causes duplicate messages in the chat.

        return content

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a manager tool call."""
        if name == "read_file":
            return await self._read_file(args.get("path", ""))
        elif name == "dispatch_task":
            # V2.3: Hard dispatch guard — reject if guard was active
            if self._dispatch_blocked:
                return "❌ Dispatch blocked — this was a status/greeting query, not an action command. Reply with text instead."
            agent = args.get("agent", "")
            # V2.3: Validate agent name
            valid_agents = {"architect", "backend-dev", "frontend-dev", "db-engineer", "qa-engineer"}
            if agent not in valid_agents:
                return f"❌ Unknown agent '{agent}'. Valid agents: {', '.join(sorted(valid_agents))}"
            return await self._dispatch_task(agent, args.get("task_description", ""))
        elif name == "get_agent_status":
            return await self._get_agent_status()
        return f"Unknown tool: {name}"

    async def _scan_codebase(self) -> str:
        """Scan the workspace — lightweight file list (no sizes to save tokens)."""
        lines: list[str] = []
        file_count = 0

        for root, dirs, files in os.walk(self._workspace):
            dirs[:] = [
                d for d in dirs
                if d not in {'.git', '.venv', 'node_modules', '__pycache__',
                             '.swarm_memory', '.next', 'dist', 'build'}
            ]
            level = root.replace(str(self._workspace), '').count(os.sep)
            if level > 3:  # Max depth 3 to keep scan lean
                continue
            indent = '  ' * level
            basename = os.path.basename(root) or str(self._workspace)
            lines.append(f"{indent}{basename}/")

            sub_indent = '  ' * (level + 1)
            for f in sorted(files):
                if f.startswith('.'):
                    continue
                lines.append(f"{sub_indent}{f}")
                file_count += 1

                if file_count > 30:  # Strict cap (was 150 — massive token waste)
                    lines.append("  ... (truncated at 30 files)")
                    break
            if file_count > 30:
                break

        return '\n'.join(lines) if lines else '(empty workspace)'

    async def _read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        full_path = self._workspace / path
        try:
            if not full_path.exists():
                return f"File not found: {path}"
            content = full_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 4000:  # V2: was 8K — manager doesn't need full files
                content = content[:4000] + "\n... (truncated at 4K chars)"
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

        # --- Session Ledger: record routing decision ---
        if self._ledger is not None:
            self._ledger.record_routing_decision(
                agent="manager",
                target_agent=agent,
                task_preview=task[:500],
                reason="Manager LLM chose this agent via dispatch_task tool call",
            )

        try:
            asyncio.create_task(self._dispatch_fn(agent, task))
            return f"✅ Task dispatched to {agent}: {task[:200]}"
        except Exception as exc:
            return f"❌ Failed to dispatch to {agent}: {exc}"

    async def _get_agent_status(self) -> str:
        """Get current status of all agents from SwarmState."""
        if self._state is None:
            return "Agent status not available — state not connected"

        try:
            snapshot = await self._state.snapshot()
            lines: list[str] = []
            for name, agent in snapshot.agents.items():
                status = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
                lines.append(
                    f"• {name}: {status} | "
                    f"iter={agent.iterations} | "
                    f"tokens={agent.token_usage.total_tokens}"
                )
            return "\n".join(lines) if lines else "No agents registered yet"
        except Exception as exc:
            return f"Error getting agent status: {exc}"

    @property
    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return self._message_count
