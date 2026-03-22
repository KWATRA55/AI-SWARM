"""ManagerChat — smart command-and-control chat interface for the swarm.

The Manager (pro model) has full visibility and control over all sub-agents:

1. **Inspect** — see agent status, token budgets, files written, progress
2. **Delegate** — dispatch tasks with exact file targets and acceptance criteria
3. **Review** — read what agents wrote and validate quality
4. **Re-delegate** — reassign failed/poor tasks to different agents
5. **Report** — live session metrics for the user

Architecture::

    ┌──────────────────────┐
    │  Dashboard Chat UI   │
    │  (WebSocket)         │
    └──────────┬───────────┘
               │  POST /api/chat
    ┌──────────▼───────────┐
    │  ManagerChat         │
    │  ├─ 8 smart tools    │
    │  ├─ intent guard     │
    │  └─ task validation  │
    └──────────┬───────────┘
               │  dispatch_dynamic_task()
    ┌──────────▼───────────┐
    │  Orchestrator        │
    │  (runs sub-agents)   │
    └──────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)


class ManagerChat:
    """Smart command-and-control interface for the swarm.

    V2.6: Upgraded from 3 basic tools to 8 smart tools with full
    agent visibility, output review, and re-delegation capabilities.
    """

    SYSTEM_PROMPT = """You are the **Manager / Lead Architect** of an AI engineering team.

## YOUR ROLE
You are the single point of control for a multi-agent swarm. YOU decide:
- Which agents get which tasks
- The exact files each agent must create/modify
- When to stop, redirect, or re-assign work
- Quality of output — you validate what agents produce

## INTENT CLASSIFICATION — MANDATORY
Before responding, classify the user's message:

### STATUS (query / report / explanation)
Trigger: "what's happening", "status", "update", "show me", any question.
Allowed tools: get_agent_status(), get_session_summary(), review_agent_output()
FORBIDDEN: dispatch_task()
Response: Data-driven answer using tool results.

### ACTION (explicit build / fix / create command)
Trigger: "build", "create", "fix", "implement", "add", "remove", "refactor" + specific target.
Allowed tools: dispatch_task(), get_token_budgets(), read_file()
Response: Confirm which agents got which tasks with file targets.

### REVIEW (check quality / validate)
Trigger: "review", "check what X wrote", "is it good", "validate".
Allowed tools: review_agent_output(), read_file(), get_agent_status()
Response: Quality assessment with specific findings.

### AMBIGUOUS (greeting, vague, unclear)
Trigger: "hello", "ok", "yes", "go ahead".
FORBIDDEN: dispatch_task()
Response: Ask what they'd like to work on.

## DISPATCH RULES
1. **ALWAYS check token budgets** before dispatching (use get_token_budgets).
2. **ALWAYS specify target_files** in dispatch_task — agents MUST know exactly which files to write.
3. Tasks must be **specific and actionable** — include:
   - Exact file paths to create/modify
   - What each file should contain
   - Clear completion criteria
4. **Batch all dispatches** in one response — don't dispatch one at a time.
5. After agents complete, use **review_agent_output** to confirm what was built.
6. **ACCEPT completed work** unless there are ACTUAL ERRORS (syntax errors, missing files, wrong language, crash-causing bugs). Minor style preferences, naming conventions, or "could be better" suggestions are NOT reasons to redispatch. The agents are competent — trust their output.
7. **NEVER redispatch** just to "improve" or "correct" working code. Only redispatch if the agent clearly FAILED (status=error, no files written, completely wrong task).
8. **NEVER dispatch multiple agents to write the SAME files.** Each file must be owned by exactly ONE agent. Split work by file ownership, not by feature. Example:
   - ✅ GOOD: agent-A writes models.py, agent-B writes api.py (no overlap)
   - ❌ BAD: agent-A writes models.py + api.py, agent-B also writes models.py + api.py (overlap = wasted tokens)
9. **Prefer fewer agents** — dispatch 1 agent for simple tasks. Only dispatch 2+ when work is clearly separable into non-overlapping file sets.
10. For tasks like "build a REST API", dispatch ONE backend-dev agent. Do NOT also dispatch db-engineer for the same models/schemas.

## QUALITY RULES
- Be concise — no walls of text
- Reference actual file contents and data, not guesses
- When agents complete successfully, give a SHORT confirmation to the user — do NOT re-analyze or nitpick

Workspace: {workspace}"""

    # ---------------------------------------------------------------
    # Tool definitions (8 total)
    # ---------------------------------------------------------------

    @staticmethod
    def _build_tools() -> list[dict]:
        """Build all 8 tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file from workspace root.",
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
                        "Assign a task to a sub-agent. ONLY use for explicit ACTION intents. "
                        "You MUST specify target_files so the agent knows exactly what to write. "
                        "Agent roles: architect (docs, architecture, design), "
                        "backend-dev (backend code, APIs, services), "
                        "frontend-dev (UI, frontend, CSS, React/Next.js), "
                        "db-engineer (database, schemas, migrations, SQL), "
                        "qa-engineer (tests, testing, quality, linting)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "Agent name: architect, backend-dev, frontend-dev, db-engineer, qa-engineer.",
                            },
                            "task_description": {
                                "type": "string",
                                "description": (
                                    "Detailed task description including EXACT file paths to create/modify, "
                                    "what each file should contain, and acceptance criteria."
                                ),
                            },
                            "target_files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of file paths the agent should create or modify.",
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
                    "description": (
                        "Get detailed status of all agents including: status, iteration count, "
                        "token usage, budget remaining, files written, and last activity."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "review_agent_output",
                    "description": (
                        "Review what a specific agent has written by reading its output files. "
                        "Use this to validate quality after an agent completes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "Agent name whose output to review.",
                            },
                        },
                        "required": ["agent"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_token_budgets",
                    "description": (
                        "Get remaining token budgets for all agents. "
                        "Use BEFORE dispatching to avoid sending tasks to agents with no budget left."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "redispatch_task",
                    "description": (
                        "Re-assign a task to a different agent when the original agent "
                        "failed, wrote incorrect output, or ran out of budget."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_agent": {
                                "type": "string",
                                "description": "Original agent who failed the task.",
                            },
                            "to_agent": {
                                "type": "string",
                                "description": "New agent to assign the task to.",
                            },
                            "task_description": {
                                "type": "string",
                                "description": (
                                    "Task description (can be refined from the original). "
                                    "Include what went wrong and what the new agent should do differently."
                                ),
                            },
                        },
                        "required": ["from_agent", "to_agent", "task_description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_session_summary",
                    "description": (
                        "Get live session metrics: total tokens, files written, "
                        "active agents, duration, and efficiency score."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_workspace_files",
                    "description": (
                        "List all files in the workspace with their sizes and last modified times. "
                        "Use to understand the project structure."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Subdirectory to list (default: root). Use '.' for root.",
                                "default": ".",
                            }
                        },
                    },
                },
            },
        ]

    # ---------------------------------------------------------------
    # Init
    # ---------------------------------------------------------------

    def __init__(
        self,
        *,
        model: str = "gemini/gemini-2.5-pro",
        workspace: Path,
        dispatch_fn: Callable[..., Coroutine] | None = None,
        dashboard_cb: Callable[..., Coroutine] | None = None,
        state: Any | None = None,
        ledger: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self._model = model
        self._workspace = workspace
        self._dispatch_fn = dispatch_fn
        self._dashboard_cb = dashboard_cb
        self._state = state
        self._ledger = ledger
        self._config = config   # SwarmConfig — for agent budgets

        # Conversation history
        self._messages: list[dict[str, str]] = [{
            "role": "system",
            "content": self.SYSTEM_PROMPT.format(workspace=workspace),
        }]
        self._message_count = 0
        self._cached_scan: str | None = None
        self._dispatch_blocked = False

        # Track dispatched tasks for review
        self._dispatched_tasks: dict[str, list[dict[str, str]]] = {}
        # Track files written per agent (populated from dashboard events)
        self._agent_files: dict[str, list[str]] = {}

    def set_dispatch_fn(self, fn: Callable[..., Coroutine]) -> None:
        """Set the function to dispatch tasks to sub-agents."""
        self._dispatch_fn = fn

    def record_file_write(self, agent: str, file_path: str) -> None:
        """Record that an agent wrote a file (called from dashboard event handler)."""
        if agent not in self._agent_files:
            self._agent_files[agent] = []
        if file_path not in self._agent_files[agent]:
            self._agent_files[agent].append(file_path)

    # ---------------------------------------------------------------
    # Main send_message
    # ---------------------------------------------------------------

    async def send_message(self, user_message: str) -> str:
        """Send a user message and get the manager's response."""
        import litellm

        self._message_count += 1
        self._messages.append({"role": "user", "content": user_message})

        # Scan codebase on first message or explicit request
        scan_keywords = {"scan", "rescan", "analyze", "review"}
        should_scan = (
            self._message_count == 1
            or any(k in user_message.lower() for k in scan_keywords)
        )

        if should_scan:
            if self._cached_scan is None or "rescan" in user_message.lower():
                await logger.info("manager_chat.scanning_codebase")
                self._cached_scan = await self._scan_codebase()
                await logger.info("manager_chat.scan_complete", ctx_length=len(self._cached_scan))
            self._messages.append({
                "role": "system",
                "content": f"--- CODEBASE SNAPSHOT ---\n{self._cached_scan}\n--- END SNAPSHOT ---",
            })

        # Build tools
        tools = self._build_tools()

        # V2.3: HARD DISPATCH GUARD
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
        _REVIEW_VERBS = {"review", "validate", "inspect", "verify"}

        _is_action = any(verb in _lower for verb in _ACTION_VERBS)
        _is_status = any(pat in _lower for pat in _STATUS_PATTERNS)
        _is_greet = any(_lower.startswith(g) for g in _GREET_PATTERNS)
        _is_review = any(verb in _lower for verb in _REVIEW_VERBS)

        # V2.5: Action verbs ALWAYS take priority.
        if _is_action:
            self._dispatch_blocked = False
        elif _is_review:
            # Reviews can dispatch re-delegation but not new tasks
            dispatch_tools = {"dispatch_task"}
            tools = [t for t in tools if t["function"]["name"] not in dispatch_tools]
            self._dispatch_blocked = True
        elif _is_status or _is_greet or not _is_action:
            dispatch_tools = {"dispatch_task", "redispatch_task"}
            tools = [t for t in tools if t["function"]["name"] not in dispatch_tools]
            self._dispatch_blocked = True
            await logger.info(
                "manager_chat.dispatch_guard_active",
                message_preview=_lower[:80],
                reason="non-action intent detected",
            )
        else:
            self._dispatch_blocked = False

        # Multi-round tool calling loop
        max_rounds = 5
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

                did_review = False
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
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
                    if fn_name == "review_agent_output":
                        did_review = True

                # After reviewing agent output, block further dispatches
                # to prevent the perfection loop (review → redispatch → review...)
                if did_review:
                    tools = [t for t in tools if t["function"]["name"] not in {"dispatch_task", "redispatch_task"}]

                continue
            else:
                content = message.content or ""
                break

        # Force text if all rounds were tool calls
        if not content:
            await logger.warning("manager_chat.forcing_text_response", rounds_exhausted=max_rounds)
            try:
                final_response = await litellm.acompletion(
                    model=self._model,
                    messages=self._messages,
                    tool_choice="none",
                    temperature=0.3,
                )
                content = final_response.choices[0].message.content or ""
            except Exception as exc:
                content = f"I analyzed the codebase but ran out of processing rounds. Error: {exc}"
                await logger.error("manager_chat.force_text_failed", error=str(exc))

        self._messages.append({"role": "assistant", "content": content})
        return content

    # ---------------------------------------------------------------
    # Tool execution router
    # ---------------------------------------------------------------

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a manager tool call."""
        if name == "read_file":
            return await self._read_file(args.get("path", ""))
        elif name == "dispatch_task":
            if self._dispatch_blocked:
                return "❌ Dispatch blocked — classify intent as ACTION first."
            agent = args.get("agent", "")
            valid_agents = {"architect", "backend-dev", "frontend-dev", "db-engineer", "qa-engineer"}
            if agent not in valid_agents:
                return f"❌ Unknown agent '{agent}'. Valid: {', '.join(sorted(valid_agents))}"
            target_files = args.get("target_files", [])
            return await self._dispatch_task(agent, args.get("task_description", ""), target_files)
        elif name == "get_agent_status":
            return await self._get_agent_status()
        elif name == "review_agent_output":
            return await self._review_agent_output(args.get("agent", ""))
        elif name == "get_token_budgets":
            return await self._get_token_budgets()
        elif name == "redispatch_task":
            if self._dispatch_blocked:
                return "❌ Redispatch blocked — only allowed on ACTION intent."
            return await self._redispatch_task(
                args.get("from_agent", ""),
                args.get("to_agent", ""),
                args.get("task_description", ""),
            )
        elif name == "get_session_summary":
            return await self._get_session_summary()
        elif name == "list_workspace_files":
            return await self._list_workspace_files(args.get("directory", "."))
        return f"Unknown tool: {name}"

    # ---------------------------------------------------------------
    # Tool implementations
    # ---------------------------------------------------------------

    async def _scan_codebase(self) -> str:
        """Scan workspace — lightweight file list."""
        lines: list[str] = []
        file_count = 0

        for root, dirs, files in os.walk(self._workspace):
            dirs[:] = [
                d for d in dirs
                if d not in {'.git', '.venv', 'node_modules', '__pycache__',
                             '.swarm_memory', '.next', 'dist', 'build', '.swarm_logs'}
            ]
            level = root.replace(str(self._workspace), '').count(os.sep)
            if level > 3:
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
                if file_count > 50:
                    lines.append("  ... (truncated)")
                    break
            if file_count > 50:
                break

        return '\n'.join(lines) if lines else '(empty workspace)'

    async def _read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        full_path = self._workspace / path
        try:
            if not full_path.exists():
                return f"File not found: {path}"
            content = full_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 4000:
                content = content[:4000] + "\n... (truncated at 4K chars)"
            return content
        except Exception as exc:
            return f"Error reading {path}: {exc}"

    async def _dispatch_task(self, agent: str, task: str, target_files: list[str] | None = None) -> str:
        """Dispatch a task to a sub-agent with file targets."""
        if not self._dispatch_fn:
            return "Task dispatch not available — orchestrator not connected"

        # ── Dispatch-time file ownership validation ──
        # Check if target_files overlap with files already claimed by other agents
        if target_files:
            from swarm.core.compliance import ToolComplianceProxy
            conflicts = []
            for f in target_files:
                owner = ToolComplianceProxy._FILE_OWNERSHIP.get(f)
                if owner and owner != agent:
                    conflicts.append(f"'{f}' (owned by {owner})")
            if conflicts:
                conflict_str = ", ".join(conflicts)
                await logger.warning(
                    "manager_chat.dispatch_overlap_blocked",
                    agent=agent, conflicts=conflicts,
                )
                return (
                    f"⚠️ DISPATCH BLOCKED: Target files overlap with another agent's work: "
                    f"{conflict_str}. Re-plan to give {agent} DIFFERENT files that no other "
                    f"agent is writing. Each file must be owned by exactly ONE agent."
                )

            # Pre-register file ownership so subsequent dispatches see the claim
            for f in target_files:
                if f not in ToolComplianceProxy._FILE_OWNERSHIP:
                    ToolComplianceProxy._FILE_OWNERSHIP[f] = agent

        # Enhance task with file targets
        enhanced_task = task
        if target_files:
            files_str = ", ".join(f"`{f}`" for f in target_files)
            enhanced_task = f"{task}\n\nTARGET FILES: {files_str}\nYou MUST create/modify these specific files."

        await logger.info(
            "manager_chat.dispatch_task",
            agent=agent,
            task=enhanced_task[:100],
            target_files=target_files,
        )

        # Record routing decision
        if self._ledger is not None:
            self._ledger.record_routing_decision(
                agent="manager",
                target_agent=agent,
                task_preview=enhanced_task[:500],
                reason="Manager LLM chose this agent via dispatch_task tool call",
            )

        # Track for review
        if agent not in self._dispatched_tasks:
            self._dispatched_tasks[agent] = []
        self._dispatched_tasks[agent].append({
            "task": task[:200],
            "target_files": ", ".join(target_files or []),
            "dispatched_at": str(int(time.time())),
        })

        try:
            # Wrap dispatch in error-catching task (prevents silent crashes)
            async def _safe_dispatch():
                try:
                    await self._dispatch_fn(agent, enhanced_task)
                except Exception as exc:
                    await logger.error(
                        "manager_chat.dispatch_crashed",
                        agent=agent,
                        error=str(exc)[:200],
                    )
                    if self._dashboard_cb:
                        await self._dashboard_cb("dispatch_error", {
                            "agent": agent,
                            "error": str(exc)[:200],
                        })

            asyncio.create_task(_safe_dispatch())
            files_note = f" (target files: {', '.join(target_files)})" if target_files else ""
            return f"✅ Task dispatched to {agent}{files_note}: {task[:200]}"
        except Exception as exc:
            return f"❌ Failed to dispatch to {agent}: {exc}"

    async def _get_agent_status(self) -> str:
        """Get detailed status of all agents."""
        if self._state is None:
            return "Agent status not available"

        try:
            snapshot = await self._state.snapshot()
            lines: list[str] = []

            for name, agent in snapshot.agents.items():
                status = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
                tokens = agent.token_usage.total_tokens

                # Get budget from config
                budget = 50_000
                if self._config:
                    for a in self._config.agents:
                        if a.name == name:
                            budget = a.token_budget
                            break

                budget_pct = round(tokens / max(budget, 1) * 100, 1)
                files_written = self._agent_files.get(name, [])
                files_str = f" | files: {', '.join(files_written)}" if files_written else ""

                lines.append(
                    f"• {name}: {status} | "
                    f"iter={agent.iterations} | "
                    f"tokens={tokens:,}/{budget:,} ({budget_pct}%)"
                    f"{files_str}"
                )

            return "\n".join(lines) if lines else "No agents registered yet"
        except Exception as exc:
            return f"Error: {exc}"

    async def _review_agent_output(self, agent: str) -> str:
        """Review files written by a specific agent."""
        files = self._agent_files.get(agent, [])

        # Fallback: if no tracked files, scan workspace for recently modified files
        if not files:
            try:
                import time as _t
                cutoff = _t.time() - 300  # Files modified in last 5 minutes
                recent: list[str] = []
                for p in self._workspace.rglob("*"):
                    if p.is_file() and p.stat().st_mtime > cutoff:
                        # Skip hidden dirs, __pycache__, .swarm_*
                        rel = str(p.relative_to(self._workspace))
                        if any(skip in rel for skip in ['__pycache__', '.swarm', '.git', 'node_modules']):
                            continue
                        recent.append(rel)
                if recent:
                    files = sorted(recent)[:10]
            except Exception:
                pass

        if not files:
            return f"Agent '{agent}' has not written any files yet."

        output: list[str] = [f"Files written by {agent}:"]
        for f in files[:5]:  # Max 5 files
            full_path = self._workspace / f
            try:
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    preview = content[:1500]  # 1.5K preview per file
                    output.append(f"\n--- {f} ({len(content)} chars) ---\n{preview}")
                    if len(content) > 1500:
                        output.append("... (truncated)")
                else:
                    output.append(f"\n--- {f}: FILE NOT FOUND ---")
            except Exception as exc:
                output.append(f"\n--- {f}: Error: {exc} ---")

        # Add task context
        tasks = self._dispatched_tasks.get(agent, [])
        if tasks:
            output.append(f"\nTasks given to {agent}:")
            for t in tasks:
                output.append(f"  • {t['task']}")

        return "\n".join(output)

    async def _get_token_budgets(self) -> str:
        """Get remaining token budgets for all agents."""
        if self._state is None:
            return "State not available"

        try:
            snapshot = await self._state.snapshot()
            lines: list[str] = ["Agent Token Budgets:"]

            for name, agent in snapshot.agents.items():
                budget = 50_000
                if self._config:
                    for a in self._config.agents:
                        if a.name == name:
                            budget = a.token_budget
                            break

                used = agent.token_usage.total_tokens
                remaining = max(budget - used, 0)
                status = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)

                bar = "█" * int(used / max(budget, 1) * 10) + "░" * (10 - int(used / max(budget, 1) * 10))
                flag = " ⚠️ LOW" if remaining < budget * 0.2 else ""
                flag = " 🔴 EXCEEDED" if used > budget else flag

                lines.append(
                    f"  {name}: [{bar}] {used:,}/{budget:,} "
                    f"(remaining: {remaining:,}){flag} — {status}"
                )

            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {exc}"

    async def _redispatch_task(self, from_agent: str, to_agent: str, task: str) -> str:
        """Re-assign a task from one agent to another."""
        if not self._dispatch_fn:
            return "Dispatch not available"

        valid_agents = {"architect", "backend-dev", "frontend-dev", "db-engineer", "qa-engineer"}
        if to_agent not in valid_agents:
            return f"❌ Unknown target agent '{to_agent}'."

        # Enhance with context about the failure
        enhanced_task = (
            f"REDISPATCHED TASK (originally assigned to {from_agent}):\n"
            f"{task}\n\n"
            f"NOTE: The previous agent failed this task. "
            f"Focus on getting it right — write the correct files."
        )

        await logger.info(
            "manager_chat.redispatch_task",
            from_agent=from_agent,
            to_agent=to_agent,
            task=task[:100],
        )

        if self._ledger is not None:
            self._ledger.record_routing_decision(
                agent="manager",
                target_agent=to_agent,
                task_preview=enhanced_task[:500],
                reason=f"Manager re-delegated from {from_agent} to {to_agent}",
            )

        try:
            asyncio.create_task(self._dispatch_fn(to_agent, enhanced_task))
            return f"🔄 Task re-delegated from {from_agent} → {to_agent}: {task[:200]}"
        except Exception as exc:
            return f"❌ Redispatch failed: {exc}"

    async def _get_session_summary(self) -> str:
        """Get live session metrics."""
        if self._state is None:
            return "State not available"

        try:
            snapshot = await self._state.snapshot()
            total_tokens = 0
            total_iters = 0
            agent_count = len(snapshot.agents)
            completed = 0
            failed = 0
            active = 0

            for _, agent in snapshot.agents.items():
                total_tokens += agent.token_usage.total_tokens
                total_iters += agent.iterations
                status = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
                if status == "completed":
                    completed += 1
                elif status in ("failed", "error"):
                    failed += 1
                elif status in ("running", "idle"):
                    active += 1

            total_files = sum(len(files) for files in self._agent_files.values())
            efficiency = round(total_tokens / max(total_files, 1))

            return (
                f"📊 Session Summary\n"
                f"  Tokens: {total_tokens:,}\n"
                f"  Iterations: {total_iters}\n"
                f"  Files written: {total_files}\n"
                f"  Agents: {completed} completed, {active} active, {failed} failed\n"
                f"  Efficiency: {efficiency:,} tokens/file {'✅' if efficiency < 10000 else '⚠️'}"
            )
        except Exception as exc:
            return f"Error: {exc}"

    async def _list_workspace_files(self, directory: str = ".") -> str:
        """List files in a workspace subdirectory."""
        target = self._workspace / directory
        if not target.exists() or not target.is_dir():
            return f"Directory not found: {directory}"

        lines: list[str] = []
        count = 0
        for root, dirs, files in os.walk(target):
            dirs[:] = [
                d for d in dirs
                if d not in {'.git', '.venv', 'node_modules', '__pycache__',
                             '.swarm_memory', '.next', 'dist', 'build', '.swarm_logs'}
            ]
            level = root.replace(str(target), '').count(os.sep)
            if level > 2:
                continue
            for f in sorted(files):
                if f.startswith('.'):
                    continue
                rel = os.path.relpath(os.path.join(root, f), self._workspace)
                size = os.path.getsize(os.path.join(root, f))
                lines.append(f"  {rel} ({size:,}b)")
                count += 1
                if count > 30:
                    lines.append("  ... (truncated)")
                    return "\n".join(lines)

        return "\n".join(lines) if lines else "(empty directory)"

    @property
    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return self._message_count
