"""WorkerAgent — the core agentic execution loop.

This is where the LLM actually thinks, plans, and acts.  Each agent runs a
``litellm``-backed loop that:

1. Receives an enhanced system prompt (with LTM context injected).
2. Sends messages to the LLM, receiving tool call requests.
3. Dispatches tool calls to the MCP ``ToolExecutor``.
4. Handles mid-loop interrupts (schema changes, context updates).
5. Compresses context when it grows too large.
6. Signals task completion or fails gracefully.

Interrupt architecture
----------------------

.. code-block:: text

    WorkerAgent.execute()
         │
         ├─── LLM call ──► response
         │         │
         │         ├── tool_calls? ──► ToolExecutor.execute()
         │         │                     │
         │         │                     ├── read_file  (direct)
         │         │                     ├── write_file (Event Bus lock)
         │         │                     └── run_command (sandbox exec)
         │         │
         │         ├── [TASK_COMPLETE]? ──► done
         │         │
         │         └── continue loop
         │
         ├─── check_interrupts() ──► schema change?
         │         │
         │         └── inject diff into context
         │             force re-evaluation
         │
         └─── check_context_size() ──► compress if over threshold

The interrupt listener runs as a parallel asyncio task that monitors the
Event Bus for schema change broadcasts.  When triggered, it sets a flag
that the main loop checks between iterations.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import structlog

from swarm.config.models import AgentConfig, CompressionConfig
from swarm.core.compressor import Compressor, count_tokens
from swarm.core.state import (
    AgentStatus,
    MessageRole,
    SwarmState,
    TaskStatus,
    TokenUsage,
)
from swarm.events.bus import EventBus, SwarmEvent
from swarm.events.channels import BroadcastChannel
from swarm.mcp.tools import ToolExecutor

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Interrupt context container
# ---------------------------------------------------------------------------


class InterruptContext:
    """Thread-safe container for pending interrupt data."""

    def __init__(self) -> None:
        self._pending: list[str] = []
        self._lock = asyncio.Lock()

    async def push(self, context: str) -> None:
        """Push an interrupt context string to the pending queue."""
        async with self._lock:
            self._pending.append(context)

    async def drain(self) -> list[str]:
        """Drain and return all pending interrupt contexts."""
        async with self._lock:
            items = list(self._pending)
            self._pending.clear()
            return items

    async def has_pending(self) -> bool:
        async with self._lock:
            return len(self._pending) > 0


# ---------------------------------------------------------------------------
# Worker Agent
# ---------------------------------------------------------------------------


class WorkerAgent:
    """Base worker agent with full agentic execution loop.

    Usage::

        agent = WorkerAgent(
            config=agent_config,
            state=swarm_state,
            tool_executor=executor,
            compressor=compressor,
        )

        result = await agent.execute(
            task="Build the REST API for user management",
            task_id="task-001",
        )
    """

    # Completion signals the agent can emit
    COMPLETION_SIGNALS = {"[TASK_COMPLETE]", "[DONE]", "[FINISHED]"}

    # Token budget per task — stop if exceeded to prevent runaway spending
    # Now config-driven via AgentConfig.token_budget; this class default is a fallback
    TOKEN_BUDGET = 50_000

    def __init__(
        self,
        *,
        config: AgentConfig,
        state: SwarmState,
        tool_executor: ToolExecutor,
        compressor: Compressor | None = None,
        event_bus: EventBus | None = None,
        dashboard_callback: Any | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._executor = tool_executor
        self._compressor = compressor
        self._event_bus = event_bus
        self._dashboard_callback = dashboard_callback  # async fn(event_type, data)

        # Interrupt handling
        self._interrupts = InterruptContext()
        self._interrupt_listener_task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()

        # Execution state
        self._messages: list[dict[str, str]] = []
        self._iteration = 0
        self._total_tokens = TokenUsage()
        self._tool_count = 0
        self._file_writes = 0
        self._file_reads = 0
        self._budget_warned = False

        # LLM Gateway for fallback routing and caching
        try:
            from swarm.core.llm_gateway import LLMGateway
            self._gateway = LLMGateway(
                fallback_models=config.fallback_models,
            )
        except Exception:
            self._gateway = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def model(self) -> str:
        return self._config.model

    # -------------------------------------------------------------------
    # Main execution
    # -------------------------------------------------------------------

    async def execute(
        self,
        *,
        task: str,
        task_id: str,
        enhanced_prompt: str = "",
    ) -> dict[str, Any]:
        """Run the full agentic loop for a task.

        Parameters
        ----------
        task:
            The task description.
        task_id:
            State-tracked task ID.
        enhanced_prompt:
            Pre-built system prompt (with LTM, workspace info, etc.).

        Returns
        -------
        dict
            Execution summary: iterations, tokens, result, errors.
        """
        start_time = time.monotonic()

        # Build system prompt if not provided
        system_prompt = enhanced_prompt or self._config.system_prompt

        # Initialise message history
        self._messages = [{"role": "system", "content": system_prompt}]
        self._iteration = 0

        # Start interrupt listener
        await self._start_interrupt_listener()

        try:
            await self._state.set_agent_status(self.name, AgentStatus.RUNNING)

            result = await self._run_loop(task, task_id)

            elapsed = time.monotonic() - start_time
            return {
                "status": "completed",
                "summary": result,
                "iterations": self._iteration,
                "tokens": self._total_tokens.model_dump(),
                "elapsed_seconds": round(elapsed, 2),
                "messages": len(self._messages),
            }

        except asyncio.CancelledError:
            await self._state.set_agent_status(self.name, AgentStatus.INTERRUPTED)
            raise

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            error_msg = f"{type(exc).__name__}: {exc}"
            await logger.error(
                "worker.execution_failed",
                agent=self.name,
                error=error_msg,
            )
            return {
                "status": "failed",
                "error": error_msg,
                "iterations": self._iteration,
                "tokens": self._total_tokens.model_dump(),
                "elapsed_seconds": round(elapsed, 2),
            }

        finally:
            await self._stop_interrupt_listener()

    # -------------------------------------------------------------------
    # Core LLM loop
    # -------------------------------------------------------------------

    async def _run_loop(self, task: str, task_id: str) -> str:
        """The inner agentic loop."""
        from litellm import acompletion  # fallback if gateway unavailable

        # Add initial task message with efficiency instructions
        self._messages.append({
            "role": "user",
            "content": (
                f"Execute the following task:\n\n{task}\n\n"
                f"**RULES:**\n"
                f"- Be efficient. Only read files directly relevant to the task.\n"
                f"- Do NOT scan the entire codebase. Read only what you need.\n"
                f"- Plan before acting: think → read key files → make changes → verify.\n"
                f"- Complete in as few iterations as possible.\n"
                f"- When finished, include '[TASK_COMPLETE]' with a brief summary.\n"
            ),
        })

        tools = self._executor.get_openai_tools()
        summary = ""
        consecutive_errors = 0
        max_consecutive_errors = 3

        while self._iteration < self._config.max_iterations:
            if self._shutdown.is_set():
                break

            # --- Token budget check ---
            budget = self._config.token_budget
            used = self._total_tokens.total_tokens

            # 90% soft warning — inject finalization prompt
            if used > budget * 0.9 and not getattr(self, '_budget_warned', False):
                self._budget_warned = True
                self._messages.append({
                    "role": "user",
                    "content": (
                        f"SYSTEM: Token budget warning — you have used "
                        f"{used:,} of {budget:,} tokens (90%). "
                        f"Wrap up your current work and finalize immediately."
                    ),
                })
                await logger.warning(
                    "worker.token_budget_warning",
                    agent=self.name,
                    tokens=used,
                    budget=budget,
                )

            # 100% hard stop
            if used > budget:
                await logger.warning(
                    "worker.token_budget_exceeded",
                    agent=self.name,
                    tokens=used,
                    budget=budget,
                )
                summary = (
                    f"[TASK_COMPLETE] Stopped — token budget of "
                    f"{budget:,} reached "
                    f"({used:,} used). "
                    f"Work done so far has been saved."
                )
                break

            self._iteration += 1
            await self._state.update_agent_iterations(self.name, self._iteration)

            # --- Check for pending interrupts ---
            await self._process_interrupts()

            # --- Check and compress context if needed ---
            await self._maybe_compress_context()

            # --- Call LLM ---
            try:
                call_kwargs: dict[str, Any] = {
                    "model": self._config.model,
                    "messages": self._messages,
                    "temperature": self._config.temperature,
                    "max_tokens": 4096,
                    "timeout": self._config.timeout_seconds,
                }

                # Only include tools if we have them
                if tools:
                    call_kwargs["tools"] = tools
                    call_kwargs["tool_choice"] = "auto"

                if self._gateway is not None:
                    response = await self._gateway.complete(
                        model=self._config.model,
                        messages=self._messages,
                        tools=tools if tools else None,
                        tool_choice="auto" if tools else None,
                        temperature=self._config.temperature,
                        max_tokens=4096,
                        timeout=self._config.timeout_seconds,
                        agent_name=self.name,
                    )
                else:
                    response = await acompletion(**call_kwargs)
                consecutive_errors = 0  # Reset on success

                # --- Broadcast LLM call metrics to dashboard ---
                llm_tokens = 0
                if response.usage:
                    llm_tokens = response.usage.total_tokens or 0
                if self._dashboard_callback:
                    try:
                        await self._dashboard_callback("llm_call", {
                            "agent": self.name,
                            "model": self._config.model,
                            "iteration": self._iteration,
                            "iterations": self._iteration,
                            "tokens": self._total_tokens.total_tokens + llm_tokens,
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                            "tools": self._tool_count,
                            "files_written": self._file_writes,
                            "files_read": self._file_reads,
                        })
                    except Exception:
                        pass

            except Exception as exc:
                consecutive_errors += 1
                error_str = str(exc)
                await logger.error(
                    "worker.llm_error",
                    agent=self.name,
                    iteration=self._iteration,
                    error=error_str,
                    consecutive_errors=consecutive_errors,
                )

                # Fatal errors: model not found, auth failure — stop immediately
                if any(kw in error_str.lower() for kw in (
                    "not_found", "not found", "no longer available",
                    "authentication", "unauthorized", "invalid api key",
                )):
                    summary = f"Agent stopped: {error_str[:200]}"
                    break

                # Too many consecutive errors — stop gracefully
                if consecutive_errors >= max_consecutive_errors:
                    summary = (
                        f"Agent stopped after {max_consecutive_errors} consecutive "
                        f"LLM errors. Last error: {error_str[:200]}"
                    )
                    await logger.warning(
                        "worker.too_many_errors",
                        agent=self.name,
                        consecutive_errors=consecutive_errors,
                    )
                    break

                # Exponential backoff on recoverable errors
                await asyncio.sleep(min(2 ** consecutive_errors, 30))
                continue

            # --- Track token usage ---
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                self._total_tokens = TokenUsage(
                    prompt_tokens=self._total_tokens.prompt_tokens + usage.prompt_tokens,
                    completion_tokens=self._total_tokens.completion_tokens + usage.completion_tokens,
                    total_tokens=self._total_tokens.total_tokens + usage.total_tokens,
                )
                await self._state.update_agent_tokens(self.name, usage)

            # --- Process response ---
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""

            # Check for tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Append assistant message with tool calls
                self._messages.append(message.model_dump())

                # Execute each tool call
                for tool_call in message.tool_calls:
                    try:
                        tool_result = await self._execute_tool_call(tool_call)
                    except Exception as tool_exc:
                        await logger.error(
                            "worker.tool_call_crashed",
                            agent=self.name,
                            tool=getattr(tool_call.function, 'name', 'unknown'),
                            error=str(tool_exc),
                        )
                        tool_result = json.dumps({
                            "success": False,
                            "output": "",
                            "error": f"Tool execution crashed: {tool_exc}",
                        })

                    # Append tool result
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })

                # Log to state
                await self._state.add_message(
                    role=MessageRole.AGENT,
                    sender=self.name,
                    recipient="orchestrator",
                    content=f"[Tool calls: {len(message.tool_calls)}]",
                )

                # Continue loop — LLM needs to process tool results
                continue

            # No tool calls — process text response
            self._messages.append({"role": "assistant", "content": content})

            await self._state.add_message(
                role=MessageRole.AGENT,
                sender=self.name,
                recipient="orchestrator",
                content=content[:500],
            )

            # Check for completion signal
            if self._is_complete(content):
                summary = content
                await logger.info(
                    "worker.task_complete",
                    agent=self.name,
                    iteration=self._iteration,
                )
                break

            # Prompt for continuation with structured self-assessment
            iteration_pct = self._iteration / self._config.max_iterations
            if iteration_pct >= 0.8:
                self._messages.append({
                    "role": "user",
                    "content": (
                        "SYSTEM: You are approaching the iteration limit "
                        f"({self._iteration}/{self._config.max_iterations}). "
                        "Finalize your work immediately. Output your final "
                        "result and include '[TASK_COMPLETE]' with a summary."
                    ),
                })
            else:
                self._messages.append({
                    "role": "user",
                    "content": (
                        "Assess your progress on the task.\n"
                        "- If the task is COMPLETE and all requirements are met, "
                        "respond with '[TASK_COMPLETE]' and a brief summary of "
                        "what was accomplished.\n"
                        "- If work REMAINS, explain exactly what needs to be "
                        "done next and proceed with the next step."
                    ),
                })

        if not summary:
            summary = f"Agent reached iteration limit ({self._config.max_iterations})."

        return summary

    # -------------------------------------------------------------------
    # Tool call execution
    # -------------------------------------------------------------------

    async def _execute_tool_call(self, tool_call: Any) -> str:
        """Execute a single tool call and return the result string."""
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments

        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid JSON arguments: {raw_args}"})

        await logger.info(
            "worker.tool_call",
            agent=self.name,
            tool=tool_name,
            args_preview=str(arguments)[:200],
        )

        # Broadcast to dashboard
        if self._dashboard_callback:
            try:
                await self._dashboard_callback("tool_call", {
                    "agent": self.name, "tool": tool_name,
                    "args_preview": arguments,
                })
            except Exception:
                pass

        result = await self._executor.execute(tool_name, arguments)
        self._tool_count += 1

        # Broadcast tool result to dashboard
        if self._dashboard_callback:
            try:
                await self._dashboard_callback("tool_executed", {
                    "agent": self.name, "tool": tool_name,
                    "success": result.success,
                    "tools": self._tool_count,
                })
                # Track file writes specifically
                if tool_name == "write_file" and result.success:
                    self._file_writes += 1
                    await self._dashboard_callback("file_write", {
                        "agent": self.name,
                        "file": arguments.get("path", ""),
                        "bytes": len(arguments.get("content", "")),
                    })
                elif tool_name == "read_file":
                    self._file_reads += 1
                    await self._dashboard_callback("file_read", {
                        "agent": self.name,
                        "file": arguments.get("path", ""),
                    })
            except Exception:
                pass

        return json.dumps({
            "success": result.success,
            "output": (result.output or "")[:8000],
            "error": result.error or "",
        })

    def force_stop(self) -> None:
        """Force-stop the agent immediately (kill switch)."""
        self._shutdown.set()

    # -------------------------------------------------------------------
    # Interrupt handling
    # -------------------------------------------------------------------

    async def _start_interrupt_listener(self) -> None:
        """Start the async interrupt listener for schema changes."""
        if self._event_bus is not None and self._event_bus.connected:
            self._interrupt_listener_task = asyncio.create_task(
                self._listen_for_interrupts(),
                name=f"interrupt-{self.name}",
            )

    async def _stop_interrupt_listener(self) -> None:
        """Stop the interrupt listener."""
        if self._interrupt_listener_task is not None:
            self._interrupt_listener_task.cancel()
            try:
                await self._interrupt_listener_task
            except asyncio.CancelledError:
                pass
            self._interrupt_listener_task = None

    async def _listen_for_interrupts(self) -> None:
        """Background task: listen for schema change broadcasts."""

        async def _handle_interrupt(event: SwarmEvent) -> None:
            """Handle an incoming interrupt event."""
            target = event.payload.get("target_agent")
            affected = event.payload.get("affected_agents", [])

            # Only process if this event targets us
            if target and target != self.name:
                return
            if affected and self.name not in affected:
                return

            file_path = event.payload.get("file_path", "unknown")
            diff = event.payload.get("diff", "")
            description = event.payload.get("description", "")

            context = (
                f"\n⚠️  SCHEMA CHANGE INTERRUPT ⚠️\n"
                f"File changed: {file_path}\n"
                f"Description: {description}\n"
                f"\nDiff:\n```\n{diff}\n```\n\n"
                f"IMPORTANT: You must update your implementation to reflect "
                f"these changes before continuing.\n"
            )

            await self._interrupts.push(context)
            await logger.info(
                "worker.interrupt_received",
                agent=self.name,
                file=file_path,
            )

        try:
            await self._event_bus.listen(
                BroadcastChannel.SCHEMA_CHANGE_INTERRUPT.value,
                _handle_interrupt,
            )
            # Also listen for generic agent interrupts
            await self._event_bus.listen(
                BroadcastChannel.AGENT_INTERRUPT.value,
                _handle_interrupt,
            )
            # Keep alive until cancelled
            while not self._shutdown.is_set():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def _process_interrupts(self) -> None:
        """Check and process any pending interrupts between iterations."""
        contexts = await self._interrupts.drain()
        if not contexts:
            return

        # Combine all pending interrupts
        combined = "\n".join(contexts)

        self._messages.append({
            "role": "user",
            "content": combined,
        })

        await logger.info(
            "worker.interrupts_injected",
            agent=self.name,
            count=len(contexts),
        )

    # -------------------------------------------------------------------
    # Context compression
    # -------------------------------------------------------------------

    async def _maybe_compress_context(self) -> None:
        """Compress the conversation context if it grows too large."""
        if self._compressor is None:
            return

        # Estimate total tokens in context
        # Guard against None content (assistant messages with tool_calls have content=None)
        total_chars = sum(len(m.get("content") or "") for m in self._messages)
        estimated_tokens = total_chars // 4  # Rough estimate

        # Only compress if over threshold (leave room for the system prompt)
        threshold = 16_000  # ~16K tokens triggers compression
        if estimated_tokens < threshold:
            return

        # Keep system prompt and last 4 messages, compress the middle
        system_msg = self._messages[0]
        recent = self._messages[-4:]
        middle = self._messages[1:-4]

        if len(middle) < 2:
            return  # Not enough to compress

        result = await self._compressor.compress_conversation(
            middle,
            context_hint=f"Agent {self.name} mid-task conversation",
            session_id=self._state.session_id,
        )

        if result.compression_ratio < 0.9:
            self._messages = [
                system_msg,
                {
                    "role": "user",
                    "content": (
                        "[CONTEXT COMPRESSED — Previous conversation summarised]\n"
                        f"{result.summary}"
                    ),
                },
                *recent,
            ]

            await logger.info(
                "worker.context_compressed",
                agent=self.name,
                original_tokens=result.original_tokens,
                compressed_tokens=result.compressed_tokens,
                ratio=round(result.compression_ratio, 3),
            )

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------

    def _is_complete(self, content: str) -> bool:
        """Check if the LLM's response contains a completion signal."""
        upper = content.upper()
        return any(signal in upper for signal in self.COMPLETION_SIGNALS)

    def shutdown(self) -> None:
        """Signal the agent to shut down gracefully."""
        self._shutdown.set()
