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
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from swarm.core.state import (
    AgentStatus,
    MessageRole,
    SwarmState,
    TokenUsage,
)
from swarm.core.telemetry import NoOpLedger
from swarm.events.channels import BroadcastChannel

if TYPE_CHECKING:
    from swarm.config.models import AgentConfig
    from swarm.core.compressor import Compressor
    from swarm.events.bus import EventBus, SwarmEvent
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
        ledger: Any | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._executor = tool_executor
        self._compressor = compressor
        self._event_bus = event_bus
        self._dashboard_callback = dashboard_callback  # async fn(event_type, data)
        self._ledger = ledger or NoOpLedger()

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
        self._last_prompt_tokens = 0  # Track ACTUAL API prompt tokens for compression

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
        """The inner agentic loop.

        V2 Cache Contract
        -----------------
        ``self._messages`` ordering is cache-safe:

        * ``messages[0]`` — system prompt (FROZEN; tagged for native
          prompt caching by LLMGateway)
        * ``messages[1]`` — task instruction + rules (FROZEN after boot)
        * ``messages[2+]`` — volatile conversation (tool calls, results,
          assistant replies, interrupts)

        Do NOT mutate ``messages[0]`` after construction.
        """
        from litellm import acompletion  # fallback if gateway unavailable

        # Add initial task message with efficiency instructions (V2.1: compressed)
        self._messages.append({
            "role": "user",
            "content": (
                f"{task}\n\n"
                f"RULES: Write ALL files in ONE response (parallel tool calls). "
                f"Plan first, then execute. Read only what's needed. "
                f"Use search_memory to check for relevant past knowledge before starting. "
                f"End with '[TASK_COMPLETE]' + summary."
            ),
        })

        summary = ""
        consecutive_errors = 0
        max_consecutive_errors = 3

        # V2.4: Auto-inject memory search before iteration 1
        # Agents kept ignoring search_memory — so we do it programmatically.
        if self._executor._compressor is not None:
            try:
                # Extract a search query from the task (first 100 chars)
                _mem_query = task[:100].strip()
                memories = await self._executor._compressor.search_memories(
                    _mem_query, limit=3,
                )
                if memories:
                    mem_lines = []
                    for mem in memories:
                        mem_lines.append(f"- [{mem.category.upper()}] {mem.title}: {mem.content[:200]}")
                    mem_context = "\n".join(mem_lines)
                    self._messages.append({
                        "role": "user",
                        "content": (
                            f"MEMORY CONTEXT (from past sessions):\n{mem_context}\n\n"
                            f"Use this context to avoid re-doing work that's already been done."
                        ),
                    })
                    self._ledger.record_memory_search(
                        agent=self.name,
                        query=_mem_query,
                        results_count=len(memories),
                        top_similarity=0.0,
                        cache_hit=False,
                    )
                    await logger.info(
                        "worker.auto_memory_injected",
                        agent=self.name,
                        memories_found=len(memories),
                    )
            except Exception as exc:
                await logger.warning(
                    "worker.auto_memory_failed",
                    agent=self.name,
                    error=str(exc),
                )

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

            # --- Dynamic tool phasing (V2) ---
            # Iteration 1: planning tools (list, read, search_memory)
            # Iteration 2+: execution tools (read, write, run_command)
            tool_phase = "planning" if self._iteration == 1 else "execution"
            tools = self._executor.get_openai_tools(phase=tool_phase)

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

                _call_start = time.monotonic()

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
                    with contextlib.suppress(Exception):
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
                self._last_prompt_tokens = response.usage.prompt_tokens or 0
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

            # --- Sliding window AFTER we have real prompt_tokens ---
            # (Must be here, not top-of-loop, because _last_prompt_tokens
            #  is only set from the response we just received.)
            await self._apply_sliding_window()

            # --- Process response ---
            if not response.choices:
                await logger.warning(
                    "worker.empty_choices",
                    agent=self.name,
                    iteration=self._iteration,
                )
                consecutive_errors += 1
                continue
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""

            # --- Session Ledger: record LLM call ---
            _tool_names: list[str] = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                _tool_names = [tc.function.name for tc in message.tool_calls]
            self._ledger.record_llm_call(
                agent=self.name,
                model=self._config.model,
                messages=self._messages,
                response_text=content,
                tool_calls=_tool_names or None,
                prompt_tokens=self._last_prompt_tokens,
                completion_tokens=getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                total_tokens=getattr(response.usage, 'total_tokens', 0) if response.usage else 0,
                cumulative_tokens=self._total_tokens.total_tokens,
                iteration=self._iteration,
                wall_clock_ms=round((time.monotonic() - _call_start) * 1000, 1) if '_call_start' in dir() else 0,
            )

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

                    # Truncate large tool outputs to save tokens (V2.2: 4K→2K)
                    _MAX_TOOL_OUTPUT_CHARS = 2_000  # ~500 tokens
                    if len(tool_result) > _MAX_TOOL_OUTPUT_CHARS:
                        tool_result = (
                            tool_result[:_MAX_TOOL_OUTPUT_CHARS]
                            + f"\n[TRUNCATED — {len(tool_result):,} chars]"
                        )

                    # Append tool result
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })

                # --- Post-write content scrubbing ---
                try:
                    self._scrub_write_file_content()
                except Exception:
                    pass  # Non-critical — don't crash the loop

                # --- Auto-complete: skip wasteful text-only TASK_COMPLETE call ---
                # If the assistant's text already contains TASK_COMPLETE, we're done.
                # Previously this would loop → make another 2-3K token text call.
                if content and self._is_complete(content):
                    summary = content
                    await logger.info(
                        "worker.task_complete_with_tools",
                        agent=self.name,
                        iteration=self._iteration,
                    )
                    break

                # Also auto-complete if agent has written files and is past iter 2
                _wrote_files = any(
                    tc.function.name == "write_file"
                    for tc in message.tool_calls
                    if hasattr(tc, 'function')
                )
                if _wrote_files and self._iteration >= 3:
                    # Inject completion nudge — saves a full LLM round trip
                    self._messages.append({
                        "role": "user",
                        "content": "Files written. If your task is complete, respond only with '[TASK_COMPLETE]' and a one-line summary.",
                    })

                # Log to state
                await self._state.add_message(
                    role=MessageRole.AGENT,
                    sender=self.name,
                    recipient="orchestrator",
                    content=f"[Tool calls: {len(message.tool_calls)}]",
                )

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

            # Only inject finalization nudge near iteration limit
            # (removed verbose self-assessment — it caused one-file-at-a-time
            # behavior, adding 8+ unnecessary iterations and 20K+ wasted tokens)
            iteration_pct = self._iteration / self._config.max_iterations
            if iteration_pct >= 0.75:
                self._messages.append({
                    "role": "user",
                    "content": (
                        f"SYSTEM: Iteration {self._iteration}/{self._config.max_iterations}. "
                        "Finalize now. Include '[TASK_COMPLETE]' with summary."
                    ),
                })

        if not summary:
            summary = f"Agent reached iteration limit ({self._config.max_iterations})."

        return summary

    # -------------------------------------------------------------------
    # Post-write content scrubbing (token economy)
    # -------------------------------------------------------------------

    def _scrub_write_file_content(self) -> None:
        """Scrub write_file content from stored assistant messages.

        When the LLM calls write_file(path, content), the full file content
        (often 3-10K tokens) persists in self._messages as the assistant's
        tool_call arguments. This content gets re-sent to the LLM on EVERY
        subsequent call, wasting massive tokens.

        This method replaces the full content with a 200-char preview.
        """
        for msg in self._messages:
            tool_calls = msg.get("tool_calls")
            if not tool_calls or msg.get("role") != "assistant":
                continue
            for tc in tool_calls:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn or not isinstance(fn, dict):
                    continue
                if fn.get("name") != "write_file":
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                    content = args.get("content", "")
                    if len(content) > 80:
                        args["content"] = (
                            content[:80]
                            + f"\n[{len(content):,}ch written]"
                        )
                        fn["arguments"] = json.dumps(args)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

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

        _tool_start = time.monotonic()

        # Broadcast to dashboard
        if self._dashboard_callback:
            with contextlib.suppress(Exception):
                await self._dashboard_callback("tool_call", {
                    "agent": self.name, "tool": tool_name,
                    "args_preview": arguments,
                })

        result = await self._executor.execute(tool_name, arguments)
        _tool_duration = (time.monotonic() - _tool_start) * 1000
        self._tool_count += 1

        # V2: Truncate error tracebacks to save tokens (max 300 chars)
        _error_str = result.error or ""
        if _error_str and len(_error_str) > 300:
            # Keep "ExceptionType: " + last meaningful line
            lines = _error_str.strip().splitlines()
            last_line = lines[-1].strip() if lines else _error_str[:200]
            _error_str = last_line[:300]

        # --- Session Ledger: record tool execution ---
        self._ledger.record_tool_execution(
            agent=self.name,
            tool_name=tool_name,
            arguments=arguments,
            result_str=json.dumps({"success": result.success, "output": (result.output or "")[:500]}),
            success=result.success,
            duration_ms=round(_tool_duration, 1),
        )

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
            "output": (result.output or "")[:2_000],  # V2: aligned to 2K (was 4K)
            "error": _error_str,
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
            with contextlib.suppress(asyncio.CancelledError):
                await self._interrupt_listener_task
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
    # Algorithmic Sliding Window (V2 — zero-cost, no LLM)
    # -------------------------------------------------------------------

    _WINDOW_KEEP_RECENT_PAIRS = 2     # assistant+tool pairs to keep in full (V2.1: was 4, too generous)
    _WINDOW_TRIGGER_TOKENS = 1_500    # start windowing above this (V2.1: was 2,500, too late)
    _STUB_MAX_CHARS = 50              # max chars for stubbed old messages

    async def _apply_sliding_window(self) -> None:
        """Cap context via pure algorithm — no LLM call needed.

        Strategy:
          1. Always keep messages[0] (system prompt) + messages[1] (task).
          2. Keep the last N assistant+tool message pairs in full.
          3. Replace all older messages with a one-line stub:
             ``[tool: write_file → ok | 200 chars]``
          4. Archive full dropped messages to Redis for post-task LTM extraction.

        This runs AFTER every LLM call because `_last_prompt_tokens` is only
        set from the response we just received.
        """
        actual = self._last_prompt_tokens
        if actual < self._WINDOW_TRIGGER_TOKENS:
            return

        # Minimum messages: system + task + at least 1 pair
        if len(self._messages) <= 4:
            return

        msgs_before = len(self._messages)

        # --- Partition messages into Atomic Blocks ---
        pinned = self._messages[:2]              # system prompt + task
        body = self._messages[2:]              # everything else

        blocks = []
        current_block = []

        for msg in body:
            role = msg.get("role")
            if role == "user" or (role == "assistant" and "tool_calls" not in msg):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                blocks.append([msg])
            elif role == "assistant" and "tool_calls" in msg:
                if current_block:
                    blocks.append(current_block)
                current_block = [msg]
            elif role == "tool":
                current_block.append(msg)
            else:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                blocks.append([msg])

        if current_block:
            blocks.append(current_block)

        # Keep last N blocks (instead of N*2 messages)
        keep_count = self._WINDOW_KEEP_RECENT_PAIRS
        if len(blocks) <= keep_count:
            return  # Nothing to drop

        blocks_to_drop = blocks[:-keep_count]
        blocks_to_keep = blocks[-keep_count:]

        # --- Archive dropped messages to telemetry ledger ---
        for block in blocks_to_drop:
            for msg in block:
                self._ledger.record_dropped_message(
                    agent=self.name,
                    role=msg.get("role", "unknown"),
                    content=(msg.get("content") or "")[:500],
                    iteration=self._iteration,
                )

        # --- Stub dropped blocks ---
        # We must replace the entire dropped block with a single plain-text stub
        # to preserve semantic lineage without violating Gemini's strict API schema.
        stubs: list[dict[str, str]] = []
        for block in blocks_to_drop:
            if not block:
                continue

            first_msg = block[0]
            role = first_msg.get("role", "unknown")
            content = first_msg.get("content") or ""

            if role == "assistant" and "tool_calls" in first_msg:
                # Atomic Tool Block: flatten into a single user message
                tool_names = []
                for tc in first_msg.get("tool_calls", []):
                    fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
                    if fn:
                        name = fn.get("name", "?") if isinstance(fn, dict) else getattr(fn, "name", "?")
                        tool_names.append(name)

                tools_str = ",".join(tool_names)
                stubs.append({
                    "role": "user",
                    "content": f"[SYSTEM CONTEXT DROPPED: The agent executed tools '{tools_str}'. Result truncated to save tokens.]"
                })
            else:
                # Normal user/assistant message: flatten into a stub
                preview = content[:self._STUB_MAX_CHARS].replace("\n", " ")
                # Map to standard roles
                stub_role = role if role in ["user", "assistant"] else "user"
                if stubs and stubs[-1]["role"] == stub_role:
                    stubs[-1]["content"] += f" | [prev: {preview}]"
                else:
                    stubs.append({"role": stub_role, "content": f"[prev: {preview}]"})

        # --- Rebuild messages ---
        to_keep = []
        for block in blocks_to_keep:
            to_keep.extend(block)

        self._messages = pinned + stubs + to_keep

        await logger.info(
            "worker.sliding_window_applied",
            agent=self.name,
            actual_prompt_tokens=actual,
            messages_before=msgs_before,
            messages_after=len(self._messages),
            dropped=len(blocks_to_drop),
            stubbed=len(stubs),
        )

        # --- Session Ledger ---
        self._ledger.record_compression(
            agent=self.name,
            original_tokens=actual,
            compressed_tokens=0,  # No LLM used — zero cost
            ratio=0.0,
            messages_before=msgs_before,
            messages_after=len(self._messages),
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
