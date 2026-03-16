"""Core orchestration engine — the main state machine and DAG runner.

This is the central nervous system of the swarm.  It:

1. Parses the ``SwarmConfig`` and provisions Docker sandboxes (via ``SandboxManager``).
2. Injects Long-Term Memory context into agent system prompts at boot.
3. Resolves the agent dependency DAG and runs agents in parallel where possible
   using ``anyio`` structured concurrency.
4. Routes inter-agent messages through the ``EventBus``.
5. Monitors execution via the ``CircuitBreakerManager``.
6. On task completion, triggers LTM knowledge extraction via the ``Compressor``.
7. Handles graceful shutdown (SIGINT/SIGTERM → drain → teardown).

Lifecycle
---------

.. code-block:: text

    ┌──────────────┐
    │  Parse       │
    │  Config      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐     ┌──────────────────┐
    │  Provision   │────►│  Inject LTM      │
    │  Sandboxes   │     │  Boot Context    │
    └──────┬───────┘     └────────┬─────────┘
           │                      │
           ▼                      ▼
    ┌──────────────────────────────────────┐
    │         Resolve DAG                  │
    │  (topological sort → parallel tiers) │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  Execute tiers concurrently          │
    │  ┌────────┐ ┌────────┐ ┌────────┐   │
    │  │Agent A │ │Agent B │ │Agent C │   │
    │  └────┬───┘ └────┬───┘ └────┬───┘   │
    │       │          │          │        │
    │       └──────────┴──────────┘        │
    │              Event Bus               │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  Extract LTM knowledge from results  │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  Teardown sandboxes                  │
    └──────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import signal
import time
from collections import defaultdict
from typing import Any

import anyio
import structlog

from swarm.config.models import AgentConfig, SwarmConfig
from swarm.core.circuit_breaker import CircuitBreakerManager, EscalationEvent
from swarm.core.compressor import Compressor
from swarm.core.manager import SwarmManager
from swarm.core.metrics import MetricsCollector
from swarm.core.state import (
    AgentStatus,
    MessageRole,
    SwarmState,
    TaskStatus,
    TokenUsage,
)
from swarm.events.bus import EventBus
from swarm.events.handlers import EventHandlerRegistry
from swarm.sandbox.manager import SandboxManager
from swarm.mcp.server import MCPServerFactory
from swarm.mcp.tools import ToolExecutor
from swarm.agents.worker import WorkerAgent
from swarm.infra.rate_limiter import SwarmRateLimiter
from swarm.infra.env import inject_sandbox_env, load_env
from swarm.core.telemetry import SessionLedger, NoOpLedger

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# DAG resolution utilities
# ---------------------------------------------------------------------------


def resolve_execution_tiers(agents: list[AgentConfig]) -> list[list[AgentConfig]]:
    """Resolve agent dependencies into parallel execution tiers.

    Uses Kahn's algorithm (topological sort) to group agents into tiers
    where each tier's agents have all dependencies satisfied by
    previous tiers.

    Returns
    -------
    list[list[AgentConfig]]
        Ordered tiers — all agents within a tier can run concurrently.

    Raises
    ------
    ValueError
        If the dependency graph contains cycles (should be caught earlier
        by config validation, but we double-check here).

    Example
    -------
    Given agents with dependencies::

        backend: []
        frontend: [backend]
        qa: [backend, frontend]

    Returns::

        [
            [backend],           # Tier 0: no dependencies
            [frontend],          # Tier 1: depends on tier 0
            [qa],                # Tier 2: depends on tiers 0 and 1
        ]
    """
    name_to_agent = {a.name: a for a in agents}
    in_degree: dict[str, int] = {a.name: 0 for a in agents}
    dependents: dict[str, list[str]] = defaultdict(list)

    for agent in agents:
        for dep in agent.depends_on:
            dependents[dep].append(agent.name)
            in_degree[agent.name] += 1

    tiers: list[list[AgentConfig]] = []
    remaining = dict(in_degree)

    while remaining:
        # Collect agents with no unresolved dependencies
        tier_names = [name for name, deg in remaining.items() if deg == 0]

        if not tier_names:
            unresolved = list(remaining.keys())
            raise ValueError(
                f"Circular dependency detected among agents: {unresolved}"
            )

        tier = [name_to_agent[n] for n in tier_names]
        tiers.append(tier)

        # Remove this tier from the graph
        for name in tier_names:
            del remaining[name]
            for dependent in dependents.get(name, []):
                if dependent in remaining:
                    remaining[dependent] -= 1

    return tiers


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SwarmOrchestrator:
    """Main orchestration engine.

    Usage::

        from swarm.config.loader import load_config

        config = load_config("swarm.yaml")
        orchestrator = SwarmOrchestrator(config)

        # Run the full pipeline
        await orchestrator.run()

        # Or with a specific task prompt
        await orchestrator.run(task="Build a REST API for user management")
    """

    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._state = SwarmState(swarm_name=config.name)
        self._circuit_breaker = CircuitBreakerManager(
            config.circuit_breaker, self._state,
        )
        self._compressor = Compressor(
            compression_config=config.compression,
            memory_config=config.memory,
            workspace=config.workspace,
        )

        # These will be initialised in run() when we have actual infra
        self._sandbox_manager: SandboxManager | None = None
        self._event_bus: EventBus | None = None
        self._mcp_factory: MCPServerFactory | None = None
        self._rate_limiter: SwarmRateLimiter | None = None
        self._handler_registry: EventHandlerRegistry | None = None

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()
        self._running = False
        self._kill_timer_task: asyncio.Task[None] | None = None

        # Dashboard (optional, set externally before run())
        self._dashboard: Any | None = None

        # Manager (micromanagement background task)
        self._manager: SwarmManager | None = None
        self._manager_task: asyncio.Task | None = None

        # Agent pause/resume coordination
        self._paused_agents: dict[str, asyncio.Event] = {}

        # Inter-tier context (output summaries from previous tiers)
        self._tier_summaries: list[str] = []

        # Metrics collector
        self._metrics = MetricsCollector()

        # Execution tiers (resolved from DAG)
        self._tiers: list[list[AgentConfig]] = []

        # Track active worker agents for kill switch
        self._active_workers: list[Any] = []  # WorkerAgent instances
        self._active_tasks: list[asyncio.Task] = []  # Running agent tasks

        # Session Replay Ledger (toggleable telemetry)
        if config.enable_session_replay:
            self._ledger: SessionLedger | NoOpLedger = SessionLedger(
                session_id=self._state.session_id,
                output_dir=config.workspace / ".swarm_logs",
                enabled=True,
                verbose=True,
            )
        else:
            self._ledger = NoOpLedger()

    @property
    def state(self) -> SwarmState:
        """Access the swarm state (for testing or external monitoring)."""
        return self._state

    @property
    def compressor(self) -> Compressor:
        """Access the compressor (for testing or external monitoring)."""
        return self._compressor

    @property
    def circuit_breaker(self) -> CircuitBreakerManager:
        """Access the circuit breaker manager."""
        return self._circuit_breaker

    def set_dashboard(self, dashboard: Any) -> None:
        """Attach a dashboard instance for real-time event broadcasting."""
        self._dashboard = dashboard

    async def _broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast an event to the dashboard (no-op if no dashboard)."""
        if self._dashboard is not None:
            try:
                await self._dashboard.broadcast_event(
                    source="orchestrator", event_type=event_type, data=data,
                )
            except Exception:
                pass  # Dashboard errors must never crash the swarm

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    async def run(
        self,
        *,
        task: str = "",
        dry_run: bool = False,
        timeout_minutes: float = 0,
        interactive: bool = False,
    ) -> dict[str, Any]:
        """Execute the full orchestration pipeline.

        Parameters
        ----------
        task:
            Optional overarching task description to inject into all agents.
        dry_run:
            If True, resolve the DAG and log the plan but don't execute.
        timeout_minutes:
            If > 0, automatically trigger graceful shutdown after this many
            minutes. Agents finish their current iteration then stop.

        Returns
        -------
        dict
            Execution summary with task results, timing, and token usage.
        """
        start_time = time.monotonic()
        self._running = True

        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()

        await logger.info(
            "orchestrator.starting",
            swarm=self._config.name,
            agents=[a.name for a in self._config.agents],
            workspace=str(self._config.workspace),
        )

        try:
            # --- Phase 1: Initialise infrastructure ---
            await self._initialize_infrastructure()

            # --- Phase 2: Resolve the execution DAG ---
            self._tiers = resolve_execution_tiers(self._config.agents)

            await logger.info(
                "orchestrator.dag_resolved",
                tiers=[
                    [a.name for a in tier] for tier in self._tiers
                ],
            )

            if dry_run:
                return await self._dry_run_report()

            # --- Phase 3: Provision sandboxes ---
            await self._provision_sandboxes()

            # --- Start kill timer ---
            if timeout_minutes > 0:
                self._kill_timer_task = asyncio.create_task(
                    self._kill_timer(timeout_minutes)
                )
                await logger.info(
                    "orchestrator.kill_switch_armed",
                    timeout_minutes=timeout_minutes,
                    msg=f"Swarm will auto-stop in {timeout_minutes} min.",
                )

            # --- Start Manager ---
            dashboard_cb = None
            if self._dashboard is not None:
                async def _mgr_cb(source: str, etype: str, data: dict) -> None:
                    await self._dashboard.broadcast_event(source, etype, data)
                dashboard_cb = _mgr_cb

            self._manager = SwarmManager(
                state=self._state,
                dashboard_cb=dashboard_cb,
                event_bus=self._event_bus,
                metrics_collector=self._metrics,
            )
            self._manager.set_helper_dispatcher(self._dispatch_helper_agent)
            self._manager_task = await self._manager.start()

            # --- Phase 4: Execute tiers (or wait in interactive mode) ---
            tier_results: dict[str, Any] = {}

            if interactive:
                # Interactive mode: don't auto-run agents.
                # The user drives work via the dashboard chat which calls
                # dispatch_dynamic_task().
                await logger.info(
                    "orchestrator.interactive_mode",
                    msg="Waiting for user input via dashboard chat.",
                )
                await self._broadcast("waiting_for_input", {
                    "message": "Interactive mode — use the chat to tell the manager what to do.",
                    "agents": [a.name for a in self._config.agents],
                })

                # Block until kill switch or Ctrl-C
                await self._shutdown_event.wait()
            else:
                for tier_index, tier in enumerate(self._tiers):
                    if self._shutdown_event.is_set():
                        await logger.warning("orchestrator.shutdown_requested")
                        break

                    await logger.info(
                        "orchestrator.executing_tier",
                        tier=tier_index,
                        agents=[a.name for a in tier],
                    )
                    await self._broadcast("tier_started", {
                        "tier": tier_index,
                        "agents": [a.name for a in tier],
                    })

                    # Inject workspace bootstrap for all tiers
                    # (tier 0 gets it too for incremental builds on existing codebases)
                    workspace_ctx = await self._build_workspace_bootstrap()

                    results = await self._execute_tier(
                        tier, task=task, workspace_context=workspace_ctx,
                    )
                    tier_results[f"tier_{tier_index}"] = results

                    # Collect tier summaries for inter-agent context
                    tier_summary_parts = []
                    for agent_name, result in results.items():
                        if isinstance(result, dict):
                            summary = result.get("result", {}).get("summary", "")
                            if summary:
                                tier_summary_parts.append(
                                    f"[{agent_name}]: {summary[:500]}"
                                )
                    if tier_summary_parts:
                        self._tier_summaries.append(
                            "\n".join(tier_summary_parts)
                        )

            # --- Stop Manager ---
            if self._manager:
                await self._manager.stop()

            # --- Phase 5: Extract long-term memories ---
            await self._post_execution_extraction()

            # --- Phase 6: Generate summary ---
            elapsed = time.monotonic() - start_time

            # Broadcast final analytics to dashboard
            try:
                analytics = self._metrics.to_dict(self._state.session_id)
                await self._broadcast("run_completed", analytics)
            except Exception:
                pass

            return await self._build_execution_report(tier_results, elapsed)

        except Exception as exc:
            await logger.error(
                "orchestrator.fatal_error",
                error=str(exc),
                exc_info=True,
            )
            raise
        finally:
            # --- Teardown ---
            self._ledger.close()  # Flush session replay ledger
            await self._teardown()
            self._running = False

    # -------------------------------------------------------------------
    # Phase 1: Infrastructure initialisation
    # -------------------------------------------------------------------

    async def _initialize_infrastructure(self) -> None:
        """Initialise the compressor, event bus, sandbox manager, and rate limiter."""
        # Long-Term Memory / Compressor
        await self._compressor.initialize()

        # Rate limiter
        self._rate_limiter = SwarmRateLimiter(self._config.rate_limits)

        # Event Bus (Redis)
        try:
            self._event_bus = EventBus(self._config.event_bus)
            await self._event_bus.connect()
            await logger.info("orchestrator.eventbus_connected")

            # Wire event bus into state for Event Sourcing (Step 9)
            self._state._event_bus = self._event_bus

            # Wire circuit breaker threshold from config (Step 12)
            self._state.set_circuit_breaker_threshold(
                self._config.circuit_breaker.max_round_trips,
            )
        except Exception as exc:
            await logger.warning(
                "orchestrator.eventbus_connect_failed",
                error=str(exc),
                msg="Running without event bus — file locks disabled.",
            )
            self._event_bus = None

        # Sandbox Manager (Docker)
        try:
            self._sandbox_manager = SandboxManager(self._config)
            await logger.info("orchestrator.sandbox_manager_ready")
        except Exception as exc:
            await logger.warning(
                "orchestrator.sandbox_manager_failed",
                error=str(exc),
                msg="Running in local mode — no Docker sandboxes.",
            )
            self._sandbox_manager = None

        # MCP Server Factory
        self._mcp_factory = MCPServerFactory(
            workspace=self._config.workspace,
            event_bus=self._event_bus,
            compressor=self._compressor,
        )

        # Event Handlers
        if self._event_bus is not None:
            self._handler_registry = EventHandlerRegistry(
                self._event_bus,
                self,
            )
            await self._handler_registry.register_all()

        await logger.info("orchestrator.infrastructure_ready")

    # -------------------------------------------------------------------
    # Phase 3: Sandbox provisioning
    # -------------------------------------------------------------------

    async def _provision_sandboxes(self) -> None:
        """Provision Docker sandboxes for all agents.

        Each sandbox mounts the shared workspace and has its MCP server
        injected.  The ``SandboxManager`` handles the Docker lifecycle.
        """
        for agent_config in self._config.agents:
            await self._state.register_agent(agent_config.name)

            if self._sandbox_manager is not None:
                try:
                    # Merge API key env vars into agent sandbox config
                    # so create_sandbox() picks them up via sandbox_cfg.env_vars
                    api_env = inject_sandbox_env(
                        agent_config.model,
                        self._config.api_keys or None,
                    )
                    if api_env:
                        agent_config.sandbox.env_vars.update(api_env)

                    sandbox_info = await self._sandbox_manager.create_sandbox(
                        agent_config,
                    )
                    await logger.info(
                        "orchestrator.sandbox_created",
                        agent=agent_config.name,
                        container_id=sandbox_info.container_id[:12],
                    )
                except Exception as exc:
                    await logger.warning(
                        "orchestrator.sandbox_create_failed",
                        agent=agent_config.name,
                        error=str(exc),
                        msg="Agent will run in local mode.",
                    )
            else:
                await logger.info(
                    "orchestrator.sandbox_skipped",
                    agent=agent_config.name,
                    msg="No sandbox manager — running in local mode.",
                )

            await self._state.set_agent_status(agent_config.name, AgentStatus.IDLE)

        await logger.info(
            "orchestrator.sandboxes_provisioned",
            agents=[a.name for a in self._config.agents],
        )

    # -------------------------------------------------------------------
    # Phase 4: Tier execution
    # -------------------------------------------------------------------

    async def _execute_tier(
        self,
        tier: list[AgentConfig],
        *,
        task: str,
        workspace_context: str = "",
    ) -> dict[str, Any]:
        """Execute all agents in a tier concurrently with fault isolation.

        Uses ``asyncio.gather(return_exceptions=True)`` so one agent
        crashing does NOT cancel its siblings.
        """
        results: dict[str, Any] = {}

        async def _safe_execute(agent_config: AgentConfig) -> None:
            """Execute one agent, catching all exceptions."""
            try:
                # Check if agent is paused
                pause_event = self._paused_agents.get(agent_config.name)
                if pause_event is not None:
                    await logger.info(
                        "orchestrator.agent_paused_waiting",
                        agent=agent_config.name,
                    )
                    await pause_event.wait()

                await self._execute_agent(
                    agent_config, task, results,
                    workspace_context=workspace_context,
                )
            except asyncio.CancelledError:
                results[agent_config.name] = {"status": "cancelled"}
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                await logger.error(
                    "orchestrator.agent_tier_failed",
                    agent=agent_config.name,
                    error=error_msg,
                )
                await self._broadcast("agent_failed", {
                    "agent": agent_config.name, "error": error_msg,
                })
                results[agent_config.name] = {
                    "status": "failed", "error": error_msg,
                }

        # Launch all agents in parallel — failures are isolated
        await asyncio.gather(
            *[_safe_execute(a) for a in tier],
            return_exceptions=True,
        )

        return results

    async def _execute_agent(
        self,
        agent_config: AgentConfig,
        task: str,
        results: dict[str, Any],
        *,
        workspace_context: str = "",
    ) -> None:
        """Execute a single agent's task lifecycle.

        1. Retrieve LTM boot context
        2. Build the enhanced system prompt
        3. Create a task in state
        4. Run the agent's LLM loop
        5. Record results
        6. Trigger LTM extraction if successful
        """
        agent_name = agent_config.name
        start_time = time.monotonic()

        try:
            # WorkerAgent.execute() handles the IDLE → RUNNING transition
            # --- 1. Retrieve LTM boot context ---
            ltm_context = await self._compressor.get_boot_context(
                agent_name=agent_name,
                agent_role=agent_config.role.value,
                task_description=task,
            )

            # --- 2. Build enhanced system prompt ---
            enhanced_prompt = self._build_enhanced_prompt(
                agent_config=agent_config,
                task=task,
                ltm_context=ltm_context,
                workspace_context=workspace_context,
                tier_context="\n".join(self._tier_summaries) if self._tier_summaries else "",
            )

            # --- 3. Create task in state ---
            task_name = f"{agent_name}:{task or 'default-task'}"
            task_id = await self._state.create_task(task_name, agent_name)
            await self._state.set_task_status(task_id, TaskStatus.RUNNING)

            # Record metrics
            self._metrics.record_agent_start(agent_name, agent_config.model)

            await logger.info(
                "orchestrator.agent_started",
                agent=agent_name,
                model=agent_config.model,
                task_id=task_id,
                ltm_entries=ltm_context.count("[") if ltm_context else 0,
            )
            await self._broadcast("agent_started", {
                "agent": agent_name, "model": agent_config.model,
                "task_id": task_id,
            })

            # --- 4. Run the agent's LLM loop ---
            agent_result = await self._run_agent_loop(
                agent_config=agent_config,
                enhanced_prompt=enhanced_prompt,
                task_id=task_id,
            )

            # --- 5. Record results ---
            elapsed = time.monotonic() - start_time
            await self._state.set_task_status(
                task_id,
                TaskStatus.COMPLETED,
                result=agent_result.get("summary", "Task completed."),
            )
            await self._state.set_agent_status(agent_name, AgentStatus.COMPLETED)

            # Record completion metrics
            self._metrics.record_agent_end(agent_name, "completed")

            results[agent_name] = {
                "status": "completed",
                "result": agent_result,
                "elapsed_seconds": round(elapsed, 2),
            }

            await logger.info(
                "orchestrator.agent_completed",
                agent=agent_name,
                elapsed=round(elapsed, 2),
            )
            await self._broadcast("agent_completed", {
                "agent": agent_name, "elapsed": round(elapsed, 2),
            })

            # Broadcast metrics update
            await self._broadcast("metrics_update", {
                "agent": agent_name,
                "status": "completed",
                "elapsed": round(elapsed, 2),
            })

        except asyncio.CancelledError:
            await self._state.set_agent_status(agent_name, AgentStatus.INTERRUPTED)
            results[agent_name] = {"status": "cancelled"}
            raise

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            error_msg = f"{type(exc).__name__}: {exc}"

            # Try to mark task as failed
            try:
                agent_record = await self._state.get_agent(agent_name)
                if agent_record and agent_record.current_task_id:
                    await self._state.set_task_status(
                        agent_record.current_task_id,
                        TaskStatus.FAILED,
                        error=error_msg,
                    )
            except Exception:
                pass

            try:
                await self._state.set_agent_status(agent_name, AgentStatus.FAILED)
            except Exception:
                pass

            results[agent_name] = {
                "status": "failed",
                "error": error_msg,
                "elapsed_seconds": round(elapsed, 2),
            }

            await logger.error(
                "orchestrator.agent_failed",
                agent=agent_name,
                error=error_msg,
                elapsed=round(elapsed, 2),
            )

    # -------------------------------------------------------------------
    # Agent LLM loop
    # -------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        *,
        agent_config: AgentConfig,
        enhanced_prompt: str,
        task_id: str,
    ) -> dict[str, Any]:
        """Run the core agentic LLM loop for a single agent.

        Uses the WorkerAgent class with full tool calling, interrupt
        handling, context compression, and rate-limited LLM calls.
        """
        agent_name = agent_config.name

        # --- Rate-limited LLM call wrapper ---
        async def _rate_limited_sandbox_exec(
            command: str, timeout: float,
        ) -> tuple[int, str]:
            """Execute command in sandbox if available."""
            if self._sandbox_manager is not None:
                return await self._sandbox_manager.exec_in_sandbox(
                    agent_name, command, timeout=timeout,
                )
            # Local fallback — subprocess
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._config.workspace),
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            return (proc.returncode or 0, output)

        # --- Create ToolExecutor ---
        tool_executor = ToolExecutor(
            workspace=self._config.workspace,
            event_bus=self._event_bus,
            compressor=self._compressor,
            sandbox_exec=_rate_limited_sandbox_exec,
            allowed_tools=agent_config.mcp.allowed_tools,
            blocked_commands=agent_config.mcp.blocked_commands,
            agent_name=agent_name,
        )

        # --- Build dashboard callback ---
        dashboard_cb = None
        if self._dashboard is not None:
            async def _cb(event_type: str, data: dict) -> None:
                await self._dashboard.broadcast_event("worker", event_type, data)
            dashboard_cb = _cb

        # --- Create WorkerAgent ---
        agent = WorkerAgent(
            config=agent_config,
            state=self._state,
            tool_executor=tool_executor,
            compressor=self._compressor,
            event_bus=self._event_bus,
            dashboard_callback=dashboard_cb,
            ledger=self._ledger,
        )

        # Track for kill switch
        self._active_workers.append(agent)

        # --- Execute ---
        try:
            return await agent.execute(
                task=task_id,
                task_id=task_id,
                enhanced_prompt=enhanced_prompt,
            )
        finally:
            # Remove from active workers
            if agent in self._active_workers:
                self._active_workers.remove(agent)

    # -------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------

    def _build_enhanced_prompt(
        self,
        *,
        agent_config: AgentConfig,
        task: str,
        ltm_context: str,
        workspace_context: str = "",
        tier_context: str = "",
    ) -> str:
        """Build the enhanced system prompt with LTM context and MCP tools.

        The prompt is composed of:
        1. The agent's original system prompt
        2. The current task description
        3. Long-Term Memory context (from past sessions)
        4. Available MCP tools description
        5. Workspace information
        6. Workspace bootstrap (directory snapshot from previous tiers)
        7. Inter-agent context (summaries from previous tier agents)
        """
        sections: list[str] = []

        # Original system prompt
        sections.append(agent_config.system_prompt.strip())

        # Task description
        if task:
            sections.append(f"\n--- CURRENT TASK ---\n{task}\n--- END TASK ---")

        # Workspace info
        sections.append(
            f"\n--- WORKSPACE ---\n"
            f"Shared workspace: {self._config.workspace}\n"
            f"Your role: {agent_config.role.value}\n"
            f"Your model: {agent_config.model}\n"
            f"Max iterations: {agent_config.max_iterations}\n"
            f"--- END WORKSPACE ---"
        )

        # MCP tools description
        tool_list = agent_config.tools
        if tool_list:
            tools_desc = "\n".join(f"  • {t}" for t in tool_list)
            sections.append(
                f"\n--- AVAILABLE TOOLS ---\n"
                f"You have access to the following tools via MCP:\n"
                f"{tools_desc}\n"
                f"\nAdditionally, you can search the team's long-term memory:\n"
                f"  • search_memory(query) — Search past session knowledge\n"
                f"--- END TOOLS ---"
            )

        # Long-Term Memory context (from past sessions)
        if ltm_context:
            sections.append(ltm_context)

        # Workspace bootstrap (directory snapshot from previous tiers)
        if workspace_context:
            sections.append(
                "\n--- WORKSPACE BOOTSTRAP ---\n"
                "The following files already exist in the workspace (created by "
                "agents in previous tiers). DO NOT re-read these with list_directory "
                "— use the information below to start working immediately:\n\n"
                f"{workspace_context}\n"
                "--- END WORKSPACE BOOTSTRAP ---"
            )

        # Inter-agent context (summaries from previous tier agents)
        if tier_context:
            sections.append(
                "\n--- PREVIOUS TIER RESULTS ---\n"
                "The following agents completed before you. Use their output "
                "as context — build on top of their work, don't duplicate it:\n\n"
                f"{tier_context}\n"
                "--- END PREVIOUS TIER RESULTS ---"
            )

        # Completion signal instruction
        sections.append(
            "\n--- COMPLETION ---\n"
            "When you have fully completed your task, include '[TASK_COMPLETE]' "
            "in your final response along with a summary of what was accomplished.\n"
            "--- END COMPLETION ---"
        )

        return "\n\n".join(sections)

    @staticmethod
    def _is_task_complete_signal(content: str) -> bool:
        """Check if the agent's response signals task completion."""
        return "[TASK_COMPLETE]" in content.upper()

    # -------------------------------------------------------------------
    # Workspace bootstrap & agent control
    # -------------------------------------------------------------------

    async def _build_workspace_bootstrap(self) -> str:
        """Generate a directory snapshot of the workspace for prompt injection.

        This eliminates the 3-4 redundant list_directory calls each agent
        makes at the start of their execution.
        """
        import os

        workspace = self._config.workspace
        lines: list[str] = []
        file_count = 0

        for root, dirs, files in os.walk(workspace):
            # Skip hidden/vendor directories
            dirs[:] = [
                d for d in dirs
                if d not in {'.git', '.venv', 'node_modules', '__pycache__',
                             '.swarm_memory', '.next'}
            ]

            level = root.replace(str(workspace), '').count(os.sep)
            indent = '  ' * level
            basename = os.path.basename(root)
            lines.append(f"{indent}{basename}/")

            sub_indent = '  ' * (level + 1)
            for file in sorted(files):
                if file.startswith('.'):
                    continue
                filepath = os.path.join(root, file)
                try:
                    size = os.path.getsize(filepath)
                    lines.append(f"{sub_indent}{file}  ({size:,} bytes)")
                    file_count += 1
                except OSError:
                    lines.append(f"{sub_indent}{file}")
                    file_count += 1

            if file_count > 100:
                lines.append('  ... (truncated at 100 files)')
                break

        return '\n'.join(lines) if lines else '(empty workspace)'

    async def _dispatch_helper_agent(
        self,
        idle_agent: str,
        struggling_agent: str,
        help_context: str,
    ) -> None:
        """Dispatch an idle agent to help a struggling peer.

        Called by the SwarmManager when it detects an agent completed
        early while a sibling is still running.
        """
        # Find the agent config
        agent_configs = {a.name: a for a in self._config.agents}
        idle_config = agent_configs.get(idle_agent)
        if idle_config is None:
            return

        await logger.info(
            "orchestrator.helper_dispatched",
            idle_agent=idle_agent,
            helping=struggling_agent,
        )
        await self._broadcast("helper_dispatched", {
            "idle_agent": idle_agent, "helping": struggling_agent,
        })

        # Reset agent counters so it doesn't trip budget/stall detection
        # (archives old token usage to global billing tracker)
        try:
            await self._state.reset_agent_for_redispatch(idle_agent)
            await self._state.set_agent_status(idle_agent, AgentStatus.IDLE)
        except Exception as exc:
            await logger.warning(
                "orchestrator.helper_reset_failed",
                agent=idle_agent, error=str(exc),
            )
            return

        # Build a helper-specific prompt with struggling agent's context
        workspace_ctx = await self._build_workspace_bootstrap()

        # Get the struggling agent's recent conversation for context sharing
        struggling_context = ""
        try:
            struggling_record = await self._state.get_agent(struggling_agent)
            if struggling_record is not None:
                # Get last few messages from struggling agent
                msgs = await self._state.get_messages(
                    sender=struggling_agent, limit=10,
                )
                if msgs:
                    struggling_context = (
                        f"\n\n--- CONTEXT FROM {struggling_agent} ---\n"
                        f"Agent '{struggling_agent}' is at iteration "
                        f"{struggling_record.iterations}. "
                        f"Here are their recent actions:\n"
                    )
                    for m in msgs:
                        role = m.role.value if hasattr(m.role, 'value') else str(m.role)
                        content = str(m.content)[:300]
                        if content.strip():
                            struggling_context += f"[{role}]: {content}\n"
                    struggling_context += f"--- END CONTEXT ---\n"
        except Exception:
            pass

        enhanced_prompt = self._build_enhanced_prompt(
            agent_config=idle_config,
            task=help_context + struggling_context,
            ltm_context="",
            workspace_context=workspace_ctx,
            tier_context="",
        )

        try:
            result = await self._run_agent_loop(
                agent_config=idle_config,
                enhanced_prompt=enhanced_prompt,
                task_id=f"helper-{idle_agent}-for-{struggling_agent}",
            )
            await logger.info(
                "orchestrator.helper_completed",
                idle_agent=idle_agent,
                result_status=result.get("status", "unknown"),
            )
        except Exception as exc:
            await logger.warning(
                "orchestrator.helper_failed",
                idle_agent=idle_agent, error=str(exc),
            )

    async def pause_agent(self, name: str, reason: str = "") -> None:
        """Pause an agent (it will wait at the next iteration boundary)."""
        event = asyncio.Event()
        self._paused_agents[name] = event
        await logger.info(
            "orchestrator.agent_paused", agent=name, reason=reason,
        )
        await self._broadcast("agent_paused", {
            "agent": name, "reason": reason,
        })

    async def resume_agent(self, name: str) -> None:
        """Resume a paused agent."""
        event = self._paused_agents.pop(name, None)
        if event is not None:
            event.set()
            await logger.info("orchestrator.agent_resumed", agent=name)
            await self._broadcast("agent_resumed", {"agent": name})

    async def dispatch_dynamic_task(self, agent_name: str, task_prompt: str) -> None:
        """Dispatch a task to a sub-agent on-demand (from Manager Chat).

        This allows the Manager (pro model) to assign new work to any
        agent dynamically from the dashboard chat.
        """
        # Find agent config
        agent_configs = {a.name: a for a in self._config.agents}
        agent_config = agent_configs.get(agent_name)

        if agent_config is None:
            await logger.warning(
                "orchestrator.dispatch_unknown_agent", agent=agent_name,
            )
            await self._broadcast("dispatch_error", {
                "agent": agent_name,
                "error": f"Unknown agent: {agent_name}",
            })
            return

        await logger.info(
            "orchestrator.dynamic_task_dispatched",
            agent=agent_name,
            task=task_prompt[:100],
        )
        await self._broadcast("task_dispatched", {
            "agent": agent_name,
            "task": task_prompt[:200],
        })

        # Reset agent status if needed
        try:
            current = await self._state.get_agent(agent_name)
            if current and current.status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                await self._state.set_agent_status(agent_name, AgentStatus.IDLE)
        except Exception:
            pass

        # Build prompt with workspace context
        workspace_ctx = await self._build_workspace_bootstrap()

        # Cap max_iterations for dynamic tasks to prevent runaway token spend
        capped_config = agent_config.model_copy(update={
            "max_iterations": min(agent_config.max_iterations, 8),  # Was 15 — too many = token waste
        })
        enhanced_prompt = self._build_enhanced_prompt(
            agent_config=capped_config,
            task=task_prompt,
            ltm_context="",
            workspace_context=workspace_ctx,
            tier_context="",
        )

        try:
            result = await self._run_agent_loop(
                agent_config=capped_config,
                enhanced_prompt=enhanced_prompt,
                task_id=f"dynamic-{agent_name}-{int(time.time())}",
            )
            # Mark agent as completed so the Manager stops watching it
            try:
                await self._state.set_agent_status(agent_name, AgentStatus.COMPLETED)
            except Exception:
                pass
            await self._broadcast("dynamic_task_completed", {
                "agent": agent_name,
                "status": result.get("status", "unknown"),
            })
        except Exception as exc:
            # Mark agent as failed so the Manager stops nudging it
            try:
                await self._state.set_agent_status(agent_name, AgentStatus.FAILED)
            except Exception:
                pass
            await logger.warning(
                "orchestrator.dynamic_task_failed",
                agent=agent_name, error=str(exc),
            )
            await self._broadcast("dynamic_task_failed", {
                "agent": agent_name, "error": str(exc),
            })

    # -------------------------------------------------------------------
    # Phase 5: Post-execution LTM extraction
    # -------------------------------------------------------------------

    async def _post_execution_extraction(self) -> None:
        """Extract long-term memories from all completed tasks."""
        if not self._config.memory.enabled or not self._config.memory.auto_extract:
            return

        completed_tasks = await self._state.get_tasks_by_status(TaskStatus.COMPLETED)

        for task_record in completed_tasks:
            # Get the agent's message history
            messages = await self._state.get_messages(
                sender=task_record.assigned_agent,
            )
            conversation_history = [
                {"role": m.role.value, "content": m.content}
                for m in messages
            ]

            try:
                extraction = await self._compressor.extract_knowledge(
                    task_name=task_record.name,
                    agent_name=task_record.assigned_agent,
                    conversation_history=conversation_history,
                    task_result=task_record.result or "",
                )

                if extraction.entries:
                    await logger.info(
                        "orchestrator.ltm_extracted",
                        task=task_record.name,
                        entries=len(extraction.entries),
                        categories=[e.category for e in extraction.entries],
                    )
            except Exception as exc:
                await logger.warning(
                    "orchestrator.ltm_extraction_failed",
                    task=task_record.name,
                    error=str(exc),
                )

    # -------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------

    async def _dry_run_report(self) -> dict[str, Any]:
        """Generate a report for dry-run mode (no execution)."""
        report = {
            "mode": "dry_run",
            "swarm": self._config.name,
            "workspace": str(self._config.workspace),
            "tiers": [
                {
                    "tier": i,
                    "agents": [
                        {
                            "name": a.name,
                            "role": a.role.value,
                            "model": a.model,
                            "depends_on": a.depends_on,
                        }
                        for a in tier
                    ],
                }
                for i, tier in enumerate(self._tiers)
            ],
            "memory_enabled": self._config.memory.enabled,
            "circuit_breaker_threshold": self._config.circuit_breaker.max_round_trips,
        }

        await logger.info("orchestrator.dry_run_complete", report=report)
        return report

    async def _build_execution_report(
        self,
        tier_results: dict[str, Any],
        elapsed: float,
    ) -> dict[str, Any]:
        """Build the final execution summary report."""
        snapshot = await self._state.snapshot()

        total_tokens = sum(
            a.token_usage.total_tokens for a in snapshot.agents.values()
        )

        return {
            "mode": "execution",
            "swarm": self._config.name,
            "session_id": snapshot.session_id,
            "elapsed_seconds": round(elapsed, 2),
            "total_tokens": total_tokens,
            "agents": {
                name: {
                    "status": agent.status.value,
                    "iterations": agent.iterations,
                    "tokens": agent.token_usage.total_tokens,
                }
                for name, agent in snapshot.agents.items()
            },
            "tasks": {
                tid: {
                    "name": task.name,
                    "status": task.status.value,
                    "agent": task.assigned_agent,
                    "result": task.result,
                    "error": task.error,
                }
                for tid, task in snapshot.tasks.items()
            },
            "tier_results": tier_results,
            "messages_total": len(snapshot.messages),
            "artifacts": snapshot.artifacts,
            "memory_entries_extracted": len(
                await self._compressor.search_memories("") if self._config.memory.enabled else []
            ),
        }

    # -------------------------------------------------------------------
    # Teardown
    # -------------------------------------------------------------------

    async def _teardown(self) -> None:
        """Gracefully tear down all sandboxes and connections."""
        await logger.info("orchestrator.teardown_starting")

        # Cancel kill timer if running
        if self._kill_timer_task and not self._kill_timer_task.done():
            self._kill_timer_task.cancel()
            try:
                await self._kill_timer_task
            except asyncio.CancelledError:
                pass

        # Event handlers are cleaned up when the event bus disconnects

        # Teardown sandboxes
        if self._sandbox_manager is not None:
            try:
                await self._sandbox_manager.teardown_all()
            except Exception as exc:
                await logger.error("orchestrator.sandbox_teardown_failed", error=str(exc))

        # Shutdown MCP servers
        if self._mcp_factory is not None:
            try:
                await self._mcp_factory.shutdown_all()
            except Exception as exc:
                await logger.error("orchestrator.mcp_teardown_failed", error=str(exc))

        # Disconnect event bus
        if self._event_bus is not None:
            try:
                await self._event_bus.disconnect()
            except Exception as exc:
                await logger.error("orchestrator.eventbus_teardown_failed", error=str(exc))

        await logger.info("orchestrator.teardown_complete")

    # -------------------------------------------------------------------
    # OrchestratorInterface methods (used by event handlers)
    # -------------------------------------------------------------------

    async def pause_agent(self, agent_name: str, reason: str) -> None:
        """Pause an agent (set status to IDLE)."""
        await self._state.set_agent_status(agent_name, AgentStatus.IDLE)
        await logger.info("orchestrator.agent_paused", agent=agent_name, reason=reason)

    async def resume_agent(self, agent_name: str) -> None:
        """Resume a paused agent."""
        await self._state.set_agent_status(agent_name, AgentStatus.RUNNING)
        await logger.info("orchestrator.agent_resumed", agent=agent_name)

    async def inject_context(self, agent_name: str, context: str) -> None:
        """Inject context into an agent's conversation."""
        await self._state.add_message(
            role=MessageRole.SYSTEM,
            sender="orchestrator",
            recipient=agent_name,
            content=context,
        )

    async def get_dependent_agents(self, agent_name: str) -> list[str]:
        """Get agents that depend on the given agent."""
        dependents: list[str] = []
        for agent in self._config.agents:
            if agent_name in agent.depends_on:
                dependents.append(agent.name)
        return dependents

    async def get_state_summary(self) -> str:
        """Get a summary of the current swarm state."""
        snapshot = await self._state.snapshot()
        lines: list[str] = [f"Session: {snapshot.session_id}"]
        for name, agent in snapshot.agents.items():
            lines.append(
                f"  {name}: {agent.status.value} "
                f"(iter={agent.iterations}, tokens={agent.token_usage.total_tokens})"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Kill switch / graceful stop
    # -------------------------------------------------------------------

    def graceful_stop(self) -> None:
        """Signal the swarm to stop — kills ALL running agents immediately.

        This is the real kill switch — agents are force-stopped,
        not left to finish current iteration.
        """
        if self._shutdown_event.is_set():
            return
        logger.warning(
            "orchestrator.graceful_stop",
            msg="Kill switch activated — force-stopping all agents.",
        )
        self._shutdown_event.set()

        # Force-stop all active worker agents
        for worker in self._active_workers:
            try:
                worker.force_stop()
            except Exception:
                pass

        # Cancel all active agent tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()

    async def _kill_timer(self, timeout_minutes: float) -> None:
        """Background task that fires the kill switch after timeout_minutes.

        Logs a countdown every 60 seconds so the user can watch progress.
        """
        total_seconds = timeout_minutes * 60
        elapsed = 0.0
        interval = min(60.0, total_seconds)

        try:
            while elapsed < total_seconds:
                remaining = total_seconds - elapsed
                mins_left = remaining / 60

                if remaining <= total_seconds:  # Always log
                    await logger.info(
                        "orchestrator.kill_timer",
                        remaining_minutes=round(mins_left, 1),
                        elapsed_minutes=round(elapsed / 60, 1),
                        total_minutes=timeout_minutes,
                    )

                wait = min(interval, remaining)
                await asyncio.sleep(wait)
                elapsed += wait

            # Time's up
            await logger.warning(
                "orchestrator.kill_switch_triggered",
                timeout_minutes=timeout_minutes,
                msg=f"⏰ {timeout_minutes} min timeout reached — stopping swarm gracefully.",
            )
            self.graceful_stop()

        except asyncio.CancelledError:
            await logger.info("orchestrator.kill_timer_cancelled")

    # -------------------------------------------------------------------
    # Signal handling
    # -------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
            except (NotImplementedError, RuntimeError):
                # Windows or running in a non-main thread
                pass

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signals gracefully."""
        logger.warning(
            "orchestrator.signal_received",
            signal=sig.name,
            msg="Initiating graceful shutdown...",
        )
        self._shutdown_event.set()

    # -------------------------------------------------------------------
    # Circuit breaker integration
    # -------------------------------------------------------------------

    async def check_agent_communication(
        self,
        sender: str,
        recipient: str,
    ) -> bool:
        """Check if an agent-to-agent message is allowed by the circuit breaker.

        Called by the event bus or message router before delivering a message.
        If the breaker trips, executes the configured escalation strategy.

        Returns True if the message should be delivered.
        """
        allowed, event = await self._circuit_breaker.check(sender, recipient)

        if not allowed and event:
            result = await self._circuit_breaker.handle_escalation(event)

            # If strategy is summarize_and_retry, compress and reset
            if result.get("requires_compression"):
                messages = await self._state.get_messages(
                    sender=sender, limit=20,
                )
                conversation = [
                    {"role": m.role.value, "content": m.content}
                    for m in messages
                ]
                await self._compressor.compress_conversation(
                    conversation,
                    context_hint=f"Circuit breaker: {sender} ↔ {recipient}",
                    session_id=self._state.session_id,
                )
                await self._circuit_breaker.attempt_reset(sender, recipient)
                return True  # Allow after reset

        return allowed
