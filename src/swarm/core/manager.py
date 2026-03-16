"""SwarmManager — the micromanaging brain of the swarm.

Runs as a background asyncio task alongside the orchestrator. Monitors agent
progress and intervenes when agents are idle, stuck, or could be helping peers.

Architecture::

    ┌─────────────────────────────────────────────────┐
    │  SwarmManager (background task)                  │
    │                                                   │
    │  ┌──────────────┐  ┌────────────────────────┐    │
    │  │ Watchdog      │  │ Idle Helper Dispatch   │    │
    │  │ (15s checks)  │  │ (re-assign completed   │    │
    │  │               │  │  agents to help peers) │    │
    │  └──────────────┘  └────────────────────────┘    │
    │  ┌──────────────┐  ┌────────────────────────┐    │
    │  │ Work Stealing │  │ Dashboard Feed         │    │
    │  │ (shared task  │  │ (push manager events)  │    │
    │  │  queue)       │  │                        │    │
    │  └──────────────┘  └────────────────────────┘    │
    └─────────────────────────────────────────────────┘

The manager never directly modifies agent state — it publishes events
on the EventBus that agents listen for. This keeps the system decoupled.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine

import structlog

from swarm.core.state import AgentStatus, SwarmState

logger = structlog.get_logger(__name__)


class SwarmManager:
    """Background manager that monitors and optimises swarm execution.

    Usage::

        manager = SwarmManager(state=state, dashboard_cb=broadcast)
        manager_task = await manager.start()

        # ... swarm runs ...

        await manager.stop()
    """

    # How often the watchdog checks agent progress (seconds)
    CHECK_INTERVAL = 10

    # If an agent hasn't progressed in this many seconds, nudge it
    STALL_THRESHOLD = 30

    # Stop nudging after this many attempts (agent is truly stuck)
    MAX_NUDGES = 10

    # Minimum iterations a peer must have before we consider it "ahead"
    HELPER_DISPATCH_MIN_ITERATIONS = 5

    def __init__(
        self,
        *,
        state: SwarmState,
        dashboard_cb: Callable[..., Coroutine] | None = None,
        event_bus: Any | None = None,
        metrics_collector: Any | None = None,
    ) -> None:
        self._state = state
        self._dashboard_cb = dashboard_cb
        self._event_bus = event_bus
        self._metrics = metrics_collector
        self._task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()

        # Tracking
        self._last_progress: dict[str, dict[str, Any]] = {}
        self._helper_tasks: dict[str, asyncio.Task] = {}
        self._completed_agents: set[str] = set()
        self._nudge_count: dict[str, int] = {}

        # Helper dispatch callback (set by orchestrator)
        self._dispatch_helper: Callable | None = None

    def set_helper_dispatcher(
        self, fn: Callable[..., Coroutine],
    ) -> None:
        """Set the callback that dispatches an idle agent to help a peer.

        The callback signature::

            async def dispatch_helper(
                idle_agent: str,
                struggling_agent: str,
                help_context: str,
            ) -> None
        """
        self._dispatch_helper = fn

    async def start(self) -> asyncio.Task[None]:
        """Start the manager background loop."""
        self._shutdown.clear()
        self._task = asyncio.create_task(
            self._run_loop(), name="swarm-manager",
        )
        await logger.info("manager.started")
        await self._broadcast("manager_started", {
            "check_interval": self.CHECK_INTERVAL,
            "stall_threshold": self.STALL_THRESHOLD,
        })
        return self._task

    async def stop(self) -> None:
        """Stop the manager."""
        self._shutdown.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await logger.info("manager.stopped")

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """The manager's main monitoring loop."""
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(self.CHECK_INTERVAL)
                if self._shutdown.is_set():
                    break
                await self._check_progress()
                # NOTE: Auto-helper dispatch disabled — too wasteful.
                # Completed agents were being re-dispatched with vague prompts,
                # burning 100k+ tokens reading the entire codebase.
                # Helpers should only be dispatched explicitly via Manager chat.
                # await self._check_for_idle_helpers()
        except asyncio.CancelledError:
            pass

    # -------------------------------------------------------------------
    # Progress watchdog
    # -------------------------------------------------------------------

    async def _check_progress(self) -> None:
        """Check each running agent's progress and nudge if stalled."""
        try:
            snapshot = await self._state.snapshot()
        except Exception:
            return

        now = time.time()

        for name, agent in snapshot.agents.items():
            if agent.status in (AgentStatus.COMPLETED, AgentStatus.FAILED,
                                AgentStatus.IDLE, AgentStatus.INTERRUPTED):
                if agent.status == AgentStatus.COMPLETED:
                    self._completed_agents.add(name)
                # Clean up tracking for finished agents
                self._last_progress.pop(name, None)
                self._nudge_count.pop(name, None)
                continue

            current_iters = agent.iterations
            current_tools = agent.token_usage.total_tokens  # proxy for activity

            prev = self._last_progress.get(name, {})
            prev_iters = prev.get("iterations", 0)
            prev_time = prev.get("time", now)

            if current_iters > prev_iters:
                # Agent is making progress — update tracking
                self._last_progress[name] = {
                    "iterations": current_iters,
                    "time": now,
                }
                self._nudge_count[name] = 0
            else:
                # No progress — check if stalled
                stall_duration = now - prev_time
                if stall_duration > self.STALL_THRESHOLD:
                    nudge_num = self._nudge_count.get(name, 0) + 1
                    self._nudge_count[name] = nudge_num

                    # Stop nudging after MAX_NUDGES — agent is truly stuck
                    if nudge_num > self.MAX_NUDGES:
                        await logger.warning(
                            "manager.agent_stuck_giving_up",
                            agent=name,
                            nudges=nudge_num,
                            msg="Max nudges reached, stopping monitoring.",
                        )
                        await self._broadcast("agent_stuck", {
                            "agent": name,
                            "nudges": nudge_num,
                            "message": f"Agent {name} stuck after {nudge_num} nudges — stopped monitoring.",
                        })
                        # Remove from tracking so we stop checking
                        self._last_progress.pop(name, None)
                        continue

                    await logger.warning(
                        "manager.agent_stalled",
                        agent=name,
                        stall_seconds=round(stall_duration, 1),
                        nudge_number=nudge_num,
                    )
                    await self._broadcast("agent_stalled", {
                        "agent": name,
                        "stall_seconds": round(stall_duration, 1),
                        "nudge_number": nudge_num,
                    })

                    # Track in metrics
                    if self._metrics:
                        self._metrics.record_nudge()

                    # Nudge the agent via event bus
                    if self._event_bus is not None:
                        try:
                            from swarm.events.bus import SwarmEvent
                            from swarm.events.channels import BroadcastChannel

                            nudge_event = SwarmEvent(
                                channel=BroadcastChannel.AGENT_INTERRUPT.value,
                                event_type="nudge",
                                sender="manager",
                                payload={
                                    "target_agent": name,
                                    "message": (
                                        f"⚡ Manager nudge #{nudge_num}: You appear "
                                        f"to be stuck for {round(stall_duration)}s. "
                                        f"Re-evaluate your approach. If waiting for "
                                        f"something, try an alternative. If unsure, "
                                        f"output [TASK_COMPLETE] with what you have."
                                    ),
                                },
                            )
                            await self._event_bus.broadcast(
                                BroadcastChannel.AGENT_INTERRUPT.value,
                                nudge_event,
                            )
                        except Exception as exc:
                            await logger.error(
                                "manager.nudge_failed",
                                agent=name, error=str(exc),
                            )

    # -------------------------------------------------------------------
    # Idle agent helper dispatch
    # -------------------------------------------------------------------

    async def _check_for_idle_helpers(self) -> None:
        """Check if any completed agents can help struggling peers."""
        if self._dispatch_helper is None:
            return

        try:
            snapshot = await self._state.snapshot()
        except Exception:
            return

        # Find running agents that are behind
        running_agents: list[tuple[str, int]] = []
        for name, agent in snapshot.agents.items():
            if agent.status == AgentStatus.RUNNING:
                running_agents.append((name, agent.iterations))

        if not running_agents:
            return

        # Find newly completed agents that haven't been dispatched yet
        newly_completed = set()
        for name, agent in snapshot.agents.items():
            if agent.status == AgentStatus.COMPLETED and name not in self._helper_tasks:
                newly_completed.add(name)

        if not newly_completed:
            return

        # Sort running agents by iterations (lowest first = most behind)
        running_agents.sort(key=lambda x: x[1])

        for idle_agent in newly_completed:
            if not running_agents:
                break

            # Help the agent that's furthest behind
            struggling_name, struggling_iters = running_agents[0]

            if struggling_iters < self.HELPER_DISPATCH_MIN_ITERATIONS:
                # Agent just started, let it work
                continue

            help_context = (
                f"Agent '{struggling_name}' is still working (iteration "
                f"{struggling_iters}). Read what they've done so far in the "
                f"workspace and help by implementing complementary parts "
                f"that they haven't gotten to yet. Focus on different files "
                f"to avoid conflicts. Check the project structure first.\n\n"
                f"IMPORTANT: Do NOT recreate files that already exist. "
                f"Only add NEW files or fix bugs in existing ones. "
                f"Use list_directory first to see what exists."
            )

            # Track in metrics
            if self._metrics:
                self._metrics.record_helper_dispatch(idle_agent, struggling_name)

            await logger.info(
                "manager.dispatching_helper",
                idle_agent=idle_agent,
                helping=struggling_name,
                struggling_iterations=struggling_iters,
            )
            await self._broadcast("helper_dispatched", {
                "idle_agent": idle_agent,
                "helping": struggling_name,
                "context": help_context,
            })

            try:
                task = asyncio.create_task(
                    self._dispatch_helper(
                        idle_agent, struggling_name, help_context,
                    ),
                    name=f"helper-{idle_agent}",
                )
                self._helper_tasks[idle_agent] = task
            except Exception as exc:
                await logger.error(
                    "manager.helper_dispatch_failed",
                    idle_agent=idle_agent, error=str(exc),
                )

            # Remove from candidate list
            running_agents.pop(0)

    # -------------------------------------------------------------------
    # Dashboard integration
    # -------------------------------------------------------------------

    async def _broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast a manager event to the dashboard."""
        if self._dashboard_cb is not None:
            try:
                await self._dashboard_cb("manager", event_type, data)
            except Exception:
                pass
