"""Event handlers — the logic that executes when events fire.

Each handler implements a specific reaction to an event.  The orchestrator
registers these handlers with the ``EventBus`` during startup.

Handler architecture
--------------------

.. code-block:: text

    EventBus (stream/pubsub)
         │
         ▼
    ┌────────────────────────┐
    │  EventHandlerRegistry  │──► routes events to handlers by channel
    └────────┬───────────────┘
             │
    ┌────────┼──────────────────────────────────┐
    │        ▼                                   │
    │  SchemaChangeHandler                       │
    │  ├── Receives SCHEMA_CHANGED events        │
    │  ├── Identifies affected agents            │
    │  ├── Pauses them via orchestrator          │
    │  ├── Injects diff into their context       │
    │  └── Broadcasts interrupt via Pub/Sub      │
    │                                            │
    │  FileWriteHandler                          │
    │  ├── Receives FILE_WRITE_REQUEST events    │
    │  ├── Acquires distributed lock             │
    │  ├── Executes the write                    │
    │  └── Publishes FILE_WRITE_COMPLETED        │
    │                                            │
    │  EscalationHandler                         │
    │  ├── Receives ESCALATION_REQUIRED events   │
    │  ├── Formats context for human review      │
    │  └── Logs / notifies (webhook, etc.)       │
    │                                            │
    │  TaskLifecycleHandler                      │
    │  ├── Receives TASK_COMPLETED/FAILED events │
    │  └── Updates state, triggers LTM extract   │
    └────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import structlog

from swarm.events.bus import EventBus, SwarmEvent
from swarm.events.channels import (
    BroadcastChannel,
    EventType,
    StreamChannel,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol for orchestrator integration
# ---------------------------------------------------------------------------


class OrchestratorInterface(Protocol):
    """Protocol defining what the handlers need from the orchestrator.

    This avoids circular imports — the handlers reference this protocol
    instead of importing the concrete ``SwarmOrchestrator`` class.
    """

    async def pause_agent(self, agent_name: str, reason: str) -> None: ...
    async def resume_agent(self, agent_name: str) -> None: ...
    async def inject_context(self, agent_name: str, context: str) -> None: ...
    async def get_dependent_agents(self, agent_name: str) -> list[str]: ...
    async def get_state_summary(self) -> str: ...


# ---------------------------------------------------------------------------
# Schema Change Handler
# ---------------------------------------------------------------------------


class SchemaChangeHandler:
    """Handles API schema changes that affect downstream agents.

    When Agent A modifies an API schema, database model, or TypeScript
    interface, this handler:

    1. Identifies agents that depend on Agent A (via the DAG).
    2. Pauses those agents immediately.
    3. Injects the schema diff into their context.
    4. Broadcasts a ``SCHEMA_CHANGE_INTERRUPT`` for real-time listeners.
    5. Resumes the agents so they re-evaluate with the updated context.

    Example scenario::

        Backend Agent modifies ``schemas/user.py``
            → SchemaChangeHandler fires
                → Frontend Agent is paused
                → Diff is injected: "User model now has 'avatar_url' field"
                → Frontend Agent resumes with updated types
    """

    def __init__(
        self,
        bus: EventBus,
        orchestrator: OrchestratorInterface,
    ) -> None:
        self._bus = bus
        self._orchestrator = orchestrator

    async def handle(self, event: SwarmEvent) -> None:
        """Process a SCHEMA_CHANGED stream event."""
        sender = event.sender
        file_path = event.payload.get("file_path", "unknown")
        diff = event.payload.get("diff", "")
        description = event.payload.get("description", "Schema updated")

        await logger.info(
            "handler.schema_changed",
            sender=sender,
            file=file_path,
        )

        # Find agents that depend on the sender
        dependent_agents = await self._orchestrator.get_dependent_agents(sender)

        if not dependent_agents:
            await logger.info(
                "handler.schema_change_no_dependents",
                sender=sender,
            )
            return

        # Pause affected agents and inject context
        for agent_name in dependent_agents:
            await logger.info(
                "handler.pausing_agent",
                agent=agent_name,
                reason=f"Schema changed by {sender}: {file_path}",
            )

            # Pause the agent
            await self._orchestrator.pause_agent(
                agent_name,
                reason=f"Schema changed by {sender}",
            )

            # Build context injection
            context = (
                f"\n⚠️  SCHEMA CHANGE INTERRUPT ⚠️\n"
                f"Agent '{sender}' has modified: {file_path}\n"
                f"Description: {description}\n"
                f"\nDiff:\n```\n{diff}\n```\n\n"
                f"You MUST update your code to reflect these changes before "
                f"continuing. Re-read the affected files and adjust your "
                f"implementation accordingly.\n"
            )
            await self._orchestrator.inject_context(agent_name, context)

            # Resume the agent with new context
            await self._orchestrator.resume_agent(agent_name)

        # Broadcast real-time interrupt for any Pub/Sub listeners
        interrupt_event = SwarmEvent(
            channel=BroadcastChannel.SCHEMA_CHANGE_INTERRUPT.value,
            event_type=EventType.SCHEMA_DIFF,
            sender=sender,
            payload={
                "file_path": file_path,
                "diff": diff,
                "description": description,
                "affected_agents": dependent_agents,
            },
        )
        await self._bus.broadcast(
            BroadcastChannel.SCHEMA_CHANGE_INTERRUPT.value,
            interrupt_event,
        )

        await logger.info(
            "handler.schema_change_processed",
            sender=sender,
            affected_agents=dependent_agents,
        )


# ---------------------------------------------------------------------------
# File Write Handler
# ---------------------------------------------------------------------------


class FileWriteHandler:
    """Serialises concurrent file write requests via distributed locks.

    This is the **absolute source of truth** for file write access in the
    shared workspace.  When an agent wants to write a file:

    1. It publishes a ``FILE_WRITE_REQUEST`` event.
    2. This handler acquires a distributed lock for that file path.
    3. Executes the write (or delegates back to the agent via callback).
    4. Releases the lock.
    5. Publishes ``FILE_WRITE_COMPLETED``.

    If the lock cannot be acquired (another agent is writing), the handler
    waits with exponential backoff until the lock is released.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._pending_writes: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def handle(self, event: SwarmEvent) -> None:
        """Process a FILE_WRITE_REQUEST stream event.

        Expected payload:
        - ``file_path``: relative path in workspace
        - ``content``: file content to write
        - ``agent``: requesting agent name
        - ``request_id``: unique request ID for correlation
        """
        file_path = event.payload.get("file_path", "")
        content = event.payload.get("content", "")
        agent = event.payload.get("agent", event.sender)
        request_id = event.payload.get("request_id", event.event_id)

        if not file_path:
            await logger.warning(
                "handler.file_write_no_path",
                event_id=event.event_id,
            )
            return

        await logger.info(
            "handler.file_write_request",
            agent=agent,
            file=file_path,
            request_id=request_id,
        )

        # Acquire distributed lock for this file
        async with self._bus.file_lock(file_path) as acquired:
            if not acquired:
                # Lock timeout — notify the agent
                await logger.warning(
                    "handler.file_write_lock_timeout",
                    agent=agent,
                    file=file_path,
                )
                await self._publish_write_result(
                    file_path=file_path,
                    agent=agent,
                    request_id=request_id,
                    success=False,
                    error="Lock acquisition timed out — another agent may be writing.",
                )
                return

            # Execute the write
            try:
                import os
                from pathlib import Path

                # Resolve absolute path (content comes in as workspace-relative)
                # The actual file writing is done by the MCP tool layer,
                # but we serialise the access here.  For direct writes:
                abs_path = Path(file_path)
                if not abs_path.is_absolute():
                    # This will be resolved by the MCP tool layer
                    pass

                await logger.info(
                    "handler.file_write_executing",
                    agent=agent,
                    file=file_path,
                )

                # If content is provided, we can write directly
                # (for cases where the handler acts as the serialiser)
                if content:
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    abs_path.write_text(content, encoding="utf-8")

                # Publish completion
                await self._publish_write_result(
                    file_path=file_path,
                    agent=agent,
                    request_id=request_id,
                    success=True,
                )

            except Exception as exc:
                await logger.error(
                    "handler.file_write_failed",
                    agent=agent,
                    file=file_path,
                    error=str(exc),
                )
                await self._publish_write_result(
                    file_path=file_path,
                    agent=agent,
                    request_id=request_id,
                    success=False,
                    error=str(exc),
                )

    async def _publish_write_result(
        self,
        *,
        file_path: str,
        agent: str,
        request_id: str,
        success: bool,
        error: str = "",
    ) -> None:
        """Publish a FILE_WRITE_COMPLETED event."""
        event = SwarmEvent(
            channel=StreamChannel.FILE_WRITE_COMPLETED.value,
            event_type=EventType.WRITE_COMPLETE if success else EventType.WRITE_REJECTED,
            sender="file_write_handler",
            payload={
                "file_path": file_path,
                "agent": agent,
                "request_id": request_id,
                "success": success,
                "error": error,
            },
        )
        await self._bus.publish(StreamChannel.FILE_WRITE_COMPLETED.value, event)


# ---------------------------------------------------------------------------
# Escalation Handler
# ---------------------------------------------------------------------------


class EscalationHandler:
    """Handles circuit breaker trips and human review requests.

    When a circuit breaker trips or a task fails critically, this handler:
    1. Formats a human-readable summary of the situation.
    2. Logs it with structured data.
    3. Optionally calls an external webhook (Slack, etc.).
    """

    def __init__(
        self,
        bus: EventBus,
        orchestrator: OrchestratorInterface,
        *,
        webhook_url: str | None = None,
    ) -> None:
        self._bus = bus
        self._orchestrator = orchestrator
        self._webhook_url = webhook_url

    async def handle(self, event: SwarmEvent) -> None:
        """Process an ESCALATION_REQUIRED stream event."""
        agent_a = event.payload.get("agent_a", "unknown")
        agent_b = event.payload.get("agent_b", "unknown")
        message_count = event.payload.get("message_count", 0)
        threshold = event.payload.get("threshold", 0)
        strategy = event.payload.get("strategy", "unknown")

        # Get current state summary for context
        state_summary = await self._orchestrator.get_state_summary()

        escalation_report = (
            f"🚨 ESCALATION REQUIRED 🚨\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Agents: {agent_a} ↔ {agent_b}\n"
            f"Messages exchanged: {message_count} (threshold: {threshold})\n"
            f"Strategy: {strategy}\n"
            f"\nCurrent State:\n{state_summary}\n"
        )

        await logger.warning(
            "handler.escalation",
            agent_a=agent_a,
            agent_b=agent_b,
            message_count=message_count,
            threshold=threshold,
            report=escalation_report,
        )

        # Webhook notification (optional)
        if self._webhook_url:
            await self._send_webhook(escalation_report)

    async def _send_webhook(self, report: str) -> None:
        """Send an escalation report to an external webhook."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(
                    self._webhook_url,  # type: ignore[arg-type]
                    json={"text": report},
                    timeout=10.0,
                )
                await logger.info(
                    "handler.escalation_webhook_sent",
                    url=self._webhook_url,
                )
        except Exception as exc:
            await logger.warning(
                "handler.escalation_webhook_failed",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Task Lifecycle Handler
# ---------------------------------------------------------------------------


class TaskLifecycleHandler:
    """Handles task completion and failure events.

    On completion: logs result and can trigger downstream actions.
    On failure: logs error and publishes escalation if needed.
    """

    def __init__(
        self,
        bus: EventBus,
        orchestrator: OrchestratorInterface,
    ) -> None:
        self._bus = bus
        self._orchestrator = orchestrator

    async def handle_completed(self, event: SwarmEvent) -> None:
        """Process a TASK_COMPLETED event."""
        agent = event.payload.get("agent", event.sender)
        task_name = event.payload.get("task_name", "unknown")
        result = event.payload.get("result", "")

        await logger.info(
            "handler.task_completed",
            agent=agent,
            task=task_name,
            result=result[:200],
        )

    async def handle_failed(self, event: SwarmEvent) -> None:
        """Process a TASK_FAILED event."""
        agent = event.payload.get("agent", event.sender)
        task_name = event.payload.get("task_name", "unknown")
        error = event.payload.get("error", "Unknown error")

        await logger.error(
            "handler.task_failed",
            agent=agent,
            task=task_name,
            error=error,
        )


# ---------------------------------------------------------------------------
# Handler Registry
# ---------------------------------------------------------------------------


class EventHandlerRegistry:
    """Registries and wires all event handlers to the event bus.

    This is the single setup point called by the orchestrator during
    initialisation.

    Usage::

        registry = EventHandlerRegistry(bus, orchestrator)
        await registry.register_all()
    """

    def __init__(
        self,
        bus: EventBus,
        orchestrator: OrchestratorInterface,
        *,
        escalation_webhook_url: str | None = None,
    ) -> None:
        self._bus = bus
        self._orchestrator = orchestrator

        # Instantiate handlers
        self.schema_change = SchemaChangeHandler(bus, orchestrator)
        self.file_write = FileWriteHandler(bus)
        self.escalation = EscalationHandler(
            bus, orchestrator, webhook_url=escalation_webhook_url,
        )
        self.task_lifecycle = TaskLifecycleHandler(bus, orchestrator)

    async def register_all(self) -> None:
        """Register all handlers with the event bus.

        After this call, the event bus will route incoming events to
        the appropriate handlers automatically.
        """
        # Stream subscriptions (durable, ordered)
        await self._bus.subscribe(
            StreamChannel.SCHEMA_CHANGED.value,
            self.schema_change.handle,
        )
        await self._bus.subscribe(
            StreamChannel.FILE_WRITE_REQUEST.value,
            self.file_write.handle,
        )
        await self._bus.subscribe(
            StreamChannel.ESCALATION_REQUIRED.value,
            self.escalation.handle,
        )
        await self._bus.subscribe(
            StreamChannel.TASK_COMPLETED.value,
            self.task_lifecycle.handle_completed,
        )
        await self._bus.subscribe(
            StreamChannel.TASK_FAILED.value,
            self.task_lifecycle.handle_failed,
        )

        await logger.info(
            "handler_registry.registered",
            stream_channels=[
                StreamChannel.SCHEMA_CHANGED.value,
                StreamChannel.FILE_WRITE_REQUEST.value,
                StreamChannel.ESCALATION_REQUIRED.value,
                StreamChannel.TASK_COMPLETED.value,
                StreamChannel.TASK_FAILED.value,
            ],
        )
