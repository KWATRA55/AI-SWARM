"""Event channel definitions for the swarm event bus.

Channels are logical topics that events are published to and subscribed from.
Each channel has a well-defined purpose and expected event types.

There are two transport modes:

* **Stream channels** — Durable, ordered events via Redis Streams with
  consumer groups.  Events are persisted and can be replayed.  Used for
  events that must not be lost (task assignments, completions).

* **Broadcast channels** — Real-time Pub/Sub events.  Fire-and-forget
  semantics.  Used for interrupt-style notifications where latency matters
  more than durability (schema changes, heartbeats).
"""

from __future__ import annotations

from enum import Enum


class StreamChannel(str, Enum):
    """Durable event channels backed by Redis Streams.

    Events on these channels are persisted, ordered, and support consumer
    groups with acknowledgment.
    """

    # --- Task lifecycle ---
    TASK_ASSIGNED = "task_assigned"
    """A new task has been assigned to an agent."""

    TASK_COMPLETED = "task_completed"
    """An agent has completed its assigned task."""

    TASK_FAILED = "task_failed"
    """An agent's task has failed."""

    # --- File operations (serialisation queue) ---
    FILE_WRITE_REQUEST = "file_write_request"
    """An agent requests to write to a file in the shared workspace.

    The ``FileWriteHandler`` processes these sequentially to prevent
    concurrent write corruption.
    """

    FILE_WRITE_COMPLETED = "file_write_completed"
    """A file write has been completed and the lock released."""

    # --- Schema & API changes ---
    SCHEMA_CHANGED = "schema_changed"
    """An agent has modified an API schema, database model, or interface
    definition that other agents depend on.

    Payload should include:
    - ``file_path``: path to the changed file
    - ``diff``: git-style diff of the changes
    - ``agent``: agent that made the change
    """

    # --- Agent lifecycle ---
    AGENT_STARTED = "agent_started"
    """An agent has been provisioned and is ready."""

    AGENT_COMPLETED = "agent_completed"
    """An agent has finished all its work."""

    AGENT_FAILED = "agent_failed"
    """An agent has encountered a fatal error."""

    # --- Escalation ---
    ESCALATION_REQUIRED = "escalation_required"
    """A circuit breaker has tripped or a human review is needed."""

    # --- Memory ---
    MEMORY_EXTRACTED = "memory_extracted"
    """New knowledge has been extracted to long-term memory."""


class BroadcastChannel(str, Enum):
    """Real-time Pub/Sub channels for interrupt-style notifications.

    These are fire-and-forget.  Listeners must be active to receive them.
    """

    SCHEMA_CHANGE_INTERRUPT = "schema_change_interrupt"
    """Immediate interrupt: an API schema has changed.

    When this fires, the orchestrator should:
    1. Pause affected agents (those depending on the changed schema).
    2. Inject the diff into their context.
    3. Force them to re-evaluate their current work.
    """

    AGENT_INTERRUPT = "agent_interrupt"
    """Generic interrupt sent to a specific agent.

    Payload should include:
    - ``target_agent``: agent name to interrupt
    - ``reason``: why the interrupt was sent
    - ``context``: additional context to inject
    """

    HEARTBEAT = "heartbeat"
    """Periodic heartbeat from agents to the orchestrator.

    Used for liveness detection.  If an agent misses N heartbeats,
    the orchestrator can mark it as failed and restart.
    """

    SHUTDOWN = "shutdown"
    """Graceful shutdown signal broadcast to all agents."""

    CONTEXT_UPDATE = "context_update"
    """New context information is available for an agent.

    Used when the compressor produces a new summary or when LTM
    retrieves relevant memories mid-execution.
    """


# ---------------------------------------------------------------------------
# Standard event type constants
# ---------------------------------------------------------------------------

class EventType:
    """Well-known event type strings used within channels."""

    # Task lifecycle
    NEW_TASK = "new_task"
    TASK_RESULT = "task_result"
    TASK_ERROR = "task_error"

    # File operations
    WRITE_REQUEST = "write_request"
    WRITE_COMPLETE = "write_complete"
    WRITE_REJECTED = "write_rejected"

    # Schema
    SCHEMA_UPDATED = "schema_updated"
    SCHEMA_DIFF = "schema_diff"

    # Agent
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    # Escalation
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    HUMAN_REVIEW_NEEDED = "human_review_needed"

    # Operational
    HEARTBEAT_PING = "heartbeat_ping"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    CONTEXT_INJECTED = "context_injected"
    MEMORY_STORED = "memory_stored"
