"""Swarm state management — the single source of truth for execution state.

The ``SwarmState`` class is a mutable, thread-safe (via asyncio lock) state
container that tracks:

* **Task states** — each task's lifecycle from ``PENDING`` → ``COMPLETED``.
* **Agent states** — whether each agent is idle, running, etc.
* **Message ledger** — ordered log of inter-agent and orchestrator messages.
* **Artifact registry** — files produced by agents.
* **Execution context** — arbitrary key-value store for cross-cutting data
  (e.g. compressed summaries, schema versions).

Design principles
-----------------
* All mutations go through explicit methods that validate transitions.
* The state can be **snapshot**-ed at any time into an immutable Pydantic model
  for safe passing to LLMs or logging without race conditions.
* A per-pair message counter powers the circuit breaker (see ``circuit_breaker.py``).
"""

from __future__ import annotations

import asyncio
import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, enum.Enum):
    """Lifecycle states for an orchestrated task."""

    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class AgentStatus(str, enum.Enum):
    """Runtime states for a worker agent."""

    INITIALISING = "initialising"
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, enum.Enum):
    """Who sent a message."""

    ORCHESTRATOR = "orchestrator"
    AGENT = "agent"
    SYSTEM = "system"
    HUMAN = "human"


# ---------------------------------------------------------------------------
# Immutable data models (used for snapshots & serialisation)
# ---------------------------------------------------------------------------


class TaskRecord(BaseModel):
    """Immutable snapshot of one task's state."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    assigned_agent: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRecord(BaseModel):
    """Immutable snapshot of one agent's runtime state."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: AgentStatus = AgentStatus.INITIALISING
    sandbox_id: str | None = None
    current_task_id: str | None = None
    iterations: int = 0
    token_usage: TokenUsage = Field(default_factory=lambda: TokenUsage())
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TokenUsage(BaseModel):
    """Accumulated token usage for an agent."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Message(BaseModel):
    """A single message in the inter-agent communication ledger."""

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: MessageRole
    sender: str = Field(description="Agent name or 'orchestrator'.")
    recipient: str = Field(description="Agent name or 'orchestrator' or '*' for broadcast.")
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SwarmStateSnapshot(BaseModel):
    """Immutable, serialisable snapshot of the entire swarm state.

    Produced by ``SwarmState.snapshot()`` — safe to pass to LLMs, log, or
    send over the event bus without race conditions.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    swarm_name: str
    tasks: dict[str, TaskRecord] = Field(default_factory=dict)
    agents: dict[str, AgentRecord] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Map of artifact name → file path.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary cross-cutting execution context.",
    )
    pair_message_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Message count per agent pair, keyed as 'agentA<->agentB'. "
            "Used by the circuit breaker."
        ),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------------------------

_VALID_TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.ESCALATED},
    TaskStatus.BLOCKED: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.ESCALATED},
    TaskStatus.COMPLETED: set(),  # terminal
    TaskStatus.FAILED: {TaskStatus.PENDING},  # allow retry
    TaskStatus.ESCALATED: {TaskStatus.PENDING, TaskStatus.FAILED},  # human can retry or abort
}

_VALID_AGENT_TRANSITIONS: dict[AgentStatus, set[AgentStatus]] = {
    AgentStatus.INITIALISING: {AgentStatus.IDLE, AgentStatus.FAILED},
    AgentStatus.IDLE: {AgentStatus.RUNNING, AgentStatus.COMPLETED, AgentStatus.FAILED},
    AgentStatus.RUNNING: {AgentStatus.IDLE, AgentStatus.WAITING, AgentStatus.INTERRUPTED, AgentStatus.COMPLETED, AgentStatus.FAILED},
    AgentStatus.WAITING: {AgentStatus.RUNNING, AgentStatus.INTERRUPTED, AgentStatus.FAILED},
    AgentStatus.INTERRUPTED: {AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.FAILED},
    AgentStatus.COMPLETED: {AgentStatus.IDLE},  # allow re-dispatch as helper
    AgentStatus.FAILED: {AgentStatus.INITIALISING, AgentStatus.IDLE},  # allow restart/recovery
}


class InvalidTransitionError(Exception):
    """Raised when a state transition violates the allowed transition graph."""


# ---------------------------------------------------------------------------
# Mutable state container
# ---------------------------------------------------------------------------


class SwarmState:
    """Mutable, async-safe state container for the orchestration engine.

    All mutations acquire a per-instance ``asyncio.Lock`` to prevent
    concurrent modification from parallel agent coroutines.

    Usage::

        state = SwarmState(swarm_name="my-project")

        # Register agents
        await state.register_agent("backend-agent")

        # Transition states
        await state.set_agent_status("backend-agent", AgentStatus.RUNNING)

        # Record tasks
        task_id = await state.create_task("implement-api", "backend-agent")
        await state.set_task_status(task_id, TaskStatus.RUNNING)

        # Take a snapshot (for passing to LLMs or logging)
        snap = await state.snapshot()
    """

    def __init__(self, swarm_name: str) -> None:
        self._lock = asyncio.Lock()
        self._session_id: str = uuid.uuid4().hex[:16]
        self._swarm_name: str = swarm_name
        self._created_at: datetime = datetime.now(timezone.utc)

        # Core state stores
        self._tasks: dict[str, TaskRecord] = {}
        self._agents: dict[str, AgentRecord] = {}
        self._messages: list[Message] = []
        self._artifacts: dict[str, str] = {}  # name → path
        self._context: dict[str, Any] = {}

        # Circuit breaker support: counts per ordered agent pair
        self._pair_message_counts: dict[str, int] = {}

    @property
    def session_id(self) -> str:
        return self._session_id

    # --- Agent management ---

    async def register_agent(self, name: str, sandbox_id: str | None = None) -> AgentRecord:
        """Register a new agent in the state. Idempotent on the same name."""
        async with self._lock:
            if name in self._agents:
                return self._agents[name]
            record = AgentRecord(name=name, sandbox_id=sandbox_id)
            self._agents[name] = record
            return record

    async def set_agent_status(self, name: str, status: AgentStatus) -> AgentRecord:
        """Transition an agent's status, enforcing valid transitions."""
        async with self._lock:
            record = self._agents.get(name)
            if record is None:
                raise KeyError(f"Unknown agent: '{name}'")

            current = record.status
            if status not in _VALID_AGENT_TRANSITIONS[current]:
                raise InvalidTransitionError(
                    f"Agent '{name}': cannot transition {current.value} → {status.value}. "
                    f"Allowed: {[s.value for s in _VALID_AGENT_TRANSITIONS[current]]}"
                )

            updates: dict[str, Any] = {"status": status}
            if status == AgentStatus.RUNNING and record.started_at is None:
                updates["started_at"] = datetime.now(timezone.utc)
            if status in {AgentStatus.COMPLETED, AgentStatus.FAILED}:
                updates["finished_at"] = datetime.now(timezone.utc)

            updated = record.model_copy(update=updates)
            self._agents[name] = updated
            return updated

    async def update_agent_iterations(self, name: str, iterations: int) -> None:
        """Update the iteration count for an agent."""
        async with self._lock:
            record = self._agents.get(name)
            if record is None:
                raise KeyError(f"Unknown agent: '{name}'")
            self._agents[name] = record.model_copy(update={"iterations": iterations})

    async def update_agent_tokens(self, name: str, usage: TokenUsage) -> None:
        """Accumulate token usage for an agent."""
        async with self._lock:
            record = self._agents.get(name)
            if record is None:
                raise KeyError(f"Unknown agent: '{name}'")
            current = record.token_usage
            merged = TokenUsage(
                prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
                completion_tokens=current.completion_tokens + usage.completion_tokens,
                total_tokens=current.total_tokens + usage.total_tokens,
            )
            self._agents[name] = record.model_copy(update={"token_usage": merged})

    async def get_agent(self, name: str) -> AgentRecord | None:
        """Get a snapshot of an agent's state (or None)."""
        async with self._lock:
            return self._agents.get(name)

    # --- Task management ---

    async def create_task(self, name: str, assigned_agent: str) -> str:
        """Create a new task and return its ID."""
        async with self._lock:
            if assigned_agent not in self._agents:
                raise KeyError(f"Cannot assign task to unknown agent: '{assigned_agent}'")
            record = TaskRecord(name=name, assigned_agent=assigned_agent)
            self._tasks[record.task_id] = record

            # Link agent to task
            agent = self._agents[assigned_agent]
            self._agents[assigned_agent] = agent.model_copy(
                update={"current_task_id": record.task_id},
            )

            return record.task_id

    async def set_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> TaskRecord:
        """Transition a task's status, enforcing valid transitions."""
        async with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise KeyError(f"Unknown task: '{task_id}'")

            current = record.status
            if status not in _VALID_TASK_TRANSITIONS[current]:
                raise InvalidTransitionError(
                    f"Task '{record.name}' ({task_id}): cannot transition "
                    f"{current.value} → {status.value}. "
                    f"Allowed: {[s.value for s in _VALID_TASK_TRANSITIONS[current]]}"
                )

            updates: dict[str, Any] = {
                "status": status,
                "updated_at": datetime.now(timezone.utc),
            }
            if result is not None:
                updates["result"] = result
            if error is not None:
                updates["error"] = error

            updated = record.model_copy(update=updates)
            self._tasks[task_id] = updated
            return updated

    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Get a snapshot of a task (or None)."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def get_tasks_by_status(self, status: TaskStatus) -> list[TaskRecord]:
        """Return all tasks matching a given status."""
        async with self._lock:
            return [t for t in self._tasks.values() if t.status == status]

    # --- Message ledger ---

    async def add_message(
        self,
        role: MessageRole,
        sender: str,
        recipient: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Record a message and increment the pair counter for circuit breaking."""
        async with self._lock:
            msg = Message(
                role=role,
                sender=sender,
                recipient=recipient,
                content=content,
                metadata=metadata or {},
            )
            self._messages.append(msg)

            # Update pair counter (sorted key so A→B and B→A share the same key)
            if sender != "orchestrator" and recipient not in {"orchestrator", "*"}:
                pair_key = "<->".join(sorted([sender, recipient]))
                self._pair_message_counts[pair_key] = (
                    self._pair_message_counts.get(pair_key, 0) + 1
                )

            return msg

    async def get_pair_message_count(self, agent_a: str, agent_b: str) -> int:
        """Return the number of messages exchanged between two specific agents."""
        async with self._lock:
            pair_key = "<->".join(sorted([agent_a, agent_b]))
            return self._pair_message_counts.get(pair_key, 0)

    async def get_messages(
        self,
        *,
        sender: str | None = None,
        recipient: str | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Query messages with optional filters."""
        async with self._lock:
            result = list(self._messages)
            if sender is not None:
                result = [m for m in result if m.sender == sender]
            if recipient is not None:
                result = [m for m in result if m.recipient == recipient]
            if limit is not None:
                result = result[-limit:]
            return result

    # --- Artifact registry ---

    async def register_artifact(self, name: str, path: str) -> None:
        """Register a file artifact produced by an agent."""
        async with self._lock:
            self._artifacts[name] = path

    async def get_artifact(self, name: str) -> str | None:
        """Look up an artifact path by name."""
        async with self._lock:
            return self._artifacts.get(name)

    # --- Execution context ---

    async def set_context(self, key: str, value: Any) -> None:
        """Set an arbitrary key in the execution context."""
        async with self._lock:
            self._context[key] = value

    async def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the execution context."""
        async with self._lock:
            return self._context.get(key, default)

    # --- Snapshot ---

    async def snapshot(self) -> SwarmStateSnapshot:
        """Produce an immutable snapshot of the entire state.

        This is safe to serialise, pass to an LLM, or log.
        """
        async with self._lock:
            return SwarmStateSnapshot(
                session_id=self._session_id,
                swarm_name=self._swarm_name,
                tasks=dict(self._tasks),
                agents=dict(self._agents),
                messages=list(self._messages),
                artifacts=dict(self._artifacts),
                context=dict(self._context),
                pair_message_counts=dict(self._pair_message_counts),
                created_at=self._created_at,
            )

    # --- Summary (for LLM context injection) ---

    async def summary(self) -> str:
        """Return a concise human-readable summary of current state."""
        snap = await self.snapshot()
        lines: list[str] = [
            f"Session: {snap.session_id} | Swarm: {snap.swarm_name}",
            f"Agents ({len(snap.agents)}):",
        ]
        for a in snap.agents.values():
            lines.append(
                f"  • {a.name}: {a.status.value} "
                f"(iter={a.iterations}, tokens={a.token_usage.total_tokens})"
            )
        lines.append(f"Tasks ({len(snap.tasks)}):")
        for t in snap.tasks.values():
            lines.append(f"  • [{t.status.value}] {t.name} → {t.assigned_agent}")
        lines.append(f"Messages: {len(snap.messages)} total")
        lines.append(f"Artifacts: {len(snap.artifacts)}")
        return "\n".join(lines)
