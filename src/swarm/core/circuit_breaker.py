"""Circuit breaker — prevents infinite ping-pong loops between agents.

The breaker monitors ``pair_message_counts`` from :class:`SwarmState` and
enforces a configurable threshold on back-and-forth exchanges between any
two agents working on the same task.

State machine
-------------

.. code-block:: text

    ┌────────┐   threshold exceeded   ┌────────┐
    │ CLOSED │ ───────────────────────►│  OPEN  │
    │(normal)│                         │(block) │
    └───┬────┘                         └───┬────┘
        │                                  │
        │            cooldown expires      │
        │                                  ▼
        │                             ┌──────────┐
        │       success               │HALF_OPEN │
        └─────────────────────────────│ (retry 1)│
                                      └──────────┘
                failure ──► back to OPEN

Escalation strategies
---------------------
* ``pause_and_notify`` — Freeze the task, emit an event for human review.
* ``auto_summarize_and_retry`` — Use the compressor to summarise the
  conversation so far, reset the counter, and retry once.
* ``abort_task`` — Mark the task as FAILED immediately.
"""

from __future__ import annotations

import asyncio
import enum
import time
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from swarm.config.models import CircuitBreakerConfig
from swarm.core.state import AgentStatus, SwarmState, TaskStatus

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Breaker states
# ---------------------------------------------------------------------------


class BreakerState(str, enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Tripped — blocking further exchanges
    HALF_OPEN = "half_open"  # Cooldown expired — allow one retry


# ---------------------------------------------------------------------------
# Per-pair breaker record
# ---------------------------------------------------------------------------


class PairBreakerRecord(BaseModel):
    """Tracks circuit breaker state for a specific agent pair."""

    model_config = ConfigDict(frozen=True)

    pair_key: str = Field(description="Sorted agent pair key, e.g. 'agentA<->agentB'.")
    state: BreakerState = BreakerState.CLOSED
    trip_count: int = Field(default=0, description="Total times this pair has tripped.")
    last_trip_time: float | None = Field(default=None, description="Epoch timestamp of last trip.")
    message_count_at_trip: int = Field(default=0, description="Message count when last tripped.")


# ---------------------------------------------------------------------------
# Escalation event payload
# ---------------------------------------------------------------------------


class EscalationEvent(BaseModel):
    """Payload emitted when a circuit breaker trips."""

    model_config = ConfigDict(frozen=True)

    pair_key: str
    agent_a: str
    agent_b: str
    message_count: int
    threshold: int
    strategy: str
    trip_count: int
    summary: str = Field(default="", description="Context summary for human review.")


# ---------------------------------------------------------------------------
# Circuit Breaker Manager
# ---------------------------------------------------------------------------


class CircuitBreakerManager:
    """Manages circuit breakers for all agent pairs in the swarm.

    Usage::

        breaker = CircuitBreakerManager(config, state)

        # Check before routing a message between two agents
        allowed, event = await breaker.check("agent-a", "agent-b")
        if not allowed:
            # Handle escalation — event contains details
            ...

        # After cooldown, attempt to reset a tripped pair
        await breaker.attempt_reset("agent-a", "agent-b")
    """

    def __init__(self, config: CircuitBreakerConfig, state: SwarmState) -> None:
        self._config = config
        self._state = state
        self._lock = asyncio.Lock()
        self._breakers: dict[str, PairBreakerRecord] = {}

    @staticmethod
    def _pair_key(agent_a: str, agent_b: str) -> str:
        """Generate a deterministic, order-independent key for an agent pair."""
        return "<->".join(sorted([agent_a, agent_b]))

    async def check(
        self,
        sender: str,
        recipient: str,
    ) -> tuple[bool, EscalationEvent | None]:
        """Check whether a message from ``sender`` to ``recipient`` is allowed.

        Returns
        -------
        (allowed, escalation_event)
            ``allowed`` is True if the message should proceed.
            If False, ``escalation_event`` contains details for handling.
        """
        pair_key = self._pair_key(sender, recipient)

        async with self._lock:
            record = self._breakers.get(
                pair_key,
                PairBreakerRecord(pair_key=pair_key),
            )

            # --- OPEN state: check if cooldown has expired ---
            if record.state == BreakerState.OPEN:
                if record.last_trip_time is not None:
                    elapsed = time.monotonic() - record.last_trip_time
                    if elapsed >= self._config.cooldown_seconds:
                        # Transition to HALF_OPEN — allow one retry
                        record = record.model_copy(update={"state": BreakerState.HALF_OPEN})
                        self._breakers[pair_key] = record
                        await logger.info(
                            "circuit_breaker.half_open",
                            pair=pair_key,
                            cooldown_elapsed=round(elapsed, 1),
                        )
                        return True, None

                # Still in cooldown — block
                event = self._build_escalation_event(pair_key, sender, recipient, record)
                return False, event

            # --- HALF_OPEN state: allow one message, then re-evaluate ---
            if record.state == BreakerState.HALF_OPEN:
                # We already allowed one retry — check if count is still high
                count = await self._state.get_pair_message_count(sender, recipient)
                if count > record.message_count_at_trip + 1:
                    # Retry failed — back to OPEN
                    record = record.model_copy(
                        update={
                            "state": BreakerState.OPEN,
                            "last_trip_time": time.monotonic(),
                            "trip_count": record.trip_count + 1,
                        }
                    )
                    self._breakers[pair_key] = record
                    event = self._build_escalation_event(pair_key, sender, recipient, record)
                    await logger.warning(
                        "circuit_breaker.re_tripped",
                        pair=pair_key,
                        trip_count=record.trip_count,
                    )
                    return False, event
                # Still within bounds — allow
                return True, None

            # --- CLOSED state: check threshold ---
            count = await self._state.get_pair_message_count(sender, recipient)

            if count >= self._config.max_round_trips:
                # Trip the breaker
                record = record.model_copy(
                    update={
                        "state": BreakerState.OPEN,
                        "trip_count": record.trip_count + 1,
                        "last_trip_time": time.monotonic(),
                        "message_count_at_trip": count,
                    }
                )
                self._breakers[pair_key] = record

                event = self._build_escalation_event(pair_key, sender, recipient, record)

                await logger.warning(
                    "circuit_breaker.tripped",
                    pair=pair_key,
                    message_count=count,
                    threshold=self._config.max_round_trips,
                    strategy=self._config.escalation_strategy,
                )

                return False, event

            # Under threshold — allow
            return True, None

    async def attempt_reset(self, agent_a: str, agent_b: str) -> bool:
        """Manually reset a tripped breaker back to CLOSED.

        Returns True if the breaker was in a tripped state and was reset.
        """
        pair_key = self._pair_key(agent_a, agent_b)

        async with self._lock:
            record = self._breakers.get(pair_key)
            if record is None or record.state == BreakerState.CLOSED:
                return False

            self._breakers[pair_key] = record.model_copy(
                update={"state": BreakerState.CLOSED}
            )
            await logger.info(
                "circuit_breaker.manual_reset",
                pair=pair_key,
                previous_state=record.state.value,
            )
            return True

    async def get_status(self) -> dict[str, PairBreakerRecord]:
        """Return a snapshot of all breaker states."""
        async with self._lock:
            return dict(self._breakers)

    async def get_pair_status(self, agent_a: str, agent_b: str) -> PairBreakerRecord | None:
        """Return the breaker record for a specific pair."""
        pair_key = self._pair_key(agent_a, agent_b)
        async with self._lock:
            return self._breakers.get(pair_key)

    def _build_escalation_event(
        self,
        pair_key: str,
        agent_a: str,
        agent_b: str,
        record: PairBreakerRecord,
    ) -> EscalationEvent:
        """Build a structured escalation event payload."""
        return EscalationEvent(
            pair_key=pair_key,
            agent_a=agent_a,
            agent_b=agent_b,
            message_count=record.message_count_at_trip,
            threshold=self._config.max_round_trips,
            strategy=self._config.escalation_strategy,
            trip_count=record.trip_count,
        )

    async def handle_escalation(
        self,
        event: EscalationEvent,
    ) -> dict[str, Any]:
        """Execute the configured escalation strategy.

        This is called by the orchestrator when a breaker trips.

        Returns
        -------
        dict
            Strategy-specific result payload.
        """
        strategy = event.strategy

        if strategy == "pause_and_notify":
            return await self._strategy_pause_and_notify(event)
        elif strategy == "auto_summarize_and_retry":
            return await self._strategy_auto_summarize_and_retry(event)
        elif strategy == "abort_task":
            return await self._strategy_abort_task(event)
        else:
            await logger.warning(
                "circuit_breaker.unknown_strategy",
                strategy=strategy,
                falling_back_to="pause_and_notify",
            )
            return await self._strategy_pause_and_notify(event)

    async def _strategy_pause_and_notify(
        self, event: EscalationEvent,
    ) -> dict[str, Any]:
        """Pause both agents and flag for human review."""
        for agent_name in (event.agent_a, event.agent_b):
            agent = await self._state.get_agent(agent_name)
            if agent and agent.status == AgentStatus.RUNNING:
                try:
                    await self._state.set_agent_status(
                        agent_name, AgentStatus.INTERRUPTED,
                    )
                except Exception:
                    pass  # Agent may already be in a terminal state

        await logger.warning(
            "circuit_breaker.escalation.paused",
            pair=event.pair_key,
            message="Both agents paused. Human review required.",
        )

        return {
            "action": "paused",
            "agents_paused": [event.agent_a, event.agent_b],
            "requires_human_review": True,
        }

    async def _strategy_auto_summarize_and_retry(
        self, event: EscalationEvent,
    ) -> dict[str, Any]:
        """Summarise the conversation, reset counter, and retry once.

        Note: The actual summarisation is delegated to the Compressor —
        the orchestrator should call ``compressor.compress_conversation()``
        and then ``breaker.attempt_reset()`` after this returns.
        """
        await logger.info(
            "circuit_breaker.escalation.summarize_and_retry",
            pair=event.pair_key,
            message="Requesting conversation summary before retry.",
        )

        return {
            "action": "summarize_and_retry",
            "requires_compression": True,
            "pair_key": event.pair_key,
            "agents": [event.agent_a, event.agent_b],
        }

    async def _strategy_abort_task(
        self, event: EscalationEvent,
    ) -> dict[str, Any]:
        """Mark related tasks as FAILED."""
        for agent_name in (event.agent_a, event.agent_b):
            agent = await self._state.get_agent(agent_name)
            if agent and agent.current_task_id:
                try:
                    await self._state.set_task_status(
                        agent.current_task_id,
                        TaskStatus.FAILED,
                        error=f"Circuit breaker tripped after {event.message_count} exchanges.",
                    )
                except Exception:
                    pass

        await logger.warning(
            "circuit_breaker.escalation.aborted",
            pair=event.pair_key,
            message="Tasks aborted due to excessive inter-agent loops.",
        )

        return {
            "action": "aborted",
            "agents_affected": [event.agent_a, event.agent_b],
        }
