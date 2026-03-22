"""Redis-backed event bus — pub/sub, streams, and distributed file locks.

Provides three communication primitives for the swarm:

1. **Streams** — Durable, ordered event log using Redis Streams with consumer
   groups and acknowledgment.  Used for task assignment, completion, and
   schema change notifications.

2. **Pub/Sub** — Real-time broadcast channel for interrupt-style events.
   Used when the orchestrator needs to immediately notify agents (e.g.
   "Schema changed — pause and re-read context").

3. **Distributed file write lock** — Redis-backed async lock that serialises
   concurrent write requests to shared workspace files.  Without this, two
   parallel agents writing to the same file simultaneously would corrupt the
   codebase.

Lock implementation
-------------------
Uses Redis ``SET key value NX PX <timeout>`` for atomic acquire and Lua
script for safe release (only the lock holder can release).  This is the
standard Redlock single-instance pattern, which is sufficient for a
single-host Docker swarm.

.. code-block:: text

    Agent A wants to write foo.py        Agent B wants to write foo.py
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                        ┌─────────────┐
    │ ACQUIRE     │──── granted ─────────► │ ACQUIRE     │
    │ file:foo.py │                        │ file:foo.py │──── denied (wait)
    └─────────────┘                        └──────┬──────┘
           │                                      │
           ▼                                      │
    ┌─────────────┐                               │
    │  WRITE      │                               │
    │  foo.py     │                               │
    └─────────────┘                               │
           │                                      │
           ▼                                      │
    ┌─────────────┐                               │
    │  RELEASE    │──── lock freed ──────────────►│
    │  file:foo.py│                        ┌──────▼──────┐
    └─────────────┘                        │ ACQUIRE     │
                                           │ file:foo.py │──── granted
                                           └─────────────┘
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from functools import wraps
from typing import Any, Callable, Coroutine

import structlog
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from swarm.config.models import EventBusConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class SwarmEvent(BaseModel):
    """A typed event flowing through the bus."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    channel: str = Field(description="Event channel/topic name.")
    event_type: str = Field(description="Specific event type within the channel.")
    sender: str = Field(description="Agent or component that emitted this event.")
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# Type alias for event handlers
EventHandler = Callable[[SwarmEvent], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Lua scripts for atomic lock operations
# ---------------------------------------------------------------------------

# Release lock only if the caller owns it (value matches)
_RELEASE_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""

# Extend lock TTL only if the caller owns it
_EXTEND_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("PEXPIRE", KEYS[1], ARGV[2])
else
    return 0
end
"""


# ---------------------------------------------------------------------------
# Distributed File Write Lock
# ---------------------------------------------------------------------------


class FileWriteLock:
    """Distributed lock for serialising file writes to the shared workspace.

    Uses Redis SET NX PX for atomic acquisition and Lua scripts for safe
    release.  Implements async context manager for convenient usage::

        async with bus.file_lock("src/api/routes.py") as acquired:
            if acquired:
                # Safe to write to the file
                ...

    V2: Includes a background **heartbeat** that extends the lock TTL
    every ``timeout_ms // 3`` ms, preventing expiry during long LLM
    generation calls (>30s).  The heartbeat is automatically started on
    ``acquire()`` and cancelled on ``release()``.
    """

    def __init__(
        self,
        redis_client: Any,
        file_path: str,
        *,
        timeout_ms: int = 30_000,
        retry_interval: float = 0.1,
        max_retries: int = 100,
        prefix: str = "swarm:filelock",
    ) -> None:
        self._redis = redis_client
        self._file_path = file_path
        self._lock_key = f"{prefix}:{file_path}"
        self._reader_key = f"{prefix}:readers:{file_path}"
        self._lock_value = uuid.uuid4().hex  # Unique owner token
        self._timeout_ms = timeout_ms
        self._retry_interval = retry_interval
        self._max_retries = max_retries
        self._acquired = False
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def acquire(self) -> bool:
        """Attempt to acquire the write lock, retrying up to max_retries.

        MRSW enforcement: the lock is only granted when both the exclusive
        write key is free AND the reader counter is zero.
        """
        for attempt in range(self._max_retries):
            # MRSW: check reader count first
            reader_count = await self._redis.get(self._reader_key)
            if reader_count and int(reader_count) > 0:
                await asyncio.sleep(self._retry_interval)
                continue

            result = await self._redis.set(
                self._lock_key,
                self._lock_value,
                nx=True,   # Only set if not exists
                px=self._timeout_ms,  # Auto-expire after timeout
            )
            if result:
                self._acquired = True
                # V2: Start heartbeat to keep TTL alive during long operations
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(),
                    name=f"lock-heartbeat-{self._file_path}",
                )
                await logger.info(
                    "filelock.acquired",
                    file=self._file_path,
                    attempt=attempt + 1,
                )
                return True

            # Lock is held by someone else — wait and retry
            await asyncio.sleep(self._retry_interval)

        await logger.warning(
            "filelock.acquire_failed",
            file=self._file_path,
            max_retries=self._max_retries,
        )
        return False

    async def release(self) -> bool:
        """Release the lock (only if we are the owner).

        V2: Cancels the heartbeat task before releasing to prevent
        orphaned background tasks.
        """
        # V2: Stop heartbeat first
        await self._stop_heartbeat()

        if not self._acquired:
            return False

        try:
            result = await self._redis.eval(
                _RELEASE_LOCK_SCRIPT,
                1,
                self._lock_key,
                self._lock_value,
            )
            self._acquired = False
            if result:
                await logger.info("filelock.released", file=self._file_path)
            return bool(result)
        except Exception as exc:
            await logger.error(
                "filelock.release_failed",
                file=self._file_path,
                error=str(exc),
            )
            return False

    async def extend(self, additional_ms: int = 10_000) -> bool:
        """Extend the lock TTL (useful for long-running writes)."""
        if not self._acquired:
            return False

        try:
            result = await self._redis.eval(
                _EXTEND_LOCK_SCRIPT,
                1,
                self._lock_key,
                self._lock_value,
                str(additional_ms),
            )
            return bool(result)
        except Exception:
            return False

    # ---- V2: Lock heartbeat -------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Background task that extends the lock TTL at 1/3 of timeout.

        This ensures the lock never expires while the owning agent is
        still actively working (e.g. waiting on a 60s LLM generation).
        """
        interval = self._timeout_ms / 3_000  # Convert ms → seconds, renew at 1/3
        try:
            while self._acquired:
                await asyncio.sleep(interval)
                if not self._acquired:
                    break
                ok = await self.extend(additional_ms=self._timeout_ms)
                if ok:
                    await logger.debug(
                        "filelock.heartbeat_extended",
                        file=self._file_path,
                        ttl_ms=self._timeout_ms,
                    )
                else:
                    await logger.warning(
                        "filelock.heartbeat_lost",
                        file=self._file_path,
                    )
                    break
        except asyncio.CancelledError:
            pass

    async def _stop_heartbeat(self) -> None:
        """Cancel the heartbeat task if running."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def __aenter__(self) -> bool:
        return await self.acquire()

    async def __aexit__(self, *_: Any) -> None:
        await self.release()


# Lua script: decrement reader count, never below zero
_DECR_READERS_SCRIPT = """
local current = redis.call("GET", KEYS[1])
if current and tonumber(current) > 0 then
    return redis.call("DECR", KEYS[1])
end
return 0
"""


class FileReadLock:
    """Distributed read lock for the MRSW pattern.

    Multiple readers can hold the lock simultaneously.  Uses a Redis
    counter key (INCR/DECR) with a TTL failsafe.  Writers check this
    counter before acquiring the exclusive write lock::

        async with bus.file_read_lock("src/api/routes.py") as acquired:
            if acquired:
                # Safe to read — writers are blocked
                content = read_file(...)
    """

    def __init__(
        self,
        redis_client: Any,
        file_path: str,
        *,
        timeout_ms: int = 30_000,
        prefix: str = "swarm:filelock",
    ) -> None:
        self._redis = redis_client
        self._file_path = file_path
        self._reader_key = f"{prefix}:readers:{file_path}"
        self._write_key = f"{prefix}:{file_path}"
        self._timeout_ms = timeout_ms
        self._acquired = False

    async def acquire(self) -> bool:
        """Acquire a read slot — blocks if a writer holds the exclusive lock."""
        for _ in range(100):
            # Check if a writer holds the exclusive lock
            write_held = await self._redis.get(self._write_key)
            if write_held:
                await asyncio.sleep(0.1)
                continue

            # Increment reader count (atomic)
            await self._redis.incr(self._reader_key)
            # Set TTL on the reader key as failsafe
            await self._redis.pexpire(self._reader_key, self._timeout_ms)
            self._acquired = True
            return True

        return False

    async def release(self) -> bool:
        """Release the read slot (decrement reader count)."""
        if not self._acquired:
            return False
        try:
            await self._redis.eval(_DECR_READERS_SCRIPT, 1, self._reader_key)
            self._acquired = False
            return True
        except Exception:
            return False

    async def __aenter__(self) -> bool:
        return await self.acquire()

    async def __aexit__(self, *_: Any) -> None:
        await self.release()


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """Redis-backed event bus with streams, pub/sub, and file locking.

    Usage::

        bus = EventBus(config)
        await bus.connect()

        # Publish an event to a stream (durable)
        await bus.publish("task_assigned", SwarmEvent(
            channel="task_assigned",
            event_type="new_task",
            sender="orchestrator",
            payload={"task_id": "abc123", "agent": "backend"},
        ))

        # Subscribe to a stream (with consumer group)
        await bus.subscribe("task_assigned", handler_function)

        # Broadcast via pub/sub (real-time, fire-and-forget)
        await bus.broadcast("schema_changed", SwarmEvent(...))

        # Listen for broadcasts
        await bus.listen("schema_changed", handler_function)

        # Acquire a file write lock
        async with bus.file_lock("src/main.py") as acquired:
            if acquired:
                write_to_file(...)

        await bus.disconnect()
    """

    def __init__(self, config: EventBusConfig, *, tenant_id: str = "") -> None:
        self._config = config
        self._tenant_id = tenant_id
        self._redis: Any | None = None
        self._pubsub: Any | None = None
        self._stream_handlers: dict[str, list[EventHandler]] = {}
        self._broadcast_handlers: dict[str, list[EventHandler]] = {}
        self._listener_tasks: list[asyncio.Task[None]] = []
        self._running = False

    @property
    def connected(self) -> bool:
        return self._redis is not None

    # -------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Redis and set up consumer groups."""
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(
            self._config.redis_url,
            decode_responses=True,
        )

        # Verify connection
        try:
            await self._redis.ping()
            await logger.info(
                "eventbus.connected",
                url=self._config.redis_url,
            )
        except Exception as exc:
            self._redis = None
            raise ConnectionError(f"Cannot connect to Redis: {exc}") from exc

        self._running = True

    async def disconnect(self) -> None:
        """Disconnect from Redis and cancel all listener tasks."""
        self._running = False

        # Cancel listener tasks
        for task in self._listener_tasks:
            task.cancel()
        if self._listener_tasks:
            await asyncio.gather(*self._listener_tasks, return_exceptions=True)
        self._listener_tasks.clear()

        # Close pub/sub
        if self._pubsub is not None:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None

        # Close Redis connection
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

        await logger.info("eventbus.disconnected")

    # -------------------------------------------------------------------
    # Stream operations (durable, ordered events)
    # -------------------------------------------------------------------

    def _stream_key(self, channel: str) -> str:
        """Build a Redis Stream key from a channel name.

        V2.5: If tenant_id is set, keys are prefixed for isolation:
        ``tenant-{id}:swarm:{channel}`` instead of ``swarm:{channel}``.
        """
        prefix = self._config.stream_prefix
        if self._tenant_id:
            prefix = f"tenant-{self._tenant_id}:{prefix}"
        return f"{prefix}:{channel}"

    async def publish(self, channel: str, event: SwarmEvent) -> str | None:
        """Publish an event to a Redis Stream.

        Returns the stream entry ID, or None on failure.
        V2: Retries up to 3× with exponential backoff on transient Redis errors.
        """
        if not self._redis:
            await logger.warning("eventbus.publish_no_connection", channel=channel)
            return None

        stream_key = self._stream_key(channel)
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "sender": event.sender,
            "payload": json.dumps(event.payload),
            "timestamp": str(event.timestamp),
        }

        try:
            entry_id = await self._publish_with_retry(stream_key, event_data)

            await logger.info(
                "eventbus.published",
                channel=channel,
                event_type=event.event_type,
                sender=event.sender,
                entry_id=entry_id,
            )
            return entry_id
        except Exception as exc:
            await logger.error(
                "eventbus.publish_failed",
                channel=channel,
                error=str(exc),
            )
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=2),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def _publish_with_retry(
        self, stream_key: str, event_data: dict[str, Any],
    ) -> str:
        """Redis XADD with tenacity retry wrapper."""
        return await self._redis.xadd(
            stream_key,
            event_data,
            maxlen=self._config.max_stream_length,
        )

    async def subscribe(
        self,
        channel: str,
        handler: EventHandler,
        *,
        consumer_name: str | None = None,
    ) -> None:
        """Subscribe to a Redis Stream with a consumer group.

        The handler is called for each new event.  Uses consumer groups
        for reliable delivery with acknowledgment.
        """
        if not self._redis:
            raise RuntimeError("EventBus not connected.")

        stream_key = self._stream_key(channel)
        group = self._config.consumer_group
        consumer = consumer_name or f"consumer-{uuid.uuid4().hex[:6]}"

        # Create consumer group (idempotent)
        try:
            await self._redis.xgroup_create(
                stream_key, group, id="0", mkstream=True,
            )
        except Exception:
            # Group already exists — that's fine
            pass

        # Register handler
        self._stream_handlers.setdefault(channel, []).append(handler)

        # Start listener task
        task = asyncio.create_task(
            self._stream_listener(stream_key, group, consumer, channel),
            name=f"stream-listener-{channel}",
        )
        self._listener_tasks.append(task)

        await logger.info(
            "eventbus.subscribed",
            channel=channel,
            group=group,
            consumer=consumer,
        )

    async def _stream_listener(
        self,
        stream_key: str,
        group: str,
        consumer: str,
        channel: str,
    ) -> None:
        """Background task that reads from a stream and dispatches to handlers."""
        while self._running and self._redis:
            try:
                # Read new messages (block for up to 1 second)
                entries = await self._redis.xreadgroup(
                    group,
                    consumer,
                    {stream_key: ">"},  # Only new messages
                    count=10,
                    block=1000,
                )

                if not entries:
                    continue

                for _stream, messages in entries:
                    for msg_id, data in messages:
                        try:
                            event = SwarmEvent(
                                event_id=data.get("event_id", ""),
                                channel=channel,
                                event_type=data.get("event_type", ""),
                                sender=data.get("sender", ""),
                                payload=json.loads(data.get("payload", "{}")),
                                timestamp=float(data.get("timestamp", 0)),
                            )

                            # Dispatch to all handlers
                            handlers = self._stream_handlers.get(channel, [])
                            for handler in handlers:
                                try:
                                    await handler(event)
                                except Exception as exc:
                                    await logger.error(
                                        "eventbus.handler_error",
                                        channel=channel,
                                        event_id=event.event_id,
                                        error=str(exc),
                                    )

                            # Acknowledge the message
                            await self._redis.xack(stream_key, group, msg_id)

                        except Exception as exc:
                            await logger.error(
                                "eventbus.message_parse_error",
                                msg_id=msg_id,
                                error=str(exc),
                            )
                            # Acknowledge bad messages to prevent redelivery loops
                            await self._redis.xack(stream_key, group, msg_id)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                await logger.error(
                    "eventbus.listener_error",
                    channel=channel,
                    error=str(exc),
                )
                await asyncio.sleep(1.0)  # Back off on errors

    # -------------------------------------------------------------------
    # Pub/Sub operations (real-time broadcasts)
    # -------------------------------------------------------------------

    def _pubsub_channel(self, channel: str) -> str:
        """Build a Redis Pub/Sub channel name."""
        prefix = self._config.stream_prefix
        if self._tenant_id:
            prefix = f"tenant-{self._tenant_id}:{prefix}"
        return f"{prefix}:pubsub:{channel}"

    async def broadcast(self, channel: str, event: SwarmEvent) -> int:
        """Broadcast an event via Redis Pub/Sub.

        Returns the number of receivers that got the message.
        Fire-and-forget — no durability guarantees.
        V2: Retries up to 3× with exponential backoff on transient Redis errors.
        """
        if not self._redis:
            return 0

        pubsub_channel = self._pubsub_channel(channel)
        event_json = event.model_dump_json()

        try:
            receivers = await self._broadcast_with_retry(pubsub_channel, event_json)
            await logger.info(
                "eventbus.broadcast",
                channel=channel,
                event_type=event.event_type,
                receivers=receivers,
            )
            return receivers
        except Exception as exc:
            await logger.error(
                "eventbus.broadcast_failed",
                channel=channel,
                error=str(exc),
            )
            return 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=2),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def _broadcast_with_retry(
        self, pubsub_channel: str, event_json: str,
    ) -> int:
        """Redis PUBLISH with tenacity retry wrapper."""
        return await self._redis.publish(pubsub_channel, event_json)

    async def listen(self, channel: str, handler: EventHandler) -> None:
        """Subscribe to real-time broadcasts on a pub/sub channel."""
        if not self._redis:
            raise RuntimeError("EventBus not connected.")

        pubsub_channel = self._pubsub_channel(channel)

        # Register handler
        self._broadcast_handlers.setdefault(channel, []).append(handler)

        # Create pub/sub connection if needed
        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()

        await self._pubsub.subscribe(pubsub_channel)

        # Start listener if not already running
        already_listening = any(
            t.get_name() == "pubsub-listener" for t in self._listener_tasks
        )
        if not already_listening:
            task = asyncio.create_task(
                self._pubsub_listener(),
                name="pubsub-listener",
            )
            self._listener_tasks.append(task)

        await logger.info("eventbus.listening", channel=channel)

    async def _pubsub_listener(self) -> None:
        """Background task that reads from pub/sub and dispatches to handlers."""
        while self._running and self._pubsub:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message and message["type"] == "message":
                    raw_channel = message["channel"]
                    # Extract our channel name from the Redis channel
                    prefix = f"{self._config.stream_prefix}:pubsub:"
                    if raw_channel.startswith(prefix):
                        channel = raw_channel[len(prefix):]
                    else:
                        channel = raw_channel

                    try:
                        event = SwarmEvent.model_validate_json(message["data"])
                    except Exception:
                        continue

                    handlers = self._broadcast_handlers.get(channel, [])
                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as exc:
                            await logger.error(
                                "eventbus.pubsub_handler_error",
                                channel=channel,
                                error=str(exc),
                            )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                await logger.error(
                    "eventbus.pubsub_listener_error",
                    error=str(exc),
                )
                await asyncio.sleep(0.5)

    # -------------------------------------------------------------------
    # File write lock
    # -------------------------------------------------------------------

    def file_lock(
        self,
        file_path: str,
        *,
        timeout_ms: int = 30_000,
    ) -> FileWriteLock:
        """Create a distributed file write lock.

        Usage::

            async with bus.file_lock("src/api/routes.py") as acquired:
                if acquired:
                    # Safe to write
                    ...

        Parameters
        ----------
        file_path:
            Relative path within the workspace (used as the lock key).
        timeout_ms:
            Auto-release timeout in milliseconds (prevents deadlocks).
        """
        if not self._redis:
            raise RuntimeError("EventBus not connected. Cannot create file lock.")

        return FileWriteLock(
            self._redis,
            file_path,
            timeout_ms=timeout_ms,
            prefix=f"{self._config.stream_prefix}:filelock",
        )

    def file_read_lock(
        self,
        file_path: str,
        *,
        timeout_ms: int = 30_000,
    ) -> FileReadLock:
        """Create a distributed file read lock (MRSW reader).

        Multiple readers can hold this simultaneously.  Writers are
        blocked while any reader holds the lock::

            async with bus.file_read_lock("src/main.py") as acquired:
                if acquired:
                    content = read_file(...)

        Parameters
        ----------
        file_path:
            Relative path within the workspace.
        timeout_ms:
            Auto-release timeout in milliseconds.
        """
        if not self._redis:
            raise RuntimeError("EventBus not connected. Cannot create file read lock.")

        return FileReadLock(
            self._redis,
            file_path,
            timeout_ms=timeout_ms,
            prefix=f"{self._config.stream_prefix}:filelock",
        )

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=2),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def stream_length(self, channel: str) -> int:
        """Get the number of entries in a stream.  V2: tenacity retry."""
        if not self._redis:
            return 0
        try:
            return await self._redis.xlen(self._stream_key(channel))
        except (ConnectionError, TimeoutError, OSError):
            raise  # Let tenacity handle these
        except Exception:
            return 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=2),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def flush_stream(self, channel: str) -> None:
        """Delete all entries in a stream (for testing).  V2: tenacity retry."""
        if self._redis:
            await self._redis.delete(self._stream_key(channel))
