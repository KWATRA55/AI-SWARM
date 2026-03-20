"""Session Replay Ledger — granular event capture for post-run analysis.

Records *everything* the swarm does during a session: LLM I/O, token
deltas, tool executions, memory hits, and routing decisions.  Output is
a ``.jsonl`` file (one JSON object per line) that can be analysed offline
to identify token waste, memory misses, and agent routing errors.

Toggleable via ``enable_session_replay: true`` in the YAML config.

Usage::

    from swarm.core.telemetry import SessionLedger

    ledger = SessionLedger(session_id="abc123", output_dir=Path("logs"))
    ledger.record_llm_call(
        agent="architect",
        model="gemini-2.5-pro",
        messages=[...],    # raw prompt
        response_text="...",
        prompt_tokens=1139,
        completion_tokens=861,
        total_tokens=2000,
    )
    ledger.record_tool_execution(
        agent="architect",
        tool_name="write_file",
        arguments={"path": "foo.py", "content": "..."},
        result_str="File written: foo.py (1234 bytes)",
        success=True,
    )
    ledger.close()
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Event schemas (Pydantic for strict typing)
# ---------------------------------------------------------------------------


class LedgerEvent(BaseModel):
    """Base ledger event — all events share these fields."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str
    event_type: str
    agent: str = ""
    wall_clock_ms: float = 0.0


class LLMCallEvent(LedgerEvent):
    """Records a single LLM API call with full I/O."""
    event_type: str = "llm_call"
    model: str = ""
    iteration: int = 0
    # Token accounting
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cumulative_tokens: int = 0
    # Context snapshot
    message_count: int = 0
    system_prompt_chars: int = 0
    # Raw I/O (truncated for sanity — full content in verbose mode)
    prompt_summary: str = ""     # First/last messages summary
    response_text: str = ""      # Full LLM response text
    tool_calls_requested: list[str] = Field(default_factory=list)
    # Compression state
    compression_triggered: bool = False


class ToolExecutionEvent(LedgerEvent):
    """Records a tool call and its result."""
    event_type: str = "tool_execution"
    tool_name: str = ""
    arguments_preview: str = ""  # Truncated args
    result_preview: str = ""     # Truncated result
    result_chars: int = 0
    success: bool = True
    duration_ms: float = 0.0


class MemorySearchEvent(LedgerEvent):
    """Records a vector DB / LTM search."""
    event_type: str = "memory_search"
    query: str = ""
    results_count: int = 0
    top_similarity: float = 0.0
    chunks_retrieved: list[str] = Field(default_factory=list)
    cache_hit: bool = False


class RoutingDecisionEvent(LedgerEvent):
    """Records a manager routing decision."""
    event_type: str = "routing_decision"
    target_agent: str = ""
    task_preview: str = ""
    reason: str = ""


class CompressionEvent(LedgerEvent):
    """Records a context compression."""
    event_type: str = "compression"
    original_tokens: int = 0
    compressed_tokens: int = 0
    ratio: float = 0.0
    messages_before: int = 0
    messages_after: int = 0


class SessionSummaryEvent(LedgerEvent):
    """Final session summary written at close."""
    event_type: str = "session_summary"
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tool_calls: int = 0
    total_files_written: int = 0
    total_files_read: int = 0
    total_memory_searches: int = 0
    total_compressions: int = 0
    agents_used: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class DroppedMessageEvent(LedgerEvent):
    """V2: Records a message dropped by the sliding window.

    Archived here so the compressor can extract LTM knowledge from
    messages that have been evicted from the active context.
    """
    event_type: str = "dropped_message"
    role: str = ""
    content_preview: str = ""
    content_chars: int = 0
    iteration_dropped: int = 0


# ---------------------------------------------------------------------------
# Session Ledger
# ---------------------------------------------------------------------------


class SessionLedger:
    """Append-only JSONL ledger for session replay.

    V2.3 FIX: Uses a **sync file handle** opened in ``__init__`` (which runs
    outside an event loop) so telemetry works immediately.  A background
    async consumer can optionally be started later via ``start_async()``.
    """

    _SENTINEL = object()  # Drain signal for graceful shutdown

    def __init__(
        self,
        *,
        session_id: str,
        output_dir: Path,
        enabled: bool = True,
        verbose: bool = False,
    ) -> None:
        self._session_id = session_id
        self._enabled = enabled
        self._verbose = verbose
        self._start_time = time.monotonic()

        # Counters for session summary
        self._llm_calls = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._tool_calls = 0
        self._files_written = 0
        self._files_read = 0
        self._memory_searches = 0
        self._compressions = 0
        self._agents: set[str] = set()

        # V2.3 FIX: Use sync file handle (works from sync __init__)
        self._file_handle = None
        self._output_path: Path | None = None

        if enabled:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_path = output_dir / f"session_replay_{session_id[:8]}_{ts}.jsonl"
            try:
                self._file_handle = open(self._output_path, "a", encoding="utf-8")
                logger.info(
                    "telemetry.ledger_started",
                    path=str(self._output_path),
                    session_id=session_id,
                )
            except OSError as exc:
                logger.warning("telemetry.ledger_open_failed", error=str(exc))
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def output_path(self) -> Path | None:
        return self._output_path if self._enabled else None

    def _write(self, event: LedgerEvent) -> None:
        """Write a single event synchronously to the JSONL file."""
        if not self._enabled or self._file_handle is None:
            return
        try:
            line = event.model_dump_json() + "\n"
            self._file_handle.write(line)
            self._file_handle.flush()
        except Exception:
            pass  # Telemetry must never crash the swarm

    # -------------------------------------------------------------------
    # LLM call recording
    # -------------------------------------------------------------------

    def record_llm_call(
        self,
        *,
        agent: str,
        model: str,
        messages: list[dict[str, Any]],
        response_text: str,
        tool_calls: list[str] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cumulative_tokens: int = 0,
        iteration: int = 0,
        compression_triggered: bool = False,
        wall_clock_ms: float = 0.0,
    ) -> None:
        """Record an LLM API call with full I/O and token accounting."""
        self._llm_calls += 1
        self._total_tokens += total_tokens
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._agents.add(agent)

        # Build prompt summary (system chars + message count)
        system_chars = 0
        if messages and messages[0].get("role") == "system":
            system_chars = len(messages[0].get("content") or "")

        prompt_summary = ""
        if self._verbose and messages:
            # In verbose mode, capture first and last user message
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                first = (user_msgs[0].get("content") or "")[:500]
                last = (user_msgs[-1].get("content") or "")[:500] if len(user_msgs) > 1 else ""
                prompt_summary = f"FIRST_USER: {first}\nLAST_USER: {last}"
        else:
            # Compact: just message roles summary
            roles = [m.get("role", "?") for m in messages]
            prompt_summary = f"roles=[{','.join(roles)}]"

        self._write(LLMCallEvent(
            session_id=self._session_id,
            agent=agent,
            model=model,
            iteration=iteration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cumulative_tokens=cumulative_tokens,
            message_count=len(messages),
            system_prompt_chars=system_chars,
            prompt_summary=prompt_summary,
            response_text=response_text[:2000],  # Cap for sanity
            tool_calls_requested=tool_calls or [],
            compression_triggered=compression_triggered,
            wall_clock_ms=wall_clock_ms,
        ))

    # -------------------------------------------------------------------
    # Tool execution recording
    # -------------------------------------------------------------------

    def record_tool_execution(
        self,
        *,
        agent: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_str: str,
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a tool call with args and result."""
        self._tool_calls += 1
        if tool_name == "write_file":
            self._files_written += 1
        elif tool_name == "read_file":
            self._files_read += 1

        # Truncate content arg for write_file to prevent huge logs
        args_preview = dict(arguments)
        if "content" in args_preview:
            c = args_preview["content"]
            args_preview["content"] = c[:200] + f"... [{len(c)} chars]" if len(c) > 200 else c

        self._write(ToolExecutionEvent(
            session_id=self._session_id,
            agent=agent,
            tool_name=tool_name,
            arguments_preview=json.dumps(args_preview)[:1000],
            result_preview=result_str[:1000],
            result_chars=len(result_str),
            success=success,
            duration_ms=duration_ms,
        ))

    # -------------------------------------------------------------------
    # Memory / vector DB recording
    # -------------------------------------------------------------------

    def record_memory_search(
        self,
        *,
        agent: str,
        query: str,
        results_count: int = 0,
        top_similarity: float = 0.0,
        chunks: list[str] | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Record a vector DB / LTM search."""
        self._memory_searches += 1
        self._write(MemorySearchEvent(
            session_id=self._session_id,
            agent=agent,
            query=query[:500],
            results_count=results_count,
            top_similarity=top_similarity,
            chunks_retrieved=[c[:200] for c in (chunks or [])[:5]],
            cache_hit=cache_hit,
        ))

    # -------------------------------------------------------------------
    # Routing decision recording
    # -------------------------------------------------------------------

    def record_routing_decision(
        self,
        *,
        agent: str = "manager",
        target_agent: str,
        task_preview: str,
        reason: str = "",
    ) -> None:
        """Record a manager → agent routing decision."""
        self._write(RoutingDecisionEvent(
            session_id=self._session_id,
            agent=agent,
            target_agent=target_agent,
            task_preview=task_preview[:500],
            reason=reason[:500],
        ))

    # -------------------------------------------------------------------
    # Compression recording
    # -------------------------------------------------------------------

    def record_compression(
        self,
        *,
        agent: str,
        original_tokens: int,
        compressed_tokens: int,
        ratio: float,
        messages_before: int,
        messages_after: int,
    ) -> None:
        """Record a context compression event."""
        self._compressions += 1
        self._write(CompressionEvent(
            session_id=self._session_id,
            agent=agent,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ratio=ratio,
            messages_before=messages_before,
            messages_after=messages_after,
        ))

    # -------------------------------------------------------------------
    # Session close
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Write final session summary and close the file handle."""
        if not self._enabled:
            return

        elapsed = time.monotonic() - self._start_time
        self._write(SessionSummaryEvent(
            session_id=self._session_id,
            total_llm_calls=self._llm_calls,
            total_tokens=self._total_tokens,
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            total_tool_calls=self._tool_calls,
            total_files_written=self._files_written,
            total_files_read=self._files_read,
            total_memory_searches=self._memory_searches,
            total_compressions=self._compressions,
            agents_used=sorted(self._agents),
            duration_seconds=round(elapsed, 2),
        ))

        # Close the file handle — all events are already flushed
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass

        # Use stdlib logging (not async structlog) since close() is sync
        import logging
        logging.getLogger(__name__).info(
            "telemetry.ledger_closed path=%s events=%d tokens=%d",
            str(self._output_path) if self._output_path else "n/a",
            self._llm_calls + self._tool_calls + self._memory_searches,
            self._total_tokens,
        )

    # -------------------------------------------------------------------
    # V2: Dropped message archival
    # -------------------------------------------------------------------

    def record_dropped_message(
        self,
        *,
        agent: str,
        role: str,
        content: str,
        iteration: int = 0,
    ) -> None:
        """Archive a message dropped by the sliding window.

        Preserves context for future LTM extraction even after the
        message is evicted from the active conversation.
        """
        self._write(DroppedMessageEvent(
            session_id=self._session_id,
            agent=agent,
            role=role,
            content_preview=content[:500],
            content_chars=len(content),
            iteration_dropped=iteration,
        ))


# ---------------------------------------------------------------------------
# No-op ledger (when disabled)
# ---------------------------------------------------------------------------


class NoOpLedger:
    """Drop-in replacement when session replay is disabled."""

    enabled = False
    output_path = None

    def record_llm_call(self, **_: Any) -> None: pass
    def record_tool_execution(self, **_: Any) -> None: pass
    def record_memory_search(self, **_: Any) -> None: pass
    def record_routing_decision(self, **_: Any) -> None: pass
    def record_compression(self, **_: Any) -> None: pass
    def record_dropped_message(self, **_: Any) -> None: pass
    def close(self) -> None: pass
