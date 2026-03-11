"""Async-safe structured logging configuration.

Configures ``structlog`` for the swarm platform with:

* **JSON output** (production): Machine-readable, parseable by log
  aggregators (ELK, Datadog, etc.).
* **Console output** (development): Human-readable, colourised output
  with timestamps and key-value pairs.
* **Async safety**: Uses ``structlog``'s built-in processor chain which
  is safe for concurrent access from multiple async tasks.
* **Agent context**: Adds ``agent_name`` and ``session_id`` to every log
  line so parallel agent outputs can be filtered and traced.

Usage::

    from swarm.infra.logging import configure_logging

    configure_logging(config.logging)

    # From any module:
    import structlog
    logger = structlog.get_logger(__name__)
    await logger.info("event.name", key="value")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from swarm.config.models import LoggingConfig, LogLevel


# ---------------------------------------------------------------------------
# Log level mapping
# ---------------------------------------------------------------------------

_LEVEL_MAP: dict[LogLevel, int] = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------


def _add_swarm_context(
    _logger: Any, _method: str, event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add swarm-specific context to every log event."""
    # These will be populated via structlog.contextvars if set
    event_dict.setdefault("service", "swarm")
    return event_dict


def _truncate_large_values(
    _logger: Any, _method: str, event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Truncate overly long string values to prevent log bloat."""
    max_len = 2000
    for key, value in event_dict.items():
        if isinstance(value, str) and len(value) > max_len:
            event_dict[key] = value[:max_len] + f"... (truncated, {len(value)} total)"
    return event_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_logging(config: LoggingConfig) -> None:
    """Configure structlog and stdlib logging for the swarm platform.

    Parameters
    ----------
    config:
        Logging configuration from the swarm YAML.
    """
    log_level = _LEVEL_MAP.get(config.level, logging.INFO)

    # --- Shared processors ---
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_swarm_context,
        _truncate_large_values,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # --- Renderer ---
    if config.json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=40,
        )

    # --- File output ---
    log_file_handle = None
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(str(log_path), "a", encoding="utf-8")  # noqa: SIM115

    # --- Configure structlog ---
    # Use WriteLoggerFactory so AsyncBoundLogger works with ainfo/aerror/awarning.
    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
        ],
        logger_factory=structlog.WriteLoggerFactory(
            file=log_file_handle or sys.stderr,
        ),
        wrapper_class=structlog.stdlib.AsyncBoundLogger,
        cache_logger_on_first_use=False,
    )

    # --- Configure stdlib logging for foreign libraries ---
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Simple handler for non-structlog loggers
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root_logger.addHandler(console_handler)

    # Suppress noisy libraries
    for lib in ("httpx", "httpcore", "urllib3", "docker", "uvicorn.access"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Context binding helpers
# ---------------------------------------------------------------------------


def bind_agent_context(agent_name: str, session_id: str = "") -> None:
    """Bind agent context to all subsequent log calls in this async task.

    Uses structlog's contextvars integration so the bound values
    propagate correctly through async/await chains.

    Usage::

        bind_agent_context("backend-agent", "session-abc123")
        await logger.info("starting work")
        # → {"agent_name": "backend-agent", "session_id": "session-abc123", ...}
    """
    structlog.contextvars.bind_contextvars(
        agent_name=agent_name,
        session_id=session_id,
    )


def unbind_agent_context() -> None:
    """Remove agent context from the current async task."""
    structlog.contextvars.unbind_contextvars("agent_name", "session_id")
