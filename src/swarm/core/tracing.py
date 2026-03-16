"""OpenTelemetry distributed tracing integration.

Provides a lightweight wrapper around the OpenTelemetry SDK so that
spans can be created from the orchestrator, worker, and tool executor
with a single import.

When the OTEL SDK is not installed, all tracing functions become no-ops
so the rest of the codebase never needs to guard imports.

Usage::

    from swarm.core.tracing import init_tracing, trace_span

    # Call once at startup (cli.py)
    init_tracing(service_name="ai-swarm")

    # In any coroutine
    async def _run_agent_loop(self):
        with trace_span("agent.loop", agent=self.name) as span:
            span.set_attribute("agent.model", self._config.model)
            ...
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import Any, Generator

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_tracer: Any | None = None
_initialized = False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_tracing(
    *,
    service_name: str = "ai-swarm",
    endpoint: str | None = None,
) -> bool:
    """Initialize the OpenTelemetry tracing pipeline.

    Parameters
    ----------
    service_name:
        The OTEL service name (appears in Jaeger / Datadog).
    endpoint:
        Optional OTLP collector endpoint (e.g. ``http://localhost:4317``).
        If not provided, falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.

    Returns
    -------
    bool
        True if tracing was successfully initialized.
    """
    global _tracer, _initialized

    if _initialized:
        return True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Try OTLP exporter first, fall back to console
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=endpoint)
            except ImportError:
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        _initialized = True

        return True

    except ImportError:
        # OTEL SDK not installed — tracing becomes a no-op
        _initialized = False
        return False


# ---------------------------------------------------------------------------
# Span creation
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Dummy span when OTEL is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *_: Any) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass


@contextmanager
def trace_span(
    name: str,
    **attributes: Any,
) -> Generator[Any, None, None]:
    """Create an OTEL span as a context manager.

    Falls back to a no-op span when OTEL is not installed::

        with trace_span("agent.llm_call", model="gemini-2.5-pro") as span:
            span.set_attribute("tokens", 1234)
            response = await acompletion(...)

    Parameters
    ----------
    name:
        Span name (e.g. ``orchestrator.run``, ``tool.read_file``).
    **attributes:
        Initial span attributes set at creation.
    """
    if _tracer is not None:
        with _tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            yield span
    else:
        yield _NoOpSpan()


def get_trace_id() -> str | None:
    """Return the current trace ID as a hex string, or None."""
    if not _initialized:
        return None
    try:
        from opentelemetry import trace
        ctx = trace.get_current_span().get_span_context()
        if ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return None
