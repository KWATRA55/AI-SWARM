"""Resilient LLM client — wraps litellm with retry, rate-limit backoff, and telemetry.

Provides a drop-in ``llm_completion`` function that:

1. Retries on transient failures (rate limits, timeouts, server errors)
   with exponential backoff via ``tenacity``.
2. Tracks per-call token usage and latency.
3. Emits structured log events for every call (model, tokens, duration).
4. Supports all litellm model strings (OpenAI, Anthropic, Google, Ollama, etc.).

Usage::

    from swarm.infra.llm import llm_completion

    response = await llm_completion(
        model="gemini/gemini-1.5-pro",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.2,
    )
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Retryable exception detection
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Determine if a litellm exception is transient and worth retrying.

    Retries on:
    - Rate limit errors (429)
    - Server errors (500, 502, 503)
    - Timeouts
    - Connection errors

    Does NOT retry on:
    - Authentication errors (401, 403)
    - Invalid request (400)
    - Model not found (404)
    - Context length exceeded (413)
    """
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()

    # Explicit litellm exception types
    retryable_types = {
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIConnectionError",
        "Timeout",
        "APITimeoutError",
    }

    if exc_name in retryable_types:
        return True

    # Check for retryable HTTP status codes in the message
    retryable_indicators = [
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "service unavailable",
        "timeout",
        "connection error",
        "overloaded",
        "capacity",
    ]

    return any(indicator in exc_str for indicator in retryable_indicators)


# ---------------------------------------------------------------------------
# Retry-wrapped completion
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(
        initial=1.0,
        max=60.0,
        jitter=2.0,
    ),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def llm_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    **kwargs: Any,
) -> Any:
    """Resilient LLM completion with automatic retry on transient failures.

    Wraps ``litellm.acompletion`` with:
    - Exponential backoff + jitter (1s → 2s → 4s → ... → 60s max)
    - Up to 5 retry attempts
    - Only retries on transient errors (rate limits, timeouts, 5xx)
    - Structured logging of every call

    Parameters
    ----------
    model:
        litellm model string (e.g., ``"gpt-4o"``, ``"gemini/gemini-1.5-pro"``).
    messages:
        OpenAI-format message list.
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens in the response.
    timeout:
        Request timeout in seconds.
    tools:
        Optional tool definitions for function calling.
    tool_choice:
        Optional tool selection strategy.
    **kwargs:
        Additional arguments passed through to litellm.

    Returns
    -------
    litellm.ModelResponse
        The LLM response object.

    Raises
    ------
    Exception
        Re-raises after exhausting retries or on non-retryable errors.
    """
    from litellm import acompletion

    start = time.monotonic()

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        **kwargs,
    }

    if tools:
        call_kwargs["tools"] = tools
    if tool_choice:
        call_kwargs["tool_choice"] = tool_choice

    try:
        response = await acompletion(**call_kwargs)

        elapsed = round(time.monotonic() - start, 3)

        # Extract token usage
        usage = response.usage if hasattr(response, "usage") and response.usage else None
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        await logger.info(
            "llm.completion",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
            has_tool_calls=bool(
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ),
        )

        return response

    except Exception as exc:
        elapsed = round(time.monotonic() - start, 3)
        await logger.warning(
            "llm.completion_error",
            model=model,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
            retryable=_is_retryable(exc),
        )
        raise


# ---------------------------------------------------------------------------
# Streaming variant
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=1.0, max=60.0, jitter=2.0),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def llm_completion_stream(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    **kwargs: Any,
) -> Any:
    """Streaming variant of ``llm_completion``.

    Returns an async generator that yields chunks.
    Retry only applies to the initial connection — once streaming
    starts, failures are not retried.
    """
    from litellm import acompletion

    response = await acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        stream=True,
        **kwargs,
    )

    return response


# ---------------------------------------------------------------------------
# Token counter utility
# ---------------------------------------------------------------------------


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for a piece of text.

    Uses tiktoken for OpenAI models, falls back to character-based
    estimation for others.
    """
    try:
        import tiktoken

        # Map litellm model strings to tiktoken encoding names
        if "gpt" in model or "o1" in model:
            enc = tiktoken.encoding_for_model(model.split("/")[-1])
        else:
            enc = tiktoken.get_encoding("cl100k_base")

        return len(enc.encode(text))
    except Exception:
        # Rough estimate: ~4 chars per token
        return len(text) // 4
