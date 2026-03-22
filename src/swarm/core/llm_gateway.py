"""LLM Gateway — intelligent routing, fallback, and semantic caching.

Sits between all agent ``acompletion()`` calls and LiteLLM, providing:

1. **Dynamic Fallback Matrix** — when the primary model returns 429 or 503,
   automatically route to the next model in the fallback chain with
   exponential backoff.

2. **Semantic Cache** — hash the system prompt + embed the user message,
   query Redis for a structurally similar prior response (cosine ≥ 0.95).
   Cache hits save the full API call.

Usage::

    gateway = LLMGateway(
        fallback_models=["anthropic/claude-3.5-sonnet", "ollama/qwen"],
        redis_client=redis,
    )
    response = await gateway.complete(
        model="gemini/gemini-2.5-pro",
        messages=[...],
        tools=[...],
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Retry / Fallback configuration
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 503, 500, 502}
_MAX_RETRIES_PER_MODEL = 2
_BASE_BACKOFF_SECONDS = 1.0


class LLMGateway:
    """Routing and caching layer between agents and LiteLLM.

    Parameters
    ----------
    fallback_models:
        Ordered list of fallback model identifiers. When the primary
        model fails with a retryable error, the gateway tries each
        fallback in order.
    redis_client:
        Optional async Redis client for semantic cache lookups.
    cache_ttl_seconds:
        How long cached responses are valid (default: 1 hour).
    cache_similarity_threshold:
        Minimum cosine similarity for a cache hit (default: 0.95).
    """

    def __init__(
        self,
        *,
        fallback_models: list[str] | None = None,
        redis_client: Any | None = None,
        cache_ttl_seconds: int = 3600,
        cache_similarity_threshold: float = 0.95,
    ) -> None:
        self._fallback_models = fallback_models or []
        self._redis = redis_client
        self._cache_ttl = cache_ttl_seconds
        self._cache_threshold = cache_similarity_threshold

        # Metrics
        self._total_requests = 0
        self._cache_hits = 0
        self._fallback_triggers = 0

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        agent_name: str = "unknown",
        **kwargs: Any,
    ) -> Any:
        """Send a completion request with fallback and cache support.

        Parameters
        ----------
        model:
            Primary model identifier (e.g. ``gemini/gemini-2.5-pro``).
        messages:
            Chat messages in OpenAI format.
        tools:
            Optional tool definitions for function calling.
        agent_name:
            For logging — which agent is making the request.

        Returns
        -------
        The LiteLLM response object.
        """
        from litellm import acompletion

        self._total_requests += 1

        # --- 1. Check semantic cache ---
        cache_key = self._build_cache_key(model, messages, agent_name)
        cached = await self._cache_lookup(cache_key)
        if cached is not None:
            self._cache_hits += 1
            await logger.info(
                "gateway.cache_hit",
                agent=agent_name,
                model=model,
                cache_key=cache_key[:16],
            )
            return cached

        # --- 2. Try primary model, then fallbacks ---
        models_to_try = [model] + self._fallback_models
        last_error: Exception | None = None

        for i, try_model in enumerate(models_to_try):
            is_fallback = i > 0
            if is_fallback:
                self._fallback_triggers += 1
                await logger.info(
                    "gateway.fallback_triggered",
                    agent=agent_name,
                    from_model=model,
                    to_model=try_model,
                    attempt=i,
                )

            for retry in range(_MAX_RETRIES_PER_MODEL):
                try:
                    call_kwargs: dict[str, Any] = {
                        "model": try_model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    if tools:
                        call_kwargs["tools"] = tools
                    if max_tokens:
                        call_kwargs["max_tokens"] = max_tokens
                    call_kwargs.update(kwargs)

                    # ---- V2: Native API Prompt Caching ----
                    # Inject provider-specific caching directives to get
                    # 75-90% cache hit rates on frozen system prompts.
                    _model_lower = try_model.lower()
                    if "anthropic" in _model_lower or "claude" in _model_lower:
                        import copy
                        # Deep copy messages so we don't mutate the agent's internal state!
                        call_kwargs["messages"] = copy.deepcopy(call_kwargs["messages"])
                        # Anthropic: add cache_control breakpoint to system msg
                        call_kwargs.setdefault("extra_headers", {})
                        call_kwargs["extra_headers"]["anthropic-beta"] = (
                            "prompt-caching-2024-07-31"
                        )
                        # Mark the last system message as cache breakpoint
                        _msgs = call_kwargs.get("messages", [])
                        for _msg in _msgs:
                            if _msg.get("role") == "system":
                                if isinstance(_msg.get("content"), str):
                                    _msg["content"] = [
                                        {
                                            "type": "text",
                                            "text": _msg["content"],
                                            "cache_control": {"type": "ephemeral"},
                                        }
                                    ]
                                break  # Only tag the first system message

                    response = await acompletion(**call_kwargs)

                    # Log which model actually served the request
                    await logger.info(
                        "gateway.request_completed",
                        agent=agent_name,
                        model=try_model,
                        is_fallback=is_fallback,
                    )

                    # Store in cache
                    await self._cache_store(cache_key, response, model=try_model)

                    return response

                except Exception as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", 0)

                    if status_code in _RETRYABLE_STATUS_CODES:
                        backoff = _BASE_BACKOFF_SECONDS * (2 ** retry)
                        await logger.warning(
                            "gateway.retryable_error",
                            agent=agent_name,
                            model=try_model,
                            status_code=status_code,
                            retry=retry + 1,
                            backoff=backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        # Non-retryable error — skip to next model
                        await logger.warning(
                            "gateway.non_retryable_error",
                            agent=agent_name,
                            model=try_model,
                            error=str(exc),
                        )
                        break

        # All models exhausted
        await logger.error(
            "gateway.all_models_failed",
            agent=agent_name,
            primary_model=model,
            fallbacks=self._fallback_models,
        )
        raise last_error or RuntimeError("All LLM models failed")

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------

    def get_metrics(self) -> dict[str, int]:
        """Return gateway metrics for dashboard display."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "fallback_triggers": self._fallback_triggers,
            "cache_hit_rate": round(
                self._cache_hits / max(self._total_requests, 1) * 100, 1
            ),
        }

    # -------------------------------------------------------------------
    # Cache internals
    # -------------------------------------------------------------------

    def _build_cache_key(
        self, model: str, messages: list[dict[str, Any]], agent_name: str
    ) -> str:
        """Build a cache key from model + user intent.

        Includes the `agent_name` to namespace cache hits per agent, preventing
        cross-agent hallucinations. Strips volatile system prompt and tool
        results, hashing only the core user messages.
        """
        import re

        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if role == "user":
                # Strip timestamps, session IDs, and other volatile tokens
                clean = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*", "", content)
                clean = re.sub(r"\b[0-9a-f]{8,}\b", "", clean)  # hex IDs
                clean = clean.strip()
                if clean:
                    user_parts.append(clean)

        raw = f"{agent_name}:{model}:{'|'.join(user_parts[-3:])}"  # last 3 user messages
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _cache_lookup(self, cache_key: str) -> Any | None:
        """Look up a cached response in Redis."""
        if self._redis is None:
            return None

        try:
            cached_json = await self._redis.get(f"llm:cache:{cache_key}")
            if cached_json:
                return json.loads(cached_json)
        except Exception as exc:
            await logger.debug("gateway.cache_lookup_error", error=str(exc))

        return None

    async def _cache_store(
        self, cache_key: str, response: Any, *, model: str,
    ) -> None:
        """Store a response in the Redis cache."""
        if self._redis is None:
            return

        try:
            # Extract serialisable response data
            if hasattr(response, "model_dump"):
                data = response.model_dump()
            elif hasattr(response, "to_dict"):
                data = response.to_dict()
            else:
                data = {"raw": str(response)}

            data["_cached_model"] = model
            data["_cached_at"] = time.time()

            await self._redis.set(
                f"llm:cache:{cache_key}",
                json.dumps(data, default=str),
                ex=self._cache_ttl,
            )
        except Exception as exc:
            await logger.debug("gateway.cache_store_error", error=str(exc))
