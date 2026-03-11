"""Async token-bucket rate limiter with per-model RPM/TPM enforcement.

Ensures the swarm never exceeds provider rate limits, even when multiple
agents share the same model.  Uses a sliding-window algorithm with
fair FIFO scheduling.

Architecture
------------

.. code-block:: text

    Agent A ──┐
    Agent B ──┼──► SwarmRateLimiter.acquire("gemini/gemini-2.5-flash", tokens=500)
    Agent C ──┘          │
                         ▼
              ModelBucket("gemini/gemini-2.5-flash")
              ┌──────────────────────────────┐
              │  RPM: 850/1000 (85% margin)  │
              │  TPM: 680K/1M               │
              │  Window: sliding 60s         │
              │  Queue: FIFO asyncio.Event   │
              └──────────────────────────────┘
                         │
                         ▼
              ✅ Acquired   or   ⏳ Wait (asyncio.sleep)

Usage::

    from swarm.infra.rate_limiter import SwarmRateLimiter
    from swarm.config.models import RateLimitConfig

    limiter = SwarmRateLimiter(RateLimitConfig())

    # Before every LLM call:
    await limiter.acquire("gemini/gemini-2.5-flash", estimated_tokens=500)
    response = await litellm.acompletion(...)

    # After the response, update actual usage:
    limiter.record_usage("gemini/gemini-2.5-flash", actual_tokens=response.usage.total_tokens)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

from swarm.config.models import RateLimitConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-model bucket
# ---------------------------------------------------------------------------


@dataclass
class _RequestRecord:
    """A single request's timestamp and token count."""
    timestamp: float
    tokens: int


class ModelBucket:
    """Sliding-window token bucket for a single model.

    Tracks both requests-per-minute (RPM) and tokens-per-minute (TPM)
    over a rolling 60-second window.
    """

    def __init__(
        self,
        model: str,
        rpm_limit: int,
        tpm_limit: int,
        safety_margin: float = 0.85,
    ) -> None:
        self.model = model
        self._rpm_limit = int(rpm_limit * safety_margin)
        self._tpm_limit = int(tpm_limit * safety_margin)
        self._window: deque[_RequestRecord] = deque()
        self._lock = asyncio.Lock()

    @property
    def rpm_limit(self) -> int:
        return self._rpm_limit

    @property
    def tpm_limit(self) -> int:
        return self._tpm_limit

    def _prune(self, now: float) -> None:
        """Remove records older than 60 seconds."""
        cutoff = now - 60.0
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

    @property
    def _current_rpm(self) -> int:
        return len(self._window)

    @property
    def _current_tpm(self) -> int:
        return sum(r.tokens for r in self._window)

    async def acquire(self, estimated_tokens: int = 500) -> float:
        """Wait until we have capacity, then reserve a slot.

        Parameters
        ----------
        estimated_tokens:
            Estimated tokens this request will consume.

        Returns
        -------
        float
            Seconds spent waiting (0.0 if acquired immediately).
        """
        waited = 0.0

        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)

                # Check both RPM and TPM
                rpm_ok = self._current_rpm < self._rpm_limit
                tpm_ok = (self._current_tpm + estimated_tokens) <= self._tpm_limit

                if rpm_ok and tpm_ok:
                    # Reserve the slot
                    self._window.append(_RequestRecord(
                        timestamp=now,
                        tokens=estimated_tokens,
                    ))
                    return waited

            # Calculate wait time based on which limit is hit
            wait_time = self._calculate_wait(estimated_tokens)
            waited += wait_time
            await asyncio.sleep(wait_time)

    def _calculate_wait(self, estimated_tokens: int) -> float:
        """Calculate how long to wait before retrying."""
        now = time.monotonic()
        self._prune(now)

        if not self._window:
            return 0.1  # Should not happen, but safe fallback

        oldest = self._window[0].timestamp
        time_until_oldest_expires = max(0.0, 60.0 - (now - oldest))

        # If RPM is the bottleneck, wait until the oldest request expires
        if self._current_rpm >= self._rpm_limit:
            return min(time_until_oldest_expires + 0.1, 5.0)

        # If TPM is the bottleneck, we need to shed enough tokens
        if (self._current_tpm + estimated_tokens) > self._tpm_limit:
            # Find how many old requests need to expire to free enough tokens
            needed = (self._current_tpm + estimated_tokens) - self._tpm_limit
            shed = 0
            for record in self._window:
                shed += record.tokens
                if shed >= needed:
                    wait = 60.0 - (now - record.timestamp) + 0.1
                    return min(max(wait, 0.1), 10.0)

        return 0.5  # Default wait

    def record_actual_usage(self, actual_tokens: int) -> None:
        """Update the most recent request with actual token count.

        Call this after the LLM response arrives with real usage data
        to improve accuracy of the sliding window.
        """
        if self._window:
            last = self._window[-1]
            # Adjust: replace the estimate with the actual
            self._window[-1] = _RequestRecord(
                timestamp=last.timestamp,
                tokens=actual_tokens,
            )

    def stats(self) -> dict[str, Any]:
        """Get current bucket statistics."""
        now = time.monotonic()
        self._prune(now)
        return {
            "model": self.model,
            "rpm_used": self._current_rpm,
            "rpm_limit": self._rpm_limit,
            "tpm_used": self._current_tpm,
            "tpm_limit": self._tpm_limit,
            "rpm_utilisation": round(self._current_rpm / max(self._rpm_limit, 1), 3),
            "tpm_utilisation": round(self._current_tpm / max(self._tpm_limit, 1), 3),
        }


# ---------------------------------------------------------------------------
# Swarm-level rate limiter
# ---------------------------------------------------------------------------


class SwarmRateLimiter:
    """Manages rate limiting across all models used by the swarm.

    Creates a ``ModelBucket`` per unique model string and provides
    ``acquire()`` / ``record_usage()`` methods for the LLM wrapper.

    Usage::

        limiter = SwarmRateLimiter(config.rate_limits)

        # Before calling litellm:
        wait = await limiter.acquire("gemini/gemini-2.5-flash", estimated_tokens=500)

        # After the response:
        limiter.record_usage("gemini/gemini-2.5-flash", actual_tokens=total)

        # Monitor utilisation:
        for stats in limiter.all_stats():
            print(stats)
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._buckets: dict[str, ModelBucket] = {}
        self._lock = asyncio.Lock()

        # Pre-create buckets for known models
        for model, limits in config.models.items():
            self._buckets[model] = ModelBucket(
                model=model,
                rpm_limit=limits.rpm,
                tpm_limit=limits.tpm,
                safety_margin=config.safety_margin,
            )

    async def acquire(
        self,
        model: str,
        estimated_tokens: int = 500,
    ) -> float:
        """Acquire rate limit capacity for a model.

        Blocks until capacity is available. Returns the wait time.

        Parameters
        ----------
        model:
            litellm model string.
        estimated_tokens:
            Estimated total tokens (prompt + completion).

        Returns
        -------
        float
            Seconds spent waiting.
        """
        if not self._config.enabled:
            return 0.0

        bucket = await self._get_or_create_bucket(model)
        waited = await bucket.acquire(estimated_tokens)

        if waited > 0:
            await logger.info(
                "rate_limiter.throttled",
                model=model,
                waited_seconds=round(waited, 2),
                **bucket.stats(),
            )

        return waited

    def record_usage(self, model: str, actual_tokens: int) -> None:
        """Update the most recent request with actual token usage."""
        bucket = self._buckets.get(model)
        if bucket:
            bucket.record_actual_usage(actual_tokens)

    async def _get_or_create_bucket(self, model: str) -> ModelBucket:
        """Get or lazily create a bucket for an unknown model."""
        if model in self._buckets:
            return self._buckets[model]

        async with self._lock:
            # Double-check after acquiring lock
            if model in self._buckets:
                return self._buckets[model]

            # Unknown model — use conservative defaults
            bucket = ModelBucket(
                model=model,
                rpm_limit=60,   # Conservative: 1 req/sec
                tpm_limit=100_000,
                safety_margin=self._config.safety_margin,
            )
            self._buckets[model] = bucket

            await logger.warning(
                "rate_limiter.unknown_model",
                model=model,
                msg="Using conservative defaults (60 RPM, 100K TPM).",
            )

            return bucket

    def all_stats(self) -> list[dict[str, Any]]:
        """Get stats for all tracked models."""
        return [bucket.stats() for bucket in self._buckets.values()]

    def get_stats(self, model: str) -> dict[str, Any] | None:
        """Get stats for a specific model."""
        bucket = self._buckets.get(model)
        return bucket.stats() if bucket else None
