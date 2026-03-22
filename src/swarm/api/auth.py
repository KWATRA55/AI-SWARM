"""API Key authentication for the Swarm gateway.

Supports three permission scopes:
  - ``dispatch`` — can trigger swarm runs and dispatch tasks.
  - ``read``     — can view sessions, telemetry, and analytics.
  - ``admin``    — full access including kill switch and config edits.

API keys are loaded from:
  1. ``SWARM_API_KEYS`` env var (comma-separated ``key:scope`` pairs).
  2. A JSON file at ``SWARM_API_KEYS_FILE`` path.

Usage::

    from swarm.api.auth import verify_api_key

    @router.post("/dispatch")
    async def dispatch(key: str = Depends(verify_api_key("dispatch"))):
        ...
"""

from __future__ import annotations

import json
import os
import secrets
from enum import Enum
from typing import Any

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = structlog.get_logger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class Scope(str, Enum):
    """Permission scopes for API keys."""

    DISPATCH = "dispatch"
    READ = "read"
    ADMIN = "admin"


# Scope hierarchy: admin > dispatch > read
_SCOPE_HIERARCHY: dict[Scope, set[Scope]] = {
    Scope.ADMIN: {Scope.ADMIN, Scope.DISPATCH, Scope.READ},
    Scope.DISPATCH: {Scope.DISPATCH, Scope.READ},
    Scope.READ: {Scope.READ},
}


class APIKeyRecord:
    """An authorised API key with its scope and optional tenant binding."""

    __slots__ = ("key", "scope", "tenant_id", "label")

    def __init__(
        self,
        key: str,
        scope: Scope = Scope.READ,
        tenant_id: str = "default",
        label: str = "",
    ) -> None:
        self.key = key
        self.scope = scope
        self.tenant_id = tenant_id
        self.label = label


class APIKeyManager:
    """Loads and validates API keys from env vars or a JSON file.

    Key sources
    -----------
    ``SWARM_API_KEYS`` env var:
        Comma-separated ``key:scope`` or ``key:scope:tenant`` strings.
        Example: ``sk-abc123:admin,sk-read456:read:tenant-acme``

    ``SWARM_API_KEYS_FILE`` env var:
        Path to a JSON file containing an array of key objects::

            [
              {"key": "sk-abc123", "scope": "admin", "tenant_id": "default"},
              {"key": "sk-read456", "scope": "read", "tenant_id": "acme"}
            ]

    If **neither** source is set, a single auto-generated admin key is created
    and logged at startup for local development convenience.
    """

    def __init__(self) -> None:
        self._keys: dict[str, APIKeyRecord] = {}
        self._load_from_env()
        self._load_from_file()

        # Dev fallback: generate an ephemeral key if none are configured
        if not self._keys:
            dev_key = f"swarm-dev-{secrets.token_hex(16)}"
            self._keys[dev_key] = APIKeyRecord(
                key=dev_key, scope=Scope.ADMIN, label="auto-generated-dev",
            )
            logger.warning(
                "api.auth.dev_key_generated",
                key=dev_key,
                msg="No API keys configured — using auto-generated dev key. "
                    "Set SWARM_API_KEYS env var for production.",
            )

    def _load_from_env(self) -> None:
        """Parse SWARM_API_KEYS env var."""
        raw = os.environ.get("SWARM_API_KEYS", "")
        if not raw:
            return

        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            key = parts[0]
            scope = Scope(parts[1]) if len(parts) > 1 else Scope.READ
            tenant_id = parts[2] if len(parts) > 2 else "default"
            self._keys[key] = APIKeyRecord(
                key=key, scope=scope, tenant_id=tenant_id,
            )

        logger.info("api.auth.keys_loaded", source="env", count=len(self._keys))

    def _load_from_file(self) -> None:
        """Load keys from SWARM_API_KEYS_FILE JSON file."""
        path = os.environ.get("SWARM_API_KEYS_FILE", "")
        if not path:
            return

        try:
            with open(path) as f:
                data = json.load(f)
            for item in data:
                key = item["key"]
                scope = Scope(item.get("scope", "read"))
                tenant_id = item.get("tenant_id", "default")
                label = item.get("label", "")
                self._keys[key] = APIKeyRecord(
                    key=key, scope=scope, tenant_id=tenant_id, label=label,
                )
            logger.info(
                "api.auth.keys_loaded", source="file",
                path=path, count=len(data),
            )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.error("api.auth.keys_file_error", path=path, error=str(exc))

    def validate(self, key: str) -> APIKeyRecord | None:
        """Return the key record if valid, else None."""
        return self._keys.get(key)

    def has_scope(self, record: APIKeyRecord, required: Scope) -> bool:
        """Check if a key record has the required scope."""
        return required in _SCOPE_HIERARCHY.get(record.scope, set())

    @property
    def key_count(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# Global singleton (initialized on first import)
# ---------------------------------------------------------------------------

_manager: APIKeyManager | None = None


def get_key_manager() -> APIKeyManager:
    """Get or create the global API key manager."""
    global _manager
    if _manager is None:
        _manager = APIKeyManager()
    return _manager


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def verify_api_key(required_scope: str = "read"):
    """FastAPI dependency factory that validates the X-API-Key header.

    Usage::

        @router.get("/data")
        async def get_data(
            key_record = Depends(verify_api_key("read"))
        ):
            ...
    """
    scope = Scope(required_scope)

    async def _dependency(
        api_key: str | None = Security(_API_KEY_HEADER),
    ) -> APIKeyRecord:
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail="Missing X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        manager = get_key_manager()
        record = manager.validate(api_key)

        if record is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if not manager.has_scope(record, scope):
            raise HTTPException(
                status_code=403,
                detail=f"API key lacks required scope: '{scope.value}'. "
                       f"Key scope: '{record.scope.value}'.",
            )

        return record

    return _dependency
