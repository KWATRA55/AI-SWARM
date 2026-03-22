"""Multi-tenant context for data isolation.

Each API key maps to a tenant, and all data paths (Redis keys, LanceDB tables,
replay files, memory store) are scoped to that tenant.

Usage::

    from swarm.api.tenant import resolve_tenant, TenantContext

    @router.get("/data")
    async def get_data(tenant: TenantContext = Depends(resolve_tenant)):
        print(tenant.tenant_id)     # "acme-corp"
        print(tenant.data_root)     # /data/tenants/acme-corp
        print(tenant.redis_prefix)  # tenant-acme-corp:swarm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends

from swarm.api.auth import APIKeyRecord, verify_api_key

logger = structlog.get_logger(__name__)

# Default data root — can be overridden via SWARM_DATA_ROOT env var
_DEFAULT_DATA_ROOT = Path.home() / ".swarm" / "data"


@dataclass(frozen=True)
class TenantContext:
    """Carries tenant identity and resolved paths through the request lifecycle.

    All data operations should use these resolved paths instead of
    constructing their own to ensure tenant isolation.
    """

    tenant_id: str
    org_id: str = "default"
    data_root: Path = field(default_factory=lambda: _DEFAULT_DATA_ROOT)

    @property
    def tenant_dir(self) -> Path:
        """Root directory for this tenant's data."""
        return self.data_root / "tenants" / self.tenant_id

    @property
    def memory_dir(self) -> Path:
        """LTM store path for this tenant."""
        return self.tenant_dir / "memory"

    @property
    def replays_dir(self) -> Path:
        """Session replay output directory."""
        return self.tenant_dir / "replays"

    @property
    def redis_prefix(self) -> str:
        """Redis key prefix for tenant isolation."""
        return f"tenant-{self.tenant_id}:swarm"

    @property
    def lancedb_prefix(self) -> str:
        """LanceDB table name prefix."""
        return f"{self.tenant_id}_"

    def ensure_dirs(self) -> None:
        """Create all tenant directories if they don't exist."""
        self.tenant_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        """Per-session subdirectory."""
        d = self.tenant_dir / "sessions" / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d


def _get_data_root() -> Path:
    """Resolve the data root from env or default."""
    import os
    raw = os.environ.get("SWARM_DATA_ROOT", "")
    if raw:
        return Path(raw)
    return _DEFAULT_DATA_ROOT


def resolve_tenant(
    key: APIKeyRecord = Depends(verify_api_key("read")),
) -> TenantContext:
    """FastAPI dependency that extracts tenant context from the API key.

    The tenant_id is embedded in the APIKeyRecord by the auth system.
    """
    data_root = _get_data_root()
    ctx = TenantContext(
        tenant_id=key.tenant_id,
        data_root=data_root,
    )
    ctx.ensure_dirs()
    return ctx
