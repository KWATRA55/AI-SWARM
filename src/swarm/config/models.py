"""Pydantic models defining the declarative YAML configuration schema.

Every agent, sandbox, and infrastructure setting is expressed here.
The ``SwarmConfig`` model is the root — it is what ``loader.py`` deserialises
the user-supplied YAML file into.

Design notes
------------
* ``AgentConfig.model`` maps directly to a ``litellm`` model string so the
  orchestrator can route tasks to different LLM providers per-agent.
* ``SandboxConfig.workspace_mount`` is populated at runtime by the loader to
  point every container at the **same** host directory (the target project repo),
  enforcing a single shared workspace with write-serialisation via the Event Bus.
* All optional fields carry sensible defaults so a minimal config is ~10 lines.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentRole(str, enum.Enum):
    """Well-known agent roles (extensible via ``CUSTOM``)."""

    BACKEND = "backend"
    FRONTEND = "frontend"
    QA = "qa"
    DEVOPS = "devops"
    DATABASE = "database"
    ARCHITECT = "architect"
    CUSTOM = "custom"


class MCPTransport(str, enum.Enum):
    """Supported MCP transport layers between host and container.

    * ``STDIO``  — host communicates via ``docker exec`` stdin/stdout pipes.
    * ``SSE``    — host connects to an HTTP+SSE endpoint exposed on a container port.

    ``SSE`` is recommended for production because it survives ``exec`` session
    drops and allows multiplexed tool calls.  ``STDIO`` is simpler for local
    development / single-tool-call workloads.
    """

    STDIO = "stdio"
    SSE = "sse"


class LogLevel(str, enum.Enum):
    """Accepted log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ResourceLimits(BaseModel):
    """Container resource constraints passed to Docker."""

    model_config = ConfigDict(frozen=True)

    memory: str = Field(
        default="2g",
        description="Memory limit, e.g. '512m', '2g'.",
    )
    cpu_count: float = Field(
        default=2.0,
        ge=0.5,
        description="Number of CPUs (fractional OK).",
    )
    storage: str = Field(
        default="10g",
        description="Tmpfs / disk limit for scratch space.",
    )


class SandboxConfig(BaseModel):
    """Per-agent container configuration.

    ``workspace_mount`` is resolved at load-time by the config loader so that
    **every** sandbox mounts the same host project directory.
    """

    model_config = ConfigDict(frozen=True)

    image: str = Field(
        default="python:3.12-slim",
        description="Docker image to use for this agent's sandbox.",
    )
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    extra_ports: list[str] = Field(
        default_factory=list,
        description=(
            "Additional host:container port mappings beyond the MCP port, "
            "e.g. ['3000:3000'] for a dev server."
        ),
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables injected into the container.",
    )
    workspace_mount: Path | None = Field(
        default=None,
        description=(
            "Absolute host path to the shared project workspace. "
            "Populated automatically by the config loader — usually not set in YAML."
        ),
    )
    dockerfile: str | None = Field(
        default=None,
        description="Optional path to a Dockerfile for building a custom image.",
    )
    network_mode: str = Field(
        default="bridge",
        description="Docker network mode (bridge | host | none | <custom>).",
    )
    startup_commands: list[str] = Field(
        default_factory=list,
        description=(
            "Shell commands to run inside the container after creation, "
            "e.g. ['pip install -r requirements.txt']."
        ),
    )


class MCPConfig(BaseModel):
    """Configuration for the MCP server injected into a sandbox."""

    model_config = ConfigDict(frozen=True)

    transport: MCPTransport = Field(
        default=MCPTransport.SSE,
        description=(
            "Transport layer for host ↔ container MCP communication. "
            "SSE is recommended for production robustness."
        ),
    )
    port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="Container-side port for the MCP SSE server (ignored when transport=stdio).",
    )
    allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "read_file",
            "write_file",
            "list_directory",
            "run_command",
            "git_diff",
            "git_commit",
        ],
        description="Whitelist of MCP tool names this agent is permitted to invoke.",
    )
    blocked_commands: list[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            "shutdown",
            "reboot",
            "mkfs",
            "dd if=/dev/zero",
        ],
        description="Shell command patterns that are unconditionally blocked.",
    )
    workspace_subpath: str = Field(
        default=".",
        description=(
            "Relative sub-directory within the shared workspace this agent "
            "is permitted to access.  '.' means full workspace access."
        ),
    )


class CircuitBreakerConfig(BaseModel):
    """Limits on inter-agent message loops."""

    model_config = ConfigDict(frozen=True)

    max_round_trips: int = Field(
        default=5,
        ge=1,
        description="Maximum back-and-forth exchanges between two agents on one task.",
    )
    cooldown_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Seconds to wait in HALF_OPEN before retrying after a trip.",
    )
    escalation_strategy: str = Field(
        default="pause_and_notify",
        description=(
            "What to do when the breaker trips: "
            "'pause_and_notify' | 'auto_summarize_and_retry' | 'abort_task'."
        ),
    )


class CompressionConfig(BaseModel):
    """Settings for the LLM-powered context compression pipeline."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True)
    token_threshold: int = Field(
        default=4000,
        ge=500,
        description="Raw content exceeding this token count gets compressed.",
    )
    compression_model: str = Field(
        default="gemini/gemini-2.0-flash",
        description="litellm model string for the fast summarisation LLM.",
    )
    archive_to_redis: bool = Field(
        default=True,
        description="Whether to store full raw content in Redis before compressing.",
    )


class MemoryConfig(BaseModel):
    """Long-Term Memory (LTM) configuration for cross-session learning.

    Agents accumulate reusable knowledge — bug resolutions, architectural
    patterns, coding preferences — in a persistent vector store.  At boot,
    relevant memories are injected into agent system prompts.  During
    execution, agents can search LTM via an MCP tool.

    Storage layout on disk::

        <store_path>/
        ├── lancedb/          # LanceDB vector tables
        │   ├── skills        # Extracted reusable skills
        │   ├── bugs          # Bug resolution records
        │   └── rules         # Architectural rules & preferences
        └── raw/              # Raw JSON memory entries (backup)
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=True,
        description="Enable long-term memory persistence.",
    )
    store_path: Path = Field(
        default=Path(".swarm_memory"),
        description=(
            "Directory for persistent memory storage. "
            "Relative paths are resolved against the workspace."
        ),
    )
    backend: str = Field(
        default="lancedb",
        description="Vector store backend: 'lancedb' (local, zero-config).",
    )
    embedding_model: str = Field(
        default="gemini/text-embedding-004",
        description=(
            "litellm-compatible embedding model for vectorising memories. "
            "Used for similarity search during retrieval."
        ),
    )
    extraction_model: str = Field(
        default="gemini/gemini-2.0-flash",
        description="LLM used to extract structured knowledge from task results.",
    )
    max_retrieval_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of memories returned per search query.",
    )
    relevance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score to include a memory in results.",
    )
    auto_extract: bool = Field(
        default=True,
        description=(
            "Automatically extract reusable knowledge when tasks complete. "
            "If False, extraction must be triggered manually."
        ),
    )
    memory_categories: list[str] = Field(
        default_factory=lambda: [
            "skills",
            "bugs",
            "rules",
            "preferences",
            "patterns",
        ],
        description="Categories for organising extracted memories.",
    )


class EventBusConfig(BaseModel):
    """Redis connection and stream configuration for the event bus."""

    model_config = ConfigDict(frozen=True)

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL.",
    )
    stream_prefix: str = Field(
        default="swarm",
        description="Prefix for Redis Stream keys (e.g. 'swarm:task_assigned').",
    )
    consumer_group: str = Field(
        default="orchestrator",
        description="Consumer group name for reliable message delivery.",
    )
    max_stream_length: int = Field(
        default=10_000,
        ge=100,
        description="Maximum entries per stream (oldest trimmed via MAXLEN).",
    )


class VerificationConfig(BaseModel):
    """Settings for the multi-modal visual verification pipeline."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False)
    vision_model: str = Field(
        default="gpt-4o",
        description="litellm model string for the vision critique LLM.",
    )
    screenshot_timeout_ms: int = Field(
        default=15_000,
        ge=1000,
        description="Timeout for Playwright page load + screenshot.",
    )
    viewport_width: int = Field(default=1280)
    viewport_height: int = Field(default=800)
    pass_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum fidelity score (0–1) to pass visual verification.",
    )


class LoggingConfig(BaseModel):
    """Structured logging settings."""

    model_config = ConfigDict(frozen=True)

    level: LogLevel = Field(default=LogLevel.INFO)
    json_output: bool = Field(
        default=True,
        description="Emit structured JSON logs (disable for human-readable dev output).",
    )
    log_file: Path | None = Field(
        default=None,
        description="Optional file path to write logs to (in addition to stderr).",
    )


class ModelRateLimitEntry(BaseModel):
    """RPM and TPM limits for a single model."""

    model_config = ConfigDict(frozen=True)

    rpm: int = Field(ge=1, description="Requests per minute.")
    tpm: int = Field(ge=1, description="Tokens per minute.")


class RateLimitConfig(BaseModel):
    """Per-model rate limiting configuration.

    Maps litellm model strings to their RPM/TPM limits.  The rate limiter
    uses a sliding-window token bucket to enforce these limits across all
    agents sharing the same model.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True)
    models: dict[str, ModelRateLimitEntry] = Field(
        default_factory=lambda: {
            "gemini/gemini-2.5-pro": ModelRateLimitEntry(rpm=150, tpm=2_000_000),
            "gemini/gemini-2.5-flash": ModelRateLimitEntry(rpm=1000, tpm=1_000_000),
            "gemini/gemini-2.0-flash": ModelRateLimitEntry(rpm=2000, tpm=4_000_000),
            "gemini/gemini-2.0-flash-lite": ModelRateLimitEntry(rpm=4000, tpm=4_000_000),
        },
        description="Map of litellm model string → rate limits.",
    )
    safety_margin: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="Use only this fraction of the limit (0.85 = leave 15%% headroom).",
    )


class TenantConfig(BaseModel):
    """Multi-tenant isolation settings."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(
        default="default",
        description="Unique tenant identifier. All data is partitioned by this.",
    )
    org_id: str = Field(
        default="default",
        description="Organisation the tenant belongs to.",
    )
    data_root: Path = Field(
        default=Path.home() / ".swarm" / "data",
        description="Root directory for tenant data storage.",
    )


class WebScraperConfig(BaseModel):
    """Secure web scraper tool configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False)
    url_whitelist: list[str] = Field(
        default_factory=list,
        description="Allowed URL patterns (glob). Empty = allow all public URLs.",
    )
    max_response_bytes: int = Field(
        default=512_000,
        ge=1_000,
        description="Maximum response size in bytes (default 500KB).",
    )
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    blocked_domains: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "0.0.0.0", "169.254.*", "10.*"],
        description="Blocked domains/IPs (internal network protection).",
    )


class SQLConfig(BaseModel):
    """Read-only SQL query tool configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False)
    connections: dict[str, str] = Field(
        default_factory=dict,
        description="Named DB connections: alias → connection string.",
    )
    read_only: bool = Field(
        default=True,
        description="Enforce read-only queries (block DDL/DML).",
    )
    max_rows: int = Field(default=100, ge=1, le=1000)
    query_timeout_seconds: int = Field(default=30, ge=5)


class WebhookConfig(BaseModel):
    """Outbound webhook tool configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False)
    url_whitelist: list[str] = Field(
        default_factory=list,
        description="Allowed webhook URLs.",
    )
    max_payload_bytes: int = Field(default=10_240, ge=100)
    timeout_seconds: int = Field(default=30, ge=5, le=120)


class HITLConfig(BaseModel):
    """Human-in-the-loop checkpoint configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="Enable HITL checkpoints.",
    )
    default_timeout_minutes: int = Field(
        default=30,
        ge=1,
        description="Minutes to wait for human approval before timing out.",
    )
    auto_approve_after: int = Field(
        default=0,
        ge=0,
        description="Auto-approve checkpoints after N minutes (0 = never).",
    )
    notify_channel: str = Field(
        default="",
        description="Optional webhook URL for checkpoint notifications.",
    )


class ToolsConfig(BaseModel):
    """Configuration for extended MCP tools."""

    model_config = ConfigDict(frozen=True)

    web_scraper: WebScraperConfig = Field(default_factory=WebScraperConfig)
    sql: SQLConfig = Field(default_factory=SQLConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Definition of a single worker agent.

    The ``model`` field maps 1-to-1 to a ``litellm`` model string so the
    orchestrator can route tasks to **any** LLM provider per-agent::

        model: "gemini/gemini-1.5-pro"          # Google
        model: "anthropic/claude-3-opus-20240229" # Anthropic
        model: "ollama/qwen"                     # Local Ollama
        model: "gpt-4o"                          # OpenAI
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique human-readable name for this agent.",
    )
    role: AgentRole = Field(
        default=AgentRole.CUSTOM,
        description="Semantic role (used for DAG ordering and display).",
    )
    model: str = Field(
        ...,
        min_length=1,
        description=(
            "litellm-compatible model identifier. Supports any provider: "
            "'gpt-4o', 'anthropic/claude-3-opus-20240229', "
            "'gemini/gemini-1.5-pro', 'ollama/qwen', etc."
        ),
    )
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System-level instructions defining this agent's behaviour.",
    )
    tools: list[str] = Field(
        default_factory=lambda: [
            "read_file",
            "write_file",
            "list_directory",
            "run_command",
        ],
        description="MCP tool names this agent is allowed to invoke.",
    )
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "Agent names that must complete before this agent starts. "
            "Used to build the task dependency DAG."
        ),
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        description="Maximum LLM loop iterations before the agent is force-stopped.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )
    timeout_seconds: float = Field(
        default=600.0,
        ge=10.0,
        description="Hard wall-clock timeout for this agent's execution.",
    )
    token_budget: int = Field(
        default=50_000,
        ge=1_000,
        description=(
            "Maximum token spend per task. At 90% a soft warning is injected; "
            "at 100% the agent is force-stopped. Set per-role: e.g. 100000 "
            "for architect, 25000 for QA."
        ),
    )
    fallback_models: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered fallback model chain for the LLM Gateway. When the primary "
            "model returns 429/503, the gateway tries each fallback in order. "
            "Example: ['anthropic/claude-3.5-sonnet', 'ollama/qwen']."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata (passed through, not interpreted).",
    )

    @field_validator("name")
    @classmethod
    def _name_is_slug(cls, v: str) -> str:
        """Enforce slug-like names (lowercase, hyphens/underscores, no spaces)."""
        import re

        if not re.match(r"^[a-z0-9][a-z0-9_-]*$", v):
            msg = (
                f"Agent name '{v}' must be lowercase alphanumeric with "
                "hyphens/underscores, starting with a letter or digit."
            )
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class SwarmConfig(BaseModel):
    """Root configuration model — deserialised from the user's YAML file.

    Usage::

        from swarm.config.loader import load_config
        cfg = load_config("swarm.yaml")
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Project / swarm session name.",
    )
    version: str = Field(
        default="1",
        description="Config schema version for forward compatibility.",
    )
    workspace: Path = Field(
        ...,
        description=(
            "Absolute path to the shared host workspace directory. "
            "All agent sandboxes mount this as their working directory."
        ),
    )
    agents: list[AgentConfig] = Field(
        ...,
        min_length=1,
        description="One or more agent definitions.",
    )

    # --- Infrastructure ---
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # --- Phase 2–5 enterprise features ---
    tenant: TenantConfig = Field(default_factory=TenantConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)

    # --- Telemetry ---
    enable_session_replay: bool = Field(
        default=False,
        description=(
            "When True, records every LLM call, tool execution, memory search, "
            "and routing decision to a JSONL file for post-run analysis."
        ),
    )

    # --- Global LLM defaults ---
    default_model: str = Field(
        default="gpt-4o",
        description=(
            "Fallback litellm model string used when an agent does not "
            "specify its own ``model``."
        ),
    )
    api_keys: dict[str, SecretStr] = Field(
        default_factory=dict,
        description=(
            "Provider API keys, keyed by provider name. "
            "Prefer environment variables; this field is for per-config overrides."
        ),
    )

    # --- Validators ---

    @model_validator(mode="after")
    def _validate_agent_names_unique(self) -> "SwarmConfig":
        """Ensure no two agents share the same name."""
        names = [a.name for a in self.agents]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            msg = f"Duplicate agent names: {dupes}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_depends_on_references(self) -> "SwarmConfig":
        """Ensure ``depends_on`` references point to existing agents."""
        known = {a.name for a in self.agents}
        for agent in self.agents:
            unknown = set(agent.depends_on) - known
            if unknown:
                msg = (
                    f"Agent '{agent.name}' depends on unknown agents: {unknown}. "
                    f"Known agents: {known}"
                )
                raise ValueError(msg)
            if agent.name in agent.depends_on:
                msg = f"Agent '{agent.name}' cannot depend on itself."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_no_dependency_cycles(self) -> "SwarmConfig":
        """Detect circular dependencies in the agent DAG."""
        adjacency: dict[str, list[str]] = {
            a.name: list(a.depends_on) for a in self.agents
        }
        visited: set[str] = set()
        in_stack: set[str] = set()

        def _dfs(node: str) -> None:
            if node in in_stack:
                msg = f"Circular dependency detected involving agent '{node}'."
                raise ValueError(msg)
            if node in visited:
                return
            in_stack.add(node)
            for dep in adjacency.get(node, []):
                _dfs(dep)
            in_stack.discard(node)
            visited.add(node)

        for name in adjacency:
            _dfs(name)

        return self

    @field_validator("workspace")
    @classmethod
    def _workspace_is_absolute(cls, v: Path) -> Path:
        if not v.is_absolute():
            msg = f"Workspace path must be absolute, got: {v}"
            raise ValueError(msg)
        return v
