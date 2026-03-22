"""Aegis Compliance Engine — Zero-Trust proxy between AI agents and tools.

The Aegis layer sits between every WorkerAgent and the ToolRegistry,
enforcing security policies, eliminating token waste, and hardening the
endpoint against malicious payloads.

Architecture::

    WorkerAgent
        │
        ▼
    ┌─────────────────────────────────────────────┐
    │  Aegis Compliance Engine                    │
    │  ├─ Anti-Redundancy Cache (mtime-aware)     │
    │  ├─ Anti-Loop Circuit Breaker               │
    │  ├─ Pydantic Schema Validator               │
    │  ├─ Output Payload Caps                     │
    │  ├─ Command Sandbox (regex denylist)         │
    │  ├─ Role-Based Tool Boundaries              │
    │  └─ Compliance Metrics                      │
    └─────────────────────────────┬───────────────┘
                                  │ ✅ PASS
                                  ▼
                          ToolRegistry
                          (actual exec)

Policies:
    DIRECTIVE 1 — Token & Redundancy Governance
        P1  READ_CACHE       mtime-aware file cache, 0-cost on re-read
        P2  ANTI_LOOP        same tool + same args + fail 3x → hard block
        P3  READ_BUDGET      max unique reads per session
        P4  DIR_CACHE        list_directory cache with mtime check

    DIRECTIVE 2 — Payload Sanitization
        P5  SCHEMA_VALIDATE  Pydantic model validation on every call
        P6  OUTPUT_CAP       truncate oversized tool output (max 8KB)

    DIRECTIVE 3 — Endpoint Hardening
        P7  CMD_SANDBOX      regex denylist for destructive / network cmds
        P8  ROLE_BOUNDARY    tier-based tool access (qa can't use architect tools)
        P9  WRITE_SCOPE      no absolute paths or ../ traversal
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import logging

logger = structlog.get_logger(__name__)          # async — use in async methods
_sync_log = logging.getLogger("swarm.aegis")     # sync — use in __init__, sync methods


# ══════════════════════════════════════════════════════════════════════════════
# Policy violation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyViolation:
    """Describes why a tool call was blocked."""
    policy: str
    reason: str
    suggestion: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Compliance metrics
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComplianceMetrics:
    """Per-agent tool usage stats for compliance decisions and reporting."""
    reads: int = 0
    writes: int = 0
    commands: int = 0
    blocked_calls: int = 0
    cache_hits: int = 0
    loop_breaks: int = 0
    schema_rejections: int = 0
    output_truncations: int = 0
    cmd_blocks: int = 0
    total_calls: int = 0
    tokens_saved_estimate: int = 0     # rough estimate of tokens saved by cache hits

    # V2 metrics
    echo_reads_blocked: int = 0        # P10: write-then-read on same file
    fuzzy_thrash_breaks: int = 0       # P11: similar failed args
    density_blocks: int = 0            # P12: minified/binary file blocks
    blind_edit_blocks: int = 0         # P13: write without prior read
    burn_rate_alerts: int = 0          # P14: >25% budget in one turn
    injection_blocks: int = 0          # P15: prompt injection in output
    hallucination_count: int = 0       # P16: ghost tool calls

    # Detailed tracking
    files_read: dict[str, int] = field(default_factory=dict)  # path → count
    files_written: set[str] = field(default_factory=set)
    cmd_consecutive_failures: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Cached file entry (for anti-redundancy)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CachedFileEntry:
    """Tracks a cached file read with its content hash and filesystem mtime."""
    path: str
    content: str
    mtime: float           # filesystem mtime at time of read
    content_hash: str      # sha256 of content
    read_count: int = 1
    tokens_estimate: int = 0  # rough char/4 token estimate
    source_agent: str = ""    # which agent originally read this file


# ══════════════════════════════════════════════════════════════════════════════
# Anti-loop tracker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoopTracker:
    """Tracks repeated tool calls with identical arguments."""
    call_key: str          # hash of (tool_name + args)
    count: int = 1
    last_success: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Command denylist (regex patterns)
# ══════════════════════════════════════════════════════════════════════════════

# These patterns match destructive, exfiltration, and escape commands
CMD_DENYLIST: list[tuple[str, re.Pattern]] = [
    # Destructive filesystem
    ("DESTRUCTIVE", re.compile(r"\brm\s+(-[rRfF]+\s+)?/", re.I)),
    ("DESTRUCTIVE", re.compile(r"\bmkfs\b", re.I)),
    ("DESTRUCTIVE", re.compile(r"\bdd\s+.*\bof=/dev/", re.I)),
    ("DESTRUCTIVE", re.compile(r"\bformat\b.*\b[A-Z]:\\", re.I)),
    ("DESTRUCTIVE", re.compile(r">\s*/dev/sd[a-z]", re.I)),
    ("DESTRUCTIVE", re.compile(r"\bshred\b", re.I)),

    # Network exfiltration
    ("NETWORK", re.compile(r"\bcurl\b.*\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("NETWORK", re.compile(r"\bwget\b.*\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("NETWORK", re.compile(r"\bnc\s+-", re.I)),       # netcat
    ("NETWORK", re.compile(r"\bncat\b", re.I)),
    ("NETWORK", re.compile(r"\bssh\s+", re.I)),
    ("NETWORK", re.compile(r"\bscp\s+", re.I)),
    ("NETWORK", re.compile(r"\brsync\b.*@", re.I)),

    # Shell escape / privilege escalation
    ("ESCAPE", re.compile(r"\bsudo\b", re.I)),
    ("ESCAPE", re.compile(r"\bsu\s+-", re.I)),
    ("ESCAPE", re.compile(r"\bchmod\s+[0-7]*777\b")),
    ("ESCAPE", re.compile(r"\bchown\b.*\broot\b", re.I)),
    ("ESCAPE", re.compile(r"\beval\b.*\$\(", re.I)),    # eval $(...)

    # Crypto / password
    ("SENSITIVE", re.compile(r"\bpasswd\b", re.I)),
    ("SENSITIVE", re.compile(r"\.ssh/", re.I)),
    ("SENSITIVE", re.compile(r"\.env\b(?!\.)", re.I)),  # .env but not .env.example
]


# P15: Prompt injection signatures to scan in tool output
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"system\s*override", re.I),
    re.compile(r"<\|im_start\|>", re.I),
    re.compile(r"<\|im_end\|>", re.I),
    re.compile(r"\[INST\]", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|in)\s+(?:unrestricted|jailbreak|DAN)", re.I),
    re.compile(r"disregard\s+(?:all|your|the)\s+(?:previous|prior|above)", re.I),
    re.compile(r"new\s+instructions?:?\s*you\s+(?:are|must|should|will)", re.I),
    re.compile(r"ADMIN\s*OVERRIDE", re.I),
]

# P12: File extensions / patterns that are minified or binary
DENSE_FILE_EXTENSIONS: set[str] = {
    '.min.js', '.min.css', '.bundle.js', '.chunk.js',
    '.map', '.wasm', '.pyc', '.pyo', '.so', '.dll',
    '.exe', '.bin', '.dat', '.db', '.sqlite', '.sqlite3',
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
    '.woff', '.woff2', '.ttf', '.eot', '.otf',
    '.zip', '.tar', '.gz', '.bz2', '.7z',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.lock',  # package-lock.json etc.
}


# ══════════════════════════════════════════════════════════════════════════════
# Role-based tool boundaries
# ══════════════════════════════════════════════════════════════════════════════

# Each role gets a tool allowlist. Tools not in the list are blocked.
# NOTE: Keys MUST match the AgentRole enum values in config.py:
# backend, frontend, qa, devops, database, architect, custom
# Available tools: read_file, write_file, list_directory, run_command, search_memory
ROLE_TOOL_BOUNDARIES: dict[str, set[str]] = {
    "architect":  {"read_file", "write_file", "list_directory", "search_memory", "run_command"},
    "backend":    {"read_file", "write_file", "list_directory", "run_command", "search_memory"},
    "frontend":   {"read_file", "write_file", "list_directory", "run_command", "search_memory"},
    "database":   {"read_file", "write_file", "list_directory", "run_command", "search_memory"},
    "qa":         {"read_file", "write_file", "list_directory", "run_command", "search_memory"},
    "devops":     {"read_file", "write_file", "list_directory", "run_command", "search_memory"},
    "custom":     {"read_file", "write_file", "list_directory", "run_command", "search_memory"},
}

# Pydantic input models per tool (imported lazily to avoid circular dep)
_TOOL_INPUT_MODELS: dict[str, str] = {
    "read_file": "ReadFileInput",
    "write_file": "WriteFileInput",
    "list_directory": "ListDirectoryInput",
    "run_command": "RunCommandInput",
    "git_diff": "GitDiffInput",
    "git_commit": "GitCommitInput",
    "search_memory": "SearchMemoryInput",
}


# ══════════════════════════════════════════════════════════════════════════════
# Aegis Engine
# ══════════════════════════════════════════════════════════════════════════════

class ToolComplianceProxy:
    """Aegis Compliance Engine — zero-trust proxy for all agent tool calls.

    Wraps a ToolRegistry with policy enforcement, caching, validation,
    and security hardening.
    """

    # Maximum output size from any tool (chars). Larger outputs are truncated.
    OUTPUT_CAP_CHARS = 8_000

    # ═══ CLASS-LEVEL SHARED FILE CACHE ═══
    # Shared across ALL agent instances so agent B benefits from agent A's reads.
    # Key: file_path → CachedFileEntry (with mtime validation).
    _SHARED_FILE_CACHE: dict[str, Any] = {}

    # ═══ CLASS-LEVEL FILE OWNERSHIP REGISTRY ═══
    # Tracks which agent "owns" each file. First agent to write a file owns it.
    # Prevents multiple agents from overwriting each other's work.
    # Resets per session via reset_session_state().
    _FILE_OWNERSHIP: dict[str, str] = {}  # path → agent_name

    def __init__(
        self,
        *,
        registry: Any,              # ToolRegistry
        agent_name: str,
        agent_role: str = "",        # e.g. "backend-dev", "qa-engineer"
        workspace: str = "",
        max_reads: int = 12,
        max_cmd_failures: int = 3,
        ledger: Any | None = None,
    ) -> None:
        self._registry = registry
        self._agent_name = agent_name
        self._agent_role = agent_role or agent_name
        self._workspace = workspace
        self._max_reads = max_reads
        self._max_cmd_failures = max_cmd_failures
        self._ledger = ledger
        self._metrics = ComplianceMetrics()

        # DIRECTIVE 1: Anti-Redundancy Cache (uses class-level _SHARED_FILE_CACHE)
        self._dir_cache: dict[str, tuple[str, float]] = {}    # path → (content, time)

        # DIRECTIVE 1: Anti-Loop Circuit Breaker
        self._loop_tracker: dict[str, LoopTracker] = {}       # call_key → tracker
        self._loop_threshold = 3  # Block on 4th identical failed call

        # V2 state tracking
        self._recent_writes: dict[str, float] = {}   # P10: path → write timestamp
        self._call_history: list[tuple[str, str, bool]] = []  # P11: (tool, args_hash, success)
        self._valid_tools: set[str] | None = None    # P16: cached tool names from registry
        self._turn_tokens_start: int = 0              # P14: tokens at start of turn
        self._agent_budget: int = 0                   # P14: total budget (set externally)

        _sync_log.info(
            "aegis.init | agent=%s role=%s workspace=%s max_reads=%d",
            self._agent_name, self._agent_role, self._workspace, self._max_reads,
        )

    @property
    def metrics(self) -> ComplianceMetrics:
        return self._metrics

    # ──────────────────────────────────────────────────────────────────────
    # Tool descriptor passthrough (with role filtering)
    # ──────────────────────────────────────────────────────────────────────

    def get_openai_tools(self, phase: str = "all", **kwargs) -> list[dict[str, Any]]:
        """Return tools filtered by phase, role, AND compliance state."""
        tools = self._registry.get_openai_tools(phase=phase, **kwargs)

        # P8: Role boundary enforcement — remove tools not in this role's allowlist
        allowed = ROLE_TOOL_BOUNDARIES.get(self._agent_role, set())
        if allowed:
            tools = [t for t in tools if t["function"]["name"] in allowed]

        # P3: Remove read_file if read budget exceeded
        if self._metrics.reads >= self._max_reads:
            tools = [t for t in tools if t["function"]["name"] != "read_file"]

        # CMD circuit breaker
        if self._metrics.cmd_consecutive_failures >= self._max_cmd_failures:
            tools = [t for t in tools if t["function"]["name"] != "run_command"]

        return tools

    def get_tool_descriptors(self) -> Any:
        return self._registry.get_tool_descriptors()

    # ──────────────────────────────────────────────────────────────────────
    # Main execution gate
    # ──────────────────────────────────────────────────────────────────────

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool after checking all Aegis policies."""
        self._metrics.total_calls += 1

        await logger.info(
            "aegis.gate",
            agent=self._agent_name,
            tool=tool_name,
            call_num=self._metrics.total_calls,
            reads=self._metrics.reads,
            writes=self._metrics.writes,
            cache_size=len(self._SHARED_FILE_CACHE),
        )

        # ── P5: Pydantic schema validation ──
        violation = self._validate_schema(tool_name, arguments)
        if violation:
            return await self._block(tool_name, arguments, violation)

        # ── P8: Role boundary ──
        violation = self._check_role_boundary(tool_name)
        if violation:
            return await self._block(tool_name, arguments, violation)

        # ── P7: Command sandbox ──
        if tool_name == "run_command":
            violation = self._check_cmd_sandbox(arguments)
            if violation:
                return await self._block(tool_name, arguments, violation)

        # ── P9: Write scope ──
        if tool_name == "write_file":
            violation = self._check_write_scope(arguments)
            if violation:
                return await self._block(tool_name, arguments, violation)

        # ── P17: File Ownership (cross-agent deconfliction) ──
        if tool_name == "write_file":
            violation = self._check_file_ownership(arguments)
            if violation:
                return await self._block(tool_name, arguments, violation)

        # ── P2: Anti-loop circuit breaker ──
        violation = self._check_loop(tool_name, arguments)
        if violation:
            return await self._block(tool_name, arguments, violation)

        # ── P1: Anti-redundancy cache (read_file) ──
        if tool_name == "read_file":
            cached = self._check_read_cache(arguments)
            if cached is not None:
                return cached

        # ── P4: Anti-redundancy cache (list_directory) ──
        if tool_name == "list_directory":
            cached = self._check_dir_cache(arguments)
            if cached is not None:
                return cached

        # ── P3: Read budget ──
        if tool_name == "read_file" and self._metrics.reads >= self._max_reads:
            violation = PolicyViolation(
                policy="READ_BUDGET",
                reason=f"Read budget exhausted ({self._metrics.reads}/{self._max_reads}). Write your output now.",
                suggestion="Start writing files immediately.",
            )
            return await self._block(tool_name, arguments, violation)

        # ═══ All V1 policies passed — check V2 pre-execution gates ═══

        # ── P16: Hallucination monitor (ghost tool detection) ──
        if self._valid_tools is None:
            try:
                raw_tools = self._registry.get_openai_tools()
                self._valid_tools = {t["function"]["name"] for t in raw_tools}
            except Exception:
                self._valid_tools = set()

        if self._valid_tools and tool_name not in self._valid_tools:
            self._metrics.hallucination_count += 1
            _sync_log.warning(
                "aegis.hallucinated_tool | agent=%s tool=%s count=%d",
                self._agent_name, tool_name, self._metrics.hallucination_count,
            )
            if self._metrics.hallucination_count >= 3:
                violation = PolicyViolation(
                    policy="HALLUCINATION_MONITOR",
                    reason=(
                        f"Agent '{self._agent_name}' has hallucinated {self._metrics.hallucination_count} "
                        f"non-existent tools. Agent is unstable — consider stopping."
                    ),
                    suggestion="Use only: read_file, write_file, list_directory, run_command.",
                )
                return await self._block(tool_name, arguments, violation)
            # Still return error but don't block yet
            from swarm.mcp.tools import ToolResult
            return ToolResult(
                success=False,
                output=f"Tool '{tool_name}' does not exist. Available tools: {', '.join(sorted(self._valid_tools))}.",
            )

        # ── P10: Echo Write-Read Suppressor ──
        if tool_name == "read_file":
            path = arguments.get("path", "")
            if path in self._recent_writes:
                elapsed = time.time() - self._recent_writes[path]
                if elapsed < 60:  # Within 60 seconds of writing
                    self._metrics.echo_reads_blocked += 1
                    _sync_log.info(
                        "aegis.echo_read_blocked | agent=%s file=%s wrote_ago=%.1fs",
                        self._agent_name, path, elapsed,
                    )
                    from swarm.mcp.tools import ToolResult
                    return ToolResult(
                        success=True,
                        output=(
                            f"[ECHO READ BLOCKED] You wrote '{path}' {elapsed:.0f}s ago. "
                            f"The file contains exactly what you wrote. No need to re-read. "
                            f"Move on to your next task."
                        ),
                        metadata={"echo_blocked": True},
                    )

        # ── P12: Semantic Density Cap (minified/binary files) ──
        if tool_name == "read_file":
            path = arguments.get("path", "")
            violation = self._check_density(path)
            if violation:
                return await self._block(tool_name, arguments, violation)

        # ── P13: Blind Edit Preventer ──
        if tool_name == "write_file":
            path = arguments.get("path", "")
            violation = self._check_blind_edit(path)
            if violation:
                return await self._block(tool_name, arguments, violation)

        # ── P11: Fuzzy Thrashing Breaker ──
        violation = self._check_fuzzy_thrash(tool_name, arguments)
        if violation:
            return await self._block(tool_name, arguments, violation)

        # ═══ All policies passed — execute through registry ═══
        result = await self._registry.execute(tool_name, arguments)

        # ── P6: Output payload cap ──
        result = self._cap_output(result)

        # ── P15: Prompt Injection Firewall (scan tool output) ──
        result = self._scan_output_injection(result)

        # ── Post-execution tracking ──
        self._track_result(tool_name, arguments, result)

        # Track call history for fuzzy matching (include success status)
        args_hash = self._make_call_key(tool_name, arguments)
        call_success = getattr(result, 'success', True)
        self._call_history.append((tool_name, args_hash, call_success))
        if len(self._call_history) > 10:
            self._call_history = self._call_history[-10:]

        # ── Data I/O log: record what went in and out ──
        out_preview = str(getattr(result, 'output', ''))[:120]
        _sync_log.info(
            "aegis.io | agent=%s tool=%s action=PASS success=%s args=%s out=%s",
            self._agent_name, tool_name, call_success,
            str(arguments)[:100], out_preview,
        )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # P1: Anti-Redundancy File Cache (mtime-aware)
    # ──────────────────────────────────────────────────────────────────────

    def _check_read_cache(self, args: dict[str, Any]) -> Any | None:
        """Return cached file content if file hasn't changed on disk.
        
        Uses the CLASS-LEVEL shared cache so all agents benefit from each
        other's reads. If agent A reads ARCHITECTURE.md, agent B gets a
        cache hit without burning tokens.
        """
        path = args.get("path", "")
        if not path or path not in self._SHARED_FILE_CACHE:
            return None

        entry = self._SHARED_FILE_CACHE[path]

        # Check if file has been modified since last read
        full_path = Path(self._workspace) / path if self._workspace else Path(path)
        try:
            current_mtime = full_path.stat().st_mtime
        except OSError:
            # File doesn't exist anymore — invalidate cache
            del self._SHARED_FILE_CACHE[path]
            return None

        if current_mtime > entry.mtime:
            # File modified — invalidate cache, allow fresh read
            del self._SHARED_FILE_CACHE[path]
            return None

        # Cache HIT — return cached content
        self._metrics.cache_hits += 1
        self._metrics.tokens_saved_estimate += entry.tokens_estimate
        entry.read_count += 1

        # Log with cross-agent attribution
        cache_source = getattr(entry, 'source_agent', 'self')
        is_cross = cache_source != self._agent_name
        _sync_log.info(
            "aegis.cache_hit | agent=%s file=%s read_count=%d tokens_saved=%d cross_agent=%s source=%s",
            self._agent_name, path, entry.read_count, entry.tokens_estimate,
            is_cross, cache_source,
        )

        from swarm.mcp.tools import ToolResult
        return ToolResult(
            success=True,
            output=(
                f"[ALREADY READ — attempt #{entry.read_count}] "
                f"You already read '{path}' earlier. The file has NOT changed since your last read. "
                f"Use the content from your earlier read — do NOT re-read this file. "
                f"Move on to writing your output files."
            ),
            metadata={"cached": True, "read_count": entry.read_count, "tokens_saved": entry.tokens_estimate},
        )

    def _cache_file_read(self, path: str, content: str) -> None:
        """Store a file read in the cache with its mtime and hash."""
        full_path = Path(self._workspace) / path if self._workspace else Path(path)
        try:
            mtime = full_path.stat().st_mtime
        except OSError:
            mtime = time.time()

        self._SHARED_FILE_CACHE[path] = CachedFileEntry(
            path=path,
            content=content,
            mtime=mtime,
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            tokens_estimate=len(content) // 4,
            source_agent=self._agent_name,
        )
        _sync_log.info(
            "aegis.cache_populated | agent=%s file=%s tokens_est=%d shared=True",
            self._agent_name, path, len(content) // 4,
        )

    # ──────────────────────────────────────────────────────────────────────
    # P4: Anti-Redundancy Directory Cache
    # ──────────────────────────────────────────────────────────────────────

    def _check_dir_cache(self, args: dict[str, Any]) -> Any | None:
        """Return cached directory listing if recent enough (30s TTL)."""
        path = args.get("path", ".")
        if path not in self._dir_cache:
            return None

        content, cached_time = self._dir_cache[path]
        if time.time() - cached_time > 30.0:
            del self._dir_cache[path]
            return None

        self._metrics.cache_hits += 1
        from swarm.mcp.tools import ToolResult
        return ToolResult(
            success=True,
            output=f"[CACHED] {content}",
            metadata={"cached": True},
        )

    # ──────────────────────────────────────────────────────────────────────
    # P2: Anti-Loop Circuit Breaker
    # ──────────────────────────────────────────────────────────────────────

    def _check_loop(self, tool_name: str, args: dict[str, Any]) -> PolicyViolation | None:
        """Block identical failing tool calls after threshold."""
        call_key = self._make_call_key(tool_name, args)
        tracker = self._loop_tracker.get(call_key)

        if tracker is None:
            return None

        # Only trigger on repeated FAILED calls
        if not tracker.last_success and tracker.count >= self._loop_threshold:
            self._metrics.loop_breaks += 1
            return PolicyViolation(
                policy="ANTI_LOOP",
                reason=(
                    f"[SYSTEM OVERRIDE: TOOL LOOP DETECTED] "
                    f"'{tool_name}' called {tracker.count} times with same args and failed each time. "
                    f"YOU MUST PLAN A DIFFERENT APPROACH."
                ),
                suggestion="Try different arguments, a different tool, or skip this step entirely.",
            )

        return None

    def _track_loop(self, tool_name: str, args: dict[str, Any], success: bool) -> None:
        """Update loop tracker after execution."""
        call_key = self._make_call_key(tool_name, args)
        tracker = self._loop_tracker.get(call_key)

        if tracker is None:
            self._loop_tracker[call_key] = LoopTracker(
                call_key=call_key, count=1, last_success=success,
            )
        else:
            if not success:
                tracker.count += 1
                tracker.last_success = False
            else:
                # Success resets the counter
                tracker.count = 0
                tracker.last_success = True

    @staticmethod
    def _make_call_key(tool_name: str, args: dict[str, Any]) -> str:
        """Create a stable hash key from tool name + arguments."""
        # Sort args for stable hashing
        args_str = str(sorted(args.items()))
        return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()[:12]

    # ──────────────────────────────────────────────────────────────────────
    # P5: Pydantic Schema Validation
    # ──────────────────────────────────────────────────────────────────────

    def _validate_schema(self, tool_name: str, args: dict[str, Any]) -> PolicyViolation | None:
        """Validate tool arguments against the Pydantic input model."""
        model_name = _TOOL_INPUT_MODELS.get(tool_name)
        if model_name is None:
            return None  # Unknown tool — let registry handle it

        try:
            import swarm.mcp.tools as tools_mod
            model_class = getattr(tools_mod, model_name, None)
            if model_class is None:
                return None

            # Validate — this strips unknown fields and checks types
            model_class(**args)
            return None  # Valid

        except Exception as exc:
            self._metrics.schema_rejections += 1
            # Clean error — no Python traceback
            error_msg = str(exc)
            # Extract just the field errors, not the full Pydantic dump
            if "validation error" in error_msg.lower():
                # Pydantic v2 style: extract field-level errors only
                lines = error_msg.split("\n")
                clean_errors = [l.strip() for l in lines if l.strip() and not l.startswith("  ")]
                error_msg = "; ".join(clean_errors[:3])  # Max 3 errors

            return PolicyViolation(
                policy="SCHEMA_VALIDATE",
                reason=f"Invalid arguments for '{tool_name}': {error_msg[:200]}",
                suggestion="Check the tool's parameter types and required fields.",
            )

    # ──────────────────────────────────────────────────────────────────────
    # P6: Output Payload Cap
    # ──────────────────────────────────────────────────────────────────────

    def _cap_output(self, result: Any) -> Any:
        """Truncate oversized tool output before it enters LLM context."""
        output = getattr(result, 'output', '')
        if len(output) > self.OUTPUT_CAP_CHARS:
            self._metrics.output_truncations += 1
            truncated = output[:self.OUTPUT_CAP_CHARS]
            truncated += f"\n\n... [OUTPUT TRUNCATED: {len(output):,} chars → {self.OUTPUT_CAP_CHARS:,} chars]"

            _sync_log.warning(
                "aegis.output_capped | agent=%s original=%d capped=%d",
                self._agent_name, len(output), self.OUTPUT_CAP_CHARS,
            )

            # Create new result with truncated output
            from swarm.mcp.tools import ToolResult
            return ToolResult(
                success=result.success,
                output=truncated,
                error=getattr(result, 'error', ''),
                metadata={**getattr(result, 'metadata', {}), "truncated": True, "original_size": len(output)},
            )
        return result

    # ──────────────────────────────────────────────────────────────────────
    # P7: Command Sandbox (regex denylist)
    # ──────────────────────────────────────────────────────────────────────

    def _check_cmd_sandbox(self, args: dict[str, Any]) -> PolicyViolation | None:
        """Check run_command against the denylist."""
        command = args.get("command", "")
        if not command:
            return None

        for category, pattern in CMD_DENYLIST:
            if pattern.search(command):
                self._metrics.cmd_blocks += 1
                return PolicyViolation(
                    policy="CMD_SANDBOX",
                    reason=(
                        f"⛔ BLOCKED: Command matches {category} denylist pattern. "
                        f"Command '{command[:60]}...' is not allowed."
                    ),
                    suggestion="Use a different approach. Destructive/network/escape commands are blocked.",
                )

        return None

    # ──────────────────────────────────────────────────────────────────────
    # P8: Role-Based Tool Boundaries
    # ──────────────────────────────────────────────────────────────────────

    def _check_role_boundary(self, tool_name: str) -> PolicyViolation | None:
        """Ensure agent only uses tools allowed for their role."""
        allowed = ROLE_TOOL_BOUNDARIES.get(self._agent_role)
        if allowed is None:
            return None  # Unknown role — allow all

        if tool_name not in allowed:
            return PolicyViolation(
                policy="ROLE_BOUNDARY",
                reason=f"Agent '{self._agent_name}' (role: {self._agent_role}) cannot use '{tool_name}'.",
                suggestion=f"Allowed tools for {self._agent_role}: {', '.join(sorted(allowed))}",
            )
        return None

    # ──────────────────────────────────────────────────────────────────────
    # P9: Write Scope Enforcement
    # ──────────────────────────────────────────────────────────────────────

    def _check_write_scope(self, args: dict[str, Any]) -> PolicyViolation | None:
        """Block writes outside workspace."""
        path = args.get("path", "")
        if path.startswith("/") or ".." in path:
            return PolicyViolation(
                policy="WRITE_SCOPE",
                reason=f"Cannot write to '{path}' — must be a relative path within workspace.",
                suggestion="Use relative paths like 'src/app.py'.",
            )
        return None

    def _check_file_ownership(self, args: dict[str, Any]) -> PolicyViolation | None:
        """Block writes to files owned by another agent (P17: Cross-Agent Deconfliction).

        First agent to successfully write a file becomes its owner.
        Other agents are blocked from overwriting with a clear message telling
        them to focus on their own files instead.
        """
        path = args.get("path", "")
        if not path:
            return None

        owner = self._FILE_OWNERSHIP.get(path)
        if owner is not None and owner != self._agent_name:
            _sync_log.warning(
                "aegis.file_ownership_blocked | agent=%s file=%s owner=%s",
                self._agent_name, path, owner,
            )
            return PolicyViolation(
                policy="FILE_OWNERSHIP",
                reason=(
                    f"[DECONFLICTION] File '{path}' is already being handled by '{owner}'. "
                    f"You MUST NOT overwrite another agent's work. "
                    f"Focus on files that no other agent is writing to."
                ),
                suggestion=(
                    f"Skip this file — '{owner}' owns it. "
                    f"Write your logic in a DIFFERENT file instead, or read the "
                    f"existing file to build on top of what '{owner}' wrote."
                ),
            )
        return None

    @classmethod
    def reset_session_state(cls) -> None:
        """Reset all class-level shared state between sessions.

        Call this at the start of each new session to prevent stale
        ownership or cached data from leaking across sessions.
        """
        cls._SHARED_FILE_CACHE.clear()
        cls._FILE_OWNERSHIP.clear()
        _sync_log.info("aegis.session_state_reset | cache=%d ownership=%d", 0, 0)

    # ──────────────────────────────────────────────────────────────────────
    # P10: Echo Write-Read Suppressor (handled inline in execute)
    # P11: Fuzzy Thrashing Breaker
    # ──────────────────────────────────────────────────────────────────────

    def _check_fuzzy_thrash(self, tool_name: str, args: dict[str, Any]) -> PolicyViolation | None:
        """Detect fuzzy thrashing: same tool, similar args, repeated FAILURES in sliding window.
        
        Only counts FAILED calls. write_file is exempt (writing multiple files is normal).
        """
        # Never block write_file — writing many different files is normal
        if tool_name == "write_file":
            return None

        if len(self._call_history) < 3:
            return None

        # Look at last 6 calls — only count FAILED calls to the same tool
        failed_same = [(t, h) for t, h, success in self._call_history[-6:]
                       if t == tool_name and not success]

        if len(failed_same) < 3:
            return None

        unique_hashes = {h for _, h in failed_same}

        # 3+ failed calls to same tool with 2+ unique arg sets → thrashing
        if len(failed_same) >= 3 and len(unique_hashes) >= 2:
            self._metrics.fuzzy_thrash_breaks += 1
            _sync_log.warning(
                "aegis.fuzzy_thrash | agent=%s tool=%s failed_calls=%d unique_args=%d",
                self._agent_name, tool_name, len(failed_same), len(unique_hashes),
            )
            return PolicyViolation(
                policy="FUZZY_THRASH",
                reason=(
                    f"[SYSTEM OVERRIDE: FUZZY THRASHING DETECTED] "
                    f"You called '{tool_name}' {len(failed_same)} times with different args and ALL FAILED. "
                    f"STOP. Explain your failure and try a COMPLETELY different approach."
                ),
                suggestion="Try a different tool or approach entirely.",
            )
        return None

    # ──────────────────────────────────────────────────────────────────────
    # P12: Semantic Density Cap
    # ──────────────────────────────────────────────────────────────────────

    def _check_density(self, path: str) -> PolicyViolation | None:
        """Block reads of minified, binary, or machine-generated files."""
        if not path:
            return None

        lower = path.lower()

        # Check compound extensions (e.g., .min.js)
        for ext in DENSE_FILE_EXTENSIONS:
            if lower.endswith(ext):
                self._metrics.density_blocks += 1
                return PolicyViolation(
                    policy="DENSITY_CAP",
                    reason=(
                        f"File '{path}' is minified/binary/machine-generated. "
                        f"Reading it will exhaust your context window with unparseable content."
                    ),
                    suggestion="Skip this file. Focus on source files (.py, .ts, .tsx, .js, .css, .html).",
                )

        # Check if file is likely minified by name pattern
        if any(marker in lower for marker in ['node_modules/', '__pycache__/', '.next/', 'dist/', 'build/']):
            self._metrics.density_blocks += 1
            return PolicyViolation(
                policy="DENSITY_CAP",
                reason=f"File '{path}' is in a build/vendor directory. Do not read generated code.",
                suggestion="Read source files instead.",
            )

        return None

    # ──────────────────────────────────────────────────────────────────────
    # P13: Blind Edit Preventer
    # ──────────────────────────────────────────────────────────────────────

    def _check_blind_edit(self, path: str) -> PolicyViolation | None:
        """Warn (but don't block) writes to files the agent hasn't read first."""
        if not path:
            return None

        # Allow new file creation (file not in workspace)
        full_path = Path(self._workspace) / path if self._workspace else Path(path)
        if not full_path.exists():
            return None  # Creating new file is OK

        # If file exists but agent hasn't read it → warn only (don't block)
        if path not in self._metrics.files_read and path not in self._metrics.files_written:
            self._metrics.blind_edit_blocks += 1
            _sync_log.warning(
                "aegis.blind_edit_warning | agent=%s file=%s",
                self._agent_name, path,
            )
            # Don't block — just track for reporting
            # (Hard-blocking was too aggressive for legitimate creation workflows)

        return None

    # ──────────────────────────────────────────────────────────────────────
    # P15: Prompt Injection Firewall
    # ──────────────────────────────────────────────────────────────────────

    def _scan_output_injection(self, result: Any) -> Any:
        """Scan tool output for prompt injection signatures and redact them."""
        output = getattr(result, 'output', '')
        if not output or len(output) < 10:
            return result

        for pattern in INJECTION_PATTERNS:
            match = pattern.search(output)
            if match:
                self._metrics.injection_blocks += 1
                _sync_log.critical(
                    "aegis.INJECTION_DETECTED | agent=%s pattern=%s matched=%s",
                    self._agent_name, pattern.pattern[:40], match.group()[:50],
                )
                # Redact the injection and return sanitized output
                sanitized = pattern.sub('[REDACTED_INJECTION]', output)
                from swarm.mcp.tools import ToolResult
                return ToolResult(
                    success=result.success,
                    output=(
                        f"[⚠️ SECURITY: Potential prompt injection detected and neutralized in tool output. "
                        f"Proceed with caution.]\n\n{sanitized}"
                    ),
                    error=getattr(result, 'error', ''),
                    metadata={**getattr(result, 'metadata', {}), "injection_redacted": True},
                )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Block helper
    # ──────────────────────────────────────────────────────────────────────

    async def _block(self, tool_name: str, args: dict[str, Any], violation: PolicyViolation) -> Any:
        """Return a synthetic ToolResult for a blocked call."""
        self._metrics.blocked_calls += 1

        await logger.warning(
            "aegis.blocked",
            agent_name=self._agent_name,
            tool=tool_name,
            policy=violation.policy,
            reason=violation.reason[:120],
        )

        if self._ledger is not None:
            try:
                self._ledger.record_tool_execution(
                    agent=self._agent_name,
                    tool_name=tool_name,
                    arguments=args,
                    success=False,
                    result_str=f"[AEGIS:{violation.policy}] {violation.reason[:200]}",
                    duration_ms=0,
                )
            except Exception:
                pass  # Don't let ledger errors block policy enforcement

        from swarm.mcp.tools import ToolResult
        msg = f"⛔ AEGIS [{violation.policy}]: {violation.reason}"
        if violation.suggestion:
            msg += f"\n💡 {violation.suggestion}"
        return ToolResult(success=False, output=msg, error=violation.reason)

    # ──────────────────────────────────────────────────────────────────────
    # Post-execution tracking
    # ──────────────────────────────────────────────────────────────────────

    def _track_result(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        """Update metrics and caches after a tool executes."""
        success = getattr(result, 'success', True)
        output = getattr(result, 'output', '')

        # Track loop pattern
        self._track_loop(tool_name, args, success)

        if tool_name == "read_file":
            path = args.get("path", "")
            self._metrics.reads += 1
            self._metrics.files_read[path] = self._metrics.files_read.get(path, 0) + 1
            # Cache the successful read
            if success and output:
                self._cache_file_read(path, output)

        elif tool_name == "write_file":
            if success:
                path = args.get("path", "")
                self._metrics.writes += 1
                self._metrics.files_written.add(path)
                # Invalidate read cache for this file (it changed)
                self._SHARED_FILE_CACHE.pop(path, None)
                # P10: Track write timestamp for echo-read suppression
                self._recent_writes[path] = time.time()
                # P17: Register file ownership (first write wins)
                if path and path not in self._FILE_OWNERSHIP:
                    self._FILE_OWNERSHIP[path] = self._agent_name
                    _sync_log.info(
                        "aegis.file_ownership_registered | agent=%s file=%s",
                        self._agent_name, path,
                    )

        elif tool_name == "list_directory":
            path = args.get("path", ".")
            if success and output:
                self._dir_cache[path] = (output, time.time())

        elif tool_name == "run_command":
            self._metrics.commands += 1
            if success:
                self._metrics.cmd_consecutive_failures = 0
            else:
                self._metrics.cmd_consecutive_failures += 1
                if self._metrics.cmd_consecutive_failures >= 2:
                    _sync_log.warning(
                        "aegis.cmd_failures_rising | agent=%s consecutive=%d threshold=%d",
                        self._agent_name, self._metrics.cmd_consecutive_failures, self._max_cmd_failures,
                    )

    # ──────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────

    def get_compliance_report(self) -> dict[str, Any]:
        """Get a full Aegis compliance report for this agent."""
        m = self._metrics
        return {
            "agent": self._agent_name,
            "role": self._agent_role,
            "total_calls": m.total_calls,
            "blocked_calls": m.blocked_calls,
            "block_rate": f"{m.blocked_calls / max(m.total_calls, 1) * 100:.1f}%",
            # V1 metrics
            "cache_hits": m.cache_hits,
            "tokens_saved": m.tokens_saved_estimate,
            "loop_breaks": m.loop_breaks,
            "schema_rejections": m.schema_rejections,
            "output_truncations": m.output_truncations,
            "cmd_blocks": m.cmd_blocks,
            "reads": m.reads,
            "writes": m.writes,
            "commands": m.commands,
            "unique_files_read": len(m.files_read),
            "files_written": list(m.files_written),
            # V2 metrics
            "echo_reads_blocked": m.echo_reads_blocked,
            "fuzzy_thrash_breaks": m.fuzzy_thrash_breaks,
            "density_blocks": m.density_blocks,
            "blind_edit_blocks": m.blind_edit_blocks,
            "injection_blocks": m.injection_blocks,
            "hallucination_count": m.hallucination_count,
            "burn_rate_alerts": m.burn_rate_alerts,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Passthrough
    # ──────────────────────────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Proxy any attribute access to the underlying registry."""
        return getattr(self._registry, name)
