# AI Swarm — Architecture & Internal Reference

> **A parallel, stack-agnostic multi-agent swarm orchestration platform.**
> Author: Shashwat · License: MIT · Python 3.11+

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Package Map](#package-map)
4. [Core Execution Flow](#core-execution-flow)
5. [Key Components Deep Dive](#key-components-deep-dive)
6. [Configuration (YAML Schema)](#configuration-yaml-schema)
7. [Tools (MCP)](#tools-mcp)
8. [Event Bus (Redis)](#event-bus-redis)
9. [Long-Term Memory (LTM)](#long-term-memory-ltm)
10. [Dashboard](#dashboard)
11. [Token Budgets & Guardrails](#token-budgets--guardrails)
12. [CLI Reference](#cli-reference)
13. [Dependencies](#dependencies)
14. [Optimization Levers](#optimization-levers)

---

## Overview

AI Swarm is a **multi-agent orchestration platform** that coordinates a team of LLM-powered agents working in parallel on a shared codebase. Each agent has a defined role (architect, backend, frontend, QA, etc.), a set of permitted tools, and connects to any LLM provider via [LiteLLM](https://docs.litellm.ai/).

**What makes it different:**
- **Declarative YAML config** — Define agents, roles, dependencies, models, and infrastructure in one file
- **DAG-based execution** — Agents run in tiers based on dependency ordering (architect → devs → QA)
- **Shared workspace** — All agents write to the same directory with Redis-backed file locking
- **Interactive mode** — Agents wait for human direction via a real-time dashboard chat
- **Memory system** — LanceDB vector store for cross-session learning (skills, bugs, patterns)

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                             │
│  swarm run config.yaml --dashboard --interactive                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Orchestrator (orchestrator.py)                 │
│                                                                  │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐              │
│  │ Config      │  │ DAG        │  │ Rate Limiter │              │
│  │ Loader      │  │ Resolver   │  │ (per-model)  │              │
│  └─────────────┘  └────────────┘  └──────────────┘              │
│                                                                  │
│  Phase 1: Parse YAML → SwarmConfig                               │
│  Phase 2: Connect Redis, init LanceDB, register agents           │
│  Phase 3: Resolve dependency DAG into execution tiers            │
│  Phase 4: Execute tiers (or wait in interactive mode)            │
│  Phase 5: Post-execution LTM extraction                          │
└────────┬──────────────┬──────────────┬───────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Worker     │ │  Worker     │ │  Worker     │  ← Parallel agents
│  Agent      │ │  Agent      │ │  Agent      │
│ (worker.py) │ │ (worker.py) │ │ (worker.py) │
│             │ │             │ │             │
│ LLM Loop:  │ │  Each has:  │ │  Signals:   │
│ think →    │ │  - model    │ │ [TASK_      │
│ tool call → │ │  - tools   │ │  COMPLETE]  │
│ observe →  │ │  - prompt   │ │  when done  │
│ repeat     │ │  - budget   │ │             │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────┴───────────────┘
                       │
                       ▼
         ┌──────────────────────────┐
         │   Shared Infrastructure   │
         │                          │
         │  📡 Event Bus (Redis)    │
         │  🔒 File Write Locks    │
         │  🧠 LTM (LanceDB)      │
         │  🛠  Tool Executor (MCP) │
         │  📊 Dashboard (WebSocket)│
         └──────────────────────────┘
```

---

## Package Map

```
src/swarm/
├── cli.py                    # Typer CLI — entry point (swarm run, validate, agents)
├── config/
│   ├── models.py             # Pydantic schema: SwarmConfig, AgentConfig, etc.
│   └── loader.py             # YAML → SwarmConfig parser with env var expansion
├── core/
│   ├── orchestrator.py       # Main engine — DAG resolution, tier exec, dispatch
│   ├── state.py              # SwarmState — async-safe state container
│   ├── manager.py            # SwarmManager — watchdog, stall detection, nudging
│   ├── manager_chat.py       # ManagerChat — LLM-backed chat interface
│   └── compressor.py         # Context compression + LTM extraction engine
├── agents/
│   └── worker.py             # WorkerAgent — the core agentic LLM loop
├── events/
│   ├── bus.py                # Redis event bus (streams, pub/sub, file locks)
│   └── channels.py           # Channel name enums
├── mcp/
│   └── tools.py              # ToolExecutor — read_file, write_file, run_command, etc.
├── dashboard/
│   ├── dashboard.py          # FastAPI app + WebSocket server
│   └── static/index.html     # Single-page dashboard UI
├── sandbox/
│   └── manager.py            # Docker sandbox manager (container lifecycle)
├── infra/
│   └── env.py                # .env loading
└── verification/
    └── (visual verification pipeline — Playwright screenshots)
```

---

## Core Execution Flow

### Normal Mode (`swarm run config.yaml`)

1. **Parse** — Load YAML, validate with Pydantic, resolve env vars
2. **Infrastructure** — Connect Redis EventBus, init LanceDB memory, register agents in SwarmState
3. **DAG** — Resolve `depends_on` into execution tiers (topological sort)
4. **Tier Execution** — Run agents in parallel per tier:
   - Tier 0: `architect` (plans the project structure)
   - Tier 1: `backend-dev`, `frontend-dev`, `db-engineer` (build in parallel)
   - Tier 2: `qa-engineer` (tests everything)
5. **LTM Extraction** — After all agents complete, extract knowledge (bugs, skills, patterns) into LanceDB

### Interactive Mode (`swarm run config.yaml --interactive`)

- **Phase 4 is skipped** — agents do NOT auto-start
- The orchestrator starts the dashboard and **waits for human input via chat**
- The Manager (chat LLM) receives instructions and calls `dispatch_dynamic_task` to assign work
- Each dynamic task is capped at **15 iterations** and **50k tokens**

---

## Key Components Deep Dive

### 1. Orchestrator (`core/orchestrator.py`)

The brain of the system. Key responsibilities:

| Method | What It Does |
|--------|-------------|
| `run()` | Main entry — phases 1–5 |
| `_resolve_dag()` | Topological sort on `depends_on` → execution tiers |
| `_execute_tier()` | Run all agents in a tier concurrently |
| `_run_agent_loop()` | Instantiate WorkerAgent + ToolExecutor, call `agent.execute()` |
| `dispatch_dynamic_task()` | Chat-driven task dispatch (capped at 15 iter) |
| `_dispatch_helper_agent()` | Re-dispatch a completed agent to help a struggling one |
| `_build_enhanced_prompt()` | Inject workspace context + LTM memories into system prompt |
| `_post_execution_extraction()` | Extract knowledge after task completion |

### 2. Worker Agent (`agents/worker.py`)

The agentic execution loop. Each agent:

1. Receives an enhanced system prompt (with LTM context)
2. Enters a `while iteration < max_iterations` loop
3. Calls the LLM with the accumulated message history + tools
4. If LLM returns **tool_calls** → executes them via ToolExecutor, appends results
5. If LLM returns **text with `[TASK_COMPLETE]`** → breaks and returns
6. If LLM returns **text without completion signal** → appends "Continue" prompt
7. Budget enforced: breaks if `total_tokens > 50,000`

**Interrupt system:** A parallel async task monitors the EventBus for schema changes. When triggered, the interrupt context is injected into the message history, forcing the agent to re-evaluate.

**Context compression:** When message history exceeds the token threshold (default 16k), the Compressor summarizes old messages via a fast LLM (Gemini Flash), archives the originals in Redis.

### 3. SwarmState (`core/state.py`)

Async-safe, lockable state container. Tracks:

- **Agent records**: status, iterations, token usage, sandbox ID
- **Task records**: lifecycle (PENDING → RUNNING → COMPLETED/FAILED)
- **Message ledger**: ordered log of all inter-agent messages
- **Artifacts**: files produced by agents
- **Pair message counts**: powers the circuit breaker

State transitions are validated against a transition graph:
```
Agent: INITIALISING → IDLE → RUNNING → COMPLETED
                                ↓
                          INTERRUPTED / FAILED
```

The `snapshot()` method produces an immutable Pydantic model safe for passing to LLMs.

### 4. SwarmManager (`core/manager.py`)

Background watchdog loop running every 10 seconds:

- **Stall detection**: If an agent hasn't progressed for 30s, nudge via EventBus
- **Max nudges**: After 10 nudges, give up (prevents infinite nudge loops)
- **Helper dispatch**: Currently disabled — was auto-dispatching completed agents as helpers but burned too many tokens

### 5. Manager Chat (`core/manager_chat.py`)

LLM-powered chat interface on the dashboard:

- **Model**: Gemini 2.5 Pro (configurable)
- **Tools available to the Manager LLM**:
  - `read_file` — read any workspace file
  - `list_directory` — browse project structure
  - `dispatch_task` — assign work to any agent
  - `get_agent_status` — query live agent status from SwarmState
- **Multi-round tool calling**: The Manager can call multiple tools in one turn (e.g., dispatch 4 agents at once)
- **Proactive reporting**: Auto-broadcasts task dispatch/complete/fail/stall events to the chat

### 6. Compressor (`core/compressor.py`)

Dual responsibility:

**Short-term compression:**
```
Raw content (logs, diffs) → token count check → if > threshold →
  archive to Redis (7-day TTL) → summarize via fast LLM → inject summary
```

**Long-term extraction (after task completion):**
```
Conversation history → extraction LLM → structured JSON →
  MemoryEntry objects → embed → persist to LanceDB
```

Categories: `skills`, `bugs`, `rules`, `preferences`, `patterns`

---

## Configuration (YAML Schema)

The root model is `SwarmConfig` (Pydantic v2, frozen). Key sections:

```yaml
name: project-name
workspace: /absolute/path/to/code

default_model: "gemini/gemini-2.5-flash"

rate_limits:
  enabled: true
  safety_margin: 0.85       # Use 85% of limit
  models:
    "gemini/gemini-2.5-pro":
      rpm: 150
      tpm: 2000000

event_bus:
  redis_url: "redis://localhost:6379/0"

memory:
  enabled: true
  auto_extract: true
  storage_dir: ".swarm_memory"   # LanceDB + raw JSON

compression:
  token_threshold: 16000         # Compress context above this

circuit_breaker:
  max_round_trips: 8             # Max back-and-forth between agents
  cooldown_seconds: 30

agents:
  - name: agent-name
    role: backend                 # backend|frontend|qa|devops|database|architect|custom
    model: "gemini/gemini-2.5-flash"
    temperature: 0.2
    max_iterations: 50            # Hard cap on LLM loop iterations
    timeout_seconds: 600          # Wall-clock timeout
    depends_on: [architect]       # DAG dependencies
    tools: [read_file, write_file, list_directory, run_command, git_diff, git_commit]
    system_prompt: |
      Your role description here...
```

**Model field is LiteLLM-compatible**, so any provider works:
- `gemini/gemini-2.5-pro` (Google)
- `anthropic/claude-3-opus-20240229` (Anthropic)
- `gpt-4o` (OpenAI)
- `ollama/qwen` (local)

---

## Tools (MCP)

The `ToolExecutor` (`mcp/tools.py`) provides agents with these tools:

| Tool | Description | Security |
|------|-------------|----------|
| `read_file` | Read file contents (truncated at 100k chars) | Workspace-scoped |
| `write_file` | Write/create files with Redis file lock | Lock-serialized |
| `list_directory` | List files/dirs (optional recursive) | Workspace-scoped |
| `run_command` | Execute shell commands | Blocked list enforced |
| `git_diff` | Show uncommitted changes | Read-only |
| `git_commit` | Commit staged changes | Workspace-scoped |
| `search_memory` | Search LTM vector store | Read-only |

**Blocked commands** (configurable): `rm -rf /`, `shutdown`, `reboot`, `mkfs`, `dd if=/dev/zero`

**Sandbox execution**: When Docker is available, commands run inside containers. The `workdir` parameter prepends `cd <path> &&` to commands.

---

## Event Bus (Redis)

Three communication primitives (`events/bus.py`):

### 1. Streams (Durable)
- Redis Streams with consumer groups and acknowledgment
- Used for: task assignment, completion, schema change notifications
- Max stream length: 10,000 entries (oldest trimmed)

### 2. Pub/Sub (Real-time)
- Fire-and-forget broadcast
- Used for: interrupts ("schema changed — re-read context")

### 3. Distributed File Locks
- Redis `SET key value NX PX <timeout>` (standard Redlock pattern)
- Lua script for safe release (only the lock holder can release)
- Prevents two agents from writing the same file simultaneously

```
Agent A writes foo.py → ACQUIRE lock → WRITE → RELEASE
Agent B writes foo.py → ACQUIRE (blocked) → wait → (granted after A releases) → WRITE → RELEASE
```

---

## Long-Term Memory (LTM)

Storage layout:
```
<workspace>/.swarm_memory/
├── lancedb/          # Vector tables
│   ├── skills        # Reusable coding patterns
│   ├── bugs          # Bug resolution records
│   ├── rules         # Architectural rules
│   ├── preferences   # Style/tool preferences
│   └── patterns      # Recurring approaches
└── raw/              # JSON backup of every entry
```

**Lifecycle:**
1. **Boot**: Orchestrator calls `compressor.get_boot_context()` → retrieves relevant memories → injects into agent system prompt
2. **Runtime**: Agents can call `search_memory` tool to query LTM
3. **Post-task**: Orchestrator calls `compressor.extract_knowledge()` → LLM extracts structured entries → persisted to LanceDB

**Search**: Vector similarity (cosine distance) with configurable relevance threshold (default 0.7).

---

## Dashboard

**Stack**: FastAPI + WebSocket + vanilla JS single-page app

**Layout** (3-column):
- **Left**: Live Events + File Changes
- **Center**: Manager Chat (primary interaction)
- **Right**: Live Metrics (tokens, files, tools, iterations) + Agent Status + Activity Log

**WebSocket protocol**: JSON events with `{source, event, data}` structure. All agent events (tool calls, LLM calls, file writes) are broadcast in real-time.

**Endpoints**:
| Route | Purpose |
|-------|---------|
| `GET /` | Dashboard HTML |
| `WS /ws` | Real-time event stream |
| `GET /api/status` | Swarm state snapshot |
| `POST /api/kill` | Kill all agents |
| `POST /api/pause` | Pause all agents |
| `POST /api/resume` | Resume all agents |

---

## Token Budgets & Guardrails

| Guardrail | Value | Location |
|-----------|-------|----------|
| Per-task token budget | **50,000** | `worker.py:TOKEN_BUDGET` |
| Dynamic task max iterations | **15** | `orchestrator.py:dispatch_dynamic_task` |
| Default max iterations | **50** | `config/models.py:AgentConfig` |
| Stall threshold | **30s** | `manager.py:STALL_THRESHOLD` |
| Max nudges before giving up | **10** | `manager.py:MAX_NUDGES` |
| Context compression threshold | **16,000 tokens** | YAML `compression.token_threshold` |
| Rate limit safety margin | **85%** | YAML `rate_limits.safety_margin` |
| Circuit breaker max round trips | **8** | YAML `circuit_breaker.max_round_trips` |
| Redis archive TTL | **7 days** | `compressor.py` |

---

## CLI Reference

```bash
# Validate a config file (syntax + dependency cycle check)
swarm validate config.yaml

# Run the swarm (normal mode — agents auto-execute in DAG order)
swarm run config.yaml --console-logs

# Run in interactive mode (agents wait for dashboard chat)
swarm run config.yaml --dashboard --interactive

# List agents defined in a config
swarm agents config.yaml

# Common flags
--dashboard          # Start the web dashboard on :8080
--console-logs       # Print structured logs to terminal
--interactive / -i   # Don't auto-start agents; wait for chat input
```

---

## Dependencies

| Category | Package | Purpose |
|----------|---------|---------|
| **Core** | `pydantic` ≥2.6 | Config schema, state models |
| **Core** | `pyyaml` ≥6.0 | YAML config parsing |
| **Core** | `anyio` ≥4.4 | Async primitives |
| **CLI** | `typer` ≥0.12 | CLI framework |
| **LLM** | `litellm` ≥1.40 | Multi-provider LLM router |
| **LLM** | `tiktoken` ≥0.7 | Token counting |
| **Messaging** | `redis` ≥5.0 | Event bus, file locks, archival |
| **Sandbox** | `docker` ≥7.1 | Container management |
| **Verification** | `playwright` ≥1.44 | Screenshot-based visual testing |
| **Logging** | `structlog` ≥24.1 | Structured JSON logging |
| **Resilience** | `tenacity` ≥8.3 | Retry with backoff |
| **Memory** | `lancedb` ≥0.6 | Vector store for LTM |
| **Memory** | `pyarrow` ≥16.0 | LanceDB dependency |
| **Dashboard** | `starlette`, `uvicorn` | FastAPI + WebSocket server |
| **HTTP** | `httpx` ≥0.27 | MCP SSE transport |

---

## Optimization Levers

Areas to tune for performance, cost, and quality:

### Token Reduction
- **`TOKEN_BUDGET`** in `worker.py` — lower for cheaper tasks (e.g., 30k for QA)
- **`compression.token_threshold`** — lower threshold triggers compression sooner
- **Agent prompts** — shorter, more focused prompts = fewer tokens per call
- **LTM boot context** — inject relevant memories to skip discovery (agents read fewer files)

### Speed
- **Model selection** — use Flash/Lite models for simple agents, Pro only for architect
- **`max_iterations`** — cap tightly for well-scoped tasks
- **Rate limits** — tune `safety_margin` closer to 1.0 if headroom exists

### Quality
- **System prompts** — more specific = better output, fewer iterations
- **`depends_on` DAG** — ensure agents get the right context from predecessors
- **LTM categories** — add domain-specific categories for better retrieval
- **`relevance_threshold`** — lower = more memories retrieved, higher = more precise

### Cost
- **Disable `auto_extract`** if LTM extraction isn't needed
- **Use cheaper models** for compression (`gemini/gemini-2.0-flash-lite`)
- **Disable helper dispatch** (already disabled — was the biggest cost multiplier)
- **Per-agent token budgets** — consider different budgets per role

### Reliability
- **`circuit_breaker.max_round_trips`** — prevent infinite agent-to-agent loops
- **`timeout_seconds`** — hard wall-clock cap per agent
- **`MAX_NUDGES`** — prevent infinite stall-nudge cycles
- **Redis `max_stream_length`** — prevent unbounded stream growth
