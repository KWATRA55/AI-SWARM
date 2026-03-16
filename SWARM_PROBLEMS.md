# AI Swarm — Technical Audit: Problems, Gaps & Optimization Questions

> **For the knowledge base.** Every issue is ranked by severity and tagged by subsystem.  
> Use the questions at the end of each section to feed your KB for deeper research.

---

## 🔴 CATEGORY 1: Sandbox & Docker — Broken in Production

### Problem 1.1: `env_override` parameter mismatch crashes sandbox provisioning
**Severity:** 🔴 Critical (currently broken)  
**File:** `orchestrator.py:513`, `sandbox/manager.py:218`

The orchestrator calls `create_sandbox(agent_config, env_override=...)` but `SandboxManager.create_sandbox()` does **not accept** an `env_override` keyword argument. This raises `TypeError` on every single sandbox creation attempt, which is why you see this in logs:

```
orchestrator.sandbox_create_failed  error=SandboxManager.create_sandbox() 
got an unexpected keyword argument 'env_override'
```

All agents are silently falling back to **local mode** (no Docker isolation). This means:
- No resource limits are enforced
- No network isolation
- `run_command` executes directly on the host

**Fix:** Either add `env_override` parameter to `SandboxManager.create_sandbox()`, or remove the kwarg from the orchestrator call and use `sandbox_cfg.env_vars` instead.

---

### Problem 1.2: Docker SDK uses synchronous blocking calls
**Severity:** 🟡 Medium  
**File:** `sandbox/manager.py`

All Docker operations (`container.start()`, `container.stop()`, `images.pull()`) are synchronous and run on the main event loop via `run_in_executor`. However, `container.start()` on line 348 and `container.stop()` on line 398 are **not** wrapped in `run_in_executor` — they block the entire async event loop.

### Problem 1.3: Container cleanup on crash
**Severity:** 🟡 Medium  

If the orchestrator process crashes (OOM, SIGKILL, unhandled exception), Docker containers stay running with `sleep infinity`. There's no session-level cleanup mechanism (e.g., Docker label-based garbage collection on restart).

**KB Questions:**
> Q10: "What is the proper way to handle Docker container lifecycle when the orchestrator crashes? Should we implement a session-cleanup-on-boot sweep using Docker labels, a periodic heartbeat to detect orphans, or both?"

> Q11: "How should `sandbox/manager.py` be refactored to make all Docker SDK calls truly async? Should we wrap every `docker-py` call in `run_in_executor`, switch to `aiodocker`, or use a subprocess-based approach?"

---

## 🔴 CATEGORY 2: Token & Cost Economy — The #1 Operational Problem

### Problem 2.1: No per-agent or per-role token budgets
**Severity:** 🔴 Critical  

The current 50k budget is **global per-task** regardless of agent role. An architect thinking about architecture should have a different budget than a QA agent running pytest. There's no way to configure this per-agent in the YAML.

### Problem 2.2: LLM calls for compression burn tokens to save tokens
**Severity:** 🟡 Medium  
**File:** `compressor.py:329-361`

Every compression call invokes the LLM (`gemini/gemini-2.0-flash`). If compression happens frequently (which it does when `token_threshold` is low), the compression LLM calls themselves become a significant cost. No tracking of compression overhead exists.

### Problem 2.3: LTM extraction runs on every task completion — even trivial ones
**Severity:** 🟡 Medium  
**File:** `compressor.py:367-442`

After every task, the orchestrator calls `extract_knowledge()` with the entire conversation history (up to 30 messages). For trivial tasks (e.g., "run pytest"), this burns tokens on extraction with nothing useful to extract. There's no heuristic to skip extraction for small/trivial tasks.

### Problem 2.4: "Continue" prompt encourages token burn
**Severity:** 🟡 Medium  
**File:** `worker.py:431-437`

When the agent returns text without `[TASK_COMPLETE]`, the worker appends `"Continue with the task..."` — this encourages the agent to keep going even when it might be done. Agents often explore unnecessarily because of this push mechanism.

### Problem 2.5: No cost tracking or billing dashboard
**Severity:** 🟡 Medium  

Token counts are tracked per-agent in `SwarmState`, but there's no conversion to dollar cost. Users have no visibility into how much each agent costs per session.

**KB Questions:**
> Q12: "How should per-agent token budgets be architected? Should budgets be defined in the YAML config per agent, or calculated dynamically based on task complexity? What's the optimal budget formula: `base_budget * model_cost_multiplier * task_complexity`?"

> Q13: "When should the 'Continue' prompt be sent versus asking the agent to self-assess completion? Would replacing the blind 'Continue' with 'Are you done? If so, say [TASK_COMPLETE]. If not, explain what remains.' reduce token waste?"

> Q14: "What heuristic should trigger LTM extraction? Should we only extract when: (a) the task involved >5 tool calls, (b) errors were encountered and resolved, (c) the conversation exceeded a minimum token count? How do we prevent extracting garbage from trivial tasks?"

---

## 🔴 CATEGORY 3: Long-Term Memory (LTM) — Silently Broken

### Problem 3.1: Embedding model mismatch — `text-embedding-3-small` is an OpenAI model
**Severity:** 🔴 Critical  
**File:** `config/models.py:278`, `compressor.py:837-854`

The default `embedding_model` is `"text-embedding-3-small"` (OpenAI). But the rest of the swarm uses **Gemini** models. Unless the user has an OpenAI API key set, **every embedding call fails silently**, falling back to a zero vector `[0.0] * 1536`.

This means:
- All LTM entries are stored with zero vectors → **search returns random/no results**
- The system appears to work (no crash) but LTM is completely non-functional
- Boot context injection returns nothing useful

**Fix:** Default should be a Gemini embedding model (e.g., `"gemini/text-embedding-004"`), or auto-detect from the configured LLM provider.

### Problem 3.2: LanceDB vector dimension is hardcoded to 1536
**Severity:** 🔴 Critical  
**File:** `compressor.py:854`

The fallback zero vector is `[0.0] * 1536`, which matches OpenAI's `text-embedding-3-small` but **not** Gemini's embedding models (768 dimensions for `text-embedding-004`). If you fix 3.1 to use Gemini, you'll get a dimension mismatch crash when storing.

### Problem 3.3: No embedding cache — every search re-embeds the query
**Severity:** 🟡 Medium  

Each `search_memories()` call runs the query through the embedding API. Identical queries (e.g., at boot time for the same agent role across sessions) re-pay the API cost every time.

### Problem 3.4: LanceDB tables have no index — full scan on every search
**Severity:** 🟡 Medium  

LanceDB supports ANN indexes (IVF-PQ) for fast approximate search, but the current code does `table.search(embedding).limit(N)` which is a brute-force scan. Fine for <1000 entries, unacceptable at scale.

### Problem 3.5: No deduplication of extracted memories
**Severity:** 🟡 Medium  

If the same task is re-run, the same knowledge is extracted and stored again. Over time, the LTM fills with duplicates that dilute search quality.

**KB Questions:**
> Q15: "What is the correct Gemini embedding model to use for LanceDB vector search? What dimension does it output? How should we handle migration when switching from OpenAI (1536-dim) to Gemini (768-dim) embeddings?"

> Q16: "How should LTM deduplication work? Should we use cosine similarity on insertion (reject if >0.95 similar entry exists), content hashing, or title-matching? What's the performance impact at 10k+ entries?"

> Q17: "At what scale does LanceDB require an ANN index? How do we create and maintain IVF-PQ indexes across the `skills`, `bugs`, `patterns` tables? Should we rebuild indexes on every N insertions?"

---

## 🟠 CATEGORY 4: Heartbeat & Liveness — Defined but Never Wired

### Problem 4.1: Heartbeat system is dead code
**Severity:** 🟠 High  
**Files:** `events/channels.py:104-109` (defines `HEARTBEAT` channel), but **no agent publishes heartbeats** and **no listener consumes them**.

The architecture doc mentions "if an agent misses N heartbeats, the orchestrator can mark it as failed and restart." But this logic doesn't exist. The stall-detection in `manager.py` relies on iteration count differences, not heartbeats.

### Problem 4.2: No crash recovery for agents
**Severity:** 🟠 High  

If an agent's LLM call hangs indefinitely (LiteLLM doesn't timeout by default for Gemini), the agent is effectively dead but still shows as `RUNNING`. There's no mechanism to detect this beyond the wall-clock `timeout_seconds`, and that only works if `asyncio.wait_for()` wraps the execution — which it does, but the resulting `asyncio.TimeoutError` is caught and logged but doesn't clean up properly.

**KB Questions:**
> Q18: "How should the heartbeat system be implemented end-to-end? Should agents emit heartbeats after every LLM call, after every tool call, or on a fixed timer? What's the correct detection threshold (missed heartbeats before declaring dead)?"

> Q19: "What happens when `timeout_seconds` fires during an LLM streaming response? Does LiteLLM's acompletion properly cancel the HTTP request? If not, we have a resource leak (hanging TCP connection). How should we handle this?"

---

## 🟠 CATEGORY 5: Circuit Breaker — Tracks but Doesn't Act

### Problem 5.1: Circuit breaker counts messages but doesn't pause agents
**Severity:** 🟠 High  
**Files:** `core/circuit_breaker.py`, `core/state.py:384-397`

The `SwarmState` tracks inter-agent message pair counts (`_pair_message_counts`), and the circuit breaker config defines `max_round_trips = 8` and `escalation_strategy = "pause_and_notify"`. But there's **no code that actually pauses agents** when the breaker trips. The escalation event is published to the EventBus but nothing subscribes to it.

In the current architecture, agents don't communicate with each other directly — they all go through the orchestrator. So the "ping-pong loop" scenario is theoretical unless helper dispatch re-enables inter-agent work.

**KB Questions:**
> Q20: "In the current orchestrator-centric architecture, what is the realistic inter-agent communication pattern? If agents don't talk to each other directly, is the circuit breaker solving the right problem? Should it instead monitor agent-to-orchestrator message volume (e.g., an agent requesting 50 file reads in a row)?"

---

## 🟠 CATEGORY 6: State Management — No Persistence, Race Risks

### Problem 6.1: SwarmState is in-memory only — dies on restart
**Severity:** 🟠 High  
**File:** `core/state.py`

All state (tasks, agents, messages, artifacts) lives in a Python dict protected by `asyncio.Lock`. If the orchestrator crashes or restarts, **everything is lost**. There's no persistence layer (Redis, SQLite, or file-based).

This means:
- No session resume after crashes
- No historical session analytics
- Dashboard loses all state on restart

### Problem 6.2: State transitions can't handle re-dispatch of completed agents
**Severity:** 🟡 Medium  
**File:** `core/state.py:180`

The transition graph allows `COMPLETED → IDLE` (for helper re-dispatch), but the helper dispatch code in `orchestrator.py:994` sets the agent to `IDLE`, then the `_run_agent_loop` sets it to `RUNNING`. This IDLE→RUNNING transition is valid, but cleaning up the old `finished_at` / `iterations` / `token_usage` fields is not handled — the re-dispatched agent inherits stale counters.

### Problem 6.3: `asyncio.Lock` is not reentrant
**Severity:** 🟡 Medium  

If any method accidentally calls another state method while holding the lock, it deadlocks. This is fragile — one code change could introduce a deadlock.

**KB Questions:**
> Q21: "How should SwarmState be persisted? Options: (a) Redis-backed with atomic operations, (b) SQLite WAL mode for local persistence, (c) periodic snapshot to JSON. What are the trade-offs for each? Which supports session resume?"

> Q22: "Should SwarmState use a reentrant lock (asyncio doesn't have one natively) or restructure to ensure no nested state calls? What's the standard Python pattern for avoiding asyncio deadlocks in state containers?"

---

## 🟡 CATEGORY 7: Dashboard & Real-time — Scaling Limits

### Problem 7.1: Dashboard broadcasts to all WebSocket clients without filtering
**Severity:** 🟡 Medium  
**File:** `dashboard/dashboard.py`

Every event (tool call, LLM call, file write, agent iteration) is broadcast to every connected WebSocket client. With 5 agents running at speed, this produces hundreds of events per second. No throttling, batching, or client-side filtering exists.

### Problem 7.2: No session management in the dashboard
**Severity:** 🟡 Medium  

There's no concept of sessions in the dashboard. If you refresh the page, you lose all chat history and event log. The dashboard doesn't fetch historical events on reconnect.

### Problem 7.3: No authentication or authorization
**Severity:** 🟡 Medium  

The dashboard is wide open — anyone on the network can connect, view events, and dispatch tasks. For enterprise use, this needs auth.

**KB Questions:**
> Q23: "How should WebSocket event broadcasting be optimized for high-throughput swarms? Should we implement: (a) server-side event batching (send every 100ms instead of per-event), (b) client-side subscription filters (only show events for selected agents), (c) event compression (gzip WebSocket frames)?"

> Q24: "What authentication model fits a local-first tool that may also be deployed on a shared server? Token-based API keys? OAuth? Session cookies? How does this interact with the WebSocket connection?"

---

## 🟡 CATEGORY 8: Agent Intelligence — Prompt Engineering Gaps

### Problem 8.1: Agents have no cross-agent visibility
**Severity:** 🟡 Medium  

When the frontend-dev runs, it has no idea what the backend-dev produced. It's told "read the architect's files" but has no summary of what actually exists. The LTM boot context helps but only if previous sessions extracted useful knowledge.

### Problem 8.2: No diff injection on schema changes
**Severity:** 🟡 Medium  

The `SCHEMA_CHANGE_INTERRUPT` broadcast channel exists and the interrupt system is wired, but the actual diff content is not injected into the agent's context. The agent is interrupted but doesn't know what changed.

### Problem 8.3: System prompts are project-specific, not portable
**Severity:** 🟡 Medium  

The YAML config hardcodes "Momentum Stock Screener" in every system prompt. The swarm is designed to be stack-agnostic, but the prompts are not. There's no template system for dynamically generating prompts based on the project type.

### Problem 8.4: No output quality verification
**Severity:** 🟡 Medium  

The `VerificationConfig` exists (Playwright screenshots + vision model scoring) but is disabled by default and doesn't appear to be wired into the execution flow. There's no automated check that the code produced actually works.

**KB Questions:**
> Q25: "How should cross-agent context sharing work? Options: (a) Each tier produces a summary that's injected into the next tier's prompt, (b) A shared 'project state' document that all agents read/write, (c) The LTM acts as the shared memory. Which approach gives the best quality with minimal token cost?"

> Q26: "How should the verification pipeline work? After QA runs pytest, should the orchestrator automatically do: (a) build the project, (b) screenshot the UI, (c) score with vision LLM, (d) re-dispatch to fix issues? What's the iteration limit for this feedback loop?"

> Q27: "How should system prompts be templated for portability? Should we support Jinja2 templates in the YAML with variables like `{{project_name}}`, `{{tech_stack}}`, `{{architecture}}`? Or should the architect agent dynamically generate prompts for other agents?"

---

## 🟡 CATEGORY 9: Infrastructure & Reliability

### Problem 9.1: Redis is a hard dependency — no fallback
**Severity:** 🟡 Medium  

If Redis is down, the EventBus can't connect and the swarm fails to start. There's no in-memory fallback for local development.

### Problem 9.2: Rate limiter doesn't share state across restarts
**Severity:** 🟡 Medium  

The rate limiter tracks RPM/TPM in-memory. If the swarm restarts mid-session, the limiter resets and could immediately exceed provider limits.

### Problem 9.3: No graceful drain on shutdown
**Severity:** 🟡 Medium  

SIGINT triggers shutdown but doesn't wait for in-flight LLM calls to complete or for agents to save their progress. Partial writes could corrupt the workspace.

### Problem 9.4: File write locks use NX (no read locks)
**Severity:** 🟡 Medium  
**File:** `events/bus.py:116-221`

The file lock only prevents concurrent writes, but doesn't prevent reads during writes. An agent could read a partially-written file.

### Problem 9.5: No retry logic for LLM failures
**Severity:** 🟡 Medium  

If the LLM returns a 429 (rate limit) or 500 (server error), the worker catches the exception but doesn't retry. The `tenacity` library is a dependency but isn't used in the worker's LLM call path.

**KB Questions:**
> Q28: "Should the EventBus have an in-memory fallback for when Redis is unavailable? Options: (a) asyncio.Queue-based in-process bus, (b) Make Redis optional and use file-based IPC, (c) Embed a Redis clone like ValKey. What are the trade-offs for local dev vs production?"

> Q29: "How should LLM call retries work with rate limiting? Should we use `tenacity` with exponential backoff that respects the rate limiter's sliding window? What's the correct retry policy: retry on (429, 500, 502, 503), max 3 retries, backoff 1s→2s→4s?"

> Q30: "How should graceful shutdown work? The ideal sequence is: (a) SIGINT received, (b) stop accepting new tasks, (c) wait up to 30s for in-flight LLM calls, (d) save agent progress to state, (e) release all file locks, (f) teardown sandboxes. Is this achievable with asyncio's signal handling?"

---

## Summary Table

| # | Issue | Severity | Subsystem | Status |
|---|-------|----------|-----------|--------|
| 1.1 | `env_override` crashes sandbox creation | 🔴 Critical | Sandbox | Broken |
| 1.2 | Docker SDK blocking calls | 🟡 Medium | Sandbox | Performance |
| 1.3 | Orphan containers on crash | 🟡 Medium | Sandbox | Missing |
| 2.1 | No per-agent token budgets | 🔴 Critical | Tokens | Missing |
| 2.2 | Compression LLM overhead | 🟡 Medium | Tokens | Design |
| 2.3 | Unnecessary LTM extraction | 🟡 Medium | Tokens | Design |
| 2.4 | "Continue" prompt burns tokens | 🟡 Medium | Tokens | Design |
| 2.5 | No cost tracking dashboard | 🟡 Medium | Tokens | Missing |
| 3.1 | Wrong default embedding model | 🔴 Critical | LTM | Broken |
| 3.2 | Hardcoded vector dimension | 🔴 Critical | LTM | Broken |
| 3.3 | No embedding cache | 🟡 Medium | LTM | Performance |
| 3.4 | No ANN index on LanceDB | 🟡 Medium | LTM | Performance |
| 3.5 | No memory deduplication | 🟡 Medium | LTM | Quality |
| 4.1 | Heartbeat system dead code | 🟠 High | Liveness | Dead code |
| 4.2 | No crash recovery | 🟠 High | Liveness | Missing |
| 5.1 | Circuit breaker doesn't act | 🟠 High | Safety | Dead code |
| 6.1 | State not persisted | 🟠 High | State | Missing |
| 6.2 | Stale counters on re-dispatch | 🟡 Medium | State | Bug |
| 6.3 | Non-reentrant lock risk | 🟡 Medium | State | Risk |
| 7.1 | WebSocket event flooding | 🟡 Medium | Dashboard | Performance |
| 7.2 | No session persistence | 🟡 Medium | Dashboard | Missing |
| 7.3 | No authentication | 🟡 Medium | Dashboard | Security |
| 8.1 | No cross-agent context | 🟡 Medium | Intelligence | Quality |
| 8.2 | No diff injection | 🟡 Medium | Intelligence | Dead code |
| 8.3 | Non-portable prompts | 🟡 Medium | Intelligence | Design |
| 8.4 | Verification not wired | 🟡 Medium | Intelligence | Dead code |
| 9.1 | Redis hard dependency | 🟡 Medium | Infra | Design |
| 9.2 | Rate limiter resets on restart | 🟡 Medium | Infra | Bug |
| 9.3 | No graceful drain | 🟡 Medium | Infra | Missing |
| 9.4 | No read locks for files | 🟡 Medium | Infra | Design |
| 9.5 | No LLM retry logic | 🟡 Medium | Infra | Missing |
