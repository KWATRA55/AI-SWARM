"""Context compressor & Long-Term Memory (LTM) extraction engine.

This module has **two complementary responsibilities**:

1. **Short-term compression** — When raw operational content (terminal logs,
   git diffs, build output) exceeds a token threshold, route it through a fast
   LLM to produce a dense executive summary.  The full content is archived to
   Redis before being replaced in-context.

2. **Long-term extraction** — When a task completes, analyse the conversation
   history and task artifacts to extract **reusable knowledge**: bug fixes,
   architectural rules, coding patterns, and skill recipes.  These are embedded
   and persisted into a LanceDB vector store for retrieval in future sessions.

Architecture
------------

.. code-block:: text

    Raw content (logs, diffs, errors)
         │
         ▼
    ┌──────────────┐
    │ Token count  │──► Under threshold? → pass through unchanged
    │   check      │
    └──────┬───────┘
           │ Over threshold
           ▼
    ┌──────────────┐     ┌──────────────┐
    │  Archive to  │────►│  Summarise   │
    │  Redis (raw) │     │ via fast LLM │
    └──────────────┘     └──────┬───────┘
                                │
                                ▼
                         Compressed summary (injected into context)


    Task completion signal
         │
         ▼
    ┌──────────────┐
    │  Extract     │──► Structured JSON: skills, bugs, rules, patterns
    │  knowledge   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐     ┌──────────────┐
    │  Embed via   │────►│  Store in    │
    │  LLM/API     │     │  LanceDB     │
    └──────────────┘     └──────────────┘
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from swarm.config.models import CompressionConfig, MemoryConfig

logger = structlog.get_logger(__name__)

# Embedding dimension for the default model (gemini/text-embedding-004)
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CompressionResult(BaseModel):
    """Result of a short-term compression operation."""

    model_config = ConfigDict(frozen=True)

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float = Field(description="compressed / original")
    summary: str
    archive_key: str | None = Field(
        default=None,
        description="Redis key where the full original content is archived.",
    )


class MemoryEntry(BaseModel):
    """A single knowledge entry extracted from a completed task."""

    model_config = ConfigDict(frozen=True)

    memory_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    category: str = Field(description="One of: skills, bugs, rules, preferences, patterns.")
    title: str = Field(description="Short descriptive title.")
    content: str = Field(description="The extracted knowledge in full.")
    tags: list[str] = Field(default_factory=list)
    source_task: str = Field(default="", description="Task name that produced this memory.")
    source_agent: str = Field(default="", description="Agent that produced this memory.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score.",
    )


class ExtractionResult(BaseModel):
    """Result of a long-term memory extraction operation."""

    model_config = ConfigDict(frozen=True)

    entries: list[MemoryEntry] = Field(default_factory=list)
    extraction_model: str
    extraction_time_seconds: float
    task_name: str
    agent_name: str


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a string using tiktoken (falls back to word estimate).

    Parameters
    ----------
    text:
        The text to count tokens for.
    model:
        Model name for tiktoken encoding selection.  Uses cl100k_base for
        most modern models.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough estimate: ~4 chars per token for English text
        return len(text) // 4


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class Compressor:
    """Short-term context compressor and long-term memory extractor.

    Usage::

        compressor = Compressor(compression_config, memory_config, workspace)

        # Short-term: compress verbose content
        result = await compressor.compress(raw_log_text)
        # result.summary contains the dense version

        # Long-term: extract knowledge after task completion
        extraction = await compressor.extract_knowledge(
            task_name="build-api",
            agent_name="backend-agent",
            conversation_history=[...],
            task_result="API built successfully with 12 endpoints.",
        )

        # Retrieve relevant memories for a new task
        memories = await compressor.search_memories("FastAPI authentication patterns")
    """

    def __init__(
        self,
        compression_config: CompressionConfig,
        memory_config: MemoryConfig,
        workspace: Path,
        redis_client: Any | None = None,
    ) -> None:
        self._comp_cfg = compression_config
        self._mem_cfg = memory_config
        self._workspace = workspace
        self._redis = redis_client

        # Resolve memory store path
        if memory_config.store_path.is_absolute():
            self._store_path = memory_config.store_path
        else:
            self._store_path = workspace / memory_config.store_path

        # LanceDB connection (lazy-initialised)
        self._db: Any | None = None
        self._tables: dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up the memory store directories and LanceDB tables."""
        if not self._mem_cfg.enabled:
            await logger.info("compressor.memory_disabled", msg="LTM is disabled.")
            return

        # Create directories
        self._store_path.mkdir(parents=True, exist_ok=True)
        (self._store_path / "raw").mkdir(exist_ok=True)

        try:
            import lancedb

            db_path = str(self._store_path / "lancedb")
            self._db = lancedb.connect(db_path)

            await logger.info(
                "compressor.lancedb_connected",
                path=db_path,
                tables=self._db.table_names(),
            )
        except ImportError:
            await logger.warning(
                "compressor.lancedb_not_installed",
                msg="LanceDB not installed. LTM storage will use JSON fallback.",
            )
        except Exception as exc:
            await logger.error(
                "compressor.lancedb_init_failed",
                error=str(exc),
            )

    # -------------------------------------------------------------------
    # Short-term compression
    # -------------------------------------------------------------------

    async def compress(
        self,
        content: str,
        *,
        context_hint: str = "",
        session_id: str = "",
    ) -> CompressionResult:
        """Compress raw content if it exceeds the token threshold.

        If the content is under threshold, returns it unchanged with a
        1.0 compression ratio.

        Parameters
        ----------
        content:
            Raw text (logs, diffs, errors) to potentially compress.
        context_hint:
            Optional context for the summariser (e.g. "backend build output").
        session_id:
            Session ID for Redis archival key construction.
        """
        original_tokens = count_tokens(content, self._comp_cfg.compression_model)

        if not self._comp_cfg.enabled or original_tokens <= self._comp_cfg.token_threshold:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                summary=content,
            )

        # Archive full content to Redis before compression
        archive_key: str | None = None
        if self._comp_cfg.archive_to_redis and self._redis is not None:
            archive_key = f"swarm:archive:{session_id}:{uuid.uuid4().hex[:8]}"
            try:
                await self._redis.set(archive_key, content, ex=86400 * 7)  # 7-day TTL
                await logger.info(
                    "compressor.archived",
                    key=archive_key,
                    tokens=original_tokens,
                )
            except Exception as exc:
                await logger.warning("compressor.archive_failed", error=str(exc))
                archive_key = None

        # Summarise via LLM
        summary = await self._call_compression_llm(content, context_hint)
        compressed_tokens = count_tokens(summary, self._comp_cfg.compression_model)

        ratio = compressed_tokens / max(original_tokens, 1)

        await logger.info(
            "compressor.compressed",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ratio=round(ratio, 3),
        )

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            summary=summary,
            archive_key=archive_key,
        )

    async def compress_conversation(
        self,
        messages: list[dict[str, str]],
        *,
        context_hint: str = "",
        session_id: str = "",
    ) -> CompressionResult:
        """Compress a conversation history (list of role/content dicts)."""
        serialised = "\n".join(
            f"[{m.get('role', 'unknown')}] {m.get('content', '')}"
            for m in messages
        )
        return await self.compress(
            serialised,
            context_hint=context_hint or "Inter-agent conversation",
            session_id=session_id,
        )

    async def _call_compression_llm(self, content: str, context_hint: str) -> str:
        """Call the fast LLM to produce a dense summary."""
        from litellm import acompletion

        system_prompt = (
            "You are a technical summariser. Produce a dense, factual executive "
            "summary of the following operational content. Preserve all critical "
            "information: error messages, file paths, version numbers, decisions "
            "made, and action items. Remove redundancy and verbose formatting.\n"
            f"Context: {context_hint}" if context_hint else ""
        )

        try:
            response = await acompletion(
                model=self._comp_cfg.compression_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarise this content:\n\n{content}"},
                ],
                temperature=0.1,
                max_tokens=min(self._comp_cfg.token_threshold, 2000),
            )
            return response.choices[0].message.content or content
        except Exception as exc:
            await logger.error(
                "compressor.llm_failed",
                error=str(exc),
                msg="Falling back to truncation.",
            )
            # Fallback: truncate to threshold
            words = content.split()
            target_words = self._comp_cfg.token_threshold // 2  # rough estimate
            return " ".join(words[:target_words]) + "\n\n[... truncated due to LLM failure]"

    # -------------------------------------------------------------------
    # Long-Term Memory: Extraction
    # -------------------------------------------------------------------

    async def extract_knowledge(
        self,
        *,
        task_name: str,
        agent_name: str,
        conversation_history: list[dict[str, str]],
        task_result: str,
        task_errors: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract reusable knowledge from a completed task.

        This is the core LTM learning loop. After a task finishes, the
        orchestrator calls this method to distil the conversation and
        artifacts into structured knowledge entries.

        Parameters
        ----------
        task_name:
            Name of the completed task.
        agent_name:
            Agent that executed the task.
        conversation_history:
            Message history (role/content dicts) from the task execution.
        task_result:
            Final result or summary of what was accomplished.
        task_errors:
            Any errors encountered during execution.
        """
        if not self._mem_cfg.enabled or not self._mem_cfg.auto_extract:
            return ExtractionResult(
                entries=[],
                extraction_model=self._mem_cfg.extraction_model,
                extraction_time_seconds=0.0,
                task_name=task_name,
                agent_name=agent_name,
            )

        # --- Pre-flight heuristic: skip trivial tasks ---
        # A task is trivial when the conversation is short, error-free, and
        # consumed minimal tokens. Extracting knowledge from these wastes ~4k
        # tokens per invocation on zero-value entries.
        total_msg_tokens = sum(
            len(m.get("content", "").split()) * 1.3  # rough token estimate
            for m in conversation_history
        )
        is_trivial = (
            len(conversation_history) <= 5
            and not task_errors
            and total_msg_tokens < 5_000
        )
        if is_trivial:
            await logger.info(
                "compressor.extraction_skipped",
                agent=agent_name,
                task=task_name,
                reason="trivial_task",
                messages=len(conversation_history),
                estimated_tokens=int(total_msg_tokens),
            )
            return ExtractionResult(
                entries=[],
                extraction_model=self._mem_cfg.extraction_model,
                extraction_time_seconds=0.0,
                task_name=task_name,
                agent_name=agent_name,
            )

        start_time = time.monotonic()

        # Build extraction context
        conversation_text = "\n".join(
            f"[{m.get('role', 'unknown')}] {m.get('content', '')}"
            for m in conversation_history[-30:]  # Last 30 messages max
        )
        errors_text = "\n".join(task_errors or [])

        # Extract via LLM
        entries = await self._call_extraction_llm(
            task_name=task_name,
            agent_name=agent_name,
            conversation_text=conversation_text,
            task_result=task_result,
            errors_text=errors_text,
        )

        elapsed = time.monotonic() - start_time

        # Persist to vector store and raw JSON
        for entry in entries:
            await self._persist_memory(entry)

        await logger.info(
            "compressor.extraction_complete",
            task=task_name,
            agent=agent_name,
            entries_extracted=len(entries),
            time_seconds=round(elapsed, 2),
        )

        return ExtractionResult(
            entries=entries,
            extraction_model=self._mem_cfg.extraction_model,
            extraction_time_seconds=elapsed,
            task_name=task_name,
            agent_name=agent_name,
        )

    async def _call_extraction_llm(
        self,
        *,
        task_name: str,
        agent_name: str,
        conversation_text: str,
        task_result: str,
        errors_text: str,
    ) -> list[MemoryEntry]:
        """Call the extraction LLM to produce structured knowledge entries."""
        from litellm import acompletion

        categories_str = ", ".join(self._mem_cfg.memory_categories)

        system_prompt = f"""You are a knowledge extraction engine for a software engineering AI team.
Analyse the following completed task and extract reusable knowledge entries.

CATEGORIES: {categories_str}

For each piece of knowledge, output a JSON object with these fields:
- "category": one of [{categories_str}]
- "title": short descriptive title (max 80 chars)
- "content": the full knowledge entry (detailed, actionable)
- "tags": list of relevant tags (technologies, concepts, file types)
- "confidence": 0.0–1.0 how confident you are this is reusable

RULES:
- Extract BUG entries when errors were encountered and resolved
- Extract SKILL entries for reusable coding patterns, API usage, or tool configurations
- Extract RULE entries for architectural decisions, conventions, or constraints discovered
- Extract PATTERN entries for recurring implementation approaches
- Extract PREFERENCE entries for coding style or technology choices made
- Only extract genuinely reusable knowledge — skip task-specific implementation details
- Be specific and actionable — include code snippets, commands, or file paths when relevant

Output a JSON array of objects. If nothing is worth extracting, output an empty array [].
"""

        user_content = f"""TASK: {task_name}
AGENT: {agent_name}
RESULT: {task_result}

ERRORS ENCOUNTERED:
{errors_text or "None"}

CONVERSATION HISTORY:
{conversation_text}
"""

        try:
            response = await acompletion(
                model=self._mem_cfg.extraction_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=4000,
            )

            raw_output = response.choices[0].message.content or "[]"
            return self._parse_extraction_output(
                raw_output, task_name=task_name, agent_name=agent_name,
            )
        except Exception as exc:
            await logger.error(
                "compressor.extraction_llm_failed",
                error=str(exc),
            )
            return []

    def _parse_extraction_output(
        self,
        raw_output: str,
        *,
        task_name: str,
        agent_name: str,
    ) -> list[MemoryEntry]:
        """Parse the LLM's JSON output into MemoryEntry objects."""
        # Strip markdown code fences if present
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON array in the output
            import re

            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(parsed, list):
            parsed = [parsed]

        entries: list[MemoryEntry] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                entry = MemoryEntry(
                    category=item.get("category", "patterns"),
                    title=item.get("title", "Untitled"),
                    content=item.get("content", ""),
                    tags=item.get("tags", []),
                    source_task=task_name,
                    source_agent=agent_name,
                    confidence=float(item.get("confidence", 0.8)),
                )
                entries.append(entry)
            except Exception:
                continue

        return entries

    # -------------------------------------------------------------------
    # Long-Term Memory: Persistence
    # -------------------------------------------------------------------

    async def _persist_memory(self, entry: MemoryEntry) -> None:
        """Persist a memory entry to both LanceDB and raw JSON backup."""
        # Raw JSON backup (always works)
        await self._persist_memory_json(entry)

        # Vector store (if available)
        if self._db is not None:
            await self._persist_memory_vector(entry)

    async def _persist_memory_json(self, entry: MemoryEntry) -> None:
        """Save a raw JSON backup of a memory entry."""
        raw_dir = self._store_path / "raw" / entry.category
        raw_dir.mkdir(parents=True, exist_ok=True)
        file_path = raw_dir / f"{entry.memory_id}.json"

        try:
            file_path.write_text(
                entry.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            await logger.error(
                "compressor.json_persist_failed",
                error=str(exc),
                memory_id=entry.memory_id,
            )

    async def _persist_memory_vector(self, entry: MemoryEntry) -> None:
        """Embed and store a memory entry in LanceDB."""
        try:
            embedding = await self._get_embedding(entry.content)

            record = {
                "memory_id": entry.memory_id,
                "category": entry.category,
                "title": entry.title,
                "content": entry.content,
                "tags": json.dumps(entry.tags),
                "source_task": entry.source_task,
                "source_agent": entry.source_agent,
                "created_at": entry.created_at,
                "confidence": entry.confidence,
                "vector": embedding,
            }

            table_name = entry.category
            if table_name in self._db.table_names():
                table = self._db.open_table(table_name)
                table.add([record])
            else:
                self._db.create_table(table_name, [record])

            await logger.info(
                "compressor.vector_persisted",
                memory_id=entry.memory_id,
                category=entry.category,
                title=entry.title,
            )
        except Exception as exc:
            await logger.error(
                "compressor.vector_persist_failed",
                error=str(exc),
                memory_id=entry.memory_id,
            )

    # -------------------------------------------------------------------
    # Long-Term Memory: Retrieval
    # -------------------------------------------------------------------

    async def search_memories(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Search the long-term memory store for relevant knowledge.

        Parameters
        ----------
        query:
            Natural language search query.
        category:
            Optional category filter (e.g. "bugs", "skills").
        limit:
            Maximum results (defaults to config's max_retrieval_results).

        Returns
        -------
        list[MemoryEntry]
            Matching memories ranked by relevance.
        """
        if not self._mem_cfg.enabled:
            return []

        max_results = limit or self._mem_cfg.max_retrieval_results

        # Try vector search first
        if self._db is not None:
            return await self._search_vector(query, category=category, limit=max_results)

        # Fallback to JSON search (keyword-based)
        return await self._search_json_fallback(query, category=category, limit=max_results)

    async def _search_vector(
        self,
        query: str,
        *,
        category: str | None,
        limit: int,
    ) -> list[MemoryEntry]:
        """Search LanceDB tables using vector similarity."""
        try:
            query_embedding = await self._get_embedding(query)
            results: list[MemoryEntry] = []

            tables_to_search = (
                [category] if category and category in self._db.table_names()
                else self._db.table_names()
            )

            for table_name in tables_to_search:
                try:
                    table = self._db.open_table(table_name)
                    search_results = (
                        table.search(query_embedding)
                        .limit(limit)
                        .to_pandas()
                    )

                    for _, row in search_results.iterrows():
                        # LanceDB returns _distance (lower = more similar)
                        distance = row.get("_distance", 1.0)
                        # Convert distance to similarity (cosine distance → similarity)
                        similarity = 1.0 - min(distance, 1.0)

                        if similarity >= self._mem_cfg.relevance_threshold:
                            tags = row.get("tags", "[]")
                            if isinstance(tags, str):
                                try:
                                    tags = json.loads(tags)
                                except json.JSONDecodeError:
                                    tags = []

                            results.append(MemoryEntry(
                                memory_id=row["memory_id"],
                                category=row["category"],
                                title=row["title"],
                                content=row["content"],
                                tags=tags,
                                source_task=row.get("source_task", ""),
                                source_agent=row.get("source_agent", ""),
                                created_at=row.get("created_at", ""),
                                confidence=row.get("confidence", 0.8),
                            ))
                except Exception as exc:
                    await logger.warning(
                        "compressor.table_search_failed",
                        table=table_name,
                        error=str(exc),
                    )

            # Sort by confidence and return top N
            results.sort(key=lambda m: m.confidence, reverse=True)
            return results[:limit]
        except Exception as exc:
            await logger.error("compressor.vector_search_failed", error=str(exc))
            return await self._search_json_fallback(query, category=category, limit=limit)

    async def _search_json_fallback(
        self,
        query: str,
        *,
        category: str | None,
        limit: int,
    ) -> list[MemoryEntry]:
        """Keyword-based search over raw JSON files (fallback when no vector DB)."""
        results: list[MemoryEntry] = []
        raw_dir = self._store_path / "raw"

        if not raw_dir.exists():
            return []

        dirs_to_search = (
            [raw_dir / category] if category else list(raw_dir.iterdir())
        )

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for cat_dir in dirs_to_search:
            if not cat_dir.is_dir():
                continue
            for json_file in cat_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    entry = MemoryEntry.model_validate(data)

                    # Simple keyword relevance scoring
                    searchable = f"{entry.title} {entry.content} {' '.join(entry.tags)}".lower()
                    matched_terms = sum(1 for t in query_terms if t in searchable)

                    if matched_terms > 0:
                        results.append(entry)
                except Exception:
                    continue

        return results[:limit]

    # -------------------------------------------------------------------
    # Long-Term Memory: Context injection
    # -------------------------------------------------------------------

    async def get_boot_context(
        self,
        agent_name: str,
        agent_role: str,
        task_description: str = "",
    ) -> str:
        """Retrieve relevant long-term memories to inject into an agent's prompt.

        Called at agent boot time by the orchestrator. Returns a formatted
        string of relevant knowledge entries that should be appended to the
        agent's system prompt.

        Parameters
        ----------
        agent_name:
            The agent requesting context.
        agent_role:
            The agent's role (e.g. "backend", "frontend").
        task_description:
            Optional description of the current task for targeted retrieval.
        """
        if not self._mem_cfg.enabled:
            return ""

        query = f"{agent_role} {task_description}".strip()
        if not query:
            return ""

        memories = await self.search_memories(query)
        if not memories:
            return ""

        lines = [
            "\n--- TEAM KNOWLEDGE BASE (from past sessions) ---",
            "The following knowledge was extracted from previous team sessions.",
            "Use it to inform your decisions and avoid repeating past mistakes.\n",
        ]
        for i, mem in enumerate(memories, 1):
            lines.append(f"[{i}] [{mem.category.upper()}] {mem.title}")
            lines.append(f"    {mem.content}")
            if mem.tags:
                lines.append(f"    Tags: {', '.join(mem.tags)}")
            lines.append("")

        lines.append("--- END TEAM KNOWLEDGE BASE ---\n")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Embedding helper
    # -------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> list[float]:
        """Get a vector embedding for the given text via litellm."""
        from litellm import aembedding

        try:
            response = await aembedding(
                model=self._mem_cfg.embedding_model,
                input=[text],
            )
            return response.data[0]["embedding"]
        except Exception as exc:
            await logger.warning(
                "compressor.embedding_failed",
                error=str(exc),
                msg="Using zero vector fallback.",
            )
            # Return a zero vector as fallback (won't match anything well)
            return [0.0] * EMBEDDING_DIM  # gemini/text-embedding-004 dimension
