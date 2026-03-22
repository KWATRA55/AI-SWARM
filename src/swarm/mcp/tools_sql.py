"""Read-only SQL query MCP tool.

Executes SQL queries against configured databases with safety guardrails:
- DDL/DML blocking (no DROP, DELETE, ALTER, INSERT, UPDATE, TRUNCATE)
- Row limit enforcement
- Query timeout
- Support for SQLite (via aiosqlite) and PostgreSQL (via asyncpg)
"""

from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Blocked SQL keywords (case-insensitive)
_BLOCKED_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|ALTER|INSERT|UPDATE|TRUNCATE|CREATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": (
            "Execute a read-only SQL query against a configured database. "
            "Returns results as JSON rows. DDL/DML statements are blocked. "
            "Use for data analysis, schema inspection, or report generation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL SELECT query to execute.",
                },
                "database": {
                    "type": "string",
                    "description": "Database connection alias (from config). Default: 'default'.",
                    "default": "default",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to return (default: 100).",
                    "default": 100,
                },
            },
            "required": ["query"],
        },
    },
}


def _validate_query(query: str, read_only: bool = True) -> tuple[bool, str]:
    """Validate that a query is safe to execute."""
    stripped = query.strip().rstrip(";")

    if not stripped:
        return False, "Empty query."

    if read_only and _BLOCKED_KEYWORDS.search(stripped):
        match = _BLOCKED_KEYWORDS.search(stripped)
        keyword = match.group(0) if match else "unknown"
        return False, f"Query contains blocked keyword: '{keyword}'. Only SELECT queries are allowed."

    # Must start with SELECT, EXPLAIN, SHOW, DESCRIBE, or PRAGMA
    first_word = stripped.split()[0].upper()
    allowed_starts = {"SELECT", "EXPLAIN", "SHOW", "DESCRIBE", "PRAGMA", "WITH"}
    if read_only and first_word not in allowed_starts:
        return False, f"Query must start with one of: {', '.join(allowed_starts)}."

    return True, ""


async def execute(
    arguments: dict[str, Any],
    *,
    config: Any = None,
) -> dict[str, Any]:
    """Execute the sql_query tool."""
    query = arguments.get("query", "")
    database = arguments.get("database", "default")
    max_rows = min(arguments.get("max_rows", 100), 1000)

    read_only = True
    connections: dict[str, str] = {}
    timeout = 30

    if config:
        read_only = getattr(config, "read_only", True)
        connections = getattr(config, "connections", {})
        timeout = getattr(config, "query_timeout_seconds", 30)

    # Validate query
    valid, reason = _validate_query(query, read_only)
    if not valid:
        return {"success": False, "error": reason, "output": ""}

    # Resolve connection string
    conn_str = connections.get(database, "")
    if not conn_str:
        available = list(connections.keys()) or ["none configured"]
        return {
            "success": False,
            "error": f"Database '{database}' not found. Available: {', '.join(available)}",
            "output": "",
        }

    # Route to appropriate driver
    if conn_str.endswith(".db") or conn_str.startswith("sqlite"):
        return await _execute_sqlite(conn_str, query, max_rows, timeout)
    elif conn_str.startswith("postgresql") or conn_str.startswith("postgres"):
        return await _execute_postgres(conn_str, query, max_rows, timeout)
    else:
        return {"success": False, "error": f"Unsupported database type: {conn_str[:20]}...", "output": ""}


async def _execute_sqlite(
    conn_str: str, query: str, max_rows: int, timeout: int,
) -> dict[str, Any]:
    """Execute query against SQLite."""
    import asyncio

    try:
        import aiosqlite
    except ImportError:
        return {"success": False, "error": "aiosqlite not installed. Run: pip install aiosqlite", "output": ""}

    # Strip sqlite:/// prefix if present
    db_path = conn_str.replace("sqlite:///", "").replace("sqlite://", "")

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await asyncio.wait_for(
                db.execute(query),
                timeout=timeout,
            )
            rows = await cursor.fetchmany(max_rows)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            results = [dict(zip(columns, row)) for row in rows]
            total = len(results)

            return {
                "success": True,
                "output": _format_results(columns, results),
                "columns": columns,
                "row_count": total,
                "truncated": total >= max_rows,
            }
    except asyncio.TimeoutError:
        return {"success": False, "error": f"Query timed out after {timeout}s", "output": ""}
    except Exception as exc:
        return {"success": False, "error": str(exc)[:300], "output": ""}


async def _execute_postgres(
    conn_str: str, query: str, max_rows: int, timeout: int,
) -> dict[str, Any]:
    """Execute query against PostgreSQL."""
    try:
        import asyncpg
    except ImportError:
        return {"success": False, "error": "asyncpg not installed. Run: pip install asyncpg", "output": ""}

    try:
        conn = await asyncpg.connect(conn_str, timeout=timeout)
        try:
            # Set read-only transaction
            await conn.execute("SET TRANSACTION READ ONLY")
            rows = await conn.fetch(query, timeout=timeout)
            if rows:
                columns = list(rows[0].keys())
                results = [dict(r) for r in rows[:max_rows]]
            else:
                columns = []
                results = []

            return {
                "success": True,
                "output": _format_results(columns, results),
                "columns": columns,
                "row_count": len(results),
                "truncated": len(results) >= max_rows,
            }
        finally:
            await conn.close()
    except Exception as exc:
        return {"success": False, "error": str(exc)[:300], "output": ""}


def _format_results(columns: list[str], results: list[dict]) -> str:
    """Format results as a readable table string."""
    if not results:
        return "No rows returned."

    # Build simple text table
    lines = [" | ".join(columns)]
    lines.append("-" * len(lines[0]))
    for row in results[:100]:  # Hard cap
        values = [str(row.get(c, ""))[:50] for c in columns]
        lines.append(" | ".join(values))

    return "\n".join(lines)
