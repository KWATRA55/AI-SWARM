"""Outbound webhook MCP tool.

Sends HTTP requests to external services with safety guardrails:
- URL whitelist enforcement
- Internal IP blocking
- Payload size limits
- Timeout enforcement
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger(__name__)

_INTERNAL_PATTERNS = re.compile(
    r"^(127\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|0\.|169\.254\.|localhost)"
)

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "webhook",
        "description": (
            "Send an HTTP request to an external webhook or API endpoint. "
            "Use for sending notifications, triggering CI/CD pipelines, "
            "or integrating with external services."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The webhook URL to send to.",
                },
                "method": {
                    "type": "string",
                    "enum": ["POST", "PUT", "PATCH"],
                    "description": "HTTP method (default: POST).",
                    "default": "POST",
                },
                "payload": {
                    "type": "object",
                    "description": "JSON payload to send in the request body.",
                    "default": {},
                },
                "headers": {
                    "type": "object",
                    "description": "Additional HTTP headers (key-value pairs).",
                    "default": {},
                },
            },
            "required": ["url"],
        },
    },
}


def _is_safe_url(url: str, whitelist: list[str]) -> tuple[bool, str]:
    """Validate URL safety."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False, f"Scheme '{parsed.scheme}' not allowed."

    hostname = parsed.hostname or ""
    if _INTERNAL_PATTERNS.match(hostname):
        return False, f"Internal address '{hostname}' is blocked."

    # If whitelist is set, URL must match at least one pattern
    if whitelist:
        matched = any(fnmatch.fnmatch(url, pattern) for pattern in whitelist)
        if not matched:
            return False, f"URL not in whitelist. Allowed patterns: {whitelist}"

    return True, ""


async def execute(
    arguments: dict[str, Any],
    *,
    config: Any = None,
) -> dict[str, Any]:
    """Execute the webhook tool."""
    import json

    import httpx

    url = arguments.get("url", "")
    method = arguments.get("method", "POST").upper()
    payload = arguments.get("payload", {})
    headers = arguments.get("headers", {})

    # Config values
    whitelist: list[str] = []
    max_payload_bytes = 10_240
    timeout = 30

    if config:
        whitelist = getattr(config, "url_whitelist", [])
        max_payload_bytes = getattr(config, "max_payload_bytes", 10_240)
        timeout = getattr(config, "timeout_seconds", 30)

    # Safety check
    safe, reason = _is_safe_url(url, whitelist)
    if not safe:
        return {"success": False, "error": reason, "output": ""}

    # Payload size check
    payload_str = json.dumps(payload)
    if len(payload_str) > max_payload_bytes:
        return {
            "success": False,
            "error": f"Payload too large: {len(payload_str)} bytes (max: {max_payload_bytes}).",
            "output": "",
        }

    # Method validation
    if method not in ("POST", "PUT", "PATCH"):
        return {"success": False, "error": f"Method '{method}' not allowed.", "output": ""}

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "SwarmOS/1.0 (Webhook Tool)"},
        ) as client:
            response = await client.request(
                method=method,
                url=url,
                json=payload,
                headers=headers,
            )

            body = response.text[:500]  # Truncate response

            return {
                "success": response.is_success,
                "output": f"HTTP {response.status_code}\n{body}",
                "status_code": response.status_code,
                "response_body": body,
            }
    except Exception as exc:
        return {"success": False, "error": str(exc)[:300], "output": ""}
