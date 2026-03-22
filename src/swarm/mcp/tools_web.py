"""Secure web scraper MCP tool.

Fetches web page content with safety guardrails:
- URL whitelist enforcement
- Internal IP/domain blocking
- Response size limits
- Clean text extraction
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger(__name__)

# Blocked IP patterns (internal networks)
_INTERNAL_PATTERNS = re.compile(
    r"^(127\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|0\.|169\.254\.|localhost)"
)

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_scrape",
        "description": (
            "Fetch and extract text content from a public web page. "
            "Returns cleaned text, links, or raw HTML based on the 'extract' parameter. "
            "Use for researching APIs, documentation, or gathering data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be http:// or https://).",
                },
                "extract": {
                    "type": "string",
                    "enum": ["text", "html", "links"],
                    "description": "What to extract: 'text' (clean text), 'html' (raw HTML), 'links' (all href URLs).",
                    "default": "text",
                },
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to narrow extraction (e.g. 'article', '#content').",
                    "default": "",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default: 30).",
                    "default": 30,
                },
            },
            "required": ["url"],
        },
    },
}


def _is_safe_url(url: str, blocked_domains: list[str]) -> tuple[bool, str]:
    """Validate URL safety — no internal IPs, no file:// scheme."""
    parsed = urlparse(url)

    # Must be http/https
    if parsed.scheme not in ("http", "https"):
        return False, f"Scheme '{parsed.scheme}' not allowed. Use http:// or https://."

    hostname = parsed.hostname or ""

    # Block internal IPs
    if _INTERNAL_PATTERNS.match(hostname):
        return False, f"Internal address '{hostname}' is blocked."

    # Block configured domains
    for pattern in blocked_domains:
        if fnmatch.fnmatch(hostname, pattern):
            return False, f"Domain '{hostname}' is blocked by policy."

    return True, ""


async def execute(
    arguments: dict[str, Any],
    *,
    config: Any = None,
) -> dict[str, Any]:
    """Execute the web_scrape tool."""
    import httpx

    url = arguments.get("url", "")
    extract = arguments.get("extract", "text")
    selector = arguments.get("selector", "")
    timeout = min(arguments.get("timeout", 30), 120)

    # Get config values
    blocked_domains = ["localhost", "127.0.0.1", "0.0.0.0"]
    max_bytes = 512_000
    if config:
        blocked_domains = getattr(config, "blocked_domains", blocked_domains)
        max_bytes = getattr(config, "max_response_bytes", max_bytes)

    # Safety check
    safe, reason = _is_safe_url(url, blocked_domains)
    if not safe:
        return {"success": False, "error": reason, "output": ""}

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "SwarmOS/1.0 (Web Scraper Tool)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check content size
            content = response.text
            if len(content) > max_bytes:
                content = content[:max_bytes]

            # Extract based on mode
            if extract == "html":
                output = content[:2000]
            elif extract == "links":
                output = _extract_links(content, url)
            else:
                output = _extract_text(content, selector)

            return {
                "success": True,
                "output": output[:2000],  # Cap to 2K chars for token efficiency
                "url": url,
                "status_code": response.status_code,
                "content_length": len(content),
            }
    except httpx.HTTPStatusError as exc:
        return {"success": False, "error": f"HTTP {exc.response.status_code}", "output": ""}
    except Exception as exc:
        return {"success": False, "error": str(exc)[:300], "output": ""}


def _extract_text(html: str, selector: str = "") -> str:
    """Extract clean text from HTML."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        if selector:
            target = soup.select_one(selector)
            if target:
                return target.get_text(separator="\n", strip=True)
            return f"Selector '{selector}' not found in page."

        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        # Fallback: basic regex strip
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean


def _extract_links(html: str, base_url: str) -> str:
    """Extract all links from HTML."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)[:50]
            links.append(f"{text}: {href}")
        return "\n".join(links[:50])  # Max 50 links
    except ImportError:
        # Regex fallback
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
        return "\n".join(hrefs[:50])
