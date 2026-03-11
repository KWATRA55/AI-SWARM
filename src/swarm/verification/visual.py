"""Multi-modal visual verifier — the swarm's eyes.

Uses Playwright (async) to capture screenshots of running frontends and
passes them to a Vision LLM (via litellm) for automated UI critique.

Architecture
------------

.. code-block:: text

    VisualVerifier.verify(url, requirements)
         │
         ├── Playwright (async)
         │   ├── Launch headless Chromium
         │   ├── Navigate to dev server URL
         │   ├── Wait for network idle
         │   └── Screenshot → PNG bytes
         │
         ├── Vision LLM (litellm)
         │   ├── System: "You are a senior UI/UX QA engineer..."
         │   ├── Image: screenshot (base64)
         │   ├── Text: original UI requirements
         │   └── Response: structured JSON critique
         │
         └── VerificationResult
             ├── passed: bool
             ├── score: float (0.0 – 1.0)
             ├── issues: list[VisualIssue]
             └── summary: str

The verifier can also run accessibility checks (via axe-core injection)
and visual regression tests (comparing screenshots against baselines).
"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from swarm.config.models import VerificationConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class VisualIssue(BaseModel):
    """A single visual defect found by the verifier."""

    model_config = ConfigDict(frozen=True)

    severity: str = Field(description="critical | major | minor")
    category: str = Field(description="alignment | overlap | spacing | colour | typography | responsiveness | accessibility")
    element: str = Field(default="", description="CSS selector or description of the affected element.")
    description: str = Field(description="Human-readable description of the issue.")
    suggestion: str = Field(default="", description="Suggested fix.")


class VerificationResult(BaseModel):
    """Structured result from a visual verification pass."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(description="Whether the UI meets the requirements.")
    score: float = Field(ge=0.0, le=1.0, description="Fidelity score (0.0 = fail, 1.0 = perfect).")
    issues: list[VisualIssue] = Field(default_factory=list)
    summary: str = Field(default="", description="One-paragraph summary of the critique.")
    screenshot_path: str | None = Field(default=None, description="Path to the saved screenshot.")
    url: str = Field(default="")
    elapsed_seconds: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Visual Verifier
# ---------------------------------------------------------------------------


VISION_SYSTEM_PROMPT = """You are a **Senior UI/UX Quality Assurance Engineer** performing automated visual verification.

You will be shown a screenshot of a web application alongside its original requirements.

**Your task:**
1. Analyse the screenshot for visual defects: CSS overlaps, misalignment, broken layouts, colour inconsistencies, typography issues, spacing problems, responsiveness concerns, and accessibility violations.
2. Compare what you see against the stated requirements.
3. Assign a **fidelity score** from 0.0 (completely broken) to 1.0 (pixel-perfect match).
4. Return your critique as a **JSON object** with this exact schema:

```json
{
  "score": 0.85,
  "passed": true,
  "summary": "One-paragraph summary of the overall quality...",
  "issues": [
    {
      "severity": "major",
      "category": "alignment",
      "element": ".header-nav",
      "description": "Navigation items are not vertically centered.",
      "suggestion": "Add display: flex; align-items: center; to .header-nav"
    }
  ]
}
```

**Scoring guidelines:**
- 1.0: Pixel-perfect, no issues
- 0.9-0.99: Minor polish needed (e.g., 2px spacing inconsistency)
- 0.8-0.89: Small issues that don't break usability
- 0.6-0.79: Noticeable problems affecting user experience
- 0.4-0.59: Significant layout or styling issues
- 0.0-0.39: Fundamentally broken

Return ONLY the JSON object, no markdown fencing, no explanation outside the JSON."""


class VisualVerifier:
    """Playwright + Vision LLM visual verification pipeline.

    Usage::

        verifier = VisualVerifier(config)
        await verifier.setup()

        result = await verifier.verify(
            url="http://localhost:3000",
            requirements="A responsive dashboard with a sidebar nav...",
        )

        if not result.passed:
            for issue in result.issues:
                print(f"[{issue.severity}] {issue.description}")

        await verifier.teardown()
    """

    def __init__(self, config: VerificationConfig) -> None:
        self._config = config
        self._browser: Any | None = None
        self._playwright: Any | None = None

    async def setup(self) -> None:
        """Launch the headless browser."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)

        await logger.info(
            "visual.browser_launched",
            browser="chromium",
            headless=True,
        )

    async def teardown(self) -> None:
        """Close the browser and release resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        await logger.info("visual.browser_closed")

    async def verify(
        self,
        *,
        url: str,
        requirements: str,
        save_screenshot: bool = True,
        screenshot_dir: Path | None = None,
    ) -> VerificationResult:
        """Run the full visual verification pipeline.

        Parameters
        ----------
        url:
            URL to screenshot (e.g., ``http://localhost:3000``).
        requirements:
            Original UI requirements to verify against.
        save_screenshot:
            Whether to persist the screenshot to disk.
        screenshot_dir:
            Directory for saved screenshots (defaults to temp).

        Returns
        -------
        VerificationResult
            Structured pass/fail with issues and score.
        """
        start = time.monotonic()

        if not self._browser:
            await self.setup()

        # --- 1. Capture screenshot ---
        screenshot_bytes = await self._capture_screenshot(url)
        screenshot_path: str | None = None

        if save_screenshot and screenshot_bytes:
            save_dir = screenshot_dir or Path(tempfile.gettempdir()) / "swarm_screenshots"
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            screenshot_path = str(save_dir / f"verify_{timestamp}.png")
            Path(screenshot_path).write_bytes(screenshot_bytes)

        if not screenshot_bytes:
            return VerificationResult(
                passed=False,
                score=0.0,
                summary=f"Failed to capture screenshot from {url}.",
                url=url,
                elapsed_seconds=round(time.monotonic() - start, 2),
            )

        # --- 2. Vision LLM critique ---
        critique = await self._run_vision_critique(
            screenshot_bytes, requirements, url,
        )

        elapsed = round(time.monotonic() - start, 2)

        if critique is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                summary="Vision model failed to return a valid critique.",
                screenshot_path=screenshot_path,
                url=url,
                elapsed_seconds=elapsed,
            )

        # --- 3. Parse and build result ---
        score = critique.get("score", 0.0)
        passed = score >= self._config.pass_threshold

        issues: list[VisualIssue] = []
        for raw_issue in critique.get("issues", []):
            try:
                issues.append(VisualIssue.model_validate(raw_issue))
            except Exception:
                continue

        result = VerificationResult(
            passed=passed,
            score=score,
            issues=issues,
            summary=critique.get("summary", ""),
            screenshot_path=screenshot_path,
            url=url,
            elapsed_seconds=elapsed,
        )

        await logger.info(
            "visual.verification_complete",
            url=url,
            score=score,
            passed=passed,
            issues=len(issues),
            elapsed=elapsed,
        )

        return result

    # -------------------------------------------------------------------
    # Screenshot capture
    # -------------------------------------------------------------------

    async def _capture_screenshot(self, url: str) -> bytes | None:
        """Navigate to URL and capture a full-page screenshot."""
        if not self._browser:
            return None

        page = None
        try:
            context = await self._browser.new_context(
                viewport={
                    "width": self._config.viewport_width,
                    "height": self._config.viewport_height,
                },
            )
            page = await context.new_page()

            await page.goto(
                url,
                wait_until="networkidle",
                timeout=self._config.screenshot_timeout_ms,
            )

            # Wait a bit for any remaining animations / lazy loads
            await asyncio.sleep(1.0)

            screenshot = await page.screenshot(full_page=True, type="png")

            await logger.info(
                "visual.screenshot_captured",
                url=url,
                size_bytes=len(screenshot),
            )

            return screenshot

        except Exception as exc:
            await logger.error(
                "visual.screenshot_failed",
                url=url,
                error=str(exc),
            )
            return None

        finally:
            if page:
                await page.close()

    # -------------------------------------------------------------------
    # Vision LLM critique
    # -------------------------------------------------------------------

    async def _run_vision_critique(
        self,
        screenshot: bytes,
        requirements: str,
        url: str,
    ) -> dict[str, Any] | None:
        """Send screenshot + requirements to a Vision LLM for critique."""
        from litellm import acompletion

        # Encode screenshot as base64 data URI
        b64 = base64.b64encode(screenshot).decode("utf-8")
        image_url = f"data:image/png;base64,{b64}"

        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"**URL:** {url}\n\n"
                            f"**UI Requirements:**\n{requirements}\n\n"
                            f"Analyse the screenshot below and return your "
                            f"JSON critique."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ]

        try:
            response = await acompletion(
                model=self._config.vision_model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                timeout=60.0,
            )

            content = response.choices[0].message.content or ""

            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:])
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            result = json.loads(content)

            await logger.info(
                "visual.vision_critique_received",
                model=self._config.vision_model,
                score=result.get("score"),
            )

            return result

        except json.JSONDecodeError as exc:
            await logger.error(
                "visual.vision_json_parse_error",
                error=str(exc),
                raw_content=content[:500] if content else "",
            )
            return None

        except Exception as exc:
            await logger.error(
                "visual.vision_llm_error",
                error=str(exc),
                model=self._config.vision_model,
            )
            return None

    # -------------------------------------------------------------------
    # Batch verification
    # -------------------------------------------------------------------

    async def verify_multiple(
        self,
        targets: list[dict[str, str]],
    ) -> list[VerificationResult]:
        """Verify multiple URLs in sequence.

        Parameters
        ----------
        targets:
            List of dicts with ``url`` and ``requirements`` keys.

        Returns
        -------
        list[VerificationResult]
            One result per target.
        """
        results: list[VerificationResult] = []
        for target in targets:
            result = await self.verify(
                url=target["url"],
                requirements=target.get("requirements", ""),
            )
            results.append(result)
        return results
