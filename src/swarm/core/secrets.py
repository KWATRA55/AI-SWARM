"""Secret management & PII masking — prevents credential leakage.

Detects sensitive environment variables by naming convention and provides
a ``mask()`` function that replaces actual secret values with safe
redaction tokens in any text output.

Usage::

    from swarm.core.secrets import SecretManager

    sm = SecretManager()
    safe_text = sm.mask("My key is sk-1234abc")
    # → "My key is [REDACTED_SECRET:SOME_API_KEY]"
"""

from __future__ import annotations

import os
import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# Patterns that indicate a key name is a secret
_SECRET_NAME_PATTERNS = re.compile(
    r"(SECRET|TOKEN|KEY|PASSWORD|CREDENTIAL|API_KEY|APIKEY|PRIVATE|AUTH)",
    re.IGNORECASE,
)

# Values shorter than this are unlikely to be real secrets
_MIN_SECRET_LENGTH = 8


class SecretManager:
    """Detects and masks sensitive values from environment and .env files.

    Parameters
    ----------
    extra_secrets:
        Additional key-value pairs to treat as secrets (e.g., from config).
    env_file:
        Optional path to a ``.env`` file to scan for secrets.
    """

    def __init__(
        self,
        *,
        extra_secrets: dict[str, str] | None = None,
        env_file: str | None = None,
    ) -> None:
        self._secrets: dict[str, str] = {}  # name → value
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = []

        # Scan environment
        self._scan_env()

        # Scan .env file
        if env_file:
            self._scan_env_file(env_file)

        # Add extra secrets
        if extra_secrets:
            self._secrets.update(extra_secrets)

        # Compile regex patterns for fast masking
        self._compile_patterns()

    def _scan_env(self) -> None:
        """Scan os.environ for secret-looking variable names."""
        for key, value in os.environ.items():
            if _SECRET_NAME_PATTERNS.search(key) and len(value) >= _MIN_SECRET_LENGTH:
                self._secrets[key] = value

    def _scan_env_file(self, path: str) -> None:
        """Parse a .env file for additional secrets."""
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if _SECRET_NAME_PATTERNS.search(key) and len(value) >= _MIN_SECRET_LENGTH:
                        self._secrets[key] = value
        except (OSError, IOError):
            pass

    def _compile_patterns(self) -> None:
        """Build compiled regex patterns sorted by value length (longest first)."""
        # Sort by value length descending to avoid partial replacements
        items = sorted(self._secrets.items(), key=lambda kv: len(kv[1]), reverse=True)
        self._compiled_patterns = [
            (name, re.compile(re.escape(value)))
            for name, value in items
            if value  # skip empty values
        ]

    def mask(self, text: str) -> str:
        """Replace all known secret values with ``[REDACTED_SECRET:<KEY_NAME>]``."""
        for name, pattern in self._compiled_patterns:
            text = pattern.sub(f"[REDACTED_SECRET:{name}]", text)
        return text

    def mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively mask all string values in a dictionary."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.mask(value)
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.mask(v) if isinstance(v, str) else v for v in value
                ]
            else:
                result[key] = value
        return result

    @property
    def secret_names(self) -> list[str]:
        """Return the list of detected secret key names."""
        return list(self._secrets.keys())

    def __repr__(self) -> str:
        return f"SecretManager(detected={len(self._secrets)} secrets)"
