"""Secure environment variable loader for the swarm platform.

Loads API keys and sensitive configuration from environment variables
and ``.env`` files without hardcoding anything.  Keys are injected into
Docker sandbox containers via environment variables.

Security model
--------------

.. code-block:: text

    Priority (highest → lowest):
    1. Explicit environment variables (set in the shell)
    2. .env file in the workspace root
    3. SwarmConfig.api_keys (from YAML — stored as SecretStr)
    4. Default empty (operation fails with a clear error)

    Keys are NEVER:
    - Logged (SecretStr masks them)
    - Written to disk (except .env which is gitignored)
    - Passed as command-line arguments
    - Embedded in Docker images (injected at runtime via env vars)

Usage::

    from swarm.infra.env import load_env, get_api_key, inject_sandbox_env

    load_env(workspace=Path("/my/project"))

    key = get_api_key("openai")
    # → reads OPENAI_API_KEY from env

    sandbox_env = inject_sandbox_env(agent_config)
    # → dict of env vars to pass to Docker container
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import SecretStr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider → env var mapping
# ---------------------------------------------------------------------------

_PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "azure": "AZURE_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}


# ---------------------------------------------------------------------------
# .env file loader
# ---------------------------------------------------------------------------


def load_env(workspace: Path | None = None) -> int:
    """Load environment variables from ``.env`` files.

    Searches for ``.env`` in:
    1. The workspace directory
    2. The current working directory
    3. The user's home directory

    Does NOT override existing environment variables.

    Returns
    -------
    int
        Number of new variables loaded.
    """
    loaded = 0
    search_paths: list[Path] = []

    if workspace:
        search_paths.append(workspace / ".env")
    search_paths.append(Path.cwd() / ".env")
    search_paths.append(Path.home() / ".env")

    for env_path in search_paths:
        if env_path.is_file():
            loaded += _parse_env_file(env_path)
            break  # Use the first .env found

    return loaded


def _parse_env_file(path: Path) -> int:
    """Parse a .env file and set variables that don't already exist."""
    loaded = 0

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Handle KEY=VALUE format
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            # Only set if not already in environment
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1

    except Exception as exc:
        # Don't fail if .env is unreadable
        logger.warning("env.parse_failed: path=%s error=%s", str(path), str(exc))

    if loaded > 0:
        logger.info("env.loaded: path=%s count=%d", str(path), loaded)

    return loaded


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def get_api_key(
    provider: str,
    config_keys: dict[str, SecretStr] | None = None,
) -> str | None:
    """Resolve an API key for a provider.

    Resolution order:
    1. Environment variable (e.g., ``OPENAI_API_KEY``)
    2. SwarmConfig.api_keys (if provided)

    Parameters
    ----------
    provider:
        Provider name (e.g., "openai", "anthropic", "gemini").
    config_keys:
        Optional dict from ``SwarmConfig.api_keys``.

    Returns
    -------
    str or None
        The API key, or None if not found.
    """
    normalised = provider.lower().strip()

    # 1. Try environment variable
    env_var = _PROVIDER_ENV_MAP.get(normalised)
    if env_var:
        value = os.environ.get(env_var)
        if value:
            return value

    # Also try the generic pattern: <PROVIDER>_API_KEY
    generic_var = f"{normalised.upper()}_API_KEY"
    value = os.environ.get(generic_var)
    if value:
        return value

    # 2. Try config keys
    if config_keys:
        secret = config_keys.get(normalised)
        if secret:
            return secret.get_secret_value()

    return None


def extract_provider(model_string: str) -> str:
    """Extract the provider name from a litellm model string.

    Examples::

        extract_provider("gpt-4o") → "openai"
        extract_provider("anthropic/claude-3-opus") → "anthropic"
        extract_provider("gemini/gemini-1.5-pro") → "gemini"
        extract_provider("ollama/qwen") → "ollama"
    """
    if "/" in model_string:
        return model_string.split("/")[0].lower()

    # Infer from model name prefix
    model_lower = model_string.lower()

    if model_lower.startswith(("gpt-", "o1", "o3", "dall-e", "text-", "chatgpt")):
        return "openai"
    elif model_lower.startswith("claude"):
        return "anthropic"
    elif model_lower.startswith("gemini"):
        return "gemini"
    elif model_lower.startswith("command"):
        return "cohere"
    elif model_lower.startswith("mistral"):
        return "mistral"

    return "openai"  # Default assumption


# ---------------------------------------------------------------------------
# Sandbox environment injection
# ---------------------------------------------------------------------------


def inject_sandbox_env(
    model_string: str,
    config_keys: dict[str, SecretStr] | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment variable dict to inject into a Docker sandbox.

    This is called by ``SandboxManager.create_sandbox()`` to securely
    pass API keys and config to the containerised agent.

    Parameters
    ----------
    model_string:
        The agent's litellm model string (to determine which key to inject).
    config_keys:
        Optional ``SwarmConfig.api_keys``.
    extra_env:
        Additional environment variables to include.

    Returns
    -------
    dict[str, str]
        Environment variables safe for Docker container injection.
    """
    env: dict[str, str] = {}

    # Add the provider's API key
    provider = extract_provider(model_string)
    api_key = get_api_key(provider, config_keys)

    if api_key:
        env_var = _PROVIDER_ENV_MAP.get(provider, f"{provider.upper()}_API_KEY")
        env[env_var] = api_key

    # For litellm model routing, also set the generic var
    if api_key:
        env["LITELLM_API_KEY"] = api_key

    # Merge extra env
    if extra_env:
        env.update(extra_env)

    return env


def validate_api_keys(
    models: list[str],
    config_keys: dict[str, SecretStr] | None = None,
) -> dict[str, bool]:
    """Check which providers have API keys available.

    Parameters
    ----------
    models:
        List of litellm model strings to check.
    config_keys:
        Optional config-level keys.

    Returns
    -------
    dict[str, bool]
        Provider → has_key mapping.
    """
    results: dict[str, bool] = {}
    for model in models:
        provider = extract_provider(model)
        key = get_api_key(provider, config_keys)
        results[provider] = key is not None
    return results
