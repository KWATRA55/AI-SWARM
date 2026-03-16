"""YAML configuration loader with environment variable interpolation.

Usage::

    from swarm.config.loader import load_config

    # From a file path
    cfg = load_config("swarm.yaml")

    # With explicit workspace override
    cfg = load_config("swarm.yaml", workspace_override="/abs/path/to/project")

The loader performs three passes:

1. **Read & interpolate** — resolve ``${ENV_VAR}`` and ``${ENV_VAR:-default}``
   placeholders against ``os.environ``.
2. **Parse & validate** — deserialise the YAML into a ``SwarmConfig`` Pydantic
   model, triggering all validators (unique names, DAG cycle check, etc.).
3. **Post-process** — inject the resolved shared workspace path into every
   agent's ``SandboxConfig.workspace_mount`` so all containers share the same
   host directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from swarm.config.models import SwarmConfig

# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

# Matches ${VAR} and ${VAR:-default_value}
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _interpolate_env(raw: str) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` with env values.

    Raises
    ------
    ValueError
        If a referenced variable is not set and no default is provided.
    """

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group("name")
        default = match.group("default")
        value = os.environ.get(var_name)
        if value is not None:
            return value
        if default is not None:
            return default
        msg = (
            f"Environment variable '${{{var_name}}}' is referenced in the "
            "config but is not set and has no default."
        )
        raise ValueError(msg)

    return _ENV_PATTERN.sub(_replace, raw)


def _interpolate_recursive(data: Any) -> Any:
    """Walk an arbitrary nested structure and interpolate all string values."""
    if isinstance(data, str):
        return _interpolate_env(data)
    if isinstance(data, dict):
        return {k: _interpolate_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate_recursive(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when config loading or validation fails."""

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def load_config(
    path: str | Path,
    *,
    workspace_override: str | Path | None = None,
) -> SwarmConfig:
    """Load, validate, and post-process a swarm YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML config file.
    workspace_override:
        If provided, overrides the ``workspace`` field in the config.
        Useful for CLI ``--workspace`` flags.

    Returns
    -------
    SwarmConfig
        Fully validated and post-processed configuration.

    Raises
    ------
    ConfigError
        On file I/O errors, YAML parse errors, env var resolution failures,
        or Pydantic validation errors.
    """
    config_path = Path(path).resolve()

    # --- 1. Read raw YAML ---
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file: {exc}") from exc

    # --- 2. Parse YAML first (before interpolation, to skip comments) ---
    try:
        raw_data: dict[str, Any] = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")

    # --- 3. Interpolate environment variables on parsed values ---
    try:
        raw_data = _interpolate_recursive(raw_data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    # --- 4. Apply workspace override ---
    if workspace_override is not None:
        raw_data["workspace"] = str(Path(workspace_override).resolve())

    # --- 5. Validate via Pydantic ---
    try:
        config = SwarmConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigError(
            f"Config validation failed with {exc.error_count()} error(s):\n{exc}",
            errors=[e for e in exc.errors()],
        ) from exc

    # --- 6. Post-process: inject shared workspace into every sandbox ---
    config = _inject_workspace_mounts(config)

    # --- 7. Resolve template variables in system prompts ---
    config = _resolve_template_vars(config)

    return config


def validate_config(path: str | Path) -> list[dict[str, Any]]:
    """Validate a config file and return a list of errors (empty = valid).

    Unlike ``load_config``, this never raises — it returns structured errors
    suitable for CLI display.
    """
    try:
        load_config(path)
        return []
    except ConfigError as exc:
        return exc.errors if exc.errors else [{"msg": str(exc)}]


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _inject_workspace_mounts(config: SwarmConfig) -> SwarmConfig:
    """Set ``workspace_mount`` on every agent's sandbox to the shared workspace.

    Because ``SwarmConfig`` and its children are frozen Pydantic models, we
    rebuild the tree using ``model_copy(update=...)``.
    """
    workspace = config.workspace

    updated_agents = []
    for agent in config.agents:
        if agent.sandbox.workspace_mount is None:
            updated_sandbox = agent.sandbox.model_copy(
                update={"workspace_mount": workspace},
            )
            updated_agent = agent.model_copy(update={"sandbox": updated_sandbox})
            updated_agents.append(updated_agent)
        else:
            updated_agents.append(agent)

    if updated_agents != list(config.agents):
        config = config.model_copy(update={"agents": updated_agents})

    return config


def _resolve_template_vars(config: SwarmConfig) -> SwarmConfig:
    """Replace ``{{variable}}`` placeholders in agent system prompts.

    Supported variables:
    - ``{{workspace}}``     — absolute workspace path
    - ``{{project_name}}``  — basename of the workspace directory
    - ``{{swarm_name}}``    — name of the swarm from the config
    """
    replacements = {
        "{{workspace}}": str(config.workspace),
        "{{project_name}}": Path(config.workspace).name,
        "{{swarm_name}}": config.name,
    }

    updated_agents = []
    changed = False
    for agent in config.agents:
        prompt = agent.system_prompt
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
        if prompt != agent.system_prompt:
            updated_agents.append(agent.model_copy(update={"system_prompt": prompt}))
            changed = True
        else:
            updated_agents.append(agent)

    if changed:
        config = config.model_copy(update={"agents": updated_agents})

    return config
