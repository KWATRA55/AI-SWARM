"""Agent plugin registry — dynamic role-to-class mapping.

Provides a simple ``@register_agent`` decorator that maps YAML role names
(e.g. ``"frontend"``, ``"backend"``, ``"database"``) to specific agent
classes.  The orchestrator uses this registry to instantiate the correct
agent class for each configured role.

Architecture
------------

.. code-block:: text

    YAML config
    ┌──────────────────────────────────┐
    │ agents:                          │
    │   - name: backend-agent          │
    │     role: backend        ────────┼──► AgentRegistry.get("backend")
    │   - name: frontend-agent        │         │
    │     role: frontend       ────────┼──► AgentRegistry.get("frontend")
    │   - name: db-agent               │         │
    │     role: database       ────────┼──► AgentRegistry.get("database")
    └──────────────────────────────────┘         │
                                                ▼
                                         WorkerAgent (or subclass)

Usage::

    from swarm.agents.registry import register_agent, AgentRegistry

    # Register a specialised agent class for a role
    @register_agent("frontend")
    class FrontendAgent(WorkerAgent):
        '''Specialised frontend agent with React-specific tooling.'''
        ...

    # The orchestrator can then resolve roles to classes
    agent_cls = AgentRegistry.get("frontend")
    agent = agent_cls(config=..., state=..., tool_executor=...)

If no specialised class is registered for a role, the registry falls back
to the base ``WorkerAgent`` class.
"""

from __future__ import annotations

from typing import Any, Type

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """Central registry mapping role names to agent classes.

    This is a class-level singleton — all registrations are global.

    Usage::

        # Register
        AgentRegistry.register("backend", BackendAgent)

        # Or use the decorator
        @register_agent("backend")
        class BackendAgent(WorkerAgent): ...

        # Resolve
        cls = AgentRegistry.get("backend")  # → BackendAgent
        cls = AgentRegistry.get("unknown")  # → WorkerAgent (default)

        # List all
        AgentRegistry.list_registered()  # → {"backend": BackendAgent, ...}
    """

    _registry: dict[str, Type[Any]] = {}
    _default: Type[Any] | None = None

    @classmethod
    def register(cls, role: str, agent_class: Type[Any]) -> None:
        """Register an agent class for a role.

        Parameters
        ----------
        role:
            The role name (matches ``AgentConfig.role`` / ``AgentRole``).
        agent_class:
            The agent class to instantiate for this role.
        """
        normalised = role.lower().strip()
        cls._registry[normalised] = agent_class
        logger.info(
            "agent_registry.registered",
            role=normalised,
            cls=agent_class.__name__,
        )

    @classmethod
    def set_default(cls, agent_class: Type[Any]) -> None:
        """Set the default agent class for unregistered roles.

        If not set, falls back to ``WorkerAgent``.
        """
        cls._default = agent_class

    @classmethod
    def get(cls, role: str) -> Type[Any]:
        """Resolve a role name to an agent class.

        Returns the registered class, or the default (``WorkerAgent``)
        if no specific class is registered for this role.

        Parameters
        ----------
        role:
            The role to look up.

        Returns
        -------
        Type[Any]
            The agent class (a callable that produces an agent instance).
        """
        normalised = role.lower().strip()
        agent_class = cls._registry.get(normalised)

        if agent_class is not None:
            return agent_class

        # Fallback to default
        if cls._default is not None:
            return cls._default

        # Import WorkerAgent lazily to avoid circular imports
        from swarm.agents.worker import WorkerAgent

        return WorkerAgent

    @classmethod
    def list_registered(cls) -> dict[str, str]:
        """List all registered role → class name mappings."""
        return {
            role: agent_cls.__name__
            for role, agent_cls in cls._registry.items()
        }

    @classmethod
    def is_registered(cls, role: str) -> bool:
        """Check if a specific role has a registered agent class."""
        return role.lower().strip() in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._registry.clear()
        cls._default = None


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def register_agent(role: str):
    """Decorator to register an agent class for a role.

    Usage::

        @register_agent("frontend")
        class FrontendAgent(WorkerAgent):
            '''Agent specialised for frontend development.'''

            async def execute(self, **kwargs):
                # Custom pre-processing for frontend tasks
                ...
                return await super().execute(**kwargs)

    Parameters
    ----------
    role:
        The role name (e.g. "backend", "frontend", "database", "qa").
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        AgentRegistry.register(role, cls)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Built-in role registrations
# ---------------------------------------------------------------------------

# These are no-ops that register the base WorkerAgent for well-known roles.
# Users can override them by decorating their own subclasses with the same
# role name — the last registration wins.


def _register_defaults() -> None:
    """Register default agent classes for well-known roles.

    This is called on import so the registry always has a baseline.
    Specialised agent subclasses can override these registrations.
    """
    from swarm.agents.worker import WorkerAgent

    for role in ("backend", "frontend", "qa", "devops", "database", "architect", "custom"):
        if not AgentRegistry.is_registered(role):
            AgentRegistry.register(role, WorkerAgent)


# Register defaults on import
_register_defaults()
