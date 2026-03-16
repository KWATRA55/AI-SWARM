"""CLI entry point — the ``swarm`` command.

Provides three sub-commands:

* ``swarm validate <config.yaml>`` — Parse and validate a swarm config
  file without running anything.
* ``swarm run <config.yaml>`` — Execute the full swarm orchestration.
* ``swarm teardown`` — Emergency cleanup of orphaned Docker containers
  and networks from previous sessions.

Usage::

    # Validate a config
    swarm validate swarm.yaml

    # Run the swarm
    swarm run swarm.yaml --dry-run

    # Clean up orphaned containers
    swarm teardown --force

Installation (via pyproject.toml)::

    [project.scripts]
    swarm = "swarm.cli:app"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="swarm",
    help="🐝 Multi-agent AI swarm orchestration engine.",
    add_completion=False,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo("swarm v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """🐝 Multi-agent AI swarm orchestration engine."""


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    config_path: Path = typer.Argument(
        ...,
        help="Path to the swarm YAML configuration file.",
        exists=True,
        readable=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full config details."),
) -> None:
    """✅ Validate a swarm configuration file without running anything."""
    from swarm.config.loader import load_config
    from swarm.infra.env import validate_api_keys

    try:
        config = load_config(config_path)
    except Exception as exc:
        typer.secho(f"❌ Validation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Display summary
    typer.secho(f"✅ Configuration valid: {config.name}", fg=typer.colors.GREEN, bold=True)
    typer.echo()

    # Agents table
    typer.secho("📋 Agents:", bold=True)
    for agent in config.agents:
        deps = f" (depends: {', '.join(agent.depends_on)})" if agent.depends_on else ""
        tools = ", ".join(agent.tools[:4])
        if len(agent.tools) > 4:
            tools += f" +{len(agent.tools) - 4} more"
        typer.echo(
            f"   • {agent.name} [{agent.role.value}]"
            f"  model={agent.model}  tools=[{tools}]{deps}"
        )
    typer.echo()

    # Infrastructure
    typer.secho("🏗️  Infrastructure:", bold=True)
    typer.echo(f"   • Workspace:      {config.workspace}")
    typer.echo(f"   • Event Bus:      {config.event_bus.redis_url}")
    typer.echo(f"   • LTM:           {'enabled' if config.memory.enabled else 'disabled'}")
    typer.echo(f"   • Verification:  {'enabled' if config.verification.enabled else 'disabled'}")
    typer.echo(f"   • Circuit Breaker: max {config.circuit_breaker.max_round_trips} round trips")
    typer.echo()

    # API key check
    models = [a.model for a in config.agents]
    key_status = validate_api_keys(models, config.api_keys or None)
    typer.secho("🔑 API Keys:", bold=True)
    all_ok = True
    for provider, has_key in key_status.items():
        icon = "✅" if has_key else "❌"
        if not has_key:
            all_ok = False
        typer.echo(f"   {icon} {provider}")

    if not all_ok:
        typer.echo()
        typer.secho(
            "⚠️  Some API keys are missing. Set them as environment variables "
            "or in a .env file in the workspace root.",
            fg=typer.colors.YELLOW,
        )

    if verbose:
        typer.echo()
        typer.secho("📄 Full config:", bold=True)
        import json
        typer.echo(config.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    config_path: Path = typer.Argument(
        ...,
        help="Path to the swarm YAML configuration file.",
        exists=True,
        readable=True,
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n",
        help="Validate and resolve the DAG without executing agents.",
    ),
    log_json: bool = typer.Option(
        True, "--json-logs/--console-logs",
        help="Emit JSON logs (default) or human-readable console logs.",
    ),
    log_level: str = typer.Option(
        "info", "--log-level", "-l",
        help="Log level: debug, info, warning, error.",
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file",
        help="Also write logs to this file.",
    ),
    timeout: float = typer.Option(
        0, "--timeout", "-t",
        help="Kill switch: auto-stop swarm after this many minutes (0 = unlimited).",
    ),
    dashboard: bool = typer.Option(
        False, "--dashboard", "-d",
        help="Launch the real-time dashboard UI at http://localhost:8080.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Start dashboard and wait for user chat — don't auto-run agents.",
    ),
    target: Optional[Path] = typer.Option(
        None, "--target",
        help="External project directory for agents to operate on (overrides workspace).",
    ),
) -> None:
    """🚀 Run the swarm orchestration engine."""
    from swarm.config.loader import load_config
    from swarm.config.models import LoggingConfig, LogLevel
    from swarm.infra.env import load_env
    from swarm.infra.logging import configure_logging

    # --- Load config ---
    try:
        config = load_config(config_path, workspace_override=target)
    except Exception as exc:
        typer.secho(f"❌ Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # --- Configure logging ---
    try:
        level = LogLevel(log_level.lower())
    except ValueError:
        level = LogLevel.INFO

    log_config = LoggingConfig(
        level=level,
        json_output=log_json,
        log_file=log_file,
    )
    configure_logging(log_config)

    # --- Load environment ---
    load_env(workspace=config.workspace)

    # --- Banner ---
    typer.secho("🐝 Swarm Orchestrator", fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.echo(f"   Project: {config.name}")
    typer.echo(f"   Agents:  {len(config.agents)}")
    typer.echo(f"   Mode:    {'DRY RUN' if dry_run else 'LIVE'}")
    if timeout > 0:
        typer.echo(f"   Timeout: {timeout} min (kill switch armed)")
    typer.echo()

    # --- Execute ---
    try:
        asyncio.run(_execute_swarm(
            config, dry_run=dry_run, timeout_minutes=timeout,
            enable_dashboard=dashboard,
            interactive=interactive,
        ))
    except KeyboardInterrupt:
        typer.secho("\n⚠️  Interrupted by user. Running teardown...", fg=typer.colors.YELLOW)
        asyncio.run(_emergency_teardown())
        raise typer.Exit(code=130)
    except Exception as exc:
        typer.secho(f"\n❌ Fatal error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


async def _execute_swarm(
    config: "SwarmConfig",  # type: ignore[name-defined]
    *,
    dry_run: bool,
    timeout_minutes: float = 0,
    enable_dashboard: bool = False,
    interactive: bool = False,
) -> None:
    """Async entrypoint that wires up and runs the orchestrator."""
    import structlog

    from swarm.core.orchestrator import SwarmOrchestrator

    logger = structlog.get_logger("swarm.cli")

    orchestrator = SwarmOrchestrator(config)

    # --- Dashboard ---
    dashboard_task = None
    if enable_dashboard:
        try:
            from swarm.dashboard.dashboard import SwarmDashboard

            dashboard = SwarmDashboard(
                orchestrator=orchestrator,
                redis_url=str(config.event_bus),
            )
            orchestrator.set_dashboard(dashboard)
            dashboard_task = await dashboard.start()

            import webbrowser
            webbrowser.open("http://localhost:8080")
        except Exception as exc:
            await logger.warning(
                "cli.dashboard_failed",
                error=str(exc),
                msg="Continuing without dashboard.",
            )

    if dry_run:
        await logger.info("cli.dry_run", msg="Resolving DAG and validating...")
        report = await orchestrator.run(dry_run=True)

        typer.secho("✅ Dry run complete", fg=typer.colors.GREEN, bold=True)
        typer.echo()

        # Show tier resolution
        if "tiers" in report:
            typer.secho("📊 DAG Resolution:", bold=True)
            for tier_info in report["tiers"]:
                tier_idx = tier_info.get("tier", "?")
                agents = tier_info.get("agents", [])
                agent_names = [a["name"] if isinstance(a, dict) else str(a) for a in agents]
                typer.echo(f"   Tier {tier_idx}: {', '.join(agent_names)}")
            typer.echo()

        # Show report summary
        for key, value in report.items():
            if key != "tiers":
                typer.echo(f"   {key}: {value}")
    else:
        if interactive:
            typer.secho(
                "🧑‍💻 Interactive mode — agents will NOT auto-start.",
                fg=typer.colors.BRIGHT_CYAN, bold=True,
            )
            typer.echo("   Use the dashboard chat to tell the manager what to do.")
            typer.echo()

        if timeout_minutes > 0:
            await logger.info(
                "cli.starting_with_timeout",
                config=config.name,
                timeout_minutes=timeout_minutes,
            )
        else:
            await logger.info("cli.starting", config=config.name)

        report = await orchestrator.run(
            timeout_minutes=timeout_minutes,
            interactive=interactive,
        )
        await logger.info("cli.completed", report=report)

        typer.secho("✅ Swarm execution complete", fg=typer.colors.GREEN, bold=True)


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


@app.command()
def teardown(
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force-remove containers even if they are still running.",
    ),
    network_cleanup: bool = typer.Option(
        True, "--network/--no-network",
        help="Also remove orphaned swarm Docker networks.",
    ),
) -> None:
    """🧹 Emergency cleanup of orphaned swarm Docker containers and networks."""
    asyncio.run(_emergency_teardown(force=force, cleanup_networks=network_cleanup))


async def _emergency_teardown(
    *,
    force: bool = True,
    cleanup_networks: bool = True,
) -> None:
    """Find and remove all Docker resources labelled with ``swarm.``."""
    try:
        import docker
        from docker.errors import DockerException
    except ImportError:
        typer.secho(
            "❌ Docker SDK not installed. Run: pip install docker",
            fg=typer.colors.RED,
        )
        return

    try:
        client = docker.from_env()
        client.ping()
    except DockerException as exc:
        typer.secho(f"❌ Cannot connect to Docker: {exc}", fg=typer.colors.RED)
        return

    # --- Containers ---
    containers = client.containers.list(
        all=True,
        filters={"label": "swarm.session"},
    )

    if containers:
        typer.secho(f"🔍 Found {len(containers)} swarm container(s):", bold=True)
        for container in containers:
            name = container.name
            status = container.status
            typer.echo(f"   • {name} ({status})")

            try:
                if force:
                    container.remove(force=True)
                else:
                    container.stop(timeout=10)
                    container.remove()
                typer.secho(f"     ✅ Removed", fg=typer.colors.GREEN)
            except Exception as exc:
                typer.secho(f"     ❌ Failed: {exc}", fg=typer.colors.RED)
    else:
        typer.echo("   No orphaned swarm containers found.")

    # --- Networks ---
    if cleanup_networks:
        networks = client.networks.list(filters={"label": "swarm.session"})
        if networks:
            typer.echo()
            typer.secho(f"🔍 Found {len(networks)} swarm network(s):", bold=True)
            for network in networks:
                typer.echo(f"   • {network.name}")
                try:
                    network.remove()
                    typer.secho(f"     ✅ Removed", fg=typer.colors.GREEN)
                except Exception as exc:
                    typer.secho(f"     ❌ Failed: {exc}", fg=typer.colors.RED)
        else:
            typer.echo("   No orphaned swarm networks found.")

    typer.echo()
    typer.secho("🧹 Teardown complete.", fg=typer.colors.GREEN, bold=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
