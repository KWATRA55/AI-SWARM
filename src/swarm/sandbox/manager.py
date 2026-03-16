"""Docker sandbox manager — container lifecycle for agent isolation.

Manages the full lifecycle of ephemeral Docker containers that serve as
isolated execution sandboxes for swarm agents.  Each container:

* Mounts the **shared host workspace** so agents can read/write the codebase.
* Exposes the MCP server port (and any additional ports from config).
* Has resource limits (CPU, memory) enforced by Docker.
* Runs startup commands (dependency installation, etc.) after creation.
* Gets an isolated network for inter-container communication.

Architecture
------------

.. code-block:: text

    Host
    ├── workspace/               ← shared project directory
    │   └── (codebase files)
    │
    ├── SandboxManager
    │   ├── create_sandbox(agent_config) ──► Docker container
    │   │   ├── Mounts workspace → /workspace (in container)
    │   │   ├── Exposes MCP port (SSE/STDIO)
    │   │   ├── Sets resource limits
    │   │   └── Runs startup_commands
    │   │
    │   ├── destroy_sandbox(sandbox_id) ──► Stop & remove
    │   ├── teardown_all() ──► Clean shutdown of all containers
    │   └── health_check() ──► Periodic liveness probe
    │
    └── Docker Network (per session)
        ├── backend-agent container
        ├── frontend-agent container
        └── qa-agent container

MCP-to-Docker Bridge
---------------------
The MCP server runs as a **host-side** process.  Communication with the
containerised agent depends on the configured transport:

* **SSE transport** (recommended): The MCP server binds to a port on the
  host.  The container exposes a corresponding port so the agent process
  inside can reach the MCP server via ``http://host.docker.internal:<port>``.
  This survives connection drops and allows multiplexed tool calls.

* **STDIO transport**: The host runs ``docker exec -i <container> <cmd>``
  and pipes JSON-RPC messages over stdin/stdout.  Simpler but limited to
  sequential tool calls.

In both cases, the ``SandboxManager`` prepares the container's network and
port mapping so the MCP layer (Phase 5) can connect seamlessly.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from swarm.config.models import AgentConfig, MCPTransport, SwarmConfig

logger = structlog.get_logger(__name__)

# Dedicated executor for Docker SDK calls — keeps them off the async event loop
_docker_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="docker")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SandboxInfo(BaseModel):
    """Runtime information about a provisioned sandbox container."""

    model_config = ConfigDict(frozen=True)

    sandbox_id: str = Field(description="Unique sandbox identifier (Docker container ID prefix).")
    container_id: str = Field(description="Full Docker container ID.")
    container_name: str = Field(description="Docker container name.")
    agent_name: str
    image: str
    status: str = Field(default="created", description="Container status: created|running|stopped|failed.")
    host_mcp_port: int | None = Field(default=None, description="Host-mapped port for MCP SSE server.")
    host_extra_ports: dict[str, int] = Field(
        default_factory=dict,
        description="Map of container_port → host_port for extra services.",
    )
    workspace_path: str = Field(description="Container-internal path to the mounted workspace.")
    network_name: str = Field(default="", description="Docker network this container is attached to.")


# ---------------------------------------------------------------------------
# Port allocator
# ---------------------------------------------------------------------------


class PortAllocator:
    """Thread-safe dynamic port allocator that avoids collisions."""

    def __init__(self, base_port: int = 18_000, max_port: int = 19_000) -> None:
        self._next_port = base_port
        self._max_port = max_port
        self._allocated: set[int] = set()
        self._lock = asyncio.Lock()

    async def allocate(self, preferred: int | None = None) -> int:
        """Allocate a host port, optionally preferring a specific one."""
        async with self._lock:
            if preferred and preferred not in self._allocated and preferred <= self._max_port:
                self._allocated.add(preferred)
                return preferred

            while self._next_port in self._allocated:
                self._next_port += 1
                if self._next_port > self._max_port:
                    raise RuntimeError(
                        f"Port pool exhausted ({self._max_port - 18_000} ports allocated)."
                    )

            port = self._next_port
            self._allocated.add(port)
            self._next_port += 1
            return port

    async def release(self, port: int) -> None:
        """Release a previously allocated port."""
        async with self._lock:
            self._allocated.discard(port)


# ---------------------------------------------------------------------------
# Sandbox Manager
# ---------------------------------------------------------------------------


class SandboxManager:
    """Manages Docker container lifecycles for agent sandboxes.

    Usage::

        manager = SandboxManager(config)
        await manager.initialize()

        # Provision a sandbox for an agent
        sandbox = await manager.create_sandbox(agent_config)
        print(f"Container running: {sandbox.container_name}")

        # Execute a command inside the sandbox
        exit_code, output = await manager.exec_in_sandbox(
            sandbox.sandbox_id, "python --version"
        )

        # Destroy when done
        await manager.destroy_sandbox(sandbox.sandbox_id)

        # Or tear down everything
        await manager.teardown_all()
    """

    # Path inside the container where the shared workspace is mounted
    CONTAINER_WORKSPACE_PATH = "/workspace"

    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._client: Any | None = None  # docker.DockerClient
        self._sandboxes: dict[str, SandboxInfo] = {}
        self._port_allocator = PortAllocator()
        self._network_name = f"swarm-{uuid.uuid4().hex[:8]}"
        self._network: Any | None = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------
    # Initialisation / teardown
    # -------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to Docker daemon and create the swarm network."""
        import docker
        from docker.errors import DockerException

        try:
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                _docker_executor, docker.from_env,
            )
            await loop.run_in_executor(_docker_executor, self._client.ping)
            version_info = await loop.run_in_executor(
                _docker_executor, self._client.version,
            )
            await logger.info(
                "sandbox.docker_connected",
                version=version_info.get("Version", "unknown"),
            )
        except DockerException as exc:
            raise RuntimeError(
                f"Cannot connect to Docker daemon. Is Docker running? Error: {exc}"
            ) from exc

        # Create an isolated bridge network for this swarm session
        try:
            self._network = await loop.run_in_executor(
                _docker_executor,
                functools.partial(
                    self._client.networks.create,
                    self._network_name,
                    driver="bridge",
                    labels={"swarm.session": self._config.name},
                ),
            )
            await logger.info(
                "sandbox.network_created",
                network=self._network_name,
            )
        except Exception as exc:
            await logger.warning(
                "sandbox.network_creation_failed",
                error=str(exc),
                msg="Falling back to default bridge network.",
            )
            self._network = None

        # Clean up orphan containers from a previous crashed session
        await self._cleanup_orphans()

    async def _cleanup_orphans(self) -> None:
        """Remove Docker containers left behind by a previous crashed session.

        Queries all containers labelled ``swarm.agent`` and force-removes any
        whose ``swarm.session`` label does not match the current session name.
        This runs once at boot before any new containers are created.
        """
        if self._client is None:
            return

        loop = asyncio.get_event_loop()
        try:
            all_swarm_containers = await loop.run_in_executor(
                _docker_executor,
                functools.partial(
                    self._client.containers.list,
                    filters={"label": "swarm.agent"},
                    all=True,
                ),
            )
        except Exception as exc:
            await logger.warning(
                "sandbox.orphan_scan_failed", error=str(exc),
            )
            return

        current_session = self._config.name
        orphans_removed = 0

        for container in all_swarm_containers:
            session_label = container.labels.get("swarm.session", "")
            if session_label != current_session:
                try:
                    await loop.run_in_executor(
                        _docker_executor,
                        functools.partial(container.remove, force=True),
                    )
                    orphans_removed += 1
                    await logger.info(
                        "sandbox.orphan_removed",
                        container=container.name,
                        stale_session=session_label,
                    )
                except Exception as exc:
                    await logger.warning(
                        "sandbox.orphan_remove_failed",
                        container=container.name,
                        error=str(exc),
                    )

        if orphans_removed > 0:
            await logger.info(
                "sandbox.orphan_cleanup_complete",
                removed=orphans_removed,
            )

    # -------------------------------------------------------------------
    # Container lifecycle
    # -------------------------------------------------------------------

    async def create_sandbox(self, agent_config: AgentConfig) -> SandboxInfo:
        """Provision a Docker container for an agent.

        Steps:
        1. Pull or verify the Docker image exists.
        2. Allocate host ports for MCP and extra services.
        3. Create the container with workspace volume, ports, resource limits.
        4. Attach to the swarm network.
        5. Start the container.
        6. Run startup commands.

        Parameters
        ----------
        agent_config:
            The agent's configuration (includes sandbox, mcp settings).

        Returns
        -------
        SandboxInfo
            Runtime info about the provisioned sandbox.
        """
        if self._client is None:
            raise RuntimeError("SandboxManager not initialized. Call initialize() first.")

        sandbox_cfg = agent_config.sandbox
        mcp_cfg = agent_config.mcp
        container_name = f"swarm-{agent_config.name}-{uuid.uuid4().hex[:6]}"

        await logger.info(
            "sandbox.creating",
            agent=agent_config.name,
            image=sandbox_cfg.image,
            container=container_name,
        )

        # --- 1. Ensure image exists ---
        await self._ensure_image(sandbox_cfg.image, sandbox_cfg.dockerfile)

        # --- 2. Allocate ports ---
        host_mcp_port: int | None = None
        port_bindings: dict[str, int] = {}

        if mcp_cfg.transport == MCPTransport.SSE:
            host_mcp_port = await self._port_allocator.allocate()
            port_bindings[f"{mcp_cfg.port}/tcp"] = host_mcp_port

        # Extra ports (e.g. "3000:3000" → container 3000 → host dynamic)
        host_extra_ports: dict[str, int] = {}
        for port_spec in sandbox_cfg.extra_ports:
            parts = port_spec.split(":")
            if len(parts) == 2:
                host_p, container_p = parts
                try:
                    allocated = await self._port_allocator.allocate(preferred=int(host_p))
                    port_bindings[f"{container_p}/tcp"] = allocated
                    host_extra_ports[container_p] = allocated
                except RuntimeError:
                    allocated = await self._port_allocator.allocate()
                    port_bindings[f"{container_p}/tcp"] = allocated
                    host_extra_ports[container_p] = allocated
            elif len(parts) == 1:
                allocated = await self._port_allocator.allocate()
                port_bindings[f"{parts[0]}/tcp"] = allocated
                host_extra_ports[parts[0]] = allocated

        # --- 3. Prepare volume mounts ---
        workspace_host = str(sandbox_cfg.workspace_mount or self._config.workspace)
        volumes = {
            workspace_host: {
                "bind": self.CONTAINER_WORKSPACE_PATH,
                "mode": "rw",  # Read-write — serialised by the Event Bus file lock
            },
        }

        # --- 4. Prepare environment ---
        environment = dict(sandbox_cfg.env_vars)
        environment.update({
            "SWARM_AGENT_NAME": agent_config.name,
            "SWARM_AGENT_ROLE": agent_config.role.value,
            "SWARM_WORKSPACE": self.CONTAINER_WORKSPACE_PATH,
            "SWARM_MCP_TRANSPORT": mcp_cfg.transport.value,
            "SWARM_MCP_PORT": str(mcp_cfg.port),
        })

        # --- 5. Parse resource limits ---
        mem_limit = sandbox_cfg.resources.memory
        nano_cpus = int(sandbox_cfg.resources.cpu_count * 1e9)

        # --- 6. Create the container ---
        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                _docker_executor,
                functools.partial(
                    self._client.containers.create,
                    image=sandbox_cfg.image,
                    name=container_name,
                    command="sleep infinity",
                    working_dir=self.CONTAINER_WORKSPACE_PATH,
                    volumes=volumes,
                    ports=port_bindings,
                    environment=environment,
                    mem_limit=mem_limit,
                    nano_cpus=nano_cpus,
                    labels={
                        "swarm.agent": agent_config.name,
                        "swarm.role": agent_config.role.value,
                        "swarm.session": self._config.name,
                    },
                    detach=True,
                    tty=True,
                    stdin_open=True,
                ),
            )
        except Exception as exc:
            await logger.error(
                "sandbox.create_failed",
                agent=agent_config.name,
                error=str(exc),
            )
            raise RuntimeError(f"Failed to create container for {agent_config.name}: {exc}") from exc

        # --- 7. Attach to swarm network ---
        if self._network is not None:
            try:
                await loop.run_in_executor(
                    _docker_executor,
                    functools.partial(self._network.connect, container),
                )
            except Exception as exc:
                await logger.warning(
                    "sandbox.network_attach_failed",
                    container=container_name,
                    error=str(exc),
                )

        # --- 8. Start the container ---
        try:
            await loop.run_in_executor(_docker_executor, container.start)
        except Exception as exc:
            await loop.run_in_executor(
                _docker_executor,
                functools.partial(container.remove, force=True),
            )
            raise RuntimeError(f"Failed to start container for {agent_config.name}: {exc}") from exc

        # --- 9. Build sandbox info ---
        sandbox_id = container.short_id
        info = SandboxInfo(
            sandbox_id=sandbox_id,
            container_id=container.id,
            container_name=container_name,
            agent_name=agent_config.name,
            image=sandbox_cfg.image,
            status="running",
            host_mcp_port=host_mcp_port,
            host_extra_ports=host_extra_ports,
            workspace_path=self.CONTAINER_WORKSPACE_PATH,
            network_name=self._network_name,
        )

        async with self._lock:
            self._sandboxes[sandbox_id] = info

        await logger.info(
            "sandbox.created",
            agent=agent_config.name,
            sandbox_id=sandbox_id,
            mcp_port=host_mcp_port,
            extra_ports=host_extra_ports,
            workspace=f"{workspace_host} → {self.CONTAINER_WORKSPACE_PATH}",
        )

        # --- 10. Run startup commands ---
        await self._run_startup_commands(sandbox_id, sandbox_cfg.startup_commands)

        return info

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """Stop and remove a sandbox container, releasing all resources."""
        async with self._lock:
            info = self._sandboxes.get(sandbox_id)
            if info is None:
                await logger.warning("sandbox.destroy_unknown", sandbox_id=sandbox_id)
                return

        if self._client is None:
            return

        loop = asyncio.get_event_loop()
        try:
            container = await loop.run_in_executor(
                _docker_executor,
                functools.partial(self._client.containers.get, info.container_id),
            )
            await loop.run_in_executor(
                _docker_executor,
                functools.partial(container.stop, timeout=10),
            )
            await loop.run_in_executor(
                _docker_executor,
                functools.partial(container.remove, force=True),
            )
            await logger.info(
                "sandbox.destroyed",
                sandbox_id=sandbox_id,
                agent=info.agent_name,
            )
        except Exception as exc:
            await logger.warning(
                "sandbox.destroy_failed",
                sandbox_id=sandbox_id,
                error=str(exc),
            )

        # Release allocated ports
        if info.host_mcp_port:
            await self._port_allocator.release(info.host_mcp_port)
        for port in info.host_extra_ports.values():
            await self._port_allocator.release(port)

        async with self._lock:
            self._sandboxes.pop(sandbox_id, None)

    async def teardown_all(self) -> None:
        """Destroy all sandboxes and clean up the swarm network."""
        async with self._lock:
            sandbox_ids = list(self._sandboxes.keys())

        await logger.info(
            "sandbox.teardown_all",
            count=len(sandbox_ids),
        )

        for sid in sandbox_ids:
            await self.destroy_sandbox(sid)

        # Remove the swarm network
        if self._network is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(_docker_executor, self._network.remove)
                await logger.info("sandbox.network_removed", network=self._network_name)
            except Exception as exc:
                await logger.warning(
                    "sandbox.network_remove_failed",
                    error=str(exc),
                )
            self._network = None

    # -------------------------------------------------------------------
    # Command execution
    # -------------------------------------------------------------------

    async def exec_in_sandbox(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout: float = 60.0,
        workdir: str | None = None,
    ) -> tuple[int, str]:
        """Execute a command inside a sandbox container.

        Parameters
        ----------
        sandbox_id:
            The sandbox to execute in.
        command:
            Shell command to run.
        timeout:
            Max seconds to wait for completion.
        workdir:
            Working directory override (defaults to /workspace).

        Returns
        -------
        (exit_code, output)
            The command's exit code and combined stdout/stderr.
        """
        async with self._lock:
            info = self._sandboxes.get(sandbox_id)
        if info is None:
            raise KeyError(f"Unknown sandbox: {sandbox_id}")
        if self._client is None:
            raise RuntimeError("Docker client not initialized.")

        loop = asyncio.get_event_loop()
        container = await loop.run_in_executor(
            _docker_executor,
            functools.partial(self._client.containers.get, info.container_id),
        )

        # Run in the dedicated Docker executor to avoid blocking the event loop
        def _exec() -> tuple[int, str]:
            exec_result = container.exec_run(
                cmd=["sh", "-c", command],
                workdir=workdir or self.CONTAINER_WORKSPACE_PATH,
                demux=False,
            )
            output = exec_result.output.decode("utf-8", errors="replace") if exec_result.output else ""
            return exec_result.exit_code, output

        try:
            exit_code, output = await asyncio.wait_for(
                loop.run_in_executor(_docker_executor, _exec),
                timeout=timeout,
            )
            return exit_code, output
        except asyncio.TimeoutError:
            await logger.warning(
                "sandbox.exec_timeout",
                sandbox_id=sandbox_id,
                command=command[:100],
                timeout=timeout,
            )
            return -1, f"Command timed out after {timeout}s"

    # -------------------------------------------------------------------
    # Health checks
    # -------------------------------------------------------------------

    async def health_check(self) -> dict[str, str]:
        """Check the status of all sandboxes. Returns sandbox_id → status."""
        results: dict[str, str] = {}

        async with self._lock:
            sandbox_items = list(self._sandboxes.items())

        loop = asyncio.get_event_loop()
        for sandbox_id, info in sandbox_items:
            try:
                if self._client:
                    container = await loop.run_in_executor(
                        _docker_executor,
                        functools.partial(
                            self._client.containers.get, info.container_id,
                        ),
                    )
                    status = container.status  # "running", "exited", etc.
                    results[sandbox_id] = status

                    # Update stored info if status changed
                    if status != info.status:
                        async with self._lock:
                            self._sandboxes[sandbox_id] = info.model_copy(
                                update={"status": status},
                            )
                else:
                    results[sandbox_id] = "unknown"
            except Exception:
                results[sandbox_id] = "unreachable"

        return results

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get info about a specific sandbox."""
        async with self._lock:
            return self._sandboxes.get(sandbox_id)

    async def get_sandbox_by_agent(self, agent_name: str) -> SandboxInfo | None:
        """Look up a sandbox by agent name."""
        async with self._lock:
            for info in self._sandboxes.values():
                if info.agent_name == agent_name:
                    return info
        return None

    async def list_sandboxes(self) -> list[SandboxInfo]:
        """List all active sandboxes."""
        async with self._lock:
            return list(self._sandboxes.values())

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    async def _ensure_image(self, image: str, dockerfile: str | None) -> None:
        """Pull or build the Docker image if not present locally."""
        if self._client is None:
            return

        if dockerfile:
            # Build from Dockerfile
            await logger.info("sandbox.building_image", dockerfile=dockerfile)
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    _docker_executor,
                    functools.partial(
                        self._client.images.build,
                        path=".",
                        dockerfile=dockerfile,
                        tag=image,
                    ),
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to build image from {dockerfile}: {exc}") from exc
        else:
            # Pull from registry
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    _docker_executor,
                    functools.partial(self._client.images.get, image),
                )
                await logger.info("sandbox.image_found_locally", image=image)
            except Exception:
                await logger.info("sandbox.pulling_image", image=image)
                try:
                    await loop.run_in_executor(
                        _docker_executor,
                        functools.partial(self._client.images.pull, image),
                    )
                    await logger.info("sandbox.image_pulled", image=image)
                except Exception as exc:
                    raise RuntimeError(f"Failed to pull image '{image}': {exc}") from exc

    async def _run_startup_commands(
        self,
        sandbox_id: str,
        commands: list[str],
    ) -> None:
        """Run startup commands inside a newly created container."""
        for cmd in commands:
            await logger.info(
                "sandbox.startup_cmd",
                sandbox_id=sandbox_id,
                command=cmd[:100],
            )
            exit_code, output = await self.exec_in_sandbox(
                sandbox_id, cmd, timeout=300.0,
            )
            if exit_code != 0:
                await logger.warning(
                    "sandbox.startup_cmd_failed",
                    sandbox_id=sandbox_id,
                    command=cmd[:100],
                    exit_code=exit_code,
                    output=output[:500],
                )
            else:
                await logger.info(
                    "sandbox.startup_cmd_ok",
                    sandbox_id=sandbox_id,
                    command=cmd[:50],
                )
