"""MetricsCollector — tracks per-agent performance data for analytics.

Records timestamps, token usage, tool call results, response times,
and helper contributions for every agent throughout the swarm run.
After the run completes, produces a structured summary for the dashboard.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class ToolCallMetric(BaseModel):
    """A single tool call measurement."""
    agent: str
    tool: str
    success: bool
    duration_ms: float
    timestamp: float
    iteration: int = 0


class LLMCallMetric(BaseModel):
    """A single LLM call measurement."""
    agent: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    timestamp: float = 0.0
    iteration: int = 0
    had_tool_calls: bool = False


class HelperMetric(BaseModel):
    """Tracks a helper agent dispatch."""
    idle_agent: str
    helped_agent: str
    dispatch_time: float
    completion_time: float | None = None
    files_written: int = 0
    tool_calls: int = 0
    success: bool = False


class AgentSummary(BaseModel):
    """Aggregated metrics for one agent."""
    name: str
    model: str = ""
    status: str = "unknown"
    total_iterations: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    tool_success_rate: float = 0.0
    total_tokens: int = 0
    avg_response_ms: float = 0.0
    min_response_ms: float = 0.0
    max_response_ms: float = 0.0
    files_written: int = 0
    files_read: int = 0
    duration_seconds: float = 0.0
    was_helper: bool = False
    helper_files_written: int = 0


class RunAnalytics(BaseModel):
    """Complete analytics for a swarm run."""
    session_id: str = ""
    total_duration_seconds: float = 0.0
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_files_written: int = 0
    agents: list[AgentSummary] = Field(default_factory=list)
    helpers: list[HelperMetric] = Field(default_factory=list)
    manager_nudges: int = 0
    timeline: list[dict[str, Any]] = Field(default_factory=list)


class MetricsCollector:
    """Collects per-agent metrics throughout a swarm run.

    Usage::

        metrics = MetricsCollector()

        # Record events
        metrics.record_llm_call("architect", model="gemini-2.5-pro", ...)
        metrics.record_tool_call("backend-dev", tool="write_file", ...)

        # After run
        analytics = metrics.build_analytics()
    """

    def __init__(self) -> None:
        self._start_time = time.time()
        self._llm_calls: list[LLMCallMetric] = []
        self._tool_calls: list[ToolCallMetric] = []
        self._helpers: list[HelperMetric] = []
        self._agent_models: dict[str, str] = {}
        self._agent_start_times: dict[str, float] = {}
        self._agent_end_times: dict[str, float] = {}
        self._agent_statuses: dict[str, str] = {}
        self._files_written: dict[str, int] = {}  # agent → count
        self._files_read: dict[str, int] = {}
        self._nudge_count = 0
        self._timeline: list[dict[str, Any]] = []

    def record_agent_start(self, agent: str, model: str = "") -> None:
        """Record when an agent starts."""
        self._agent_start_times[agent] = time.time()
        if model:
            self._agent_models[agent] = model
        self._timeline.append({
            "time": time.time() - self._start_time,
            "agent": agent,
            "event": "started",
            "model": model,
        })

    def record_agent_end(self, agent: str, status: str = "completed") -> None:
        """Record when an agent finishes."""
        self._agent_end_times[agent] = time.time()
        self._agent_statuses[agent] = status
        self._timeline.append({
            "time": time.time() - self._start_time,
            "agent": agent,
            "event": "completed",
            "status": status,
        })

    def record_llm_call(
        self,
        agent: str,
        *,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: float = 0.0,
        iteration: int = 0,
        had_tool_calls: bool = False,
    ) -> None:
        """Record a single LLM API call."""
        self._llm_calls.append(LLMCallMetric(
            agent=agent,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            timestamp=time.time(),
            iteration=iteration,
            had_tool_calls=had_tool_calls,
        ))

    def record_tool_call(
        self,
        agent: str,
        *,
        tool: str,
        success: bool,
        duration_ms: float,
        iteration: int = 0,
    ) -> None:
        """Record a single tool call."""
        self._tool_calls.append(ToolCallMetric(
            agent=agent,
            tool=tool,
            success=success,
            duration_ms=duration_ms,
            timestamp=time.time(),
            iteration=iteration,
        ))

    def record_file_write(self, agent: str) -> None:
        """Record a file write."""
        self._files_written[agent] = self._files_written.get(agent, 0) + 1

    def record_file_read(self, agent: str) -> None:
        """Record a file read."""
        self._files_read[agent] = self._files_read.get(agent, 0) + 1

    def record_helper_dispatch(self, idle_agent: str, helped_agent: str) -> None:
        """Record a helper agent being dispatched."""
        self._helpers.append(HelperMetric(
            idle_agent=idle_agent,
            helped_agent=helped_agent,
            dispatch_time=time.time(),
        ))

    def record_helper_complete(self, idle_agent: str, success: bool, files: int = 0, tools: int = 0) -> None:
        """Mark a helper dispatch as complete."""
        for h in reversed(self._helpers):
            if h.idle_agent == idle_agent and h.completion_time is None:
                h.completion_time = time.time()
                h.success = success
                h.files_written = files
                h.tool_calls = tools
                break

    def record_nudge(self) -> None:
        """Record a manager nudge."""
        self._nudge_count += 1

    def build_analytics(self, session_id: str = "") -> RunAnalytics:
        """Build the final analytics summary."""
        end_time = time.time()
        all_agents = set()
        for c in self._llm_calls:
            all_agents.add(c.agent)
        for c in self._tool_calls:
            all_agents.add(c.agent)
        for a in self._agent_start_times:
            all_agents.add(a)

        agent_summaries: list[AgentSummary] = []
        total_tokens = 0
        total_files = 0

        for agent in sorted(all_agents):
            agent_llm = [c for c in self._llm_calls if c.agent == agent]
            agent_tools = [c for c in self._tool_calls if c.agent == agent]
            response_times = [c.duration_ms for c in agent_llm if c.duration_ms > 0]
            tool_successes = [c for c in agent_tools if c.success]
            agent_tokens = sum(c.total_tokens for c in agent_llm)
            files_w = self._files_written.get(agent, 0)
            files_r = self._files_read.get(agent, 0)

            start = self._agent_start_times.get(agent, self._start_time)
            end = self._agent_end_times.get(agent, end_time)

            helper_files = 0
            was_helper = False
            for h in self._helpers:
                if h.idle_agent == agent:
                    was_helper = True
                    helper_files += h.files_written

            summary = AgentSummary(
                name=agent,
                model=self._agent_models.get(agent, ""),
                status=self._agent_statuses.get(agent, "unknown"),
                total_iterations=max((c.iteration for c in agent_llm), default=0),
                total_llm_calls=len(agent_llm),
                total_tool_calls=len(agent_tools),
                tool_success_rate=(
                    len(tool_successes) / len(agent_tools) * 100
                    if agent_tools else 0.0
                ),
                total_tokens=agent_tokens,
                avg_response_ms=(
                    sum(response_times) / len(response_times)
                    if response_times else 0.0
                ),
                min_response_ms=min(response_times) if response_times else 0.0,
                max_response_ms=max(response_times) if response_times else 0.0,
                files_written=files_w,
                files_read=files_r,
                duration_seconds=round(end - start, 1),
                was_helper=was_helper,
                helper_files_written=helper_files,
            )
            agent_summaries.append(summary)
            total_tokens += agent_tokens
            total_files += files_w

        return RunAnalytics(
            session_id=session_id,
            total_duration_seconds=round(end_time - self._start_time, 1),
            total_tokens=total_tokens,
            total_llm_calls=len(self._llm_calls),
            total_tool_calls=len(self._tool_calls),
            total_files_written=total_files,
            agents=agent_summaries,
            helpers=self._helpers,
            manager_nudges=self._nudge_count,
            timeline=self._timeline,
        )

    def to_dict(self, session_id: str = "") -> dict[str, Any]:
        """Build analytics and return as dict (JSON-serializable)."""
        return self.build_analytics(session_id).model_dump()
