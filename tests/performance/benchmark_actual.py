import sys
from unittest.mock import MagicMock, AsyncMock

class MockModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @classmethod
    def model_validate(cls, args):
        return cls(**args)
    @classmethod
    def model_json_schema(cls):
        return {}

# Mock dependencies before any imports
for mod in ["structlog", "pydantic", "tenacity", "swarm.events.bus", "swarm.events.channels"]:
    sys.modules[mod] = MagicMock()

# Handle async logger
async_logger = MagicMock()
async_logger.info = AsyncMock()
async_logger.warning = AsyncMock()
async_logger.error = AsyncMock()
async_logger.debug = AsyncMock()
sys.modules["structlog"].get_logger.return_value = async_logger

sys.modules["pydantic"].BaseModel = MockModel
sys.modules["pydantic"].Field = MagicMock()

import asyncio
import time
import statistics
from pathlib import Path
import tempfile

# Import the module to be tested
import swarm.mcp.tools

# Monkeypatch the models
swarm.mcp.tools.ReadFileInput = MockModel
swarm.mcp.tools.WriteFileInput = MockModel
swarm.mcp.tools.ListDirectoryInput = MockModel

from swarm.mcp.tools import ToolExecutor

async def monitor_loop_lag(stop_event, lags):
    interval = 0.001  # 1ms
    while not stop_event.is_set():
        start = time.perf_counter()
        await asyncio.sleep(interval)
        end = time.perf_counter()
        lag = (end - start) - interval
        lags.append(max(0, lag))

async def run_benchmark():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        executor = ToolExecutor(workspace=workspace)

        # Create some files
        content = "some content " * 10000 # ~130KB
        for i in range(20):
            (workspace / f"file_{i}.txt").write_text(content)

        lags = []
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_loop_lag(stop_event, lags))

        print(f"Starting benchmark on ACTUAL code...")

        start_time = time.perf_counter()

        tasks = []
        for i in range(100):
            tasks.append(executor.execute("write_file", {"path": f"new_file_{i}.txt", "content": content, "create_dirs": True}))
            tasks.append(executor.execute("read_file", {"path": f"file_{i % 20}.txt", "start_line": None, "end_line": None, "structure_only": False}))
            tasks.append(executor.execute("list_directory", {"path": ".", "max_depth": 3, "recursive": False}))

        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        stop_event.set()
        await monitor_task

        duration = end_time - start_time
        avg_lag = statistics.mean(lags) if lags else 0
        max_lag = max(lags) if lags else 0

        print(f"Benchmark finished in {duration:.4f}s")
        print(f"Average loop lag: {avg_lag*1000:.4f}ms")
        print(f"Max loop lag: {max_lag*1000:.4f}ms")

        return avg_lag, max_lag

if __name__ == "__main__":
    asyncio.run(run_benchmark())
