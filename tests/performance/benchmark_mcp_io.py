import asyncio
import time
import statistics
from pathlib import Path
import tempfile

# Simplified ToolExecutor that mimics the I/O behavior of the real one
class MockToolExecutor:
    def __init__(self, workspace: Path):
        self._workspace = workspace

    async def execute(self, tool_name: str, arguments: dict):
        if tool_name == "write_file":
            return await self._direct_write(self._workspace / arguments["path"], arguments["content"])
        elif tool_name == "read_file":
            return await self._exec_read_file(self._workspace / arguments["path"])
        elif tool_name == "list_directory":
            return await self._exec_list_directory(self._workspace / arguments["path"])

    async def _direct_write(self, target: Path, content: str) -> bool:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return True

    async def _exec_read_file(self, target: Path) -> str:
        if not target.exists(): return ""
        return target.read_text(encoding="utf-8")

    async def _exec_list_directory(self, target: Path) -> list:
        if not target.exists(): return []
        entries = []
        def _walk(tp):
            for item in tp.iterdir():
                entries.append(str(item))
                if item.is_dir(): _walk(item)
        _walk(target)
        return entries

async def monitor_loop_lag(stop_event, lags):
    interval = 0.001  # 1ms for higher precision in this synthetic benchmark
    while not stop_event.is_set():
        start = time.perf_counter()
        await asyncio.sleep(interval)
        end = time.perf_counter()
        lag = (end - start) - interval
        lags.append(max(0, lag))

async def run_benchmark():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        executor = MockToolExecutor(workspace=workspace)

        # Create some files
        content = "some content " * 10000 # ~130KB
        for i in range(20):
            (workspace / f"file_{i}.txt").write_text(content)

        lags = []
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_loop_lag(stop_event, lags))

        print(f"Starting benchmark (synthetic)...")

        start_time = time.perf_counter()

        tasks = []
        for i in range(100):
            tasks.append(executor.execute("write_file", {"path": f"new_file_{i}.txt", "content": content}))
            tasks.append(executor.execute("read_file", {"path": f"file_{i % 20}.txt"}))
            tasks.append(executor.execute("list_directory", {"path": "."}))

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
