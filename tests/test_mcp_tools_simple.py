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
import swarm.mcp.tools

# Simple functional test
async def test_functionality():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Monkeypatch models
        swarm.mcp.tools.ReadFileInput = MockModel
        swarm.mcp.tools.WriteFileInput = MockModel
        swarm.mcp.tools.ListDirectoryInput = MockModel

        executor = swarm.mcp.tools.ToolExecutor(workspace=workspace)

        # Test Write
        res = await executor.execute("write_file", {"path": "test.txt", "content": "hello world", "create_dirs": True})
        assert res.success, res.error
        assert (workspace / "test.txt").read_text() == "hello world"

        # Test Read
        res = await executor.execute("read_file", {"path": "test.txt", "start_line": None, "end_line": None, "structure_only": False})
        assert res.success, res.error
        assert res.output == "hello world"

        # Test List
        res = await executor.execute("list_directory", {"path": ".", "max_depth": 3, "recursive": False})
        assert res.success, res.error
        assert "[FILE] test.txt" in res.output

if __name__ == "__main__":
    asyncio.run(test_functionality())
    print("Functionality tests passed!")
