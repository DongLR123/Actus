import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.external.task.redis_stream_task import RedisStreamTask


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure clean registry before/after each test."""
    RedisStreamTask._task_registry.clear()
    yield
    RedisStreamTask._task_registry.clear()


@pytest.mark.anyio
async def test_destroy_does_not_raise_on_concurrent_registry_mutation():
    """destroy() must not raise RuntimeError when cancel() mutates _task_registry."""
    mock_runner_1 = MagicMock()
    mock_runner_1.destroy = AsyncMock()
    mock_runner_2 = MagicMock()
    mock_runner_2.destroy = AsyncMock()

    task_1 = RedisStreamTask(task_runner=mock_runner_1)
    task_1._execution_task = asyncio.create_task(asyncio.sleep(999))

    task_2 = RedisStreamTask(task_runner=mock_runner_2)
    task_2._execution_task = asyncio.create_task(asyncio.sleep(999))

    assert len(RedisStreamTask._task_registry) == 2

    # Should NOT raise RuntimeError: dictionary changed size during iteration
    await RedisStreamTask.destroy()

    assert len(RedisStreamTask._task_registry) == 0
    task_1._execution_task.cancel()
    task_2._execution_task.cancel()
