import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.anyio
async def test_destroy_calls_flow_close():
    """destroy() should call self._flow.close() if the flow has a close method."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._sandbox = None

    mock_flow = MagicMock()
    mock_flow.close = AsyncMock()
    runner._flow = mock_flow

    runner._cleanup_tools = AsyncMock()

    await runner.destroy()

    mock_flow.close.assert_awaited_once()


@pytest.mark.anyio
async def test_destroy_skips_close_when_flow_lacks_method():
    """destroy() should not fail if the flow does not have a close method."""
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._sandbox = None

    mock_flow = MagicMock(spec=[])  # no close attribute
    runner._flow = mock_flow

    runner._cleanup_tools = AsyncMock()

    await runner.destroy()  # should not raise
