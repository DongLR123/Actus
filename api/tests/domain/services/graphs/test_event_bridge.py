"""Tests for GraphEventBridge."""

import pytest
from unittest.mock import AsyncMock
from app.domain.models.event import MessageEvent, DoneEvent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestGraphEventBridge:
    async def test_yields_events_from_graph_result(self):
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        msg_event = MessageEvent(role="assistant", message="hello")
        done_event = DoneEvent()

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "events": [msg_event, done_event],
            "flow_status": "completed",
        }

        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(mock_graph, {"message": "test"}):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], MessageEvent)
        assert isinstance(events[1], DoneEvent)

    async def test_returns_final_state(self):
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "events": [],
            "flow_status": "completed",
            "plan": None,
        }

        bridge = GraphEventBridge()
        async for _ in bridge.run(mock_graph, {}):
            pass

        assert bridge.final_state["flow_status"] == "completed"

    async def test_empty_events(self):
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"events": []}

        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(mock_graph, {}):
            events.append(event)

        assert events == []
