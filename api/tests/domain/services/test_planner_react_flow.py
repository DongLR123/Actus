"""Tests for PlannerReActFlow — LangGraph-based implementation."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph.checkpoint.memory import MemorySaver

from app.domain.models.app_config import AgentConfig
from app.domain.models.event import DoneEvent, PlanEvent, WaitEvent
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step
from app.domain.services.flows.planner_react import PlannerReActFlow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_llm():
    """Mock BaseChatModel LLM."""
    llm = MagicMock(spec=BaseChatModel)
    # with_structured_output returns a runnable whose ainvoke returns a parsed model
    mock_structured = AsyncMock()
    llm.with_structured_output = MagicMock(return_value=mock_structured)
    # Default: return None (callers override per-test as needed)
    mock_structured.ainvoke = AsyncMock(return_value=None)
    type(llm).model_name = PropertyMock(return_value="gpt-4o")
    return llm


@pytest.fixture
def mock_uow():
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=False)
    uow.session = AsyncMock()
    uow.session.get_skill_graph_state = AsyncMock(return_value=None)
    uow.session.get_memory = AsyncMock(return_value=Memory())
    uow.session.get_summary = AsyncMock(return_value=[])
    uow.session.save_memory = AsyncMock()
    uow.session.save_summary = AsyncMock()
    return uow


def _make_flow(mock_llm, mock_uow, **overrides):
    """Helper to create a PlannerReActFlow with standard test params."""
    kwargs = dict(
        uow_factory=MagicMock(return_value=mock_uow),
        llm=mock_llm,
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="test-session",
        browser=AsyncMock(),
        sandbox=AsyncMock(),
        search_engine=AsyncMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        checkpointer=MemorySaver(),
    )
    kwargs.update(overrides)
    return PlannerReActFlow(**kwargs)


def test_planner_react_flow_constructs_successfully(mock_llm, mock_uow):
    """Flow can be constructed with all required parameters (graphs not built yet)."""
    flow = _make_flow(mock_llm, mock_uow)
    assert flow.done is True
    assert flow.plan is None
    # 延迟绑定：构造时不构建图
    assert flow._graphs_built is False


async def test_planner_react_flow_invoke_produces_events(mock_llm, mock_uow):
    """Flow.invoke() should yield events including DoneEvent."""
    flow = _make_flow(mock_llm, mock_uow)

    # Mock the graph to produce a DoneEvent without real LLM calls
    mock_bridge = MagicMock()
    mock_bridge.final_state = {
        "plan": Plan(title="Test", goal="test", language="en",
                     steps=[Step(description="s1")], message="ok"),
        "messages": [SystemMessage(content="sys"), HumanMessage(content="hi"),
                     AIMessage(content="done")],
        "original_request": "test",
        "should_interrupt": False,
        "flow_status": "completed",
    }

    async def mock_run(graph, input_state, config=None):
        from app.domain.models.event import TitleEvent, MessageEvent, PlanEvent, PlanEventStatus
        yield TitleEvent(title="Test")
        yield MessageEvent(role="assistant", message="ok")
        yield PlanEvent(
            plan=mock_bridge.final_state["plan"],
            status=PlanEventStatus.CREATED,
        )
        yield DoneEvent()

    mock_bridge.run = mock_run

    events = []
    with patch(
        "app.domain.services.flows.planner_react.GraphEventBridge",
        return_value=mock_bridge,
    ), patch.object(flow, "_ensure_graphs", new_callable=AsyncMock):
        async for event in flow.invoke(Message(message="help me test")):
            events.append(event)

    assert len(events) > 0
    assert any(isinstance(e, DoneEvent) for e in events)


async def test_planner_react_flow_produces_plan_event(mock_llm, mock_uow):
    """Flow should produce PlanEvent during execution."""
    flow = _make_flow(mock_llm, mock_uow)

    mock_bridge = MagicMock()
    plan = Plan(title="Test", goal="test", language="en",
                steps=[Step(description="s1")], message="ok")
    mock_bridge.final_state = {
        "plan": plan,
        "messages": [SystemMessage(content="sys"), AIMessage(content="done")],
        "original_request": "test",
        "should_interrupt": False,
        "flow_status": "completed",
    }

    async def mock_run(graph, input_state, config=None):
        from app.domain.models.event import PlanEvent, PlanEventStatus
        yield PlanEvent(plan=plan, status=PlanEventStatus.CREATED)
        yield DoneEvent()

    mock_bridge.run = mock_run

    events = []
    with patch(
        "app.domain.services.flows.planner_react.GraphEventBridge",
        return_value=mock_bridge,
    ), patch.object(flow, "_ensure_graphs", new_callable=AsyncMock):
        async for event in flow.invoke(Message(message="help me test")):
            events.append(event)

    plan_events = [e for e in events if isinstance(e, PlanEvent)]
    assert len(plan_events) >= 1


def test_planner_react_flow_set_skill_context(mock_llm, mock_uow):
    """set_skill_context should store the context."""
    flow = _make_flow(mock_llm, mock_uow)
    flow.set_skill_context("test context")
    assert flow._skill_context == "test context"


async def test_persist_after_graph_saves_memory_on_interrupt(
    mock_llm, mock_uow,
):
    """_persist_after_graph should save Memory when should_interrupt=True.
    Checkpointer handles state persistence — no InterruptState needed."""
    flow = _make_flow(mock_llm, mock_uow)

    plan = Plan(title="t", goal="g", language="zh", steps=[Step(description="s1")], message="m")
    step = plan.steps[0]
    msgs = [SystemMessage(content="sys"), HumanMessage(content="u")]
    final_state = {
        "plan": plan,
        "current_step": step,
        "messages": msgs,
        "original_request": "g",
        "should_interrupt": True,
        "flow_status": "executing",
    }

    await flow._persist_after_graph(final_state, summaries=[])

    # Memory should be saved
    mock_uow.session.save_memory.assert_called_once()
    # Flow status should be EXECUTING for resume
    from app.domain.services.flows.base import FlowStatus
    assert flow.status == FlowStatus.EXECUTING


async def test_ensure_graphs_builds_tools_lazily(mock_llm, mock_uow):
    """_ensure_graphs should build tools from MCP/A2A at call time, not __init__ time."""
    mock_mcp = MagicMock()
    # MCP returns tools only after initialization
    mock_mcp.get_tools = MagicMock(return_value=[
        {"function": {"name": "notion_search", "description": "Search Notion"}},
    ])
    mock_a2a = MagicMock()
    mock_a2a.manager = None  # A2A not initialized

    flow = _make_flow(mock_llm, mock_uow,
                      mcp_tool=mock_mcp, a2a_tool=mock_a2a)

    # Before _ensure_graphs: no graphs
    assert flow._graphs_built is False
    assert flow._react_graph is None

    # Call _ensure_graphs: should pick up MCP tools
    await flow._ensure_graphs()

    assert flow._graphs_built is True
    assert flow._react_graph is not None
    assert flow._main_graph is not None
    # Verify MCP tools were queried
    mock_mcp.get_tools.assert_called()


async def test_generator_early_close_still_persists(
    mock_llm, mock_uow,
):
    """Simulates WaitEvent early-return: closing the generator triggers finally persistence."""
    flow = _make_flow(mock_llm, mock_uow)

    plan = Plan(title="t", goal="g", language="zh", steps=[Step(description="s1")], message="m")
    wait_event = WaitEvent()

    # Patch GraphEventBridge to yield a WaitEvent and set should_interrupt in final_state
    mock_bridge_instance = MagicMock()
    mock_bridge_instance.final_state = {
        "plan": plan,
        "current_step": plan.steps[0],
        "messages": [SystemMessage(content="sys")],
        "original_request": "g",
        "should_interrupt": True,
        "flow_status": "executing",
    }

    async def mock_run(graph, input_state, config=None):
        yield wait_event

    mock_bridge_instance.run = mock_run

    with patch(
        "app.domain.services.flows.planner_react.GraphEventBridge",
        return_value=mock_bridge_instance,
    ), patch.object(flow, "_ensure_graphs", new_callable=AsyncMock):
        gen = flow.invoke(Message(message="test"))
        # Consume only the first event (WaitEvent), then close — simulating early return
        first_event = await gen.__anext__()
        assert isinstance(first_event, WaitEvent)
        await gen.aclose()  # Simulates the consumer abandoning the generator

    # Memory should be saved via finally block, even on early close
    mock_uow.session.save_memory.assert_called()
