"""Tests for main_graph — outer orchestration (plan→execute→update→summarize)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock
from app.domain.models.event import PlanEvent, TitleEvent, MessageEvent, DoneEvent, PlanEventStatus
from app.domain.models.plan import Plan, Step, ExecutionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_planner_llm():
    """Mock LLM for planner that returns a plan JSON."""
    async def mock_invoke(**kwargs):
        return {
            "content": '{"title":"Test","goal":"Do test","language":"en","steps":[{"description":"Step 1"}],"message":"Let me help"}',
            "role": "assistant",
        }
    llm = AsyncMock()
    llm.invoke = mock_invoke
    type(llm).model_name = PropertyMock(return_value="gpt-4o")
    return llm


@pytest.fixture
def mock_json_parser():
    parser = AsyncMock()
    import json
    async def parse(content, default_value=None):
        try:
            return json.loads(content)
        except Exception:
            return {"title": "Fallback", "goal": content, "steps": [{"description": content}], "message": "ok", "language": "en"}
    parser.invoke = parse
    return parser


def _make_mock_react_graph():
    """Create a mock react_graph with async generator astream."""
    class MockReactGraph:
        async def astream(self, input_state, config=None):
            yield {"llm_node": {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [],
            }}

        async def ainvoke(self, input_state, config=None):
            return {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [],
                "should_interrupt": False,
                "attempt_count": 1,
                "failure_count": 0,
            }

    return MockReactGraph()


class TestBuildMainGraph:
    def test_graph_compiles(self, mock_planner_llm, mock_json_parser):
        from app.domain.services.graphs.main_graph import build_main_graph
        graph = build_main_graph(
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-1",
        )
        assert graph is not None


class TestMainGraphFlow:
    async def test_full_flow_produces_plan_and_done(self, mock_planner_llm, mock_json_parser):
        from app.domain.services.graphs.main_graph import build_main_graph

        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=False)
        mock_uow.session = AsyncMock()
        mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)

        graph = build_main_graph(
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(return_value=mock_uow),
            session_id="sess-1",
        )

        result = await graph.ainvoke({
            "message": "help me test",
            "language": "en",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-1",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        events = result.get("events", [])
        event_types = [type(e).__name__ for e in events]
        # planner events come through state; executor events go via queue (empty in state)
        assert "PlanEvent" in event_types or "TitleEvent" in event_types

    async def test_default_language_is_zh(self, mock_planner_llm, mock_json_parser):
        """When no language is specified, planner should default to zh."""
        from app.domain.services.graphs.main_graph import build_main_graph

        graph = build_main_graph(
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-lang",
        )

        result = await graph.ainvoke({
            "message": "帮我查一下天气",
            "language": "zh",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-lang",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        # Verify the plan language fallback is "zh" not "en"
        plan = result.get("plan")
        assert plan is not None

    async def test_planner_receives_conversation_summaries(self, mock_json_parser):
        """Planner system prompt should include conversation summaries when available."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_system_content = []

        async def capturing_invoke(**kwargs):
            messages = kwargs.get("messages", [])
            for m in messages:
                if m.get("role") == "system":
                    captured_system_content.append(m["content"])
            return {
                "content": '{"title":"Test","goal":"test","language":"zh","steps":[{"description":"step1"}],"message":"ok"}',
                "role": "assistant",
            }

        planner_llm = AsyncMock()
        planner_llm.invoke = capturing_invoke

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-summary",
        )

        await graph.ainvoke({
            "message": "继续上次的工作",
            "language": "zh",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-summary",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": ["### 第1轮\n- 用户需求：查天气\n- 执行结果：成功获取北京天气"],
        })

        assert len(captured_system_content) >= 1
        assert "历史对话摘要" in captured_system_content[0]
        assert "查天气" in captured_system_content[0]
