import pytest
from typing import Any

from app.domain.models.app_config import AgentConfig
from app.domain.models.tool_result import ToolResult
from app.domain.models.event import (
    ControlEvent,
    ControlScope,
    MessageEvent,
    StepEvent,
    ToolEvent,
    ToolEventStatus,
    WaitEvent,
)
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step
from app.domain.services.agents.react import ReActAgent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _DummySessionRepo:
    def __init__(self) -> None:
        self._memory = Memory()

    async def get_memory(self, _session_id: str, _agent_name: str) -> Memory:
        return self._memory

    async def save_memory(
        self, _session_id: str, _agent_name: str, memory: Memory
    ) -> None:
        self._memory = memory


class _DummyUoW:
    def __init__(self) -> None:
        self.session = _DummySessionRepo()

    async def __aenter__(self) -> "_DummyUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _DummyJsonParser:
    async def invoke(self, _payload: Any) -> Any:
        # 模拟异常场景：解析器返回空字符串，而非结构化字典
        return ""


class _DummyLLM:
    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": "unused"}


async def test_execute_step_degrades_when_parser_returns_non_dict() -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield MessageEvent(role="assistant", message="这是一段普通文本结果")

    # 仅测试 execute_step 的结构化降级逻辑
    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="测试步骤")])
    step = plan.steps[0]
    assert step is not None
    message = Message(message="请执行测试步骤")

    events = [event async for event in agent.execute_step(plan, step, message)]

    step_events = [event for event in events if isinstance(event, StepEvent)]
    message_events = [event for event in events if isinstance(event, MessageEvent)]

    assert len(step_events) == 2
    assert step_events[-1].step.status.value == "completed"
    assert step_events[-1].step.result == "这是一段普通文本结果"
    assert len(message_events) == 1
    assert message_events[0].message == "这是一段普通文本结果"


async def test_execute_step_message_ask_user_without_takeover_emits_wait_event() -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-wait",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请确认是否继续", "suggest_user_takeover": "none"},
            status=ToolEventStatus.CALLING,
        )
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请确认是否继续", "suggest_user_takeover": "none"},
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="测试步骤")])
    step = plan.steps[0]
    message = Message(message="请继续")
    events = [event async for event in agent.execute_step(plan, step, message)]

    assert any(isinstance(event, WaitEvent) for event in events)
    assert not any(isinstance(event, ControlEvent) for event in events)


async def test_execute_step_message_ask_user_without_takeover_field_emits_wait_event() -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-wait-default",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请确认是否继续"},
            status=ToolEventStatus.CALLING,
        )
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请确认是否继续"},
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="测试步骤")])
    step = plan.steps[0]
    message = Message(message="请继续")
    events = [event async for event in agent.execute_step(plan, step, message)]

    assert any(isinstance(event, WaitEvent) for event in events)
    assert not any(isinstance(event, ControlEvent) for event in events)


@pytest.mark.parametrize("scope", ["shell", "browser"])
async def test_execute_step_message_ask_user_takeover_emits_control_requested(scope: str) -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-control",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请接管", "suggest_user_takeover": scope},
            status=ToolEventStatus.CALLING,
        )
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请接管", "suggest_user_takeover": scope},
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="测试步骤")])
    step = plan.steps[0]
    message = Message(message="请继续")
    events = [event async for event in agent.execute_step(plan, step, message)]

    control_events = [event for event in events if isinstance(event, ControlEvent)]
    assert len(control_events) == 1
    assert control_events[0].action.value == "requested"
    assert control_events[0].scope == ControlScope(scope)
    assert control_events[0].source.value == "agent"
    assert not any(isinstance(event, WaitEvent) for event in events)


async def test_execute_step_message_ask_user_takeover_scope_with_spaces_emits_control_requested() -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-control-strip",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请接管", "suggest_user_takeover": " Shell "},
            status=ToolEventStatus.CALLING,
        )
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="message",
            function_name="message_ask_user",
            function_args={"text": "请接管", "suggest_user_takeover": " Shell "},
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="测试步骤")])
    step = plan.steps[0]
    message = Message(message="请继续")
    events = [event async for event in agent.execute_step(plan, step, message)]

    control_events = [event for event in events if isinstance(event, ControlEvent)]
    assert len(control_events) == 1
    assert control_events[0].scope == ControlScope.SHELL


def test_ask_user_soft_hint_before_attempt_threshold() -> None:
    """未达阈值时第一次调用返回 SOFT_HINT（success=True）"""
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-gating-blocked",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )
    agent._step_tool_attempt_rounds = 0
    agent._step_failed_tool_calls = 0
    agent._step_ask_user_soft_hint_count = 0

    result = agent._intercept_tool_call("message_ask_user", {"text": "需要确认"})
    assert result is not None
    assert result.success is True
    assert result.data["code"] == "ASK_USER_SOFT_HINT"
    assert agent._step_ask_user_soft_hint_count == 1


def test_ask_user_allowed_after_attempt_threshold() -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-gating-allowed",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )
    agent._step_tool_attempt_rounds = 3
    agent._step_failed_tool_calls = 0

    assert agent._intercept_tool_call("message_ask_user", {"text": "需要确认"}) is None


def test_soft_hint_ask_user_does_not_increment_attempt_counters() -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-gating-no-count",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )
    agent._step_tool_attempt_rounds = 0
    agent._step_failed_tool_calls = 0
    agent._step_ask_user_soft_hint_count = 0

    result = agent._intercept_tool_call("message_ask_user", {"text": "需要确认"})
    assert result is not None
    assert result.success is True

    agent._on_tool_result("message_ask_user", result)

    assert agent._step_tool_attempt_rounds == 0
    assert agent._step_failed_tool_calls == 0


from app.domain.services.tools.base import BaseTool, tool


class _ConfirmableTool(BaseTool):
    name: str = "confirmable"

    @tool(
        name="dangerous_action",
        description="A dangerous action",
        parameters={"arg": {"type": "string", "description": "arg"}},
        required=["arg"],
        require_confirmation=True,
    )
    async def dangerous_action(self, arg: str) -> ToolResult:
        return ToolResult(success=True, data="done")

    @tool(
        name="safe_action",
        description="A safe action",
        parameters={"arg": {"type": "string", "description": "arg"}},
        required=["arg"],
    )
    async def safe_action(self, arg: str) -> ToolResult:
        return ToolResult(success=True, data="done")


def test_tool_decorator_require_confirmation_stored_on_function():
    t = _ConfirmableTool()
    dangerous = None
    safe = None
    import inspect
    for _, method in inspect.getmembers(t, inspect.ismethod):
        if getattr(method, "_tool_name", None) == "dangerous_action":
            dangerous = method
        if getattr(method, "_tool_name", None) == "safe_action":
            safe = method
    assert dangerous is not None
    assert dangerous._require_confirmation is True
    assert safe is not None
    assert safe._require_confirmation is False


def test_base_tool_get_tool_confirmation_required():
    t = _ConfirmableTool()
    assert t.get_tool_confirmation_required("dangerous_action") is True
    assert t.get_tool_confirmation_required("safe_action") is False
    assert t.get_tool_confirmation_required("nonexistent") is False


def test_intercept_tool_call_blocks_require_confirmation_tool():
    t = _ConfirmableTool()
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-confirm",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[t],
    )
    agent._step_tool_attempt_rounds = 0

    result = agent._intercept_tool_call("dangerous_action", {"arg": "test"})
    assert result is not None
    assert result.success is False
    assert result.data["code"] == "TOOL_CONFIRMATION_REQUIRED"
    assert result.data["tool_name"] == "dangerous_action"


def test_intercept_tool_call_passes_safe_tool():
    t = _ConfirmableTool()
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-safe",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[t],
    )
    result = agent._intercept_tool_call("safe_action", {"arg": "test"})
    assert result is None


def test_confirmation_blocked_tool_does_not_increment_counters():
    t = _ConfirmableTool()
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-confirm-count",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[t],
    )
    agent._step_tool_attempt_rounds = 0
    agent._step_failed_tool_calls = 0

    result = agent._intercept_tool_call("dangerous_action", {"arg": "test"})
    assert result is not None
    agent._on_tool_result("dangerous_action", result)

    assert agent._step_tool_attempt_rounds == 0
    assert agent._step_failed_tool_calls == 0


def test_ask_user_allowed_on_second_call_after_soft_hint():
    """收到一次 SOFT_HINT 后第二次调用直接放行"""
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-second-ask",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )
    agent._step_tool_attempt_rounds = 0
    agent._step_failed_tool_calls = 0
    agent._step_ask_user_soft_hint_count = 1

    result = agent._intercept_tool_call("message_ask_user", {"text": "需要确认"})
    assert result is None


async def test_execute_step_tool_confirmation_required_emits_wait_event():
    t = _ConfirmableTool()
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-tool-confirm",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[t],
    )

    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="confirmable",
            function_name="dangerous_action",
            function_args={"arg": "rm -rf /"},
            status=ToolEventStatus.CALLING,
        )
        yield ToolEvent(
            tool_call_id="tool-1",
            tool_name="confirmable",
            function_name="dangerous_action",
            function_args={"arg": "rm -rf /"},
            function_result=ToolResult(
                success=False,
                message="TOOL_CONFIRMATION_REQUIRED",
                data={
                    "code": "TOOL_CONFIRMATION_REQUIRED",
                    "tool_name": "dangerous_action",
                    "function_args": {"arg": "rm -rf /"},
                },
            ),
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="测试步骤")])
    step = plan.steps[0]
    message = Message(message="执行命令")
    events = [event async for event in agent.execute_step(plan, step, message)]

    message_events = [e for e in events if isinstance(e, MessageEvent)]
    assert len(message_events) == 1
    assert "dangerous_action" in message_events[0].message
    assert any(isinstance(e, WaitEvent) for e in events)
