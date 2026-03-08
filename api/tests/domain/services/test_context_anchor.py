import pytest

from app.domain.models.memory import Memory
from app.domain.services.agents.base import BaseAgent

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _DummySessionRepo:
    def __init__(self) -> None:
        self._memory = Memory()

    async def get_memory(self, _sid: str, _name: str) -> Memory:
        return self._memory

    async def save_memory(self, _sid: str, _name: str, memory: Memory) -> None:
        self._memory = memory


class _DummyUoW:
    def __init__(self) -> None:
        self.session = _DummySessionRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _DummyLLM:
    async def invoke(self, **kwargs):
        return {"role": "assistant", "content": "ok"}


class _DummyJSONParser:
    async def invoke(self, text: str):
        return {}


def _make_agent() -> BaseAgent:
    from app.domain.models.app_config import AgentConfig

    uow = _DummyUoW()
    agent = BaseAgent.__new__(BaseAgent)
    agent._uow_factory = lambda: uow
    agent._uow = uow
    agent._session_id = "test-session"
    agent._agent_config = AgentConfig()
    agent._llm = _DummyLLM()
    agent._memory = Memory()
    agent._json_parser = _DummyJSONParser()
    agent._tools = []
    agent._runtime_system_context = ""
    agent._conversation_summaries = []
    agent.name = "test"
    agent._system_prompt = "system"
    agent._format = None
    agent._tool_choice = None
    agent._retry_interval = 0.01
    return agent


def test_inject_context_anchor_running_full() -> None:
    """RUNNING 场景：注入完整锚点"""
    agent = _make_agent()

    agent.inject_context_anchor(
        session_status="running",
        user_message="新的问题",
        original_request="帮我分析数据",
        completed_steps=["步骤1: 读取文件", "步骤2: 分析数据"],
    )

    last_msg = agent._memory.get_last_message()
    assert last_msg is not None
    assert last_msg["role"] == "user"
    assert "[上下文回顾]" in last_msg["content"]
    assert "帮我分析数据" in last_msg["content"]
    assert "步骤1: 读取文件" in last_msg["content"]
    assert "新的问题" in last_msg["content"]


def test_inject_context_anchor_waiting_lightweight() -> None:
    """WAITING 场景：仅注入轻量提醒"""
    agent = _make_agent()

    agent.inject_context_anchor(
        session_status="waiting",
        user_message="是的，继续",
        original_request="帮我分析数据",
        completed_steps=["步骤1: 读取文件"],
    )

    last_msg = agent._memory.get_last_message()
    assert last_msg is not None
    assert "[上下文回顾]" in last_msg["content"]
    assert "是的，继续" in last_msg["content"]
    assert "步骤1" not in last_msg["content"]


def test_inject_context_anchor_no_duplicate() -> None:
    """锚点不会重复注入"""
    agent = _make_agent()

    agent.inject_context_anchor(
        session_status="running",
        user_message="msg1",
        original_request="req",
        completed_steps=[],
    )
    agent.inject_context_anchor(
        session_status="running",
        user_message="msg2",
        original_request="req",
        completed_steps=[],
    )

    anchor_count = sum(
        1
        for m in agent._memory.messages
        if m.get("role") == "user"
        and m.get("content", "").startswith("[上下文回顾]")
    )
    assert anchor_count == 1


def test_get_latest_assistant_content() -> None:
    """get_latest_assistant_content 应从 memory 中取最近的 assistant content"""
    agent = _make_agent()
    agent._memory.add_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "第一轮回复"},
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "这是最终的执行结果，包含一些详细的信息"},
        ]
    )

    result = agent.get_latest_assistant_content(max_chars=10)
    assert result == "这是最终的执行结果，"
    assert len(result) == 10


def test_get_latest_assistant_content_empty_memory() -> None:
    """memory 为空时返回空字符串"""
    agent = _make_agent()
    assert agent.get_latest_assistant_content() == ""


def test_get_latest_assistant_content_no_assistant_message() -> None:
    """没有 assistant 消息时返回空字符串"""
    agent = _make_agent()
    agent._memory.add_message({"role": "user", "content": "hello"})
    assert agent.get_latest_assistant_content() == ""
