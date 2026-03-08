from typing import Any

import pytest

from app.domain.models.app_config import AgentConfig
from app.domain.models.conversation_summary import ConversationSummary
from app.domain.models.event import MessageEvent
from app.domain.models.memory import Memory
from app.domain.models.tool_result import ToolResult
from app.domain.services.agents.base import BaseAgent
from app.domain.services.tools.base import BaseTool, tool

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _InMemorySessionRepo:
    def __init__(self) -> None:
        self._memories: dict[tuple[str, str], Memory] = {}
        self._summaries: dict[str, list[ConversationSummary]] = {}

    async def get_memory(self, session_id: str, agent_name: str) -> Memory:
        return self._memories.get((session_id, agent_name), Memory())

    async def save_memory(
        self, session_id: str, agent_name: str, memory: Memory
    ) -> None:
        self._memories[(session_id, agent_name)] = memory

    async def get_summary(self, session_id: str) -> list[ConversationSummary]:
        return list(self._summaries.get(session_id, []))

    async def save_summary(
        self, session_id: str, summaries: list[ConversationSummary]
    ) -> None:
        self._summaries[session_id] = list(summaries)


class _InMemoryUoW:
    def __init__(self, repo: _InMemorySessionRepo) -> None:
        self.session = repo

    async def __aenter__(self) -> "_InMemoryUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _SequenceLLM:
    def __init__(self) -> None:
        self._index = 0

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        self._index += 1
        if self._index == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool-browser-1",
                        "function": {
                            "name": "browser_navigate",
                            "arguments": "{}",
                        },
                    }
                ],
            }

        return {"role": "assistant", "content": "已打开页面", "tool_calls": []}


class _AnswerOnlyLLM:
    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": "收到", "tool_calls": []}


class _JsonParser:
    async def invoke(self, payload: Any) -> Any:
        if payload == "{}":
            return {}
        return payload


class _BrowserTool(BaseTool):
    name = "browser"

    @tool(
        name="browser_navigate",
        description="打开网页",
        parameters={},
        required=[],
    )
    async def browser_navigate(self) -> ToolResult:
        return ToolResult(
            success=True,
            message=(
                "<html><head><title>百度一下</title></head>"
                "<body>搜索引擎内容</body></html>"
            ),
        )


class _DummyAgent(BaseAgent):
    name = "dummy"
    _system_prompt = "test system prompt"


def _build_agent(repo: _InMemorySessionRepo, llm: Any) -> _DummyAgent:
    return _DummyAgent(
        uow_factory=lambda: _InMemoryUoW(repo),
        session_id="s-memory",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=llm,
        json_parser=_JsonParser(),
        tools=[_BrowserTool()],
    )


async def test_agent_memory_flow_preserves_summary_and_browser_compaction() -> None:
    repo = _InMemorySessionRepo()
    await repo.save_summary(
        "s-memory",
        [
            ConversationSummary(
                round_number=1,
                user_intent="分析数据",
                plan_summary="读取 CSV",
                execution_results=["成功读取文件"],
                decisions=["按月聚合"],
                unresolved=[],
            )
        ],
    )

    agent = _build_agent(repo, _SequenceLLM())
    agent.set_conversation_summaries(await repo.get_summary("s-memory"))

    events = [event async for event in agent.invoke("打开网页")]
    await agent.compact_memory(keep_summary=True)

    memory = await repo.get_memory("s-memory", agent.name)
    tool_message = next(
        message for message in memory.messages if message.get("role") == "tool"
    )

    assert isinstance(events[-1], MessageEvent)
    assert events[-1].message == "已打开页面"
    assert "## 历史对话摘要" in memory.messages[0]["content"]
    assert "第1轮" in memory.messages[0]["content"]
    assert "分析数据" in memory.messages[0]["content"]
    assert tool_message["function_name"] == "browser_navigate"
    assert "[已执行] browser_navigate:" in tool_message["content"]
    assert "百度一下" in tool_message["content"]
    assert "<html>" not in tool_message["content"]
    assert agent.get_latest_assistant_content() == "已打开页面"


async def test_persisted_summary_and_context_anchor_share_same_memory() -> None:
    repo = _InMemorySessionRepo()
    await repo.save_summary(
        "s-memory",
        [
            ConversationSummary(
                round_number=1,
                user_intent="分析数据",
                plan_summary="读取 CSV",
                execution_results=["成功读取文件"],
                decisions=[],
                unresolved=[],
            )
        ],
    )

    agent = _build_agent(repo, _AnswerOnlyLLM())
    agent.set_conversation_summaries(await repo.get_summary("s-memory"))

    _ = [event async for event in agent.invoke("开始处理")]
    agent.inject_context_anchor(
        session_status="running",
        user_message="继续",
        original_request="分析数据",
        completed_steps=["读取文件"],
    )
    agent.inject_context_anchor(
        session_status="running",
        user_message="继续",
        original_request="分析数据",
        completed_steps=["读取文件"],
    )

    memory = await repo.get_memory("s-memory", agent.name)
    anchors = [
        message["content"]
        for message in memory.messages
        if message.get("role") == "user"
        and message.get("content", "").startswith("[上下文回顾]")
    ]

    assert "## 历史对话摘要" in memory.messages[0]["content"]
    assert len(anchors) == 1
    assert "用户原始需求：分析数据" in anchors[0]
    assert "已完成步骤：读取文件" in anchors[0]
    assert "用户新消息：继续" in anchors[0]
