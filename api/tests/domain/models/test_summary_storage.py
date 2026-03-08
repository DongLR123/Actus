import pytest

from app.domain.models.conversation_summary import ConversationSummary
from app.domain.models.memory import Memory

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _DummySessionRepo:
    """模拟内存存储的 SessionRepository，测试 summary 独立于 memory"""

    def __init__(self) -> None:
        self._memories: dict = {}
        self._summaries: dict = {}

    async def get_memory(self, session_id: str, agent_name: str) -> Memory:
        return self._memories.get((session_id, agent_name), Memory())

    async def save_memory(
        self, session_id: str, agent_name: str, memory: Memory
    ) -> None:
        self._memories[(session_id, agent_name)] = memory

    async def get_summary(self, session_id: str) -> list[ConversationSummary]:
        return self._summaries.get(session_id, [])

    async def save_summary(
        self, session_id: str, summaries: list[ConversationSummary]
    ) -> None:
        self._summaries[session_id] = summaries


async def test_save_and_get_summary_roundtrip() -> None:
    repo = _DummySessionRepo()

    summary = ConversationSummary(
        round_number=1,
        user_intent="分析数据",
        plan_summary="读取并分析",
        execution_results=["完成"],
        decisions=[],
        unresolved=[],
    )

    await repo.save_summary("s1", [summary])
    result = await repo.get_summary("s1")

    assert len(result) == 1
    assert result[0].user_intent == "分析数据"


async def test_summary_does_not_interfere_with_memory() -> None:
    repo = _DummySessionRepo()

    memory = Memory(messages=[{"role": "user", "content": "hello"}])
    await repo.save_memory("s1", "react", memory)

    summary = ConversationSummary(
        round_number=1,
        user_intent="test",
        plan_summary="test plan",
        execution_results=[],
        decisions=[],
        unresolved=[],
    )
    await repo.save_summary("s1", [summary])

    loaded_memory = await repo.get_memory("s1", "react")
    assert len(loaded_memory.messages) == 1
    assert loaded_memory.messages[0]["content"] == "hello"

    loaded_summaries = await repo.get_summary("s1")
    assert len(loaded_summaries) == 1


async def test_get_summary_returns_empty_when_not_exists() -> None:
    repo = _DummySessionRepo()
    result = await repo.get_summary("nonexistent")
    assert result == []
