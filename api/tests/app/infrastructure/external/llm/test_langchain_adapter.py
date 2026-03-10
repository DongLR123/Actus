"""Tests for LLMAdapter — bridges Actus LLM Protocol to LangChain BaseChatModel."""

import pytest
from unittest.mock import AsyncMock, PropertyMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_llm():
    """Create a mock Actus LLM Protocol implementation."""
    llm = AsyncMock()
    type(llm).model_name = PropertyMock(return_value="gpt-4o")
    type(llm).temperature = PropertyMock(return_value=0.7)
    type(llm).max_tokens = PropertyMock(return_value=4096)
    return llm


@pytest.fixture
def adapter(mock_llm):
    from app.infrastructure.external.llm.langchain_adapter import LLMAdapter
    return LLMAdapter(llm=mock_llm)


class TestLLMAdapterProperties:
    def test_model_name(self, adapter):
        assert adapter.model_name == "gpt-4o"

    def test_llm_type(self, adapter):
        assert adapter._llm_type == "actus-llm-adapter"


class TestLangChainMessageConversion:
    """Test converting LangChain messages to Actus dict format."""

    def test_human_message(self, adapter):
        msgs = [HumanMessage(content="hello")]
        result = adapter._to_actus_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_system_message(self, adapter):
        msgs = [SystemMessage(content="you are helpful")]
        result = adapter._to_actus_messages(msgs)
        assert result == [{"role": "system", "content": "you are helpful"}]

    def test_ai_message_plain(self, adapter):
        msgs = [AIMessage(content="sure")]
        result = adapter._to_actus_messages(msgs)
        assert result == [{"role": "assistant", "content": "sure"}]

    def test_ai_message_with_tool_calls(self, adapter):
        msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "shell_execute",
                "args": {"command": "ls"},
            }],
        )
        result = adapter._to_actus_messages([msg])
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["id"] == "call_1"

    def test_tool_message(self, adapter):
        msgs = [ToolMessage(content="output", tool_call_id="call_1")]
        result = adapter._to_actus_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"


class TestAGenerate:
    async def test_plain_response(self, adapter, mock_llm):
        mock_llm.invoke.return_value = {
            "content": "Hello!",
            "role": "assistant",
        }
        result = await adapter.ainvoke([HumanMessage(content="hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

    async def test_tool_call_response(self, adapter, mock_llm):
        mock_llm.invoke.return_value = {
            "content": "",
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "shell_execute", "arguments": '{"command":"ls"}'},
                "type": "function",
            }],
        }
        result = await adapter.ainvoke([HumanMessage(content="list files")])
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "shell_execute"


class TestBindTools:
    def test_bind_tools_returns_new_adapter(self, adapter):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        bound = adapter.bind_tools([dummy_tool])
        assert bound is not adapter
        assert bound._tools is not None
