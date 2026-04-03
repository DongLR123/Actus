"""Tests for ActusChatModel — BaseChatModel wrapping OpenAI Chat Completions API.

TDD: these tests are written BEFORE the implementation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult

from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers — mock OpenAI response objects
# ---------------------------------------------------------------------------


def _make_chat_completion(
    content: str | None = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a mock ChatCompletion response matching OpenAI SDK structure."""
    message_fields: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
    }
    if tool_calls:
        message_fields["tool_calls"] = [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc["arguments"],
                ),
            )
            for tc in tool_calls
        ]
    message = SimpleNamespace(**message_fields)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        id="chatcmpl-test",
        choices=[choice],
        model="test-model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_stream_chunks(
    content: str = "Hello!",
) -> list[SimpleNamespace]:
    """Build mock streaming chunks for a simple text response."""
    chunks = []
    for char in content:
        delta = SimpleNamespace(content=char, role=None, tool_calls=None)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice]))
    # Final chunk with finish_reason
    final_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
    final_choice = SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")
    chunks.append(SimpleNamespace(choices=[final_choice]))
    return chunks


def _make_stream_tool_call_chunks(
    tool_id: str = "call_123",
    tool_name: str = "get_weather",
    arguments: str = '{"location": "Paris"}',
) -> list[SimpleNamespace]:
    """Build mock streaming chunks for a tool call response."""
    chunks = []

    # First chunk: tool call start with name
    tc = SimpleNamespace(
        index=0,
        id=tool_id,
        type="function",
        function=SimpleNamespace(name=tool_name, arguments=""),
    )
    delta = SimpleNamespace(content=None, role="assistant", tool_calls=[tc])
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
    chunks.append(SimpleNamespace(choices=[choice]))

    # Argument chunks
    for char in arguments:
        tc = SimpleNamespace(
            index=0,
            id=None,
            type=None,
            function=SimpleNamespace(name=None, arguments=char),
        )
        delta = SimpleNamespace(content=None, role=None, tool_calls=[tc])
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice]))

    # Final chunk
    final_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
    final_choice = SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")
    chunks.append(SimpleNamespace(choices=[final_choice]))
    return chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ActusChatModel:
    """Create an ActusChatModel with test configuration."""
    return ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
        supports_response_format=True,
    )


@pytest.fixture
def model_no_response_format() -> ActusChatModel:
    """Create an ActusChatModel with supports_response_format=False."""
    return ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
        supports_response_format=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMType:
    """Test _llm_type property."""

    def test_llm_type_value(self, model: ActusChatModel) -> None:
        assert model._llm_type == "actus-chat"


class TestSyncGenerate:
    """_generate should raise NotImplementedError (async-only project)."""

    def test_generate_raises(self, model: ActusChatModel) -> None:
        with pytest.raises(NotImplementedError):
            model._generate([HumanMessage(content="hi")])


class TestAGenerate:
    """Test _agenerate — async OpenAI Chat Completions calls."""

    async def test_basic_text_response(self, model: ActusChatModel) -> None:
        """Mock AsyncOpenAI to return a simple text response."""
        mock_response = _make_chat_completion(content="Hello from LLM!")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="Hi")])

        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello from LLM!"
        assert msg.tool_calls == []

    async def test_tool_calls_response(self, model: ActusChatModel) -> None:
        """Mock AsyncOpenAI to return a response with tool calls."""
        mock_response = _make_chat_completion(
            content="",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            ],
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="Weather?")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"location": "Paris"}

    async def test_multiple_message_types(self, model: ActusChatModel) -> None:
        """Ensure SystemMessage, HumanMessage, AIMessage, ToolMessage all convert."""
        mock_response = _make_chat_completion(content="Done")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(
                content="Let me check",
                tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "test"}}],
            ),
            ToolMessage(content="result data", tool_call_id="tc1"),
        ]

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate(messages)

        # Verify the call was made with properly converted messages
        call_kwargs = mock_client.chat.completions.create.call_args
        sent_messages = call_kwargs.kwargs["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["role"] == "user"
        assert sent_messages[2]["role"] == "assistant"
        assert sent_messages[2]["tool_calls"][0]["function"]["name"] == "search"
        assert sent_messages[3]["role"] == "tool"
        assert sent_messages[3]["tool_call_id"] == "tc1"

    async def test_response_format_passed_when_supported(
        self, model: ActusChatModel
    ) -> None:
        """response_format should be passed when supports_response_format=True."""
        mock_response = _make_chat_completion(content='{"answer": 42}')
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="answer")],
                response_format={"type": "json_object"},
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    async def test_response_format_not_passed_when_unsupported(
        self, model_no_response_format: ActusChatModel
    ) -> None:
        """response_format should NOT be passed when supports_response_format=False."""
        mock_response = _make_chat_completion(content="text only")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(
            model_no_response_format, "_get_client", return_value=mock_client
        ):
            await model_no_response_format._agenerate(
                [HumanMessage(content="answer")],
                response_format={"type": "json_object"},
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs

    async def test_tools_from_kwargs(self, model: ActusChatModel) -> None:
        """Per-call tools passed via kwargs should be sent to OpenAI."""
        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tools=tools,
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    async def test_tool_choice_from_kwargs(self, model: ActusChatModel) -> None:
        """tool_choice from kwargs should be passed to OpenAI."""
        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tool_choice="auto",
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "auto"

    async def test_tool_choice_any_normalized_to_required(self, model: ActusChatModel) -> None:
        """LangChain's 'any' tool_choice should be normalized to 'required' for OpenAI API."""
        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tool_choice="any",
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"


class TestBindTools:
    """Test bind_tools returns new instance with tools bound."""

    def test_bind_tools_returns_new_instance(self, model: ActusChatModel) -> None:
        """bind_tools should return a new ActusChatModel, not mutate the original."""
        from langchain_core.tools import tool

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        bound = model.bind_tools([dummy_tool])
        assert isinstance(bound, ActusChatModel)
        assert bound is not model

    def test_bind_tools_stores_converted_tools(self, model: ActusChatModel) -> None:
        """bind_tools should convert tools to OpenAI format and store them."""
        from langchain_core.tools import tool

        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Sunny in {location}"

        bound = model.bind_tools([get_weather])
        assert bound._bound_tools is not None
        assert len(bound._bound_tools) == 1
        assert bound._bound_tools[0]["type"] == "function"
        assert bound._bound_tools[0]["function"]["name"] == "get_weather"

    def test_original_has_no_tools(self, model: ActusChatModel) -> None:
        """Original model should remain unchanged after bind_tools."""
        from langchain_core.tools import tool

        @tool
        def dummy(x: str) -> str:
            """Dummy."""
            return x

        model.bind_tools([dummy])
        assert model._bound_tools is None or model._bound_tools == []

    async def test_bound_tools_sent_to_openai(self, model: ActusChatModel) -> None:
        """After bind_tools, the bound tools should be sent in _agenerate calls."""
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        bound = model.bind_tools([search])

        mock_response = _make_chat_completion(content="found it")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(bound, "_get_client", return_value=mock_client):
            await bound._agenerate([HumanMessage(content="search something")])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "search"

    async def test_bound_tools_merged_with_kwargs_tools(
        self, model: ActusChatModel
    ) -> None:
        """Per-call tools from kwargs should be merged with bound tools."""
        from langchain_core.tools import tool

        @tool
        def bound_tool(x: str) -> str:
            """A bound tool."""
            return x

        bound = model.bind_tools([bound_tool])

        extra_tool = {
            "type": "function",
            "function": {"name": "extra_tool", "parameters": {}},
        }

        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(bound, "_get_client", return_value=mock_client):
            await bound._agenerate(
                [HumanMessage(content="test")],
                tools=[extra_tool],
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        sent_tools = call_kwargs["tools"]
        tool_names = [t["function"]["name"] for t in sent_tools]
        assert "bound_tool" in tool_names
        assert "extra_tool" in tool_names


class TestAStream:
    """Test _astream — streaming from OpenAI Chat Completions."""

    async def test_stream_text_response(self, model: ActusChatModel) -> None:
        """Streaming a simple text response should yield AIMessageChunk objects."""
        from langchain_core.messages import AIMessageChunk

        chunks = _make_stream_chunks("Hi!")

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = mock_stream

        with patch.object(model, "_get_client", return_value=mock_client):
            collected = []
            async for gen_chunk in model._astream([HumanMessage(content="hello")]):
                collected.append(gen_chunk)

        # Should have yielded chunks with content
        contents = [
            c.message.content for c in collected if c.message.content
        ]
        assert "".join(contents) == "Hi!"

    async def test_stream_tool_call_response(self, model: ActusChatModel) -> None:
        """Streaming a tool call response should yield chunks that aggregate."""
        from langchain_core.messages import AIMessageChunk

        chunks = _make_stream_tool_call_chunks(
            tool_id="call_456",
            tool_name="get_weather",
            arguments='{"city": "NYC"}',
        )

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = mock_stream

        with patch.object(model, "_get_client", return_value=mock_client):
            collected = []
            async for gen_chunk in model._astream([HumanMessage(content="weather")]):
                collected.append(gen_chunk)

        # At least one chunk should have tool_call_chunks
        has_tool_chunks = any(
            c.message.tool_call_chunks for c in collected
            if hasattr(c.message, "tool_call_chunks") and c.message.tool_call_chunks
        )
        assert has_tool_chunks


class TestModelProperties:
    """Test that model configuration fields are accessible."""

    def test_model_name_accessible(self, model: ActusChatModel) -> None:
        assert model.model_name == "test-model"

    def test_temperature_accessible(self, model: ActusChatModel) -> None:
        assert model.temperature == 0.5

    def test_max_tokens_accessible(self, model: ActusChatModel) -> None:
        assert model.max_tokens == 1024

    def test_supports_response_format_accessible(self, model: ActusChatModel) -> None:
        assert model.supports_response_format is True


# ---------------------------------------------------------------------------
# Tests — Fallback tool call extraction from content
# ---------------------------------------------------------------------------

def _make_model_with_tools(tool_names: list[str]) -> ActusChatModel:
    """Create an ActusChatModel with _bound_tools and _bound_tool_names set."""
    model = ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key",
        model_name="test-model",
    )
    model._bound_tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in tool_names
    ]
    model._bound_tool_names = frozenset(tool_names)
    return model


class TestExtractToolCallsFromContentXML:
    """Test XML <invoke> pattern extraction."""

    def test_minimax_style_xml(self) -> None:
        """MiniMax-style XML with :tool_call marker and <invoke> block."""
        model = _make_model_with_tools(["message_notify_user"])
        content = (
            "💭 思考过程\n用户要求分析图片。\n\n"
            "minimax:tool_call\n"
            '<invoke name="message_notify_user">'
            "<end_turn>true</end_turn>"
            "<text>正在调用工具...</text>"
            "</invoke>"
        )
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "message_notify_user"
        assert tool_calls[0]["args"] == {"text": "正在调用工具..."}
        assert "end_turn" not in json.dumps(tool_calls[0]["args"])
        assert tool_calls[0]["id"].startswith("fallback_")
        # Cleaned content should not contain the XML or marker
        assert "<invoke" not in cleaned
        assert "minimax:tool_call" not in cleaned
        # But should preserve the thinking text
        assert "思考过程" in cleaned

    def test_multiple_xml_invocations(self) -> None:
        """Multiple <invoke> blocks should all be extracted."""
        model = _make_model_with_tools(["file_read", "shell_execute"])
        content = (
            '<invoke name="file_read"><filepath>/tmp/a.txt</filepath></invoke>\n'
            '<invoke name="shell_execute"><command>ls -la</command></invoke>'
        )
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "file_read"
        assert tool_calls[0]["args"] == {"filepath": "/tmp/a.txt"}
        assert tool_calls[1]["name"] == "shell_execute"
        assert tool_calls[1]["args"] == {"command": "ls -la"}

    def test_xml_with_numeric_arg(self) -> None:
        """Numeric values in XML args should be parsed as numbers."""
        model = _make_model_with_tools(["shell_wait_process"])
        content = '<invoke name="shell_wait_process"><seconds>5</seconds></invoke>'
        tool_calls, _ = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["args"]["seconds"] == 5

    def test_xml_with_boolean_arg(self) -> None:
        """Boolean values in XML args should be parsed as booleans."""
        model = _make_model_with_tools(["file_write"])
        content = '<invoke name="file_write"><filepath>/tmp/f</filepath><append>true</append></invoke>'
        tool_calls, _ = model._extract_tool_calls_from_content(content)
        assert tool_calls[0]["args"]["append"] is True

    def test_xml_unknown_tool_name_ignored(self) -> None:
        """XML invoke with a tool name not in bound tools should be ignored."""
        model = _make_model_with_tools(["file_read"])
        content = '<invoke name="nonexistent_tool"><x>1</x></invoke>'
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []
        assert cleaned == content

    def test_xml_end_turn_tag_excluded(self) -> None:
        """The <end_turn> control tag should not appear in extracted args."""
        model = _make_model_with_tools(["search_web"])
        content = '<invoke name="search_web"><end_turn>true</end_turn><query>test</query></invoke>'
        tool_calls, _ = model._extract_tool_calls_from_content(content)
        assert "end_turn" not in tool_calls[0]["args"]
        assert tool_calls[0]["args"]["query"] == "test"


class TestExtractToolCallsFromContentJSON:
    """Test JSON-in-content extraction."""

    def test_basic_json_tool_call(self) -> None:
        """JSON object with name + arguments should be extracted."""
        model = _make_model_with_tools(["search_web"])
        content = (
            'I will search for you.\n'
            '{"name": "search_web", "arguments": {"query": "python tutorial"}}'
        )
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "search_web"
        assert tool_calls[0]["args"] == {"query": "python tutorial"}
        assert "search_web" not in cleaned
        assert "I will search for you." in cleaned

    def test_json_in_code_fence(self) -> None:
        """JSON inside ```json ... ``` code fence."""
        model = _make_model_with_tools(["file_read"])
        content = (
            "Let me read that file:\n"
            '```json\n'
            '{"name": "file_read", "arguments": {"filepath": "/etc/hosts"}}\n'
            '```'
        )
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "file_read"
        assert tool_calls[0]["args"]["filepath"] == "/etc/hosts"

    def test_json_unknown_tool_ignored(self) -> None:
        """JSON object with unknown tool name should not be extracted."""
        model = _make_model_with_tools(["file_read"])
        content = '{"name": "unknown_tool", "arguments": {"x": 1}}'
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []
        assert cleaned == content

    def test_json_missing_arguments_ignored(self) -> None:
        """JSON object without 'arguments' dict should not be extracted."""
        model = _make_model_with_tools(["file_read"])
        content = '{"name": "file_read", "args": {"filepath": "/tmp"}}'
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []

    def test_multiple_json_tool_calls(self) -> None:
        """Multiple JSON tool call objects in content."""
        model = _make_model_with_tools(["file_read", "shell_execute"])
        content = (
            '{"name": "file_read", "arguments": {"filepath": "/a.txt"}}\n'
            '{"name": "shell_execute", "arguments": {"command": "ls"}}'
        )
        tool_calls, _ = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 2


class TestExtractToolCallsNoFalsePositives:
    """Ensure normal content without tool calls is not misinterpreted."""

    def test_normal_text_unchanged(self) -> None:
        """Plain text content should not trigger extraction."""
        model = _make_model_with_tools(["file_read", "shell_execute"])
        content = "The file was read successfully. Here are the results..."
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []
        assert cleaned == content

    def test_empty_content(self) -> None:
        """Empty content returns empty."""
        model = _make_model_with_tools(["file_read"])
        tool_calls, cleaned = model._extract_tool_calls_from_content("")
        assert tool_calls == []
        assert cleaned == ""

    def test_no_bound_tools(self) -> None:
        """Without bound tools, no extraction should happen."""
        model = ActusChatModel(base_url="https://x", api_key="k", model_name="m")
        model._bound_tools = None
        model._bound_tool_names = frozenset()
        content = '<invoke name="file_read"><x>1</x></invoke>'
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []
        assert cleaned == content

    def test_json_object_without_name_key(self) -> None:
        """Random JSON in content that isn't a tool call."""
        model = _make_model_with_tools(["file_read"])
        content = 'The config is: {"host": "localhost", "port": 8080}'
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []
        assert cleaned == content

    def test_xml_like_content_without_invoke(self) -> None:
        """Content with XML-like tags but not <invoke> should be left alone."""
        model = _make_model_with_tools(["file_read"])
        content = "<result><status>ok</status></result>"
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []
        assert cleaned == content

    def test_content_exceeding_max_scan_length(self) -> None:
        """Content longer than _MAX_CONTENT_TO_SCAN is skipped entirely."""
        model = _make_model_with_tools(["file_read"])
        # Create content just over the limit
        content = "x" * (model._MAX_CONTENT_TO_SCAN + 1)
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert tool_calls == []
        assert cleaned == content

    def test_tool_call_marker_not_corrupting_normal_text(self) -> None:
        """Text containing 'tool_call' as a substring should NOT be removed."""
        model = _make_model_with_tools(["file_read"])
        content = (
            'See the tool_call field for details.\n'
            '<invoke name="file_read"><filepath>/tmp/a</filepath></invoke>'
        )
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        # The normal text should be preserved — marker regex is line-anchored
        assert "tool_call" in cleaned

    def test_only_accepted_xml_blocks_removed(self) -> None:
        """Unrecognised invoke blocks should remain in content."""
        model = _make_model_with_tools(["file_read"])
        content = (
            '<invoke name="file_read"><filepath>/tmp/a</filepath></invoke>\n'
            '<invoke name="unknown_tool"><x>1</x></invoke>'
        )
        tool_calls, cleaned = model._extract_tool_calls_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "file_read"
        # The unknown_tool invoke should still be in the cleaned content
        assert "unknown_tool" in cleaned


class TestFallbackIntegrationInAgenerate:
    """Test that _agenerate uses fallback extraction when API returns no tool_calls."""

    async def test_xml_tool_call_in_content_extracted(self) -> None:
        """When API returns no tool_calls but content has XML, fallback extracts it."""
        model = _make_model_with_tools(["mcp_MiniMax_understand_image"])

        xml_content = (
            "让我调用工具来分析图片。\n\n"
            "minimax:tool_call\n"
            '<invoke name="mcp_MiniMax_understand_image">'
            "<image_url>http://example.com/img.png</image_url>"
            "</invoke>"
        )
        mock_response = _make_chat_completion(content=xml_content, tool_calls=None)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="分析图片")])

        msg = result.generations[0].message
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "mcp_MiniMax_understand_image"
        assert msg.tool_calls[0]["args"]["image_url"] == "http://example.com/img.png"
        # Content should be cleaned
        assert "<invoke" not in msg.content

    async def test_structured_tool_calls_take_precedence(self) -> None:
        """When API returns structured tool_calls, fallback is NOT invoked."""
        model = _make_model_with_tools(["get_weather"])

        # Content also has XML, but structured tool_calls are present
        content_with_xml = '<invoke name="get_weather"><location>NYC</location></invoke>'
        mock_response = _make_chat_completion(
            content=content_with_xml,
            tool_calls=[{
                "id": "call_real",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            }],
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="weather?")])

        msg = result.generations[0].message
        # Should use the structured tool call, not the XML one
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["id"] == "call_real"
        assert msg.tool_calls[0]["args"]["location"] == "Paris"
        # Content should still contain the XML (fallback not invoked)
        assert "<invoke" in msg.content
