"""Tests for LangChain MCP adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models.tool_result import ToolResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestCreateMCPTools:
    async def test_creates_tools_from_mcp_tool(self):
        from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.get_tools.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_tool_1",
                    "description": "A test MCP tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        mock_mcp_tool.invoke = AsyncMock(return_value=ToolResult(
            success=True, message="ok", data={"result": "done"}
        ))

        tools = create_mcp_langchain_tools(mock_mcp_tool)
        assert len(tools) == 1
        assert tools[0].name == "mcp_tool_1"

    async def test_empty_tools_returns_empty_list(self):
        from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.get_tools.return_value = []

        tools = create_mcp_langchain_tools(mock_mcp_tool)
        assert tools == []
