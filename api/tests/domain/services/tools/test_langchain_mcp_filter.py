"""Tests for create_mcp_langchain_tools tool_names filter."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools


def _make_mcp_tool_with_tools(tool_names: list[str]):
    """Create a mock MCPTool returning specified tools."""
    mock = MagicMock()
    mock.get_tools.return_value = [
        {
            "function": {
                "name": f"mcp_server_{name}",
                "description": f"Tool {name}",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        for name in tool_names
    ]
    return mock


class TestCreateMcpLangchainToolsFilter:

    def test_no_filter_returns_all(self):
        """Without tool_names, all tools returned (backward compat)."""
        mcp = _make_mcp_tool_with_tools(["weather", "geo", "search"])
        tools = create_mcp_langchain_tools(mcp)
        assert len(tools) == 3

    def test_none_filter_returns_all(self):
        """Explicit None returns all tools."""
        mcp = _make_mcp_tool_with_tools(["weather", "geo", "search"])
        tools = create_mcp_langchain_tools(mcp, tool_names=None)
        assert len(tools) == 3

    def test_filter_returns_subset(self):
        """Only tools in tool_names set are returned."""
        mcp = _make_mcp_tool_with_tools(["weather", "geo", "search"])
        tools = create_mcp_langchain_tools(
            mcp, tool_names={"mcp_server_weather", "mcp_server_geo"}
        )
        names = {t.name for t in tools}
        assert names == {"mcp_server_weather", "mcp_server_geo"}

    def test_filter_empty_set_returns_none(self):
        """Empty set means no tools."""
        mcp = _make_mcp_tool_with_tools(["weather", "geo"])
        tools = create_mcp_langchain_tools(mcp, tool_names=set())
        assert len(tools) == 0

    def test_filter_nonexistent_names_ignored(self):
        """Names not in MCP tools are silently ignored."""
        mcp = _make_mcp_tool_with_tools(["weather"])
        tools = create_mcp_langchain_tools(
            mcp, tool_names={"mcp_server_weather", "mcp_server_nonexistent"}
        )
        assert len(tools) == 1
        assert tools[0].name == "mcp_server_weather"
