"""Wrap existing MCPTool schemas as LangChain StructuredTool instances.

Instead of using langchain-mcp-adapters (which requires direct MCP server access),
we wrap our existing MCPTool.get_tools() schemas and MCPTool.invoke() dispatcher
into LangChain tools. This preserves the existing MCP client management.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from app.domain.services.tools.mcp import MCPTool


def _make_mcp_coroutine(mcp_tool: MCPTool, tool_name: str):
    """为每个 MCP tool 创建独立的协程，通过闭包绑定 tool_name。"""

    async def _invoke(**kwargs: Any) -> str:
        result = await mcp_tool.invoke(tool_name, **kwargs)
        if hasattr(result, "message") and result.message:
            return result.message
        if hasattr(result, "data") and result.data:
            return json.dumps(result.data)
        return str(result)

    return _invoke


def create_mcp_langchain_tools(mcp_tool: MCPTool) -> list[StructuredTool]:
    """Convert MCPTool's registered tools into LangChain StructuredTool instances."""
    tools: list[StructuredTool] = []

    for schema in mcp_tool.get_tools():
        fn_def = schema.get("function", {})
        name = fn_def.get("name", "")
        description = fn_def.get("description", "")

        if not name:
            continue

        tool = StructuredTool.from_function(
            coroutine=_make_mcp_coroutine(mcp_tool, name),
            name=name,
            description=description,
        )
        tools.append(tool)

    return tools
