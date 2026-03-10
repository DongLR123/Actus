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


def create_mcp_langchain_tools(mcp_tool: MCPTool) -> list[StructuredTool]:
    """Convert MCPTool's registered tools into LangChain StructuredTool instances."""
    tools: list[StructuredTool] = []

    for schema in mcp_tool.get_tools():
        fn_def = schema.get("function", {})
        name = fn_def.get("name", "")
        description = fn_def.get("description", "")

        if not name:
            continue

        # Capture name in closure
        _name = name

        async def _invoke(tool_name: str = _name, **kwargs: Any) -> str:
            result = await mcp_tool.invoke(tool_name, **kwargs)
            if hasattr(result, "message") and result.message:
                return result.message
            if hasattr(result, "data") and result.data:
                return json.dumps(result.data)
            return str(result)

        tool = StructuredTool.from_function(
            coroutine=lambda tool_name=_name, **kw: _invoke(tool_name=tool_name, **kw),
            name=name,
            description=description,
        )
        tools.append(tool)

    return tools
