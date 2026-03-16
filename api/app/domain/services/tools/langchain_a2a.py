"""Wrap existing A2ATool as LangChain StructuredTool instances.

Similar to langchain_mcp.py, we wrap A2ATool's methods into LangChain tools,
preserving the existing A2A client management.
"""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool, tool as lc_tool

from app.domain.services.tools.a2a import A2ATool


def create_a2a_langchain_tools(a2a_tool: A2ATool) -> list[StructuredTool]:
    """Convert A2ATool's methods into LangChain StructuredTool instances.

    Returns empty list if A2ATool has no initialized manager (not yet connected).
    """
    if not getattr(a2a_tool, "manager", None):
        return []

    @lc_tool
    async def get_remote_agent_cards() -> str:
        """获取可远程调用的Agent卡片信息, 包含Agent id、名称、描述、技能、请求端点等。"""
        result = await a2a_tool.get_remote_agent_cards()
        if hasattr(result, "success") and not result.success:
            raise RuntimeError(getattr(result, "message", None) or str(result))
        if hasattr(result, "data") and result.data is not None:
            return json.dumps(result.data, ensure_ascii=False)
        if hasattr(result, "message") and result.message:
            return result.message
        return str(result)

    @lc_tool
    async def call_remote_agent(id: str, query: str) -> str:
        """根据传递的id+query(分配给远程Agent完成的任务query)调用远程Agent完成对应需求"""
        result = await a2a_tool.call_remote_agent(id=id, query=query)
        if hasattr(result, "success") and not result.success:
            raise RuntimeError(getattr(result, "message", None) or str(result))
        if hasattr(result, "data") and result.data is not None:
            return json.dumps(result.data, ensure_ascii=False)
        if hasattr(result, "message") and result.message:
            return result.message
        return str(result)

    return [get_remote_agent_cards, call_remote_agent]
