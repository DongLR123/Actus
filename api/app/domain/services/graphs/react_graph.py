"""react_graph — inner ReAct loop as a LangGraph StateGraph.

Replaces BaseAgent.invoke() and ReActAgent.execute_step().
Nodes: llm_node, tool_node
Edges: START → llm_node → route_after_llm → (tool_node → llm_node) | END

Reference: docs/plans/2026-03-10-langchain-langgraph-migration-design.md §4.3-4.4
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.application.errors.exceptions import ServerRequestsError
from app.domain.models.event import (
    MessageEvent,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.tool_result import ToolResult

from .state import ReactGraphState

logger = logging.getLogger(__name__)

# Max ReAct iterations to prevent infinite loops
MAX_ITERATIONS = 30

# Tool name → category mapping (mirrors agent_task_runner._classify_tool_name)
_TOOL_CATEGORY_PREFIXES = {
    "browser_": "browser",
    "shell_": "shell",
    "file_": "file",
    "search_": "search",
    "message_": "message",
}
_KNOWN_CATEGORIES = frozenset(
    {"browser", "shell", "file", "search", "message", "mcp", "a2a", "skill"}
)


def _classify_tool_name(tool_name: str) -> str:
    """Extract tool category from LangChain tool name.

    e.g., "browser_navigate" → "browser", "shell_execute" → "shell".
    Keeps names like "mcp", "browser" as-is if already a category.
    """
    if tool_name in _KNOWN_CATEGORIES:
        return tool_name
    for prefix, category in _TOOL_CATEGORY_PREFIXES.items():
        if tool_name.startswith(prefix):
            return category
    return tool_name


def build_react_graph(llm: Any, tools: list, agent_config: Any = None) -> Any:
    """Build and compile the inner ReAct loop graph.

    Parameters
    ----------
    llm : LangChain BaseChatModel (or LLMAdapter) — must support bind_tools.
    tools : List of LangChain tools.
    agent_config : Optional AgentConfig for iteration limits etc.
    """
    # Build tool lookup
    tool_map: dict[str, Any] = {t.name: t for t in tools}

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    # ---- Nodes --------------------------------------------------------- #

    async def llm_node(state: ReactGraphState) -> dict:
        """Call the LLM with current messages."""
        messages = state["messages"]

        response: AIMessage = await llm_with_tools.ainvoke(messages)

        new_events = []

        # Emit ToolEvent(CALLING) for each tool call
        if response.tool_calls:
            for tc in response.tool_calls:
                func_name = tc["name"]
                new_events.append(
                    ToolEvent(
                        tool_call_id=tc["id"],
                        tool_name=_classify_tool_name(func_name),
                        function_name=func_name,
                        function_args=tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"]),
                        status=ToolEventStatus.CALLING,
                    )
                )

        # 最终回答（无 tool_calls 且有内容）发射 MessageEvent，使前端实时收到
        if not response.tool_calls and response.content:
            new_events.append(
                MessageEvent(role="assistant", message=response.content)
            )

        return {
            "messages": [response],
            "events": new_events,
        }

    async def tool_node(state: ReactGraphState) -> dict:
        """Execute tool calls from the last assistant message.

        Special handling for ``message_ask_user``:
        - If ``suggest_user_takeover`` is "browser"/"shell" → set should_interrupt
          (handled by confirmation_check, but also guard here).
        - Otherwise, first call returns SOFT_HINT (agent should try to solve
          autonomously). If a SOFT_HINT was already returned in this step
          and the LLM calls again, it truly needs user input → interrupt.
        """
        messages = state["messages"]
        last_msg = messages[-1]

        # AIMessage.tool_calls is a list of dicts with id/name/args
        tool_calls = last_msg.tool_calls if isinstance(last_msg, AIMessage) else []

        # Check if a SOFT_HINT was already returned in this step
        has_prior_soft_hint = state.get("soft_hint_sent", False)

        new_messages = []
        new_events = []
        should_interrupt = False
        new_failures = 0

        for tc in tool_calls:
            tool_name = tc["name"]
            args = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
            call_id = tc["id"]

            # ---- message_ask_user: SOFT_HINT gating ---- #
            tool_success = True
            if tool_name == "message_ask_user":
                suggest = str(args.get("suggest_user_takeover", "none")).strip().lower()
                if suggest in {"browser", "shell"}:
                    # Takeover request → always interrupt
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                elif not has_prior_soft_hint:
                    # First non-takeover ask → return SOFT_HINT
                    result_str = "SOFT_HINT"
                    logger.info("message_ask_user: returning SOFT_HINT (first attempt)")
                else:
                    # Second call after SOFT_HINT → truly needs user input
                    result_str = "WAITING_FOR_USER"
                    should_interrupt = True
                    logger.info("message_ask_user: user input required (after SOFT_HINT)")
            else:
                # ---- Normal tool execution ---- #
                tool_fn = tool_map.get(tool_name)
                if tool_fn is None:
                    result_str = f"Error: Unknown tool '{tool_name}'"
                    tool_success = False
                else:
                    try:
                        result_str = await tool_fn.ainvoke(args)
                        if not isinstance(result_str, str):
                            result_str = str(result_str)
                    except Exception as exc:
                        result_str = f"Error executing {tool_name}: {exc}"
                        tool_success = False

            if not tool_success:
                new_failures += 1

            # Prefix error messages so the LLM can clearly identify failures
            content = f"[TOOL_ERROR] {result_str}" if not tool_success else result_str

            new_messages.append(ToolMessage(
                content=content,
                tool_call_id=call_id,
                name=tool_name,
            ))

            # Emit ToolEvent(CALLED) with correct success status
            new_events.append(
                ToolEvent(
                    tool_call_id=call_id,
                    tool_name=_classify_tool_name(tool_name),
                    function_name=tool_name,
                    function_args=args,
                    function_result=ToolResult(success=tool_success, message=result_str),
                    status=ToolEventStatus.CALLED,
                )
            )

        result: dict = {
            "messages": new_messages,
            "events": new_events,
            "attempt_count": state["attempt_count"] + 1,
            "failure_count": state["failure_count"] + new_failures,
        }
        if should_interrupt:
            result["should_interrupt"] = True
        if not has_prior_soft_hint and any(
            m.content == "SOFT_HINT" and m.name == "message_ask_user"
            for m in new_messages
        ):
            result["soft_hint_sent"] = True
        return result

    # ---- Routing ------------------------------------------------------- #

    def route_after_llm(state: ReactGraphState) -> str:
        """Route after LLM call: tool calls → tool_node, else END."""
        if state.get("should_interrupt"):
            return END

        messages = state["messages"]
        if not messages:
            return END

        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tool_node"

        return END

    def route_after_tool(state: ReactGraphState) -> str:
        """Route after tool execution: back to LLM."""
        if state.get("should_interrupt"):
            return END
        if state.get("attempt_count", 0) >= MAX_ITERATIONS:
            return END
        return "llm_node"

    # ---- Build Graph --------------------------------------------------- #

    g: StateGraph = StateGraph(ReactGraphState)

    # RetryPolicy for transient LLM errors (ServerRequestsError → RuntimeError)
    llm_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=2.0,
        backoff_factor=2.0,
        retry_on=ServerRequestsError,
    )

    g.add_node("llm_node", llm_node, retry_policy=llm_retry)
    g.add_node("tool_node", tool_node)

    g.add_edge(START, "llm_node")
    g.add_conditional_edges("llm_node", route_after_llm)
    g.add_conditional_edges("tool_node", route_after_tool)

    return g.compile()
