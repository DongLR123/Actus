"""Message conversion utilities between LangChain BaseMessage and Actus dict format.

Used at the boundary between LangGraph (BaseMessage) and Memory/raw-LLM (dict).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def dicts_to_messages(dicts: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert Actus dict messages to LangChain BaseMessage list."""
    messages: list[BaseMessage] = []
    for d in dicts:
        role = d.get("role", "user")
        content = d.get("content", "")

        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tool_calls_raw = d.get("tool_calls") or []
            tool_calls = []
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "args": args,
                })
            messages.append(AIMessage(
                content=content,
                tool_calls=tool_calls if tool_calls else [],
            ))
        elif role == "tool":
            messages.append(ToolMessage(
                content=content,
                tool_call_id=d.get("tool_call_id", ""),
                name=d.get("function_name", ""),
            ))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def dedup_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Deduplicate messages by ID — later message with same ID replaces earlier.

    Replicates langgraph.graph.message.add_messages dedup semantics.
    Messages without an id (or id=None) are always appended without dedup.
    """
    seen: dict[str, int] = {}  # id -> index in result
    result: list[BaseMessage] = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if msg_id and msg_id in seen:
            result[seen[msg_id]] = msg  # replace
        else:
            if msg_id:
                seen[msg_id] = len(result)
            result.append(msg)
    return result


def messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain BaseMessage list to Actus dict format (for Memory/raw-LLM)."""
    dicts: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            dicts.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            dicts.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            d: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                            if isinstance(tc["args"], dict)
                            else tc["args"],
                        },
                    }
                    for tc in msg.tool_calls
                ]
            dicts.append(d)
        elif isinstance(msg, ToolMessage):
            dicts.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
                "function_name": msg.name or "",
            })
        else:
            dicts.append({"role": "user", "content": str(msg.content)})
    return dicts
