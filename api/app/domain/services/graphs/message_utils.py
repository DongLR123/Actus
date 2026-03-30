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


_IMAGE_VISION_HINT = (
    '\n\n【多模态识图提示】上述附件中的图片已直接嵌入本消息中，你可以直接看到图片内容。'
    '请基于你直接看到的图片进行分析，无需使用 file_read、浏览器或其他工具来查看或打开图片。'
)


def format_attachments_text(
    attachments: list[str],
    has_image_blocks: bool = False,
) -> str:
    """Format attachments list as prompt text, with conditional image vision hint.

    When image blocks are present, appends a hint telling the LLM that images
    are directly embedded in the message and it should analyze them visually
    instead of trying to open them with tools.
    """
    text = ", ".join(attachments) if attachments else "无"
    if has_image_blocks:
        text += _IMAGE_VISION_HINT
    return text


def build_multimodal_content(
    text: str, image_blocks: list[dict] | None = None,
) -> str | list[dict]:
    """Build HumanMessage content, optionally with image content blocks.

    Returns plain text string if no image blocks, or a list of content blocks
    (OpenAI Chat Completions multimodal format) when images are present.
    """
    if not image_blocks:
        return text
    return [{"type": "text", "text": text}] + list(image_blocks)


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


def _flatten_multimodal_content(content: Any) -> str:
    """Extract text from multimodal content blocks for Memory persistence.

    When HumanMessage.content is a list[dict] (multimodal), extract only
    the text blocks and discard image blocks to avoid storing large base64
    payloads in Memory/database.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts) if text_parts else str(content)
    return str(content)


def messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain BaseMessage list to Actus dict format (for Memory/raw-LLM).

    Note: Multimodal content (image blocks) in HumanMessage is flattened to
    text-only to avoid persisting large base64 image data in Memory/database.
    """
    dicts: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            dicts.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            dicts.append({"role": "user", "content": _flatten_multimodal_content(msg.content)})
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


def truncate_tool_content(content: str, max_chars: int = 8000) -> str:
    """对超长工具结果执行 head+tail 截断，保证返回长度 <= max_chars。

    当 len(content) <= max_chars 时原样返回。
    超限时先扣除截断标记开销，再将剩余预算均分给 head 和 tail。
    """
    if len(content) <= max_chars:
        return content
    # 用 len(content) 作为被截断字符数的上界，确保标记位数足够
    marker_overhead = len(f"\n...(已截断 {len(content)} 字符)\n")
    budget = max(0, max_chars - marker_overhead)
    head = budget // 2
    tail = budget - head
    removed = len(content) - head - tail
    return f"{content[:head]}\n...(已截断 {removed} 字符)\n{content[-tail:]}"
