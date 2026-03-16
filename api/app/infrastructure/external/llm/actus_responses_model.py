"""ActusResponsesModel -- BaseChatModel wrapping OpenAI Responses API.

Direct LangChain BaseChatModel implementation that calls the AsyncOpenAI
Responses endpoint (client.responses.create). Replaces the old LLM Protocol +
OpenAIResponsesLLM indirection with a single unified class.

Key differences from ActusChatModel (Chat Completions):
- Uses ``responses.create()`` instead of ``chat.completions.create()``
- ``max_output_tokens`` instead of ``max_tokens``
- ``input`` param instead of ``messages``
- ``response_format`` -> ``text.format`` mapping
- Function_call items instead of tool_calls in messages
- Needs JSON schema sanitization (array items:{})

Implements:
- _generate: raises NotImplementedError (project is async-only)
- _agenerate: calls Responses API, returns ChatResult with AIMessage
- _astream: fallback to _agenerate (yields single ChatGenerationChunk)
- bind_tools: returns new instance with tools bound, tools converted via _convert_tools
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ActusResponsesModel(BaseChatModel):
    """BaseChatModel that wraps the OpenAI Responses API directly.

    Configuration fields (Pydantic):
        base_url: OpenAI-compatible API base URL.
        api_key: API key for authentication.
        model_name: Model identifier (e.g. "gpt-5.4-pro").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate (mapped to max_output_tokens internally).
    """

    # ---- Pydantic config fields ------------------------------------------ #

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-5.4-pro"
    temperature: float = 0.7
    max_tokens: int = 8192

    # Tools bound via bind_tools() -- None means no tools bound
    _bound_tools: Optional[list] = None

    # ---- Properties ------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        return "actus-responses"

    # ---- Client factory -------------------------------------------------- #

    def _get_client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client. Extracted as method for testability."""
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    # ------------------------------------------------------------------
    # JSON Schema sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_declares_array(schema_type: Any) -> bool:
        """Check if a schema type declares an array (string or list form)."""
        if schema_type == "array":
            return True
        if isinstance(schema_type, list):
            return "array" in schema_type
        return False

    @classmethod
    def _sanitize_json_schema(cls, schema: Any, path: str = "root") -> Any:
        """Recursively fix incomplete JSON Schema for the stricter Responses API.

        Adds missing ``"items": {}`` for array-type properties.
        """
        if isinstance(schema, list):
            return [cls._sanitize_json_schema(item, f"{path}[]") for item in schema]

        if not isinstance(schema, dict):
            return schema

        sanitized = {
            key: cls._sanitize_json_schema(value, f"{path}.{key}")
            for key, value in schema.items()
        }

        if cls._schema_declares_array(sanitized.get("type")) and "items" not in sanitized:
            logger.warning("检测到数组 schema 缺少 items，已自动补齐: %s", path)
            sanitized["items"] = {}

        return sanitized

    # ------------------------------------------------------------------
    # Tool format conversion: Chat Completions -> Responses API
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Convert Chat Completions tool format to Responses API format.

        Chat Completions: {"type": "function", "function": {"name": ..., "parameters": ...}}
        Responses API:    {"type": "function", "name": ..., "parameters": ..., "strict": False}
        """
        converted: List[dict[str, Any]] = []
        for tool in tools:
            func = tool.get("function", {})
            parameters = ActusResponsesModel._sanitize_json_schema(func.get("parameters", {}))
            converted.append({
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": parameters,
                "strict": False,
            })
        return converted

    # ------------------------------------------------------------------
    # Message conversion: LangChain BaseMessage -> Responses API input
    # ------------------------------------------------------------------

    def _to_openai_messages(self, messages: List[BaseMessage]) -> list[dict]:
        """Convert LangChain BaseMessage list to OpenAI Chat dict format.

        Intermediate step before _convert_input_messages transforms them
        to Responses API input items.
        """
        result: list[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    entry["tool_calls"] = [
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
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                })
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    @staticmethod
    def _convert_input_messages_from_dicts(messages: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Convert Chat-style message dicts to Responses API input items.

        Handles:
        - role "tool" -> type "function_call_output"
        - assistant tool_calls -> "function_call" items
        - Other messages pass through unchanged
        """
        converted: List[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")

            if role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                })
                continue

            if role == "assistant" and message.get("tool_calls"):
                content = message.get("content")
                if content is not None and content != "":
                    converted.append({
                        "role": "assistant",
                        "content": content,
                    })

                for tool_call in message.get("tool_calls", []):
                    function = tool_call.get("function", {})
                    converted.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    })
                continue

            converted.append(message)

        return converted

    def _convert_input_messages(self, messages: List[BaseMessage]) -> List[dict[str, Any]]:
        """Convert LangChain BaseMessage list to Responses API input items.

        Two-step process:
        1. Convert BaseMessage -> Chat Completions dict format
        2. Convert Chat dicts -> Responses API input items
        """
        chat_dicts = self._to_openai_messages(messages)
        return self._convert_input_messages_from_dicts(chat_dicts)

    # ------------------------------------------------------------------
    # Response normalization: Responses API output -> message dict
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_response(response: Any) -> dict[str, Any]:
        """Normalize Responses API response to Chat Completions-compatible message dict.

        Responses API output array may contain:
        - type="message": text message (content contains type="output_text" items)
        - type="function_call": tool call
        """
        dumped = response.model_dump() if hasattr(response, "model_dump") else response
        output_items = dumped.get("output", [])

        content_text = ""
        tool_calls: List[dict[str, Any]] = []

        for item in output_items:
            item_type = item.get("type")

            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content_text += part.get("text", "")

            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    # ------------------------------------------------------------------
    # Parse tool calls from normalized response into LangChain format
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[dict]:
        """Parse tool_calls from normalized response into LangChain format."""
        if not raw_tool_calls:
            return []

        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", "{}")

            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            tool_calls.append({
                "id": tc.get("id", ""),
                "name": fn_name,
                "args": fn_args,
            })
        return tool_calls

    # ---- LangChain interface: _generate (sync) --------------------------- #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async interface (_agenerate). Project is async-only.")

    # ---- LangChain interface: _agenerate (async) ------------------------- #

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call AsyncOpenAI Responses API and return ChatResult."""
        client = self._get_client()
        input_items = self._convert_input_messages(messages)

        # Build request params
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "input": input_items,
        }

        # response_format -> text.format mapping
        response_format = kwargs.get("response_format")
        if response_format is not None:
            params["text"] = {"format": response_format}

        # Merge tools: bound tools + per-call tools from kwargs
        all_tools = list(self._bound_tools or [])
        extra_tools = kwargs.get("tools")
        if extra_tools:
            # Per-call tools come in Chat Completions format, convert them
            all_tools.extend(self._convert_tools(extra_tools))
        if all_tools:
            params["tools"] = all_tools
            logger.info("调用Responses API并携带工具信息: %s", self.model_name)
        else:
            logger.info("调用Responses API未携带工具: %s", self.model_name)

        # tool_choice from kwargs
        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        logger.info("ActusResponsesModel._agenerate: model=%s, tools=%d",
                     self.model_name, len(all_tools))

        response = await client.responses.create(**params)

        # Normalize Responses API output to Chat Completions-compatible dict
        normalized = self._normalize_response(response)
        content = normalized.get("content") or ""
        tool_calls = self._parse_tool_calls(normalized.get("tool_calls"))

        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ---- LangChain interface: _astream (async streaming) ----------------- #

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Streaming fallback: calls _agenerate and yields a single chunk.

        The Responses API does not use the same streaming interface as
        Chat Completions. This method provides compatibility by wrapping
        the non-streaming result as a single ChatGenerationChunk.
        """
        result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        msg = result.generations[0].message

        ai_chunk = AIMessageChunk(
            content=msg.content,
            tool_call_chunks=[
                {
                    "index": i,
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                }
                for i, tc in enumerate(msg.tool_calls)
            ] if msg.tool_calls else [],
        )
        gen_chunk = ChatGenerationChunk(message=ai_chunk)

        if run_manager:
            await run_manager.on_llm_new_token(msg.content, chunk=gen_chunk)

        yield gen_chunk

    # ---- bind_tools ------------------------------------------------------ #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusResponsesModel":
        """Return a new ActusResponsesModel with tool schemas bound for LLM calls.

        Uses LangChain's convert_to_openai_tool to normalize tool definitions,
        then converts from Chat Completions format to Responses API format.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        # First convert to standard OpenAI Chat Completions format
        chat_format = [convert_to_openai_tool(t) for t in tools]
        # Then convert to Responses API format
        responses_format = self._convert_tools(chat_format)

        # Create a new instance with the same config but tools bound
        new_model = ActusResponsesModel(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        new_model._bound_tools = responses_format
        return new_model
