"""ActusChatModel — BaseChatModel wrapping OpenAI Chat Completions API.

Direct LangChain BaseChatModel implementation that calls the AsyncOpenAI
Chat Completions endpoint. Replaces the old LLM Protocol + LLMAdapter
indirection with a single unified class.

Implements:
- _generate: raises NotImplementedError (project is async-only)
- _agenerate: calls AsyncOpenAI chat.completions.create, returns ChatResult
- _astream: calls AsyncOpenAI with stream=True, yields ChatGenerationChunk
- bind_tools: returns new instance with tools bound via convert_to_openai_tool
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


class ActusChatModel(BaseChatModel):
    """BaseChatModel that wraps the OpenAI Chat Completions API directly.

    Configuration fields (Pydantic):
        base_url: OpenAI-compatible API base URL.
        api_key: API key for authentication.
        model_name: Model identifier (e.g. "gpt-4o", "deepseek-chat").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        supports_response_format: Whether the model supports response_format param.
    """

    # ---- Pydantic config fields ------------------------------------------ #

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 8192
    supports_response_format: bool = True

    # Tools bound via bind_tools() — None means no tools bound
    _bound_tools: Optional[list] = None

    # ---- Properties ------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        return "actus-chat"

    # ---- Client factory -------------------------------------------------- #

    def _get_client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client. Extracted as method for testability."""
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    # ---- Message conversion (private) ------------------------------------ #

    def _to_openai_messages(self, messages: List[BaseMessage]) -> list[dict]:
        """Convert LangChain BaseMessage list to OpenAI Chat dict format."""
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
                # Fallback for unknown message types
                result.append({"role": "user", "content": str(msg.content)})
        return result

    # ---- Response conversion --------------------------------------------- #

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[dict]:
        """Parse OpenAI tool_calls from response into LangChain format."""
        if not raw_tool_calls:
            return []

        tool_calls = []
        for tc in raw_tool_calls:
            # Handle both object and dict formats
            if hasattr(tc, "function"):
                fn = tc.function
                fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                fn_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
            else:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", "{}")
                tc_id = tc.get("id", "")

            # Deserialize JSON arguments
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            tool_calls.append({
                "id": tc_id,
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
        """Call AsyncOpenAI Chat Completions API and return ChatResult."""
        client = self._get_client()
        openai_messages = self._to_openai_messages(messages)

        # Build request params
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        }

        # Merge tools: bound tools + per-call tools from kwargs
        all_tools = list(self._bound_tools or [])
        extra_tools = kwargs.get("tools")
        if extra_tools:
            all_tools.extend(extra_tools)
        if all_tools:
            params["tools"] = all_tools

        # tool_choice from kwargs
        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        # response_format: only pass if supported and provided
        response_format = kwargs.get("response_format")
        if response_format is not None and self.supports_response_format:
            params["response_format"] = response_format

        # stop sequences
        if stop:
            params["stop"] = stop

        logger.info("ActusChatModel._agenerate: model=%s, tools=%d",
                     self.model_name, len(all_tools))

        response = await client.chat.completions.create(**params)

        # Extract message from response
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls = self._parse_tool_calls(message.tool_calls)

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
        """Call AsyncOpenAI Chat Completions with stream=True, yield ChatGenerationChunk."""
        client = self._get_client()
        openai_messages = self._to_openai_messages(messages)

        # Build request params
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
            "stream": True,
        }

        # Merge tools
        all_tools = list(self._bound_tools or [])
        extra_tools = kwargs.get("tools")
        if extra_tools:
            all_tools.extend(extra_tools)
        if all_tools:
            params["tools"] = all_tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        response_format = kwargs.get("response_format")
        if response_format is not None and self.supports_response_format:
            params["response_format"] = response_format

        if stop:
            params["stop"] = stop

        logger.info("ActusChatModel._astream: model=%s", self.model_name)

        response = client.chat.completions.create(**params)
        # AsyncOpenAI with stream=True returns an awaitable that resolves to
        # an async iterator. Some mocks may return an async generator directly.
        if hasattr(response, "__aiter__"):
            stream = response
        else:
            stream = await response

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Extract content
            content = delta.content or ""

            # Extract tool_call_chunks for streaming aggregation
            tool_call_chunks = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    fn = tc.function if hasattr(tc, "function") else None
                    tool_call_chunks.append({
                        "index": tc.index if hasattr(tc, "index") else 0,
                        "id": tc.id if hasattr(tc, "id") and tc.id else None,
                        "name": fn.name if fn and hasattr(fn, "name") and fn.name else None,
                        "args": fn.arguments if fn and hasattr(fn, "arguments") else "",
                    })

            ai_chunk = AIMessageChunk(
                content=content,
                tool_call_chunks=tool_call_chunks if tool_call_chunks else [],
            )
            gen_chunk = ChatGenerationChunk(message=ai_chunk)

            if run_manager:
                await run_manager.on_llm_new_token(content, chunk=gen_chunk)

            yield gen_chunk

    # ---- bind_tools ------------------------------------------------------ #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusChatModel":
        """Return a new ActusChatModel with tool schemas bound for LLM calls.

        Uses LangChain's convert_to_openai_tool to normalize tool definitions.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        converted = [convert_to_openai_tool(t) for t in tools]

        # Create a new instance with the same config but tools bound
        new_model = ActusChatModel(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            supports_response_format=self.supports_response_format,
        )
        new_model._bound_tools = converted
        return new_model
