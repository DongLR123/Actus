"""ActusFallbackChatModel — BaseChatModel with primary/fallback delegation.

Replaces ``primary.with_fallbacks([fallback])`` to avoid a langchain-core bug
where ``RunnableWithFallbacks.__getattr__`` calls ``typing.get_type_hints()``
on ``BaseChatModel.with_structured_output``, which fails because ``builtins``
is imported under ``TYPE_CHECKING`` only in langchain-core.

This model delegates to ``primary`` first; on any exception it retries with
``fallback``. Both ``bind_tools`` and ``with_structured_output`` propagate to
both inner models so that tool schemas stay consistent.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)


class ActusFallbackChatModel(BaseChatModel):
    """BaseChatModel that tries *primary* first and falls back to *fallback*.

    Both ``primary`` and ``fallback`` must be ``BaseChatModel`` instances.
    """

    primary: BaseChatModel
    fallback: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return "actus-fallback"

    # ---- sync (not used — project is async-only) ------------------------- #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async interface. Project is async-only.")

    # ---- async ----------------------------------------------------------- #

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await self.primary._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )
        except Exception as primary_exc:
            logger.warning(
                "Primary LLM (%s) failed, falling back: %s",
                self.primary._llm_type, primary_exc,
            )
            return await self.fallback._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        try:
            async for chunk in self.primary._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            ):
                yield chunk
        except Exception as primary_exc:
            logger.warning(
                "Primary LLM stream (%s) failed, falling back: %s",
                self.primary._llm_type, primary_exc,
            )
            async for chunk in self.fallback._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            ):
                yield chunk

    # ---- bind_tools / with_structured_output ----------------------------- #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusFallbackChatModel":
        return ActusFallbackChatModel(
            primary=self.primary.bind_tools(tools, **kwargs),
            fallback=self.fallback.bind_tools(tools, **kwargs),
        )
