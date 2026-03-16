from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.models.llm_responses import ContinuationIntent
from app.domain.services.prompts.continuation_classifier import (
    CONTINUATION_CLASSIFIER_SYSTEM_PROMPT,
    CONTINUATION_CLASSIFIER_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class ContinuationIntentClassifier:
    """基于LLM的续写意图二分类器。"""

    def __init__(
        self,
        llm: BaseChatModel,
        timeout_seconds: float = 3.0,
    ) -> None:
        self._llm = llm
        self._timeout_seconds = max(0.1, float(timeout_seconds))

    async def classify(
        self,
        current_message: str,
        previous_substantive_message: str,
    ) -> bool:
        if not current_message or not previous_substantive_message:
            return False

        messages = [
            SystemMessage(content=CONTINUATION_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=CONTINUATION_CLASSIFIER_USER_PROMPT.format(
                    previous_substantive_message=previous_substantive_message,
                    current_message=current_message,
                )
            ),
        ]

        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._llm.ainvoke(
                    messages,
                    response_format={"type": "json_object"},
                    tool_choice="none",
                )
        except Exception as exc:
            logger.warning("续写意图LLM判定失败，回退false: %s", str(exc))
            return False

        try:
            parsed = ContinuationIntent.model_validate_json(response.content)
        except Exception as exc:
            logger.warning("续写意图JSON解析失败，回退false: %s", str(exc))
            return False

        return parsed.is_continuation
