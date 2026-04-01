"""Token estimator with char/hybrid/provider_api strategies.

Activated by ContextOverflowConfig.token_estimator (previously unused).
See spec: docs/superpowers/specs/2026-03-30-token-estimator-upgrade-design.md
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# --- hybrid strategy parameters ---
ASCII_TOKENS_PER_CHAR = 0.25
CJK_TOKENS_PER_CHAR = 1.5
OTHER_NON_ASCII_TOKENS_PER_CHAR = 1.0

# --- char strategy parameters ---
CHARS_PER_TOKEN_SIMPLE = 3

# --- message structure overhead ---
MESSAGE_OVERHEAD_TOKENS = 3
IMAGE_TOKEN_ESTIMATE = 2000

# --- CJK Unicode ranges ---
_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x2E80, 0x2EFF),    # CJK Radicals Supplement
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
)


def _is_cjk(cp: int) -> bool:
    """Check if a Unicode code point is in CJK range."""
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


class TokenEstimator:
    """Token estimator supporting char/hybrid/provider_api strategies.

    Created once per PlannerReActFlow from ContextOverflowConfig.
    Thread-safe after __init__ (no mutable state).
    """

    def __init__(
        self,
        strategy: Literal["char", "hybrid", "provider_api"] = "hybrid",
        model_name: str = "",
    ) -> None:
        self._requested_strategy = strategy
        self._model_name = model_name
        self._encoding = None

        if strategy == "provider_api":
            self._encoding = self._try_load_tiktoken(model_name)
            if self._encoding is None:
                self._strategy = "hybrid"
            else:
                self._strategy = "provider_api"
        else:
            self._strategy = strategy

        logger.info(
            "TokenEstimator: requested=%s, effective=%s, model=%s",
            self._requested_strategy,
            self._strategy,
            self._model_name,
        )

    @property
    def effective_strategy(self) -> str:
        """Return the strategy actually in use (after potential fallback)."""
        return self._strategy

    def estimate(self, text: str) -> int:
        """Estimate token count for a text string."""
        if not text:
            return 0

        if self._strategy == "char":
            return len(text) // CHARS_PER_TOKEN_SIMPLE

        if self._strategy == "provider_api" and self._encoding is not None:
            return len(self._encoding.encode(text))

        # hybrid (default, also fallback for provider_api)
        total = 0.0
        for ch in text:
            cp = ord(ch)
            if cp <= 127:
                total += ASCII_TOKENS_PER_CHAR
            elif _is_cjk(cp):
                total += CJK_TOKENS_PER_CHAR
            else:
                total += OTHER_NON_ASCII_TOKENS_PER_CHAR
        return max(round(total), 1)

    def estimate_message(self, msg: BaseMessage) -> int:
        """Estimate token count for a single LangChain message."""
        from langchain_core.messages import AIMessage

        tokens = MESSAGE_OVERHEAD_TOKENS
        content = msg.content

        if isinstance(content, str):
            tokens += self.estimate(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image_url":
                        tokens += IMAGE_TOKEN_ESTIMATE
                    elif block.get("type") == "text":
                        tokens += self.estimate(block.get("text", ""))
                    else:
                        tokens += self.estimate(str(block))
                else:
                    tokens += self.estimate(str(block))
        else:
            tokens += self.estimate(str(content))

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tokens += self.estimate(str(tc.get("args", {})))

        return tokens

    def estimate_messages(self, msgs: Sequence[BaseMessage]) -> int:
        """Estimate total token count for a list of messages."""
        return sum(self.estimate_message(m) for m in msgs)

    @staticmethod
    def _try_load_tiktoken(model_name: str):
        """Attempt to load tiktoken encoding for model. Returns None on failure."""
        try:
            import tiktoken
        except ImportError:
            logger.info("tiktoken not installed, provider_api falls back to hybrid")
            return None
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.info(
                "tiktoken has no encoding for model '%s', falling back to hybrid",
                model_name,
            )
            return None
