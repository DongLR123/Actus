"""Tests for message_utils helper functions."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class TestDedupMessages:
    def test_same_id_replaced_by_later_message(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        msg1 = HumanMessage(content="first", id="msg-1")
        msg2 = HumanMessage(content="updated", id="msg-1")
        result = dedup_messages([msg1, msg2])

        assert len(result) == 1
        assert result[0].content == "updated"

    def test_no_id_messages_appended(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        sys = SystemMessage(content="system")
        human = HumanMessage(content="hello")
        result = dedup_messages([sys, human])

        assert len(result) == 2
        assert result[0].content == "system"
        assert result[1].content == "hello"

    def test_empty_list(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        result = dedup_messages([])
        assert result == []

    def test_none_id_messages_always_appended(self):
        """Messages with id=None are never deduped (LangChain auto-generates UUIDs,
        so id=None must be set explicitly to trigger this path)."""
        from app.domain.services.graphs.message_utils import dedup_messages

        sys = SystemMessage(content="sys", id=None)
        h1 = HumanMessage(content="v1", id="h-1")
        ai = AIMessage(content="response", id="ai-1")
        h2 = HumanMessage(content="v2", id="h-1")  # replaces h1
        result = dedup_messages([sys, h1, ai, h2])

        assert len(result) == 3
        assert result[0].content == "sys"
        assert result[1].content == "v2"  # replaced
        assert result[2].content == "response"

    def test_preserves_order(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        msgs = [
            HumanMessage(content="a", id="1"),
            AIMessage(content="b", id="2"),
            HumanMessage(content="c", id="3"),
        ]
        result = dedup_messages(msgs)

        assert len(result) == 3
        assert [m.content for m in result] == ["a", "b", "c"]


class TestTruncateToolContent:
    def test_no_truncation_within_limit(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "a" * 8000
        result = truncate_tool_content(content, max_chars=8000)
        assert result == content

    def test_truncation_head_tail(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "H" * 5000 + "M" * 2000 + "T" * 5000  # 12000 chars
        result = truncate_tool_content(content, max_chars=8000)
        assert result.startswith("H")
        assert result.endswith("T")
        assert "已截断" in result
        assert len(result) <= 8000

    def test_boundary_exact_limit(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "x" * 8000
        result = truncate_tool_content(content, max_chars=8000)
        assert result == content

    def test_truncation_with_small_threshold(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "A" * 1000 + "B" * 2000 + "C" * 1000  # 4000 chars
        result = truncate_tool_content(content, max_chars=2000)
        assert result.startswith("A")
        assert result.endswith("C")
        assert len(result) <= 2000

    def test_truncation_marker_contains_char_count(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "x" * 10000
        result = truncate_tool_content(content, max_chars=8000)
        assert "已截断" in result
        assert len(result) <= 8000

    def test_result_never_exceeds_max_chars(self):
        """Strict guarantee: result length <= max_chars for all inputs."""
        from app.domain.services.graphs.message_utils import truncate_tool_content

        for max_chars in [50, 100, 200, 500, 2000, 8000]:
            for input_len in [max_chars + 1, max_chars * 2, max_chars * 10]:
                content = "x" * input_len
                result = truncate_tool_content(content, max_chars=max_chars)
                assert len(result) <= max_chars, (
                    f"max_chars={max_chars}, input={input_len}, result={len(result)}"
                )


class TestToolResultMaxCharsConfig:
    """Verify tool_result_max_chars config field defaults and projection."""

    def test_llm_config_default(self):
        from app.domain.models.app_config import LLMConfig
        config = LLMConfig()
        assert config.tool_result_max_chars == 8000

    def test_overflow_config_default(self):
        from app.domain.models.context_overflow_config import ContextOverflowConfig
        config = ContextOverflowConfig()
        assert config.tool_result_max_chars == 8000

    def test_from_llm_config_projection(self):
        from app.domain.models.app_config import LLMConfig
        from app.domain.models.context_overflow_config import ContextOverflowConfig
        llm = LLMConfig(tool_result_max_chars=5000)
        overflow = ContextOverflowConfig.from_llm_config(llm)
        assert overflow.tool_result_max_chars == 5000

    def test_from_llm_config_default_projection(self):
        from app.domain.models.app_config import LLMConfig
        from app.domain.models.context_overflow_config import ContextOverflowConfig
        llm = LLMConfig()
        overflow = ContextOverflowConfig.from_llm_config(llm)
        assert overflow.tool_result_max_chars == 8000
