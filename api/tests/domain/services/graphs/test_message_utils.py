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
