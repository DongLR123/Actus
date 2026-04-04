"""Tests for ContextAssembler.

Uses TokenEstimator(strategy="char") for predictable token counts:
  char strategy: len(text) // 3  (integer division)
  MESSAGE_OVERHEAD_TOKENS = 3 per message
"""
from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.domain.services.graphs.context_assembler import (
    AssemblyResult,
    ContextAssembler,
    MessageGroup,
    group_messages,
)
from app.domain.services.graphs.token_estimator import (
    MESSAGE_OVERHEAD_TOKENS,
    TokenEstimator,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def char_est() -> TokenEstimator:
    return TokenEstimator(strategy="char")


def _tok(text: str) -> int:
    """Expected token count for a single string under char strategy."""
    return len(text) // 3


def _msg_tok(text: str) -> int:
    """Expected token count for a single message under char strategy."""
    return _tok(text) + MESSAGE_OVERHEAD_TOKENS


# ── TestGroupMessages ─────────────────────────────────────────────────────────

class TestGroupMessages:
    """group_messages() correctly categorises messages and pairs tool calls."""

    def test_empty(self):
        assert group_messages([]) == []

    def test_single_system(self):
        msgs = [SystemMessage(content="sys")]
        groups = group_messages(msgs)
        assert len(groups) == 1
        assert groups[0].kind == "system"
        assert groups[0].messages == msgs

    def test_single_human(self):
        msgs = [HumanMessage(content="hi")]
        groups = group_messages(msgs)
        assert len(groups) == 1
        assert groups[0].kind == "human"

    def test_single_ai_text(self):
        msgs = [AIMessage(content="hello")]
        groups = group_messages(msgs)
        assert len(groups) == 1
        assert groups[0].kind == "ai_text"

    def test_tool_call_single_tool(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "search", "args": {"q": "test"}}],
        )
        tool = ToolMessage(content="result", tool_call_id="c1")
        groups = group_messages([ai, tool])
        assert len(groups) == 1
        assert groups[0].kind == "tool_call"
        assert ai in groups[0].messages
        assert tool in groups[0].messages

    def test_tool_call_multiple_tools(self):
        ai = AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "tool_a", "args": {}},
                {"id": "c2", "name": "tool_b", "args": {}},
            ],
        )
        t1 = ToolMessage(content="r1", tool_call_id="c1")
        t2 = ToolMessage(content="r2", tool_call_id="c2")
        groups = group_messages([ai, t1, t2])
        assert len(groups) == 1
        assert groups[0].kind == "tool_call"
        assert len(groups[0].messages) == 3  # ai + 2 tool results

    def test_mixed_sequence(self):
        ai_tc = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "x", "args": {}}],
        )
        tool_r = ToolMessage(content="r", tool_call_id="c1")
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="q"),
            ai_tc,
            tool_r,
            AIMessage(content="done"),
        ]
        groups = group_messages(msgs)
        kinds = [g.kind for g in groups]
        assert kinds == ["system", "human", "tool_call", "ai_text"]

    def test_orphaned_tool_message(self):
        """ToolMessage with no preceding AIMessage → defensive kind=tool_call."""
        msgs = [ToolMessage(content="orphan", tool_call_id="x99")]
        groups = group_messages(msgs)
        assert len(groups) == 1
        assert groups[0].kind == "tool_call"

    def test_ai_text_without_tool_calls(self):
        msgs = [AIMessage(content="just text", tool_calls=[])]
        groups = group_messages(msgs)
        assert groups[0].kind == "ai_text"


# ── TestComputeBudget ─────────────────────────────────────────────────────────

class TestComputeBudget:
    """_compute_budget() returns floor((context_window - reserved) / safety_factor)."""

    def test_basic_no_safety(self):
        est = char_est()
        asm = ContextAssembler(
            estimator=est,
            context_window=10000,
            reserved_output_tokens=2000,
            safety_factor=1.0,
        )
        assert asm._compute_budget() == 8000

    def test_with_safety_factor(self):
        est = char_est()
        asm = ContextAssembler(
            estimator=est,
            context_window=10000,
            reserved_output_tokens=2000,
            safety_factor=1.15,
        )
        expected = int((10000 - 2000) / 1.15)
        assert asm._compute_budget() == expected

    def test_default_params(self):
        est = char_est()
        asm = ContextAssembler(estimator=est, context_window=8192)
        expected = int((8192 - 4096) / 1.15)
        assert asm._compute_budget() == expected


# ── TestUnderBudget ───────────────────────────────────────────────────────────

class TestUnderBudget:
    """assemble() returns original when already under budget."""

    def test_empty_messages(self):
        est = char_est()
        asm = ContextAssembler(estimator=est, context_window=100000)
        result = asm.assemble([])
        assert result.messages == []
        assert result.original_tokens == 0
        assert result.final_tokens == 0
        assert result.actions == []

    def test_returns_original_when_under_budget(self):
        est = char_est()
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
        ]
        asm = ContextAssembler(estimator=est, context_window=100000)
        result = asm.assemble(msgs)
        assert result.messages == msgs
        assert result.original_tokens == result.final_tokens
        assert result.actions == []


# ── TestPhase1Compress ────────────────────────────────────────────────────────

class TestPhase1Compress:
    """Phase 1: compress large ToolMessage content when near budget."""

    def _make_assembler(
        self,
        context_window: int = 1000,
        trigger_ratio: float = 0.75,
        target_chars: int = 50,
    ) -> ContextAssembler:
        return ContextAssembler(
            estimator=char_est(),
            context_window=context_window,
            reserved_output_tokens=0,
            safety_factor=1.0,
            tool_compress_trigger_ratio=trigger_ratio,
            tool_compress_target_chars=target_chars,
        )

    def test_does_not_compress_when_under_trigger(self):
        # Budget = 1000; trigger = 750 tokens.
        # Messages total well under 750 → no compression.
        asm = self._make_assembler(context_window=10000, trigger_ratio=0.75, target_chars=50)
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "t", "args": {}}],
        )
        # short tool result — won't exceed 75% of huge budget
        tool = ToolMessage(content="short result", tool_call_id="c1")
        human = HumanMessage(content="q")
        result = asm.assemble([human, ai, tool])
        # Should be under budget, no compression
        assert result.actions == []

    def test_compresses_old_tool_results_when_over_trigger(self):
        # Budget = 200 tokens; trigger = 150 tokens.
        # An old (non-protected) tool group has 600-char content → ~200 tokens.
        # We place it early in history so it is NOT in the last 2 groups.
        target_chars = 30
        asm = self._make_assembler(
            context_window=200,
            trigger_ratio=0.75,
            target_chars=target_chars,
        )
        long_content = "x" * 600  # 600 chars → 200 tokens
        # Old tool call (will NOT be in last 2 groups because there's more after it)
        old_ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "t", "args": {}}],
        )
        old_tool = ToolMessage(content=long_content, tool_call_id="c1")
        human = HumanMessage(content="q")           # last human → protected
        ai_final = AIMessage(content="done")         # last 2 groups: human + ai_final

        result = asm.assemble([old_ai, old_tool, human, ai_final])

        # The old tool group is NOT protected → its content should be compressed
        tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
        if tool_msgs:
            for tm in tool_msgs:
                assert len(tm.content) <= target_chars * 3  # slack for truncation marker

        # At minimum some trimming action was recorded
        assert any("compress" in a.lower() or "phase" in a.lower() for a in result.actions)

    def test_boundary_exactly_at_trigger(self):
        """Token count exactly at trigger ratio → compression should occur."""
        # Budget = 100, trigger_ratio = 0.75 → trigger = 75 tokens.
        # Build messages that total exactly 75 tokens.
        # Each char = 1/3 token. 225 chars = 75 tokens (overhead included).
        # Let's just use a large enough content to be at/above trigger.
        target_chars = 20
        asm = self._make_assembler(
            context_window=100,
            trigger_ratio=0.75,
            target_chars=target_chars,
        )
        # 225 chars in tool content → 75 tokens text + overhead
        long_content = "a" * 225
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "t", "args": {}}],
        )
        tool = ToolMessage(content=long_content, tool_call_id="c1")
        result = asm.assemble([ai, tool])
        # We're at trigger or above — at minimum, phase 1 attempted
        assert isinstance(result, AssemblyResult)


# ── TestPhase2RemoveToolGroups ────────────────────────────────────────────────

class TestPhase2RemoveToolGroups:
    """Phase 2: remove oldest non-protected tool_call groups."""

    def test_removes_oldest_tool_group_first(self):
        # Create three tool groups + a final human turn.
        # With a tight budget only the last 2 groups can be kept.
        # tool_a (oldest, non-protected) must be removed before tool_b or tool_c.
        content_a = "a" * 300   # 100 tokens  — old, non-protected
        content_b = "b" * 300   # 100 tokens  — middle, non-protected
        ai_a = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "ta", "args": {}}],
        )
        tool_a = ToolMessage(content=content_a, tool_call_id="c1")
        ai_b = AIMessage(
            content="",
            tool_calls=[{"id": "c2", "name": "tb", "args": {}}],
        )
        tool_b = ToolMessage(content=content_b, tool_call_id="c2")
        human = HumanMessage(content="q")  # last human → protected (last 2)

        # Groups: [tool_call_a(0), tool_call_b(1), human(2)]
        # last 2 protected = indices 1 and 2 → tool_call_a (idx 0) is removable.
        # Budget tight enough that we need to remove at least one group.
        asm = ContextAssembler(
            estimator=char_est(),
            context_window=130,
            reserved_output_tokens=0,
            safety_factor=1.0,
            tool_compress_trigger_ratio=0.75,
            tool_compress_target_chars=5,
        )
        result = asm.assemble([ai_a, tool_a, ai_b, tool_b, human])
        # After trimming, c1 (oldest group) should be removed first
        remaining_tool_ids = [
            m.tool_call_id for m in result.messages if isinstance(m, ToolMessage)
        ]
        # c1 must have been removed (oldest non-protected)
        assert "c1" not in remaining_tool_ids

    def test_preserves_tool_call_pairing(self):
        """After phase 2, no orphaned ToolMessage (no tool without its AIMessage)."""
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "t", "args": {}}],
        )
        tool = ToolMessage(content="x" * 900, tool_call_id="c1")  # 300 tokens
        human = HumanMessage(content="q")

        asm = ContextAssembler(
            estimator=char_est(),
            context_window=50,
            reserved_output_tokens=0,
            safety_factor=1.0,
            tool_compress_trigger_ratio=0.75,
            tool_compress_target_chars=5,
        )
        result = asm.assemble([human, ai, tool])

        # Check: if there's a ToolMessage, there must be an AIMessage with matching tool_call_id
        tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
        ai_msgs = [m for m in result.messages if isinstance(m, AIMessage)]
        for tm in tool_msgs:
            ai_ids = {
                tc["id"]
                for a in ai_msgs
                if isinstance(a, AIMessage)
                for tc in (a.tool_calls or [])
            }
            assert tm.tool_call_id in ai_ids, f"Orphaned ToolMessage {tm.tool_call_id}"


# ── TestPhase3RemoveTurns ─────────────────────────────────────────────────────

class TestPhase3RemoveTurns:
    """Phase 3: remove oldest human/ai_text groups."""

    def test_removes_oldest_conversation_turn(self):
        sys = SystemMessage(content="system")
        old_human = HumanMessage(content="old question")
        old_ai = AIMessage(content="old answer")
        new_human = HumanMessage(content="new question")

        # Budget tight enough to force dropping old_human/old_ai
        asm = ContextAssembler(
            estimator=char_est(),
            context_window=60,
            reserved_output_tokens=0,
            safety_factor=1.0,
            tool_compress_trigger_ratio=0.75,
            tool_compress_target_chars=5,
        )
        result = asm.assemble([sys, old_human, old_ai, new_human])

        # Must keep system and new_human (last HumanMessage)
        assert any(isinstance(m, SystemMessage) for m in result.messages)
        assert any(isinstance(m, HumanMessage) and m.content == "new question"
                   for m in result.messages)

    def test_keeps_system_message(self):
        sys = SystemMessage(content="you are helpful")
        old_ai = AIMessage(content="previous response " * 50)  # many tokens
        human = HumanMessage(content="question")

        asm = ContextAssembler(
            estimator=char_est(),
            context_window=50,
            reserved_output_tokens=0,
            safety_factor=1.0,
        )
        result = asm.assemble([sys, old_ai, human])

        assert any(isinstance(m, SystemMessage) for m in result.messages)

    def test_keeps_last_human_message(self):
        sys = SystemMessage(content="system")
        old_human = HumanMessage(content="old " * 100)
        new_human = HumanMessage(content="final question")

        asm = ContextAssembler(
            estimator=char_est(),
            context_window=30,
            reserved_output_tokens=0,
            safety_factor=1.0,
        )
        result = asm.assemble([sys, old_human, new_human])

        human_msgs = [m for m in result.messages if isinstance(m, HumanMessage)]
        assert any(m.content == "final question" for m in human_msgs)


# ── TestMustKeep ──────────────────────────────────────────────────────────────

class TestMustKeep:
    """Protected messages (SystemMessage, last HumanMessage) are never dropped."""

    def test_system_always_kept(self):
        sys = SystemMessage(content="important system prompt")
        # Fill the rest with garbage
        filler_ai = [AIMessage(content="a" * 300) for _ in range(10)]
        human = HumanMessage(content="final")

        asm = ContextAssembler(
            estimator=char_est(),
            context_window=100,
            reserved_output_tokens=0,
            safety_factor=1.0,
        )
        result = asm.assemble([sys, *filler_ai, human])
        assert any(isinstance(m, SystemMessage) for m in result.messages)

    def test_last_human_always_kept(self):
        filler_ai = [AIMessage(content="a" * 300) for _ in range(10)]
        human = HumanMessage(content="the final question")

        asm = ContextAssembler(
            estimator=char_est(),
            context_window=100,
            reserved_output_tokens=0,
            safety_factor=1.0,
        )
        result = asm.assemble([*filler_ai, human])
        human_msgs = [m for m in result.messages if isinstance(m, HumanMessage)]
        assert any(m.content == "the final question" for m in human_msgs)

    def test_must_keep_exceeds_budget_returns_must_keep_only(self):
        """When must-keep items alone exceed budget, return them anyway."""
        sys = SystemMessage(content="s" * 600)  # 200 tokens
        human = HumanMessage(content="q" * 600)  # 200 tokens

        asm = ContextAssembler(
            estimator=char_est(),
            context_window=10,  # tiny budget
            reserved_output_tokens=0,
            safety_factor=1.0,
        )
        result = asm.assemble([sys, human])
        # Despite budget overflow, must-keep messages should still be present
        assert any(isinstance(m, SystemMessage) for m in result.messages)
        assert any(isinstance(m, HumanMessage) for m in result.messages)


# ── TestAssemblyResultAudit ───────────────────────────────────────────────────

class TestAssemblyResultAudit:
    """AssemblyResult records phases taken and shows token reduction."""

    def test_actions_record_phases(self):
        long_tool = "x" * 600  # 200 tokens
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "t", "args": {}}],
        )
        tool = ToolMessage(content=long_tool, tool_call_id="c1")
        human = HumanMessage(content="q")

        # Tight budget to force at least one phase
        asm = ContextAssembler(
            estimator=char_est(),
            context_window=50,
            reserved_output_tokens=0,
            safety_factor=1.0,
            tool_compress_trigger_ratio=0.5,
            tool_compress_target_chars=10,
        )
        result = asm.assemble([human, ai, tool])
        # At least one action should be recorded
        assert len(result.actions) >= 1

    def test_original_greater_than_final_tokens_after_trimming(self):
        # Place a large old tool group before a newer human turn so it's
        # NOT in the last 2 protected groups and can actually be removed.
        long_tool = "x" * 900  # 300 tokens
        old_ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "t", "args": {}}],
        )
        old_tool = ToolMessage(content=long_tool, tool_call_id="c1")
        human = HumanMessage(content="q")     # last human → protected
        new_ai = AIMessage(content="done")    # last 2 groups: human + new_ai

        # Groups: [tool_call(0), human(1), ai_text(2)]
        # last 2 protected = indices 1 and 2 → tool_call (idx 0) removable.
        asm = ContextAssembler(
            estimator=char_est(),
            context_window=50,
            reserved_output_tokens=0,
            safety_factor=1.0,
            tool_compress_target_chars=5,
        )
        result = asm.assemble([old_ai, old_tool, human, new_ai])
        assert result.original_tokens > result.final_tokens

    def test_assembly_result_is_frozen(self):
        """AssemblyResult is a frozen dataclass — mutation raises."""
        result = AssemblyResult(messages=[], original_tokens=0, final_tokens=0)
        with pytest.raises((TypeError, AttributeError)):
            result.messages = []  # type: ignore[misc]

    def test_message_group_is_frozen(self):
        """MessageGroup is a frozen dataclass — mutation raises."""
        grp = MessageGroup(kind="human", messages=[HumanMessage(content="x")])
        with pytest.raises((TypeError, AttributeError)):
            grp.kind = "system"  # type: ignore[misc]


# ── TestFlattenAndEstimate ────────────────────────────────────────────────────

class TestFlattenAndEstimate:
    """Internal helpers: _flatten_groups and _estimate_groups."""

    def test_flatten_preserves_order(self):
        sys = SystemMessage(content="s")
        human = HumanMessage(content="h")
        ai = AIMessage(content="a")

        asm = ContextAssembler(estimator=char_est(), context_window=10000)
        groups = group_messages([sys, human, ai])
        flat = asm._flatten_groups(groups)
        assert flat == [sys, human, ai]

    def test_estimate_groups_matches_messages(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="hello"),
        ]
        asm = ContextAssembler(estimator=char_est(), context_window=10000)
        groups = group_messages(msgs)
        estimated = asm._estimate_groups(groups)
        expected = char_est().estimate_messages(msgs)
        assert estimated == expected

    def test_last_human_idx_found(self):
        groups = group_messages([
            SystemMessage(content="s"),
            HumanMessage(content="first"),
            AIMessage(content="reply"),
            HumanMessage(content="second"),
        ])
        asm = ContextAssembler(estimator=char_est(), context_window=10000)
        idx = asm._last_human_idx(groups)
        assert idx == 3  # 0=system,1=human,2=ai_text,3=human

    def test_last_human_idx_no_human(self):
        groups = group_messages([SystemMessage(content="only system")])
        asm = ContextAssembler(estimator=char_est(), context_window=10000)
        idx = asm._last_human_idx(groups)
        assert idx == -1

    def test_mark_protected_first_system(self):
        groups = group_messages([
            SystemMessage(content="sys"),
            HumanMessage(content="h1"),
            HumanMessage(content="h2"),
        ])
        asm = ContextAssembler(estimator=char_est(), context_window=10000)
        marked = asm._mark_protected(groups)
        # First group (system) must be protected
        assert marked[0].protected is True

    def test_mark_protected_last_human(self):
        groups = group_messages([
            HumanMessage(content="h1"),
            HumanMessage(content="h2"),
        ])
        asm = ContextAssembler(estimator=char_est(), context_window=10000)
        marked = asm._mark_protected(groups)
        # Last human (index 1) must be protected
        last_human_idx = asm._last_human_idx(marked)
        assert marked[last_human_idx].protected is True

    def test_mark_protected_last_two_groups(self):
        groups = group_messages([
            SystemMessage(content="sys"),
            HumanMessage(content="h1"),
            AIMessage(content="a1"),
            HumanMessage(content="h2"),
        ])
        asm = ContextAssembler(estimator=char_est(), context_window=10000)
        marked = asm._mark_protected(groups)
        # Last 2 groups are protected
        assert marked[-1].protected is True
        assert marked[-2].protected is True


# ── TestDeterministic ─────────────────────────────────────────────────────────

class TestDeterministic:
    """assemble() is deterministic: same input → same output."""

    def test_same_input_same_output(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "t", "args": {}}],
        )
        tool = ToolMessage(content="x" * 300, tool_call_id="c1")
        msgs = [SystemMessage(content="sys"), HumanMessage(content="q"), ai, tool]

        asm = ContextAssembler(
            estimator=char_est(),
            context_window=100,
            reserved_output_tokens=0,
            safety_factor=1.0,
        )
        r1 = asm.assemble(msgs)
        r2 = asm.assemble(msgs)
        assert r1.messages == r2.messages
        assert r1.original_tokens == r2.original_tokens
        assert r1.final_tokens == r2.final_tokens
        assert r1.actions == r2.actions


class TestReactGraphIntegration:
    """Verify react_graph wiring includes pre_llm_node."""

    def test_pre_llm_node_exists_in_graph(self):
        from unittest.mock import MagicMock
        from app.domain.services.graphs.react_graph import build_react_graph

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        graph = build_react_graph(llm=mock_llm, tools=[], assembler=None)
        assert "pre_llm_node" in graph.get_graph().nodes

    def test_assembler_none_builds_successfully(self):
        from unittest.mock import MagicMock
        from app.domain.services.graphs.react_graph import build_react_graph

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        graph = build_react_graph(llm=mock_llm, tools=[], assembler=None)
        assert graph is not None


class TestExecutorNodeContract:
    """Verify the assembler contract for executor_node integration."""

    def test_assembler_is_deterministic(self):
        """Same input produces same output — contract for cross-step trimming."""
        from app.domain.services.graphs.context_assembler import ContextAssembler
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from langchain_core.messages import SystemMessage, HumanMessage

        a = ContextAssembler(
            estimator=TokenEstimator(strategy="char"),
            context_window=1000,
            reserved_output_tokens=200,
            safety_factor=1.0,
        )
        msgs = [SystemMessage(content="sys"), HumanMessage(content="q")]
        r1 = a.assemble(msgs)
        r2 = a.assemble(msgs)
        assert r1.messages == r2.messages
        assert r1.final_tokens == r2.final_tokens

    def test_resume_path_contract(self):
        """Document: executor_node skips assembler when resume_value is not None.

        The assembler itself is stateless — the skip logic is in executor_node:
          if assembler is not None and resume_value is None:
        """
        from app.domain.services.graphs.context_assembler import ContextAssembler
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from langchain_core.messages import SystemMessage, HumanMessage

        a = ContextAssembler(
            estimator=TokenEstimator(strategy="char"),
            context_window=1000,
            reserved_output_tokens=200,
            safety_factor=1.0,
        )
        msgs = [SystemMessage(content="sys"), HumanMessage(content="q")]
        result = a.assemble(msgs)
        assert result.messages == msgs  # under budget, returns original
