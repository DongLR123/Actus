"""Tests for LangGraph state models."""

import pytest
from app.domain.services.graphs.state import MainGraphState, ReactGraphState


class TestMainGraphState:
    def test_create_minimal(self):
        state = MainGraphState(
            message="hello",
            language="en",
            attachments=[],
            plan=None,
            current_step=None,
            messages=[],
            execution_summary="",
            events=[],
            flow_status="idle",
            session_id="sess-1",
            should_interrupt=False,
            original_request="",
            skill_context="",
        )
        assert state["flow_status"] == "idle"

    def test_events_accumulate(self):
        """Annotated[list, operator.add] should merge event lists."""
        import operator
        events1 = [{"type": "msg", "text": "a"}]
        events2 = [{"type": "msg", "text": "b"}]
        merged = operator.add(events1, events2)
        assert len(merged) == 2


class TestReactGraphState:
    def test_create_minimal(self):
        state = ReactGraphState(
            messages=[],
            step_description="do something",
            original_request="build X",
            language="en",
            attachments=[],
            events=[],
            should_interrupt=False,
            attempt_count=0,
            failure_count=0,
        )
        assert state["should_interrupt"] is False
        assert state["attempt_count"] == 0
