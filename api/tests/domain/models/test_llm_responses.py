# api/tests/domain/models/test_llm_responses.py
"""LLM response model 容错性测试。"""
import pytest
from app.domain.models.llm_responses import (
    StepDef, PlanResponse, PlanUpdateResponse,
    SummarizerOutput, ConversationSummaryResponse,
    ContinuationIntent,
)


class TestPlanResponse:
    def test_full_fields(self):
        data = {
            "message": "msg", "goal": "g", "title": "t",
            "language": "en", "steps": [{"id": "1", "description": "do"}],
        }
        r = PlanResponse.model_validate(data)
        assert r.title == "t"
        assert len(r.steps) == 1

    def test_empty_input(self):
        """LLM 返回空 JSON 不应抛异常。"""
        r = PlanResponse.model_validate({})
        assert r.steps == []
        assert r.message == ""

    def test_extra_fields_ignored(self):
        r = PlanResponse.model_validate({"extra": 1, "title": "t"})
        assert r.title == "t"


class TestSummarizerOutput:
    def test_message_and_attachments(self):
        r = SummarizerOutput.model_validate({"message": "done", "attachments": ["/f.md"]})
        assert r.message == "done"
        assert r.attachments == ["/f.md"]

    def test_empty(self):
        r = SummarizerOutput.model_validate({})
        assert r.message == ""
        assert r.attachments == []


class TestConversationSummaryResponse:
    def test_partial_fields(self):
        r = ConversationSummaryResponse.model_validate({"user_intent": "x"})
        assert r.user_intent == "x"
        assert r.decisions == []


class TestContinuationIntent:
    def test_defaults_to_false(self):
        r = ContinuationIntent.model_validate({})
        assert r.is_continuation is False
