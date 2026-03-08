from datetime import datetime

from app.domain.models.conversation_summary import ConversationSummary


def test_conversation_summary_creation() -> None:
    summary = ConversationSummary(
        round_number=1,
        user_intent="分析CSV文件的销售趋势",
        plan_summary="读取文件并生成折线图",
        execution_results=["成功读取文件", "生成了折线图"],
        decisions=["选择按月聚合数据"],
        unresolved=[],
    )

    assert summary.round_number == 1
    assert summary.user_intent == "分析CSV文件的销售趋势"
    assert len(summary.execution_results) == 2
    assert isinstance(summary.timestamp, datetime)


def test_conversation_summary_serialization() -> None:
    summary = ConversationSummary(
        round_number=1,
        user_intent="test",
        plan_summary="plan",
        execution_results=["result1"],
        decisions=[],
        unresolved=["pending item"],
    )

    data = summary.model_dump(mode="json")
    restored = ConversationSummary.model_validate(data)

    assert restored.round_number == summary.round_number
    assert restored.user_intent == summary.user_intent
    assert restored.unresolved == ["pending item"]


def test_conversation_summary_to_prompt_text() -> None:
    summary = ConversationSummary(
        round_number=2,
        user_intent="加上同比增长率",
        plan_summary="修改图表",
        execution_results=["已添加同比列"],
        decisions=["使用百分比格式"],
        unresolved=[],
    )

    text = summary.to_prompt_text()

    assert "第2轮" in text
    assert "加上同比增长率" in text
    assert "已添加同比列" in text
    assert "使用百分比格式" in text
