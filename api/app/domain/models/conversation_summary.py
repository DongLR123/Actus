from datetime import datetime

from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    """对话摘要 - 每轮对话结束后生成"""

    round_number: int
    user_intent: str
    plan_summary: str
    execution_results: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_prompt_text(self) -> str:
        """将摘要转为可注入 system prompt 的文本"""
        lines = [f"### 第{self.round_number}轮"]
        lines.append(f"- 用户需求：{self.user_intent}")
        if self.execution_results:
            lines.append(f"- 执行结果：{'；'.join(self.execution_results)}")
        if self.decisions:
            lines.append(f"- 关键决策：{'；'.join(self.decisions)}")
        if self.unresolved:
            lines.append(f"- 未解决：{'；'.join(self.unresolved)}")
        return "\n".join(lines)
