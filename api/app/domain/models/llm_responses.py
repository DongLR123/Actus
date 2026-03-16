# api/app/domain/models/llm_responses.py
"""LLM 结构化输出的 Pydantic Response Models。

用于 with_structured_output(Model) 调用，所有字段使用宽松默认值
以容忍 LLM 返回部分字段。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StepDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = ""
    description: str = ""


class PlanResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str = ""
    goal: str = ""
    title: str = ""
    language: str = "zh"
    steps: list[StepDef] = []


class PlanUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    steps: list[StepDef] = []


class SummarizerOutput(BaseModel):
    """summarizer_node 的 LLM 返回（用户最终交付：message + attachments）。"""
    model_config = ConfigDict(extra="ignore")
    message: str = ""
    attachments: list[str] = []


class ConversationSummaryResponse(BaseModel):
    """_generate_summary 的 LLM 返回（ConversationSummary 持久化）。"""
    model_config = ConfigDict(extra="ignore")
    user_intent: str = ""
    plan_summary: str = ""
    execution_results: list[str] = []
    decisions: list[str] = []
    unresolved: list[str] = []


class ContinuationIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    is_continuation: bool = False
