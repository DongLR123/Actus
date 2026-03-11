from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.plan import Plan, Step


class InterruptState(BaseModel):
    """中断恢复状态 — 当 WaitEvent 触发 should_interrupt 时持久化到 DB。

    存储在 session.memories["_interrupt_state"]，跨 AgentTaskRunner 实例恢复。
    """

    model_config = ConfigDict(extra="ignore")

    plan: Plan | None = None
    current_step: Step | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    original_request: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
