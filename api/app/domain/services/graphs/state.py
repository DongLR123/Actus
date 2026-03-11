"""LangGraph state models for main_graph and react_graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from app.domain.models.plan import Plan, Step
from app.domain.services.flows.base import FlowStatus


class MainGraphState(TypedDict):
    """State for the outer orchestration graph (planner → executor → updater → summarizer).

    ``messages`` stores LangChain BaseMessage objects (overwrite semantics — each
    executor_node returns the full conversation so far). The type annotation is
    ``list`` to avoid import-time dependency on langchain_core.
    """

    # Input
    message: str
    language: str
    attachments: list[str]

    # Planning
    plan: Plan | None
    current_step: Step | None

    # Execution — LangChain BaseMessage list (overwrite, not append)
    messages: list
    execution_summary: str

    # Events produced by nodes (accumulated across nodes)
    events: Annotated[list, operator.add]

    # Control
    flow_status: str  # FlowStatus.value — typed routing via FlowStatus enum
    session_id: str
    should_interrupt: bool
    is_resuming: bool

    # Context
    original_request: str
    skill_context: str
    conversation_summaries: list[str]  # 历史对话摘要文本（to_prompt_text() 输出）


class ReactGraphState(TypedDict):
    """State for the inner ReAct loop graph (LLM → tool → LLM → ...).

    ``messages`` stores LangChain BaseMessage objects (overwrite semantics —
    each node returns the full list including new messages).
    """

    # Conversation — LangChain BaseMessage list
    messages: list

    # Step context
    step_description: str
    original_request: str
    language: str
    attachments: list[str]

    # Events produced by nodes (accumulated)
    events: Annotated[list, operator.add]

    # Control
    should_interrupt: bool

    # Gating counters (for message_ask_user soft-hint throttling)
    attempt_count: int
    failure_count: int
