"""LangGraph state models for main_graph and react_graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from app.domain.models.plan import Plan, Step


class MainGraphState(TypedDict):
    """State for the outer orchestration graph (planner → executor → updater → summarizer)."""

    # Input
    message: str
    language: str
    attachments: list[str]

    # Planning
    plan: Plan | None
    current_step: Step | None

    # Execution
    messages: list[dict[str, Any]]  # LLM conversation history
    execution_summary: str

    # Events produced by nodes (accumulated across nodes)
    events: Annotated[list, operator.add]

    # Control
    flow_status: str  # idle | planning | executing | updating | summarizing | completed
    session_id: str
    should_interrupt: bool

    # Context
    original_request: str
    skill_context: str


class ReactGraphState(TypedDict):
    """State for the inner ReAct loop graph (LLM → tool → LLM → ...)."""

    # Conversation
    messages: list[dict[str, Any]]

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
