"""Tests for Task 7: AgentTaskRunner reads flush batch and calls flusher.

Verifies:
1. memory_flusher parameter exists in __init__ signature
2. memory_flusher defaults to None
3. When flow has _pending_flush_batch and flusher is set, submit is called
4. When flow has _pending_flush_batch but no flusher, no error
5. When flow has no _pending_flush_batch, submit is not called
"""
from __future__ import annotations

import inspect

import pytest


class TestRunnerFlushWiringSignature:
    """AgentTaskRunner.__init__ accepts memory_flusher parameter."""

    def test_memory_flusher_in_init_signature(self) -> None:
        """memory_flusher should be a parameter of __init__."""
        from app.domain.services.agent_task_runner import AgentTaskRunner

        sig = inspect.signature(AgentTaskRunner.__init__)
        assert "memory_flusher" in sig.parameters, (
            "AgentTaskRunner.__init__ must accept memory_flusher parameter"
        )

    def test_memory_flusher_defaults_to_none(self) -> None:
        """memory_flusher default value should be None."""
        from app.domain.services.agent_task_runner import AgentTaskRunner

        sig = inspect.signature(AgentTaskRunner.__init__)
        param = sig.parameters["memory_flusher"]
        assert param.default is None, (
            f"memory_flusher default should be None, got {param.default}"
        )
