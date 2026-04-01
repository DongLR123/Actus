"""Integration tests for _check_overflow with GradualCompactor."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.services.graphs.compaction import CompactionResult, GradualCompactor
from app.domain.services.graphs.token_estimator import TokenEstimator


class TestCheckOverflowIntegration:
    @pytest.mark.anyio
    async def test_guard_disabled_returns_none(self):
        """context_overflow_guard_enabled=False → early return None."""
        from app.domain.services.flows.planner_react import PlannerReActFlow

        flow = MagicMock(spec=PlannerReActFlow)
        flow._overflow_config = ContextOverflowConfig(context_overflow_guard_enabled=False)
        result = await PlannerReActFlow._check_overflow(flow, MagicMock())
        assert result is None

    @pytest.mark.anyio
    async def test_stores_last_compaction_result(self):
        """_check_overflow stores result on self._last_compaction_result."""
        from app.domain.services.flows.planner_react import PlannerReActFlow

        flow = MagicMock(spec=PlannerReActFlow)
        flow._overflow_config = ContextOverflowConfig(
            context_overflow_guard_enabled=True,
            model_name="test",
            # context_window omitted (optional, None) — patched via resolve_context_window
        )
        flow._token_estimator = TokenEstimator(strategy="char")
        flow._compactor = GradualCompactor(
            token_estimator=flow._token_estimator,
            token_safety_factor=1.0,
        )
        flow._summary_llm = None
        flow._session_id = "test-session"
        flow._uow_factory = AsyncMock()

        memory = MagicMock()
        memory.messages = [{"role": "system", "content": "sys"}]

        # resolve_context_window is a local import inside _check_overflow, so patch
        # the function in its original module.
        with patch(
            "app.domain.services.context.model_context_window.resolve_context_window",
            return_value=128_000,
        ), patch(
            "app.domain.services.flows.planner_react.dicts_to_messages",
            return_value=[SystemMessage(content="sys")],
        ):
            result = await PlannerReActFlow._check_overflow(flow, memory)

        assert result is not None
        assert result.level_applied == 0  # small content, no compaction needed
        assert flow._last_compaction_result == result
