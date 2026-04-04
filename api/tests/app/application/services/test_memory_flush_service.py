"""Tests for Task 8: MemoryFlushService scheduling skeleton.

Verifies:
1. MemoryFlushService satisfies MemoryFlusher protocol
2. submit creates a background task
3. Background task completes and self-discards from _pending_tasks
4. shutdown empty (no error)
5. shutdown waits for pending tasks
6. Circuit breaker initial state
7. Circuit breaker blocks submit when threshold exceeded
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.domain.models.memory_chunk import FlushBatch, RawChunk


def _make_batch(**overrides) -> FlushBatch:
    """Create a minimal FlushBatch for testing."""
    defaults = {
        "session_id": "test-session",
        "user_id": "test-user",
        "from_cursor": 0,
        "target_cursor": 5,
        "chunks": (
            RawChunk(
                content="test content",
                session_id="test-session",
                user_id="test-user",
                source="test",
                metadata={},
                content_hash="abc123",
            ),
        ),
    }
    defaults.update(overrides)
    return FlushBatch(**defaults)


class TestMemoryFlushServiceProtocol:
    """MemoryFlushService satisfies MemoryFlusher protocol."""

    def test_satisfies_memory_flusher_protocol(self) -> None:
        """MemoryFlushService should be a structural subtype of MemoryFlusher."""
        from app.application.services.memory_flush_service import MemoryFlushService
        from app.domain.external.memory_flusher import MemoryFlusher

        service = MemoryFlushService()
        # Protocol structural check: the instance must have submit(batch) -> None
        assert hasattr(service, "submit")
        assert callable(service.submit)

        # runtime_checkable Protocol check
        assert isinstance(service, MemoryFlusher)


@pytest.mark.anyio
class TestMemoryFlushServiceSubmit:
    """submit creates background task."""

    async def test_submit_creates_task(self) -> None:
        """submit should create and track an asyncio.Task."""
        from app.application.services.memory_flush_service import MemoryFlushService

        service = MemoryFlushService()
        batch = _make_batch()

        service.submit(batch)

        assert len(service._pending_tasks) == 1

    async def test_task_completes_and_self_discards(self) -> None:
        """After task completes, it should be removed from _pending_tasks."""
        from app.application.services.memory_flush_service import MemoryFlushService

        service = MemoryFlushService()
        batch = _make_batch()

        service.submit(batch)
        assert len(service._pending_tasks) == 1

        # Wait for the task to complete
        await asyncio.sleep(0.1)

        assert len(service._pending_tasks) == 0


@pytest.mark.anyio
class TestMemoryFlushServiceShutdown:
    """shutdown behavior."""

    async def test_shutdown_empty_no_error(self) -> None:
        """shutdown with no pending tasks should not raise."""
        from app.application.services.memory_flush_service import MemoryFlushService

        service = MemoryFlushService()
        await service.shutdown()  # Should not raise

    async def test_shutdown_waits_for_pending(self) -> None:
        """shutdown should wait for pending tasks to complete."""
        from app.application.services.memory_flush_service import MemoryFlushService

        service = MemoryFlushService()
        batch = _make_batch()

        service.submit(batch)
        assert len(service._pending_tasks) == 1

        await service.shutdown()
        assert len(service._pending_tasks) == 0


class TestMemoryFlushServiceCircuitBreaker:
    """Circuit breaker behavior."""

    def test_initial_state(self) -> None:
        """Circuit breaker should start in closed state (failures=0)."""
        from app.application.services.memory_flush_service import MemoryFlushService

        service = MemoryFlushService()
        assert service._consecutive_failures == 0
        assert service._last_failure_time is None

    @pytest.mark.anyio
    async def test_circuit_breaker_blocks_submit(self) -> None:
        """When consecutive_failures >= threshold, submit should skip."""
        from app.application.services.memory_flush_service import MemoryFlushService

        service = MemoryFlushService(circuit_breaker_threshold=3)
        # Simulate reaching circuit breaker threshold
        service._consecutive_failures = 3
        service._last_failure_time = time.monotonic()

        batch = _make_batch()
        service.submit(batch)

        # No task should be created when circuit breaker is open
        assert len(service._pending_tasks) == 0
