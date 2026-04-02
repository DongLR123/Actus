import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.models.memory_chunk import FlushBatch

logger = logging.getLogger(__name__)
_CIRCUIT_BREAKER_RECOVERY_SECONDS = 300.0


class MemoryFlushService:
    """记忆刷写服务（app.state 单例）。"""

    def __init__(self, max_retries: int = 3, circuit_breaker_threshold: int = 3):
        self._pending_tasks: set[asyncio.Task] = set()
        self._consecutive_failures: int = 0
        self._last_failure_time: float | None = None
        self._max_retries = max_retries
        self._circuit_breaker_threshold = circuit_breaker_threshold

    def submit(self, batch: "FlushBatch") -> None:
        """Submit FlushBatch to background flush queue. Sync, non-blocking."""
        # Circuit breaker: check + time-based auto-recovery
        if self._consecutive_failures >= self._circuit_breaker_threshold:
            if (
                self._last_failure_time
                and (time.monotonic() - self._last_failure_time)
                > _CIRCUIT_BREAKER_RECOVERY_SECONDS
            ):
                self._consecutive_failures = 0
                self._last_failure_time = None
            else:
                logger.warning("MemoryFlushService: circuit breaker open, skipping")
                return

        task = asyncio.create_task(self._do_flush(batch))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _do_flush(self, batch: "FlushBatch") -> None:
        """Execute flush with retry. C5.1 replaces stub body with real logic."""
        last_exc = None
        for attempt in range(1 + self._max_retries):
            try:
                logger.info(
                    "MemoryFlushService._do_flush: session=%s from=%d to=%d chunks=%d attempt=%d",
                    batch.session_id,
                    batch.from_cursor,
                    batch.target_cursor,
                    len(batch.chunks),
                    attempt,
                )
                # C5.1: await self._embed_and_insert(batch)
                # C5.1: await self._update_cursor(batch)
                self._consecutive_failures = 0
                self._last_failure_time = None
                return
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)

        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        logger.warning("MemoryFlushService flush failed: %s", last_exc)

    async def shutdown(self) -> None:
        if not self._pending_tasks:
            return
        done, pending = await asyncio.wait(self._pending_tasks, timeout=5)
        for task in pending:
            task.cancel()
        self._pending_tasks.clear()
