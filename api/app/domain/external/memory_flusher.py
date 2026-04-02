from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.domain.models.memory_chunk import FlushBatch


@runtime_checkable
class MemoryFlusher(Protocol):
    """记忆刷写调度器协议。Domain 层仅依赖此接口。"""

    def submit(self, batch: "FlushBatch") -> None:
        """提交 FlushBatch 到后台 flush 队列。同步方法，不阻塞。"""
        ...
