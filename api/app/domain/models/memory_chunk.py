from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawChunk:
    """对话分块，不含 embedding。由 flow 产出，传递给 flush service。"""

    content: str
    session_id: str
    user_id: str
    source: str
    metadata: dict[str, Any]
    content_hash: str


@dataclass(frozen=True)
class FlushBatch:
    """一次 flush 提交的完整上下文。"""

    session_id: str
    user_id: str
    from_cursor: int
    target_cursor: int
    chunks: tuple[RawChunk, ...]
