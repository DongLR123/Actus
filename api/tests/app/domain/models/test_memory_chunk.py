"""Tests for RawChunk and FlushBatch dataclasses (Task 1)."""
from __future__ import annotations

import pytest

from app.domain.models.memory_chunk import FlushBatch, RawChunk


class TestRawChunk:
    def test_fields_accessible(self) -> None:
        """RawChunk 字段可以正常访问。"""
        chunk = RawChunk(
            content="hello world",
            session_id="sess-1",
            user_id="user-1",
            source="react_graph",
            metadata={"key": "value"},
            content_hash="abc123",
        )
        assert chunk.content == "hello world"
        assert chunk.session_id == "sess-1"
        assert chunk.user_id == "user-1"
        assert chunk.source == "react_graph"
        assert chunk.metadata == {"key": "value"}
        assert chunk.content_hash == "abc123"

    def test_frozen_immutability(self) -> None:
        """RawChunk 是 frozen dataclass，修改字段应抛出 FrozenInstanceError。"""
        chunk = RawChunk(
            content="hello",
            session_id="sess-1",
            user_id="user-1",
            source="react_graph",
            metadata={},
            content_hash="abc123",
        )
        with pytest.raises(Exception):  # FrozenInstanceError (subclass of AttributeError)
            chunk.content = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """相同字段的 RawChunk 应相等。"""
        chunk_a = RawChunk(
            content="hello",
            session_id="sess-1",
            user_id="user-1",
            source="react_graph",
            metadata={"k": "v"},
            content_hash="abc123",
        )
        chunk_b = RawChunk(
            content="hello",
            session_id="sess-1",
            user_id="user-1",
            source="react_graph",
            metadata={"k": "v"},
            content_hash="abc123",
        )
        assert chunk_a == chunk_b


class TestFlushBatch:
    def _make_chunk(self, content: str = "test content") -> RawChunk:
        return RawChunk(
            content=content,
            session_id="sess-1",
            user_id="user-1",
            source="react_graph",
            metadata={},
            content_hash="hash_" + content[:8],
        )

    def test_fields_accessible(self) -> None:
        """FlushBatch 字段可以正常访问。"""
        chunk = self._make_chunk()
        batch = FlushBatch(
            session_id="sess-1",
            user_id="user-1",
            from_cursor=0,
            target_cursor=5,
            chunks=(chunk,),
        )
        assert batch.session_id == "sess-1"
        assert batch.user_id == "user-1"
        assert batch.from_cursor == 0
        assert batch.target_cursor == 5
        assert len(batch.chunks) == 1
        assert batch.chunks[0] is chunk

    def test_frozen_immutability(self) -> None:
        """FlushBatch 是 frozen dataclass，修改字段应抛出 FrozenInstanceError。"""
        batch = FlushBatch(
            session_id="sess-1",
            user_id="user-1",
            from_cursor=0,
            target_cursor=5,
            chunks=(),
        )
        with pytest.raises(Exception):
            batch.session_id = "other"  # type: ignore[misc]

    def test_from_cursor_and_target_cursor(self) -> None:
        """from_cursor 和 target_cursor 正确存储游标值。"""
        batch = FlushBatch(
            session_id="sess-1",
            user_id="user-1",
            from_cursor=10,
            target_cursor=20,
            chunks=(),
        )
        assert batch.from_cursor == 10
        assert batch.target_cursor == 20

    def test_empty_chunks_tuple(self) -> None:
        """FlushBatch 支持空 chunks 元组。"""
        batch = FlushBatch(
            session_id="sess-1",
            user_id="user-1",
            from_cursor=0,
            target_cursor=0,
            chunks=(),
        )
        assert batch.chunks == ()
        assert len(batch.chunks) == 0

    def test_multiple_chunks(self) -> None:
        """FlushBatch 支持多个 RawChunk。"""
        chunks = tuple(self._make_chunk(f"content {i}") for i in range(3))
        batch = FlushBatch(
            session_id="sess-1",
            user_id="user-1",
            from_cursor=0,
            target_cursor=3,
            chunks=chunks,
        )
        assert len(batch.chunks) == 3
