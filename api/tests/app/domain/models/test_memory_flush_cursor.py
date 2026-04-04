"""Tests for Memory.flush_cursor field (Task 3)."""
from __future__ import annotations

from app.domain.models.memory import Memory


class TestMemoryFlushCursor:
    def test_default_flush_cursor_is_zero(self) -> None:
        """Memory 默认 flush_cursor 为 0。"""
        memory = Memory()
        assert memory.flush_cursor == 0

    def test_custom_flush_cursor_value(self) -> None:
        """Memory 可以设置自定义 flush_cursor 值。"""
        memory = Memory(flush_cursor=42)
        assert memory.flush_cursor == 42

    def test_serialization_roundtrip_preserves_flush_cursor(self) -> None:
        """序列化和反序列化后 flush_cursor 保持不变。"""
        memory = Memory(
            messages=[{"role": "user", "content": "hello"}],
            flush_cursor=7,
        )
        serialized = memory.model_dump()
        restored = Memory(**serialized)
        assert restored.flush_cursor == 7

    def test_backward_compatible_old_data_defaults_to_zero(self) -> None:
        """从不含 flush_cursor 的旧数据恢复时默认为 0（向后兼容）。"""
        old_data = {"messages": [{"role": "user", "content": "legacy"}]}
        memory = Memory(**old_data)
        assert memory.flush_cursor == 0

    def test_flush_cursor_json_roundtrip(self) -> None:
        """JSON 序列化和反序列化后 flush_cursor 保持不变。"""
        memory = Memory(flush_cursor=15)
        json_str = memory.model_dump_json()
        restored = Memory.model_validate_json(json_str)
        assert restored.flush_cursor == 15
