"""Verify that @lc_tool does NOT serialize FileProcessResult to string.

This is the critical assumption of the file_view design: tool_node must
receive the original FileProcessResult object, not a str(FileProcessResult).
"""
import asyncio

import pytest
from langchain_core.tools import tool as lc_tool

from app.domain.external.file_processor import FileProcessResult


def test_lc_tool_returns_file_process_result_object():
    """@lc_tool must pass FileProcessResult through without str() conversion."""

    @lc_tool
    async def fake_file_view(filepath: str) -> FileProcessResult:
        """Fake tool for testing."""
        return FileProcessResult(
            text="[Image: test.png, 100x200]",
            image_blocks=({"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},),
        )

    result = asyncio.get_event_loop().run_until_complete(
        fake_file_view.ainvoke({"filepath": "/tmp/test.png"})
    )

    # CRITICAL: result must be the FileProcessResult object, not a string
    assert isinstance(result, FileProcessResult), (
        f"@lc_tool serialized FileProcessResult to {type(result).__name__}: {result!r}. "
        f"The file_view design requires the original object to reach tool_node."
    )
    assert result.text == "[Image: test.png, 100x200]"
    assert len(result.image_blocks) == 1


def test_lc_tool_string_return_still_works():
    """Existing tools returning str must continue to work."""

    @lc_tool
    async def fake_tool(x: str) -> str:
        """Fake tool."""
        return "hello"

    result = asyncio.get_event_loop().run_until_complete(
        fake_tool.ainvoke({"x": "test"})
    )
    assert isinstance(result, str)
    assert result == "hello"
