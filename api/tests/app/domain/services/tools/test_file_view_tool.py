import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.domain.external.file_processor import FileProcessResult


class FakeProcessor:
    async def process(self, sandbox_path, filename, mime_type, supports_vision):
        return FileProcessResult(
            text=f"[Image: {filename}, 100x200]",
            image_blocks=({"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},),
        )


class FakeLookup:
    def get_processor(self, mime_type):
        if mime_type.startswith("image/"):
            return FakeProcessor()
        return None


def _make_sandbox_mock(mime_output: str = "image/png"):
    sandbox = AsyncMock()
    mock_result = MagicMock()
    mock_result.__str__ = lambda self: mime_output
    mock_result.success = True
    sandbox.exec_command = AsyncMock(return_value=mock_result)
    return sandbox


class TestFileViewTool:
    def test_file_view_returns_file_process_result(self):
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        tools = _make_file_view_tools(_make_sandbox_mock(), FakeLookup(), supports_vision=True)
        file_view = tools[0]

        result = asyncio.get_event_loop().run_until_complete(
            file_view.ainvoke({"filepath": "/home/ubuntu/test.png"})
        )
        assert isinstance(result, FileProcessResult)
        assert "100x200" in result.text

    def test_file_view_unsupported_type_returns_string(self):
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        tools = _make_file_view_tools(_make_sandbox_mock("text/plain"), FakeLookup(), supports_vision=True)
        file_view = tools[0]

        result = asyncio.get_event_loop().run_until_complete(
            file_view.ainvoke({"filepath": "/home/ubuntu/readme.txt"})
        )
        assert isinstance(result, str)
        assert "Unsupported" in result

    def test_file_view_extension_fallback(self):
        """When `file --mime-type` returns octet-stream, fall back to extension."""
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        tools = _make_file_view_tools(
            _make_sandbox_mock("application/octet-stream"), FakeLookup(), supports_vision=True,
        )
        file_view = tools[0]

        result = asyncio.get_event_loop().run_until_complete(
            file_view.ainvoke({"filepath": "/home/ubuntu/photo.jpg"})
        )
        # Extension .jpg maps to image/jpeg → FakeLookup matches image/ prefix
        assert isinstance(result, FileProcessResult)

    def test_create_native_tools_includes_file_view(self):
        from app.domain.services.tools.langchain_tools import create_native_tools

        tools = create_native_tools(
            sandbox=AsyncMock(), browser=AsyncMock(), search_engine=AsyncMock(),
            processor_lookup=FakeLookup(), supports_vision=True,
        )
        tool_names = [t.name for t in tools]
        assert "file_view" in tool_names

    def test_create_native_tools_without_lookup_has_no_file_view(self):
        from app.domain.services.tools.langchain_tools import create_native_tools

        tools = create_native_tools(
            sandbox=AsyncMock(), browser=AsyncMock(), search_engine=AsyncMock(),
        )
        tool_names = [t.name for t in tools]
        assert "file_view" not in tool_names
