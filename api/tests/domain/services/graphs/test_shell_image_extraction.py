"""Tests for _extract_shell_images (M1d)."""
import base64
import pytest


class TestExtractShellImages:
    def test_no_images_returns_original(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        text = "Hello world\ncommand output"
        cleaned, blocks = _extract_shell_images(text)
        assert cleaned == text
        assert blocks == []

    def test_extracts_valid_png_data_url(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        b64 = base64.b64encode(raw).decode()
        text = f"before data:image/png;base64,{b64} after"
        cleaned, blocks = _extract_shell_images(text)
        assert "[image extracted]" in cleaned
        assert "before" in cleaned
        assert "after" in cleaned
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"

    def test_invalid_base64_preserved(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        text = "data:image/png;base64,AAA"  # 3 chars, not valid base64
        cleaned, blocks = _extract_shell_images(text)
        assert blocks == []
        assert "data:image/png;base64,AAA" in cleaned

    def test_oversized_image_skipped(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        from app.infrastructure.external.llm.message_sanitizer import MAX_IMAGE_BYTES
        # Build base64 that decodes to just over the limit
        oversized_bytes = b"\x00" * (MAX_IMAGE_BYTES + 100)
        import base64 as b64mod
        huge_b64 = b64mod.b64encode(oversized_bytes).decode()
        text = f"data:image/jpeg;base64,{huge_b64}"
        cleaned, blocks = _extract_shell_images(text)
        assert blocks == []
        assert "too large" in cleaned

    def test_text_after_limit_preserved(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        from app.domain.external.file_processor import MAX_FILE_VIEW_IMAGES
        # Use >= 16 bytes so payload passes minimum length check
        raw = b"\x89PNG" + b"\x00" * 20  # 24 bytes → 32 base64 chars
        b64 = base64.b64encode(raw).decode()
        parts = [f"data:image/png;base64,{b64}" for _ in range(15)]
        text = "prefix " + " mid ".join(parts) + " suffix"
        cleaned, blocks = _extract_shell_images(text)
        assert len(blocks) == MAX_FILE_VIEW_IMAGES  # exactly at cap
        assert "suffix" in cleaned

    def test_no_base64_marker_skipped(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        text = "data:image/png;encoding=utf8,not-base64"
        cleaned, blocks = _extract_shell_images(text)
        assert blocks == []

    def test_fast_path_no_prefix(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        text = "x" * 100000
        cleaned, blocks = _extract_shell_images(text)
        assert cleaned == text
        assert blocks == []

    def test_empty_payload_rejected(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        text = "data:image/png;base64,"
        cleaned, blocks = _extract_shell_images(text)
        assert blocks == []
        # Original text preserved since payload too short

    def test_short_payload_rejected(self):
        from app.domain.services.graphs.react_graph import _extract_shell_images
        text = "data:image/png;base64,AAAA"  # only 4 chars, < 16 minimum
        cleaned, blocks = _extract_shell_images(text)
        assert blocks == []
