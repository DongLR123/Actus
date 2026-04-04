"""Tests for shared multimodal content block sanitizer."""
from __future__ import annotations

import pytest

from app.infrastructure.external.llm.message_sanitizer import sanitize_multimodal_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_block(text: str = "hello") -> dict:
    return {"type": "text", "text": text}


def _image_block(b64_data: str | None = None, url: str | None = None) -> dict:
    if b64_data is not None:
        img_url = f"data:image/jpeg;base64,{b64_data}"
    elif url is not None:
        img_url = url
    else:
        img_url = "https://example.com/image.jpg"
    return {"type": "image_url", "image_url": {"url": img_url}}


def _file_block(b64_data: str | None = None, filename: str = "doc.pdf") -> dict:
    if b64_data is None:
        b64_data = "AAAA"  # minimal valid placeholder
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "file_data": f"data:application/pdf;base64,{b64_data}",
        },
    }


def _big_b64(mb: float) -> str:
    """Return a valid base64 string of approximately *mb* megabytes."""
    import base64
    byte_count = int(mb * 1024 * 1024)
    return base64.b64encode(b"\x00" * byte_count).decode()


# ---------------------------------------------------------------------------
# 1. Text blocks pass through unchanged
# ---------------------------------------------------------------------------

def test_text_block_passes_through():
    blocks = [_text_block("some text")]
    result = sanitize_multimodal_blocks(blocks, supports_vision=False, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "some text"}]


# ---------------------------------------------------------------------------
# 2. Image stripped when vision disabled
# ---------------------------------------------------------------------------

def test_image_stripped_when_vision_disabled():
    blocks = [_image_block(url="https://example.com/img.png")]
    result = sanitize_multimodal_blocks(blocks, supports_vision=False, supports_pdf_input=False)
    # Only the fallback text block should remain (all blocks filtered)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 3. Image kept when vision enabled
# ---------------------------------------------------------------------------

def test_image_kept_when_vision_enabled():
    block = _image_block(url="https://example.com/img.png")
    blocks = [block]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=False)
    assert result == [block]


# ---------------------------------------------------------------------------
# 4. Oversized base64 image stripped (> 5 MB)
# ---------------------------------------------------------------------------

def test_oversized_base64_image_stripped():
    big_b64 = _big_b64(6)  # 6 MB worth of base64 chars
    blocks = [_image_block(b64_data=big_b64)]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=False)
    # Image was stripped as oversized; fallback kicks in
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 5. Normal base64 image kept (< 5 MB)
# ---------------------------------------------------------------------------

def test_normal_base64_image_kept():
    small_b64 = _big_b64(1)  # 1 MB — well within 5 MB limit
    block = _image_block(b64_data=small_b64)
    blocks = [block]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=False)
    assert result == [block]


# ---------------------------------------------------------------------------
# 6. File block stripped when pdf_input disabled
# ---------------------------------------------------------------------------

def test_file_block_stripped_when_pdf_disabled():
    blocks = [_file_block()]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 7. File block kept when pdf_input enabled
# ---------------------------------------------------------------------------

def test_file_block_kept_when_pdf_enabled():
    block = _file_block()
    blocks = [block]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=True)
    assert result == [block]


# ---------------------------------------------------------------------------
# 8. File block with wrong media type stripped
# ---------------------------------------------------------------------------

def test_file_block_wrong_media_type_stripped():
    # file_data has wrong prefix (not application/pdf)
    bad_block = {
        "type": "file",
        "file": {
            "filename": "doc.txt",
            "file_data": "data:text/plain;base64,AAAA",
        },
    }
    blocks = [bad_block]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 9. Oversized file block stripped (> 50 MB)
# ---------------------------------------------------------------------------

def test_oversized_file_block_stripped():
    big_b64 = _big_b64(55)  # 55 MB worth of base64
    block = _file_block(b64_data=big_b64)
    blocks = [block]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 10. File block missing file_data key stripped
# ---------------------------------------------------------------------------

def test_file_block_missing_file_data_stripped():
    bad_block = {"type": "file", "file": {"filename": "doc.pdf"}}  # no file_data
    blocks = [bad_block]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 11. File block with empty file_data stripped
# ---------------------------------------------------------------------------

def test_file_block_empty_file_data_stripped():
    bad_block = {"type": "file", "file": {"filename": "doc.pdf", "file_data": ""}}
    blocks = [bad_block]
    result = sanitize_multimodal_blocks(blocks, supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 12. Empty input list returns fallback
# ---------------------------------------------------------------------------

def test_empty_blocks_returns_fallback():
    result = sanitize_multimodal_blocks([], supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 13. All blocks stripped returns fallback with "[content filtered]"
# ---------------------------------------------------------------------------

def test_all_stripped_returns_content_filtered():
    blocks = [
        _image_block(url="https://example.com/img.png"),  # stripped: vision disabled
        _file_block(),  # stripped: pdf disabled
    ]
    result = sanitize_multimodal_blocks(blocks, supports_vision=False, supports_pdf_input=False)
    assert len(result) == 1
    assert result[0] == {"type": "text", "text": "[content filtered]"}


# ---------------------------------------------------------------------------
# 14. Mixed blocks: some kept, some stripped
# ---------------------------------------------------------------------------

def test_mixed_blocks_some_kept_some_stripped():
    text = _text_block("description")
    image = _image_block(url="https://example.com/img.png")
    blocks = [text, image]
    # vision disabled → image stripped; text kept
    result = sanitize_multimodal_blocks(blocks, supports_vision=False, supports_pdf_input=False)
    assert result == [text]


# ---------------------------------------------------------------------------
# 15-18. Malformed block structures
# ---------------------------------------------------------------------------

def test_malformed_image_url_not_dict_stripped():
    """image_url is a string instead of dict → stripped, not AttributeError."""
    block = {"type": "image_url", "image_url": "https://example.com/img.png"}
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_malformed_image_url_none_stripped():
    block = {"type": "image_url", "image_url": None}
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_empty_url_image_stripped():
    """Empty string URL → stripped."""
    block = {"type": "image_url", "image_url": {"url": ""}}
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_malformed_file_not_dict_stripped():
    """file is a string instead of dict → stripped, not AttributeError."""
    block = {"type": "file", "file": "not-a-dict"}
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_malformed_file_missing_key_stripped():
    block = {"type": "file"}  # no "file" key at all
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 19-20. Invalid base64 payloads
# ---------------------------------------------------------------------------

def test_invalid_base64_image_stripped():
    """Syntactically broken base64 in image data URL → stripped."""
    block = _image_block(url="data:image/png;base64,@@@@not-valid-base64@@@@")
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_invalid_base64_file_stripped():
    """Syntactically broken base64 in file data URL → stripped."""
    block = {"type": "file", "file": {
        "filename": "doc.pdf",
        "file_data": "data:application/pdf;base64,!!!invalid!!!",
    }}
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 21-22. Empty payload data URLs
# ---------------------------------------------------------------------------

def test_empty_payload_image_data_url_stripped():
    """data:image/png;base64, with nothing after comma → stripped."""
    block = _image_block(url="data:image/png;base64,")
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_empty_payload_file_data_url_stripped():
    """data:application/pdf;base64, with nothing after comma → stripped."""
    block = {"type": "file", "file": {
        "filename": "doc.pdf",
        "file_data": "data:application/pdf;base64,",
    }}
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=True)
    assert result == [{"type": "text", "text": "[content filtered]"}]


# ---------------------------------------------------------------------------
# 23. Non-raster image MIME (SVG) stripped
# ---------------------------------------------------------------------------

def test_svg_data_url_image_stripped():
    """SVG data URL should be rejected by sanitizer (not a supported raster format)."""
    import base64
    svg_b64 = base64.b64encode(b"<svg></svg>").decode()
    block = _image_block(url=f"data:image/svg+xml;base64,{svg_b64}")
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_svg_remote_url_image_stripped():
    """Remote SVG URL should also be rejected."""
    block = _image_block(url="https://example.com/diagram.svg")
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_svg_remote_url_with_query_stripped():
    """Remote SVG URL with query params should also be rejected."""
    block = _image_block(url="https://example.com/chart.svg?v=2")
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]


def test_svgz_remote_url_stripped():
    """Remote .svgz URL should also be rejected."""
    block = _image_block(url="https://example.com/icon.svgz")
    result = sanitize_multimodal_blocks([block], supports_vision=True, supports_pdf_input=False)
    assert result == [{"type": "text", "text": "[content filtered]"}]
