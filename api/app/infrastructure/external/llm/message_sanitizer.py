"""Shared multimodal content block sanitizer for LLM adapters."""
from __future__ import annotations

import base64 as _b64
import logging

logger = logging.getLogger(__name__)

# --- Shared size limits (exported for processors to stay aligned) ---
MAX_IMAGE_BYTES = 5 * 1024 * 1024    # 5 MiB decoded
MAX_PDF_BYTES = 50 * 1024 * 1024     # 50 MiB decoded

# Allowed raster image MIME types (SVG and other non-raster formats rejected)
_ALLOWED_IMAGE_MIMES = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp",
})


def _bytes_to_max_b64_chars(max_bytes: int) -> int:
    """Convert a decoded-byte limit to the maximum valid base64 string length.

    base64 encodes ceil(n/3)*4 chars for n bytes, so the max base64 length
    for max_bytes decoded is ((max_bytes + 2) // 3) * 4.
    """
    return ((max_bytes + 2) // 3) * 4


_MAX_IMAGE_B64_CHARS = _bytes_to_max_b64_chars(MAX_IMAGE_BYTES)
_MAX_PDF_B64_CHARS = _bytes_to_max_b64_chars(MAX_PDF_BYTES)


def sanitize_multimodal_blocks(
    blocks: list[dict],
    supports_vision: bool,
    supports_pdf_input: bool,
) -> list[dict]:
    """Filter and validate multimodal content blocks before sending to LLM.

    Rules:
    - image_url blocks: stripped when supports_vision is False, when the nested
      value is not a dict, when embedded base64 exceeds 5 MB, or when base64
      is syntactically invalid.
    - file blocks: stripped when supports_pdf_input is False, when file_data
      is missing/empty, when the media type is not application/pdf, when the
      payload exceeds 50 MB, or when base64 is syntactically invalid.
    - All other block types (e.g. "text") pass through unchanged.

    If every block is stripped (or the input list is empty), returns a single
    fallback text block ``{"type": "text", "text": "[content filtered]"}``.
    """
    sanitized: list[dict] = []

    for block in blocks:
        block_type = block.get("type", "")

        if block_type == "image_url":
            if not supports_vision:
                logger.warning("Stripping image block: model does not support vision")
                continue
            image_url_val = block.get("image_url")
            if not isinstance(image_url_val, dict):
                logger.warning("Stripping malformed image block: image_url is not a dict")
                continue
            url = image_url_val.get("url", "")
            if not isinstance(url, str) or not url:
                logger.warning("Stripping malformed image block: url missing or not a string")
                continue
            if url.startswith("data:") and ";base64," in url:
                # Check image MIME — reject non-raster types (e.g. SVG)
                mime_part = url.split(";base64,", 1)[0]  # "data:image/png"
                mime = mime_part[5:] if mime_part.startswith("data:") else ""  # "image/png"
                if mime not in _ALLOWED_IMAGE_MIMES:
                    logger.warning("Stripping image block: unsupported MIME %s", mime)
                    continue
                b64_part = url.split(";base64,", 1)[1]
                if not b64_part:
                    logger.warning("Stripping image block: empty base64 payload")
                    continue
                if len(b64_part) > _MAX_IMAGE_B64_CHARS:
                    logger.warning("Stripping oversized image: %d base64 chars", len(b64_part))
                    continue
                if not _is_valid_base64(b64_part):
                    logger.warning("Stripping image block: invalid base64 payload")
                    continue
            else:
                # Remote URL: check for non-raster extensions (e.g. .svg)
                url_lower = url.lower().split("?")[0]  # strip query params
                if url_lower.endswith(".svg") or url_lower.endswith(".svgz"):
                    logger.warning("Stripping image block: SVG URL not supported")
                    continue

        elif block_type == "file":
            if not supports_pdf_input:
                logger.warning("Stripping file block: model does not support PDF input")
                continue
            file_val = block.get("file")
            if not isinstance(file_val, dict):
                logger.warning("Stripping malformed file block: file is not a dict")
                continue
            file_data = file_val.get("file_data", "")
            if not isinstance(file_data, str) or not file_data:
                logger.warning("Stripping file block: missing or non-string file_data")
                continue
            if not file_data.startswith("data:application/pdf;base64,"):
                logger.warning("Stripping file block: unsupported media type")
                continue
            b64_part = file_data.split(";base64,", 1)[1]
            if not b64_part:
                logger.warning("Stripping file block: empty base64 payload")
                continue
            if len(b64_part) > _MAX_PDF_B64_CHARS:
                logger.warning("Stripping oversized document: %d base64 chars", len(b64_part))
                continue
            if not _is_valid_base64(b64_part):
                logger.warning("Stripping file block: invalid base64 payload")
                continue

        sanitized.append(block)

    if not sanitized:
        return [{"type": "text", "text": "[content filtered]"}]
    return sanitized


def _is_valid_base64(data: str) -> bool:
    """Check if a string is valid base64 (cheap: only validates, does not decode fully)."""
    try:
        _b64.b64decode(data, validate=True)
        return True
    except Exception:
        return False
