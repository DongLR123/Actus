"""Image detection, compression, and metadata extraction.

Called exclusively from the storage layer (infrastructure).
All Pillow operations are synchronous — callers must wrap in asyncio.to_thread().
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import ClassVar

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (aligned with Claude Code)
# ---------------------------------------------------------------------------
IMAGE_TARGET_RAW_SIZE: int = 3_932_160  # 3.75 MB
IMAGE_MAX_DIMENSION: int = 2000  # px
IMAGE_FALLBACK_DIMENSION: int = 1000  # px
IMAGE_MAX_FILE_SIZE: int = 20_971_520  # 20 MB
IMAGE_LOW_DETAIL_THRESHOLD: int = 512  # px

MEDIA_TYPE_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

_FORMAT_TO_MEDIA_TYPE: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


class ImageProcessError(Exception):
    """Raised when image processing fails irrecoverably."""


@dataclass(frozen=True)
class ImageProcessResult:
    buffer: bytes
    media_type: str
    width: int
    height: int
    original_width: int
    original_height: int


class ImageProcessor:
    """Static utility for image detection and processing."""

    MAX_FILE_SIZE: ClassVar[int] = IMAGE_MAX_FILE_SIZE

    _SIGNATURES: ClassVar[list[tuple[bytes, int]]] = [
        (b"\x89PNG\r\n\x1a\n", 0),
        (b"\xff\xd8\xff", 0),
        (b"GIF87a", 0),
        (b"GIF89a", 0),
    ]

    @staticmethod
    def is_image(raw_bytes: bytes, mime_type: str) -> bool:
        """Detect image by magic bytes. MIME type is secondary."""
        if len(raw_bytes) < 12:
            return False

        for sig, offset in ImageProcessor._SIGNATURES:
            if raw_bytes[offset : offset + len(sig)] == sig:
                return True

        # WebP: RIFF at 0, WEBP at 8
        if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
            return True

        return False

    @staticmethod
    def process(
        raw_bytes: bytes, original_size: int, mime_type: str
    ) -> ImageProcessResult:
        """Compress image and extract metadata.

        Raises ImageProcessError on corrupt/unreadable images.
        """
        try:
            img = Image.open(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ImageProcessError(f"Cannot open image: {exc}") from exc

        # Capture format before exif_transpose (which may return a copy with format=None)
        fmt = img.format or "PNG"

        try:
            # Only re-save when EXIF orientation actually requires correction.
            # exif_transpose() returns a new object even for normal-orientation
            # images in some Pillow versions, so we check the tag directly.
            exif_orientation = img.getexif().get(0x0112, 1)  # 1 = normal
            if exif_orientation != 1:
                img = ImageOps.exif_transpose(img)
                raw_bytes = ImageProcessor._save(img, fmt)
                original_size = len(raw_bytes)
        except Exception:
            pass  # Not all formats have EXIF

        original_width, original_height = img.size
        current_bytes = raw_bytes
        current_size = original_size
        width, height = original_width, original_height

        # Step 1: Resize if dimensions exceed limit
        if width > IMAGE_MAX_DIMENSION or height > IMAGE_MAX_DIMENSION:
            img = ImageProcessor._proportional_resize(img, IMAGE_MAX_DIMENSION)
            width, height = img.size
            current_bytes = ImageProcessor._save(img, fmt)
            current_size = len(current_bytes)

        # Step 2: Compress if size exceeds limit
        if current_size > IMAGE_TARGET_RAW_SIZE:
            current_bytes, fmt = ImageProcessor._progressive_compress(
                img, fmt, IMAGE_TARGET_RAW_SIZE
            )
            current_size = len(current_bytes)
            img = Image.open(io.BytesIO(current_bytes))
            width, height = img.size

        # Step 3: Ultimate fallback
        if current_size > IMAGE_TARGET_RAW_SIZE:
            img = ImageProcessor._proportional_resize(img, IMAGE_FALLBACK_DIMENSION)
            img = ImageProcessor._ensure_rgb(img)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=20)
            current_bytes = buf.getvalue()
            current_size = len(current_bytes)
            fmt = "JPEG"
            width, height = img.size

        if current_size > IMAGE_TARGET_RAW_SIZE:
            raise ImageProcessError(
                f"Image still {current_size} bytes after all compression attempts"
            )

        media_type = _FORMAT_TO_MEDIA_TYPE.get(fmt, "image/png")
        return ImageProcessResult(
            buffer=current_bytes,
            media_type=media_type,
            width=width,
            height=height,
            original_width=original_width,
            original_height=original_height,
        )

    @staticmethod
    def _proportional_resize(img: Image.Image, max_dim: int) -> Image.Image:
        w, h = img.size
        if w <= max_dim and h <= max_dim:
            return img
        ratio = min(max_dim / w, max_dim / h)
        new_size = (int(w * ratio), int(h * ratio))
        return img.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _ensure_rgb(img: Image.Image) -> Image.Image:
        """Convert any mode with transparency to RGB with white background.

        Handles:
        - RGBA/LA/PA: alpha channel composited onto white
        - P with transparency index: convert to RGBA first, then white-bg composite
        - Other non-RGB modes: direct convert
        """
        # P-mode with transparency index (common in GIF/PNG icons)
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
            # Fall through to RGBA handling below

        if img.mode in ("RGBA", "LA", "PA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            return bg
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    @staticmethod
    def _save(img: Image.Image, fmt: str) -> bytes:
        buf = io.BytesIO()
        save_kwargs: dict = {}
        if fmt == "PNG":
            save_kwargs = {"optimize": True}
        elif fmt == "JPEG":
            save_kwargs = {"quality": 80}
        img.save(buf, format=fmt, **save_kwargs)
        return buf.getvalue()

    @staticmethod
    def _progressive_compress(
        img: Image.Image, fmt: str, target_size: int
    ) -> tuple[bytes, str]:
        """Try progressively aggressive compression."""
        # PNG: try optimized PNG first
        if fmt == "PNG":
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True, compress_level=9)
            if len(buf.getvalue()) <= target_size:
                return buf.getvalue(), "PNG"

        # JPEG at decreasing quality
        rgb_img = ImageProcessor._ensure_rgb(img)
        for quality in (80, 60, 40, 20):
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=quality)
            if len(buf.getvalue()) <= target_size:
                return buf.getvalue(), "JPEG"

        # Return best effort
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=20)
        return buf.getvalue(), "JPEG"
