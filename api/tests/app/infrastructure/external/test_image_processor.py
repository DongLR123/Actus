import io

import pytest
from PIL import Image

from app.infrastructure.external.image_processor import (
    IMAGE_MAX_DIMENSION,
    ImageProcessError,
    ImageProcessResult,
    ImageProcessor,
)

# -- Magic byte test fixtures --
_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
_JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 100
_GIF89A_HEADER = b"GIF89a" + b"\x00" * 100
_WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100
_PDF_HEADER = b"%PDF-1.4" + b"\x00" * 100
_EMPTY = b""


class TestIsImage:
    def test_png(self) -> None:
        assert ImageProcessor.is_image(_PNG_HEADER, "image/png") is True

    def test_jpeg(self) -> None:
        assert ImageProcessor.is_image(_JPEG_HEADER, "image/jpeg") is True

    def test_gif(self) -> None:
        assert ImageProcessor.is_image(_GIF89A_HEADER, "image/gif") is True

    def test_webp(self) -> None:
        assert ImageProcessor.is_image(_WEBP_HEADER, "image/webp") is True

    def test_pdf_not_image(self) -> None:
        assert ImageProcessor.is_image(_PDF_HEADER, "application/pdf") is False

    def test_empty_bytes(self) -> None:
        assert ImageProcessor.is_image(_EMPTY, "") is False

    def test_wrong_mime_but_valid_magic(self) -> None:
        assert ImageProcessor.is_image(_PNG_HEADER, "application/octet-stream") is True

    def test_image_mime_but_invalid_magic(self) -> None:
        assert ImageProcessor.is_image(_PDF_HEADER, "image/png") is False


class TestImageProcessResult:
    def test_fields(self) -> None:
        r = ImageProcessResult(
            buffer=b"data",
            media_type="image/png",
            width=100,
            height=200,
            original_width=400,
            original_height=800,
        )
        assert r.width == 100
        assert r.original_width == 400
        assert r.media_type == "image/png"


# -- Helpers for creating real images --


def _make_png(width: int, height: int, color: tuple = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgba_png(width: int, height: int) -> bytes:
    img = Image.new("RGBA", (width, height), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), (0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class TestProcess:
    def test_small_image_passthrough(self) -> None:
        raw = _make_png(800, 600)
        result = ImageProcessor.process(raw, len(raw), "image/png")
        assert result.original_width == 800
        assert result.original_height == 600
        assert result.width == 800
        assert result.height == 600
        assert result.media_type == "image/png"

    def test_oversized_dimension_resized(self) -> None:
        raw = _make_png(4000, 3000)
        result = ImageProcessor.process(raw, len(raw), "image/png")
        assert result.width <= IMAGE_MAX_DIMENSION
        assert result.height <= IMAGE_MAX_DIMENSION
        assert result.original_width == 4000
        assert result.original_height == 3000
        assert abs(result.width / result.height - 4000 / 3000) < 0.01

    def test_small_jpeg_passthrough_no_reencode(self) -> None:
        """Small JPEG without EXIF → buffer returned byte-identical (no lossy re-encode)."""
        raw = _make_jpeg(500, 500)
        result = ImageProcessor.process(raw, len(raw), "image/jpeg")
        assert result.buffer is raw, "Passthrough image should return original bytes object"

    def test_small_png_passthrough_no_reencode(self) -> None:
        """Small PNG without EXIF → buffer returned byte-identical."""
        raw = _make_png(400, 300)
        result = ImageProcessor.process(raw, len(raw), "image/png")
        assert result.buffer is raw, "Passthrough image should return original bytes object"

    def test_rgba_to_jpeg_white_background(self) -> None:
        raw = _make_rgba_png(100, 100)
        result = ImageProcessor.process(raw, len(raw), "image/png")
        assert result.width == 100

    def test_corrupt_image_raises(self) -> None:
        corrupt = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        with pytest.raises(ImageProcessError):
            ImageProcessor.process(corrupt, len(corrupt), "image/png")

    def test_exif_orientation_normalized(self) -> None:
        img = Image.new("RGB", (200, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()
        result = ImageProcessor.process(raw, len(raw), "image/jpeg")
        assert result.original_width == 200
        assert result.original_height == 100

    def test_jpeg_passthrough(self) -> None:
        raw = _make_jpeg(500, 500)
        result = ImageProcessor.process(raw, len(raw), "image/jpeg")
        assert result.media_type == "image/jpeg"
        assert result.width == 500

    def test_exif_orientation_bytes_rewritten(self) -> None:
        """EXIF transpose must rewrite output bytes, not just metadata."""
        # Create a 200x100 JPEG with EXIF Orientation=6 (rotated 90° CW)
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        # Build EXIF with Orientation=6 using Pillow's built-in Exif class
        exif = img.getexif()
        exif[0x0112] = 6  # 0x0112 = Orientation tag, 6 = 90° CW
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        raw_with_exif = buf.getvalue()

        result = ImageProcessor.process(
            raw_with_exif, len(raw_with_exif), "image/jpeg"
        )

        # After transpose: 200x100 with Orientation=6 → 100x200 visual
        assert result.original_width == 100
        assert result.original_height == 200

        # Output bytes must also reflect the transposed orientation
        reopened = Image.open(io.BytesIO(result.buffer))
        assert reopened.size == (100, 200)

    def test_p_mode_transparency_white_background(self) -> None:
        """P-mode PNG with transparency index → white background, not black."""
        # Create P-mode image with transparency
        img = Image.new("P", (50, 50))
        # Set palette: index 0 = red, rest = default
        palette = [255, 0, 0] + [0, 0, 0] * 255
        img.putpalette(palette)
        # Mark index 0 as transparent
        img.info["transparency"] = 0
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()

        result = ImageProcessor.process(raw, len(raw), "image/png")

        # Verify: if we force JPEG conversion, transparent pixels should be white
        from app.infrastructure.external.image_processor import ImageProcessor as IP

        test_img = Image.open(io.BytesIO(raw))
        rgb = IP._ensure_rgb(test_img)
        # The transparent pixel (index 0) should now be white (255, 255, 255)
        pixel = rgb.getpixel((0, 0))
        assert pixel == (255, 255, 255), f"Expected white bg, got {pixel}"
