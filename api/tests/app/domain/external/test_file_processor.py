import pytest

from app.domain.external.file_processor import FileProcessResult


class TestFileProcessResult:
    def test_text_only(self):
        r = FileProcessResult(text="[Image: test.png, 100x200]")
        assert r.text == "[Image: test.png, 100x200]"
        assert r.image_blocks == ()

    def test_with_image_blocks(self):
        blocks = ({"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},)
        r = FileProcessResult(text="[Image: test.png]", image_blocks=blocks)
        assert len(r.image_blocks) == 1

    def test_frozen(self):
        r = FileProcessResult(text="test")
        with pytest.raises(AttributeError):
            r.text = "changed"
