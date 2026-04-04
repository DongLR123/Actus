import pytest
from app.domain.external.file_processor import FileProcessResult


class TestFileProcessResult:
    def test_document_blocks_default_empty(self):
        r = FileProcessResult(text="hello")
        assert r.document_blocks == ()

    def test_document_blocks_set(self):
        doc = ({"type": "file", "file": {"filename": "x.pdf", "file_data": "data:..."}},)
        r = FileProcessResult(text="pdf", document_blocks=doc)
        assert len(r.document_blocks) == 1

    def test_frozen(self):
        r = FileProcessResult(text="hi")
        with pytest.raises(AttributeError):
            r.text = "bye"

    def test_backward_compatible(self):
        r = FileProcessResult(text="old", image_blocks=({"type": "image_url"},))
        assert r.document_blocks == ()
        assert len(r.image_blocks) == 1
