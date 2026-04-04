"""Tests for ActusResponsesModel multimodal content conversion."""
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel


class TestConvertContentBlocksForResponses:
    """Test _convert_content_blocks_for_responses with multimodal blocks."""

    def test_text_block_converted_to_input_text(self):
        blocks = [{"type": "text", "text": "hello world"}]
        result = ActusResponsesModel._convert_content_blocks_for_responses(blocks)
        assert result == [{"type": "input_text", "text": "hello world"}]

    def test_image_url_converted_to_input_image(self):
        blocks = [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        result = ActusResponsesModel._convert_content_blocks_for_responses(blocks)
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert result[0]["image_url"] == "https://example.com/img.png"

    def test_file_block_converted_to_input_file(self):
        blocks = [
            {
                "type": "file",
                "file": {
                    "filename": "doc.pdf",
                    "file_data": "data:application/pdf;base64,AAAA",
                },
            },
        ]
        result = ActusResponsesModel._convert_content_blocks_for_responses(blocks)
        assert len(result) == 1
        assert result[0]["type"] == "input_file"
        assert result[0]["filename"] == "doc.pdf"
        assert result[0]["file_data"] == "data:application/pdf;base64,AAAA"

    def test_mixed_blocks_all_converted(self):
        blocks = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "https://x.com/i.png"}},
            {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,BB"}},
        ]
        result = ActusResponsesModel._convert_content_blocks_for_responses(blocks)
        assert len(result) == 3
        assert result[0]["type"] == "input_text"
        assert result[1]["type"] == "input_image"
        assert result[2]["type"] == "input_file"

    def test_unknown_block_passed_through(self):
        blocks = [{"type": "custom", "data": "stuff"}]
        result = ActusResponsesModel._convert_content_blocks_for_responses(blocks)
        assert result == blocks

    def test_file_block_missing_file_key_still_converts(self):
        """Converter is a pure format translator — malformed blocks should be
        caught upstream by sanitize_multimodal_blocks(), not here.
        This test verifies the converter doesn't crash on partial input."""
        blocks = [{"type": "file"}]
        result = ActusResponsesModel._convert_content_blocks_for_responses(blocks)
        assert result[0]["type"] == "input_file"


class TestResponsesModelBindToolsPreservesFields:
    def test_bind_tools_preserves_vision_fields(self):
        model = ActusResponsesModel(
            base_url="https://test.com/v1",
            api_key="test",
            supports_vision=False,
            supports_pdf_input=True,
        )
        bound = model.bind_tools([])
        assert bound.supports_vision is False
        assert bound.supports_pdf_input is True
