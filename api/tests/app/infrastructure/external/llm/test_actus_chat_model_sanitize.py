"""Integration tests for ActusChatModel multimodal sanitization."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage

from app.infrastructure.external.llm.actus_chat_model import ActusChatModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(**kwargs) -> ActusChatModel:
    defaults = dict(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model_name="gpt-4o",
    )
    defaults.update(kwargs)
    return ActusChatModel(**defaults)


def _image_block(url: str = "https://example.com/img.png") -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _text_block(text: str = "describe") -> dict:
    return {"type": "text", "text": text}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vision_disabled_strips_images():
    """When supports_vision=False, image_url blocks must be stripped."""
    model = _make_model(supports_vision=False)
    msg = HumanMessage(content=[_text_block("hello"), _image_block()])
    messages = model._to_openai_messages([msg])

    assert len(messages) == 1
    content = messages[0]["content"]
    # Image block should have been stripped; only the text block remains
    assert isinstance(content, list)
    block_types = [b.get("type") for b in content]
    assert "image_url" not in block_types
    assert "text" in block_types


def test_vision_enabled_keeps_images():
    """When supports_vision=True, image_url blocks pass through."""
    model = _make_model(supports_vision=True)
    image = _image_block()
    msg = HumanMessage(content=[_text_block("describe"), image])
    messages = model._to_openai_messages([msg])

    assert len(messages) == 1
    content = messages[0]["content"]
    assert isinstance(content, list)
    block_types = [b.get("type") for b in content]
    assert "image_url" in block_types


def test_bind_tools_preserves_fields():
    """bind_tools() must propagate supports_vision and supports_pdf_input."""
    model = _make_model(supports_vision=False, supports_pdf_input=True)

    # Minimal mock tool
    tool = MagicMock()
    tool.name = "test_tool"

    with patch(
        "app.infrastructure.external.llm.actus_chat_model.ActusChatModel.bind_tools",
        wraps=model.bind_tools,
    ):
        # Use a real LangChain tool-like dict to avoid conversion errors
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        bound = model.bind_tools([dummy_tool])

    assert bound.supports_vision is False
    assert bound.supports_pdf_input is True


def test_string_content_not_affected():
    """String content (non-multimodal) must pass through unchanged."""
    model = _make_model(supports_vision=False)
    msg = HumanMessage(content="plain text message")
    messages = model._to_openai_messages([msg])

    assert len(messages) == 1
    assert messages[0]["content"] == "plain text message"
