from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.external.embedding.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_provider(**kwargs) -> OpenAIEmbeddingProvider:
    defaults = dict(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        model="text-embedding-3-small",
        dimensions=256,
    )
    defaults.update(kwargs)
    return OpenAIEmbeddingProvider(**defaults)


class TestOpenAIEmbeddingProvider:
    def test_model_name_returns_configured_model(self):
        provider = _make_provider(model="custom-embed-model")
        assert provider.model_name == "custom-embed-model"

    def test_dimensions_returns_configured_value(self):
        provider = _make_provider(dimensions=1024)
        assert provider.dimensions == 1024

    @pytest.mark.anyio
    async def test_embed_calls_openai_and_returns_vectors(self):
        provider = _make_provider()

        mock_embeddings = [
            SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
            SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_response = SimpleNamespace(data=mock_embeddings)

        provider._client = MagicMock()
        provider._client.embeddings = MagicMock()
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed(["hello", "world"])

        provider._client.embeddings.create.assert_awaited_once_with(
            input=["hello", "world"],
            model="text-embedding-3-small",
            dimensions=256,
        )
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
