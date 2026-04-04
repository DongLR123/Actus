from __future__ import annotations

from openai import AsyncOpenAI

from app.domain.external.embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding provider."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 256,
    ):
        self._client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            input=texts, model=self._model, dimensions=self._dimensions
        )
        return [d.embedding for d in resp.data]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model
