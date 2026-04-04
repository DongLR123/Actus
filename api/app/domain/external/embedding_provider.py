"""Embedding provider protocol for skill semantic search."""
from __future__ import annotations
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """向量化文本的抽象接口。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表。"""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """向量维度。"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型标识符，用于缓存 key 构建。"""
        ...
