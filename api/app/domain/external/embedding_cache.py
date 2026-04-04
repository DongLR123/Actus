"""Embedding cache protocol for persistent vector storage."""
from __future__ import annotations
from abc import ABC, abstractmethod


class EmbeddingCache(ABC):
    """向量缓存的抽象接口。"""

    @abstractmethod
    async def get(self, key: str) -> list[float] | None:
        """根据 key 获取已缓存的向量，未命中返回 None。"""
        ...

    @abstractmethod
    async def set(self, key: str, vector: list[float]) -> None:
        """缓存向量。"""
        ...
