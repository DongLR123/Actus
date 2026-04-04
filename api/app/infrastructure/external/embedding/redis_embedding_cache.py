from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.domain.external.embedding_cache import EmbeddingCache

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisEmbeddingCache(EmbeddingCache):
    PREFIX = "skill_emb:"

    def __init__(self, redis_client: Redis, ttl: int = 86400) -> None:
        self._redis = redis_client
        self._ttl = ttl

    async def get(self, key: str) -> list[float] | None:
        try:
            raw = await self._redis.get(f"{self.PREFIX}{key}")
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.debug("Redis embedding cache get failed for key=%s", key, exc_info=True)
            return None

    async def set(self, key: str, vector: list[float]) -> None:
        try:
            await self._redis.set(f"{self.PREFIX}{key}", json.dumps(vector), ex=self._ttl)
        except Exception:
            logger.debug("Redis embedding cache set failed for key=%s", key, exc_info=True)
