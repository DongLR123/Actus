from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.external.embedding.redis_embedding_cache import RedisEmbeddingCache


def _make_redis_mock() -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock()
    redis.set = AsyncMock()
    return redis


@pytest.mark.anyio
async def test_get_returns_none_on_miss():
    redis = _make_redis_mock()
    redis.get.return_value = None

    cache = RedisEmbeddingCache(redis, ttl=3600)
    result = await cache.get("missing_key")

    assert result is None
    redis.get.assert_awaited_once_with("skill_emb:missing_key")


@pytest.mark.anyio
async def test_get_returns_vector_on_hit():
    vector = [0.1, 0.2, 0.3]
    redis = _make_redis_mock()
    redis.get.return_value = json.dumps(vector)

    cache = RedisEmbeddingCache(redis, ttl=3600)
    result = await cache.get("hit_key")

    assert result == vector
    redis.get.assert_awaited_once_with("skill_emb:hit_key")


@pytest.mark.anyio
async def test_set_stores_json_with_prefix_and_ttl():
    redis = _make_redis_mock()
    ttl = 7200
    vector = [0.4, 0.5, 0.6]

    cache = RedisEmbeddingCache(redis, ttl=ttl)
    await cache.set("store_key", vector)

    redis.set.assert_awaited_once_with(
        "skill_emb:store_key",
        json.dumps(vector),
        ex=ttl,
    )
