"""Tests for SkillEmbeddingIndex."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import numpy as np
import pytest

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.infrastructure.external.embedding.skill_embedding_index import (
    SkillEmbeddingIndex,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_skill(
    skill_id: str,
    name: str,
    description: str = "",
    tools: list[dict] | None = None,
) -> Skill:
    manifest: dict = {}
    if tools:
        manifest["tools"] = tools
    return Skill(
        id=skill_id,
        slug=name.lower().replace(" ", "-"),
        name=name,
        description=description,
        source_type=SkillSourceType.LOCAL,
        source_ref="local",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest=manifest,
    )


def _make_provider(dims: int = 4) -> MagicMock:
    provider = MagicMock()
    type(provider).dimensions = PropertyMock(return_value=dims)
    type(provider).model_name = PropertyMock(return_value="test-model")
    provider.embed = AsyncMock()
    return provider


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


async def test_build_with_empty_skills():
    provider = _make_provider(dims=4)
    index = SkillEmbeddingIndex(provider)

    await index.build([])

    result = await index.query("anything")
    assert result == []
    assert index._embeddings is not None
    assert index._embeddings.shape == (0, 4)


async def test_build_and_query_returns_ranked_results():
    provider = _make_provider(dims=2)
    # Two skills: embeddings close to [1,0] and [0,1]
    provider.embed = AsyncMock(
        side_effect=[
            # build call: two skill embeddings
            [[1.0, 0.0], [0.0, 1.0]],
            # query call: query vector close to skill 1
            [[0.9, 0.1]],
        ]
    )

    s1 = _build_skill("s1", "Alpha")
    s2 = _build_skill("s2", "Beta")
    index = SkillEmbeddingIndex(provider)

    await index.build([s1, s2])
    results = await index.query("find alpha", top_k=2)

    assert len(results) == 2
    # s1 should rank first (closer to [1,0])
    assert results[0][0] == "s1"
    assert results[1][0] == "s2"
    assert results[0][1] > results[1][1]


async def test_query_before_build_returns_empty():
    provider = _make_provider(dims=4)
    index = SkillEmbeddingIndex(provider)

    result = await index.query("hello")
    assert result == []


async def test_add_increments_index():
    provider = _make_provider(dims=2)
    provider.embed = AsyncMock(
        side_effect=[
            # build: 1 skill
            [[1.0, 0.0]],
            # add: 1 skill
            [[0.0, 1.0]],
            # query
            [[0.6, 0.8]],
        ]
    )

    s1 = _build_skill("s1", "Alpha")
    s2 = _build_skill("s2", "Beta")
    index = SkillEmbeddingIndex(provider)

    await index.build([s1])
    await index.add(s2)

    results = await index.query("test", top_k=10)
    result_ids = [r[0] for r in results]
    assert "s1" in result_ids
    assert "s2" in result_ids
    assert len(results) == 2


async def test_remove_decrements_index():
    provider = _make_provider(dims=2)
    provider.embed = AsyncMock(
        side_effect=[
            # build: 2 skills
            [[1.0, 0.0], [0.0, 1.0]],
            # query after remove
            [[0.5, 0.5]],
        ]
    )

    s1 = _build_skill("s1", "Alpha")
    s2 = _build_skill("s2", "Beta")
    index = SkillEmbeddingIndex(provider)

    await index.build([s1, s2])
    await index.remove("s1")

    results = await index.query("test", top_k=10)
    result_ids = [r[0] for r in results]
    assert "s2" in result_ids
    assert "s1" not in result_ids
    assert len(results) == 1


async def test_cache_hit_skips_embed():
    provider = _make_provider(dims=2)
    cache = MagicMock()
    # All cache hits: return pre-computed vectors
    cache.get = AsyncMock(side_effect=[[1.0, 0.0], [0.0, 1.0]])
    cache.set = AsyncMock()

    s1 = _build_skill("s1", "Alpha")
    s2 = _build_skill("s2", "Beta")
    index = SkillEmbeddingIndex(provider, cache=cache)

    await index.build([s1, s2])

    # provider.embed should NOT have been called since all texts were cached
    provider.embed.assert_not_awaited()
    assert cache.get.await_count == 2


def test_skill_to_text_includes_name_description_tools():
    skill = _build_skill(
        skill_id="s1",
        name="MySkill",
        description="Does amazing things",
        tools=[
            {"name": "tool_a", "description": "Tool A desc"},
            {"name": "tool_b", "description": "Tool B desc"},
        ],
    )
    skill.manifest["activation"] = {"keywords": ["magic", "wizard"]}

    text = SkillEmbeddingIndex._skill_to_text(skill)

    assert "MySkill" in text
    assert "Does amazing things" in text
    assert "tool_a" in text
    assert "Tool A desc" in text
    assert "tool_b" in text
    assert "Tool B desc" in text
    assert "magic" in text
    assert "wizard" in text
