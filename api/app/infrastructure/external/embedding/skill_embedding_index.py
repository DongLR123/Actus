"""Vector index for skill semantic search."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.domain.external.embedding_cache import EmbeddingCache
    from app.domain.external.embedding_provider import EmbeddingProvider
    from app.domain.models.skill import Skill

logger = logging.getLogger(__name__)


class SkillEmbeddingIndex:
    """In-memory numpy-backed vector index for skill semantic search."""

    BATCH_SIZE = 256

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._lock = asyncio.Lock()
        self._skill_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(self, skills: list[Skill]) -> None:
        """Build the full index from a list of skills."""
        if not skills:
            dims = self._provider.dimensions
            self._skill_ids = []
            self._embeddings = np.empty((0, dims), dtype=np.float32)
            return

        texts = [self._skill_to_text(s) for s in skills]
        vectors = await self._embed_with_cache(texts)

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        self._skill_ids = [s.id for s in skills]
        self._embeddings = matrix

    async def query(
        self,
        message: str,
        top_k: int = 12,
    ) -> list[tuple[str, float]]:
        """Return top-k (skill_id, score) pairs ranked by cosine similarity."""
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return []

        raw = await self._provider.embed([message])
        q_vec = np.array(raw[0], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        scores = self._embeddings @ q_vec  # cosine similarity via dot product
        k = min(top_k, len(self._skill_ids))
        top_indices = np.argsort(scores)[::-1][:k]

        return [(self._skill_ids[i], float(scores[i])) for i in top_indices]

    async def add(self, skill: Skill) -> None:
        """Add a single skill to the index (thread-safe)."""
        async with self._lock:
            text = self._skill_to_text(skill)
            raw = await self._provider.embed([text])
            vec = np.array(raw[0], dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            self._skill_ids.append(skill.id)
            if self._embeddings is None or self._embeddings.shape[0] == 0:
                self._embeddings = vec
            else:
                self._embeddings = np.vstack([self._embeddings, vec])

    async def remove(self, skill_id: str) -> None:
        """Remove a skill from the index by id (thread-safe)."""
        async with self._lock:
            if skill_id not in self._skill_ids:
                return
            idx = self._skill_ids.index(skill_id)
            self._skill_ids.pop(idx)
            if self._embeddings is not None:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skill_to_text(skill: Skill) -> str:
        """Combine skill metadata into a single searchable text string."""
        manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
        context_blob = str(manifest.get("context_blob") or "")

        parts: list[str] = [
            skill.name,
            skill.description,
            context_blob[:500],
        ]

        for tool in manifest.get("tools", []):
            if isinstance(tool, dict):
                parts.append(str(tool.get("name") or ""))
                parts.append(str(tool.get("description") or ""))

        activation = manifest.get("activation") or {}
        if isinstance(activation, dict):
            for keyword in activation.get("keywords", []) or []:
                parts.append(str(keyword))

        return " ".join(parts)

    def _cache_key(self, text: str) -> str:
        """Deterministic cache key based on provider model + text content."""
        raw = f"{self._provider.model_name}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _embed_with_cache(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed texts, using cache for hits and batching misses."""
        results: list[list[float] | None] = [None] * len(texts)
        to_embed_indices: list[int] = []
        to_embed_texts: list[str] = []

        # Check cache for each text
        if self._cache is not None:
            for i, text in enumerate(texts):
                key = self._cache_key(text)
                cached = await self._cache.get(key)
                if cached is not None:
                    results[i] = cached
                else:
                    to_embed_indices.append(i)
                    to_embed_texts.append(text)
        else:
            to_embed_indices = list(range(len(texts)))
            to_embed_texts = list(texts)

        # Batch embed uncached texts
        if to_embed_texts:
            all_vectors: list[list[float]] = []
            for batch_start in range(0, len(to_embed_texts), self.BATCH_SIZE):
                batch = to_embed_texts[batch_start : batch_start + self.BATCH_SIZE]
                batch_vectors = await self._provider.embed(batch)
                all_vectors.extend(batch_vectors)

            for j, idx in enumerate(to_embed_indices):
                results[idx] = all_vectors[j]
                # Store in cache
                if self._cache is not None:
                    key = self._cache_key(to_embed_texts[j])
                    await self._cache.set(key, all_vectors[j])

        if any(v is None for v in results):
            logger.warning("Embedding results incomplete, %d/%d missing",
                           sum(1 for v in results if v is None), len(results))
            # Fill missing with zero vectors as fallback
            dims = self._provider.dimensions
            results = [v if v is not None else [0.0] * dims for v in results]

        return results  # type: ignore[return-value]
