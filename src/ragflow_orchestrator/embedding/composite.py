from __future__ import annotations

from collections import OrderedDict
from typing import Iterable


class CachedEmbedder:
    """Wraps an embedder with a small in-memory LRU cache for repeated texts."""

    def __init__(self, base_embedder: object, max_items: int = 4096) -> None:
        self.base_embedder = base_embedder
        self.max_items = max(16, int(max_items))
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def dimensions(self) -> int:
        return int(getattr(self.base_embedder, "dimensions"))

    @property
    def model_name(self) -> str:
        inner = getattr(self.base_embedder, "model_name", None) or getattr(self.base_embedder, "model", None)
        if isinstance(inner, str) and inner:
            return f"cached:{inner}"
        return f"cached:{type(self.base_embedder).__name__}"

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        vector = list(self.base_embedder.embed(text))
        self._cache[text] = vector
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return vector

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class FallbackEmbedder:
    """Uses primary embedder and falls back to secondary provider on failures/empty vectors."""

    def __init__(self, primary: object, secondary: object) -> None:
        self.primary = primary
        self.secondary = secondary
        primary_dim = int(getattr(primary, "dimensions"))
        secondary_dim = int(getattr(secondary, "dimensions"))
        if primary_dim != secondary_dim:
            raise ValueError("Primary and secondary embedders must have the same dimensions")
        self._dimensions = primary_dim

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        primary_name = getattr(self.primary, "model_name", None) or getattr(self.primary, "model", type(self.primary).__name__)
        secondary_name = getattr(self.secondary, "model_name", None) or getattr(self.secondary, "model", type(self.secondary).__name__)
        return f"fallback:{primary_name}|{secondary_name}"

    def embed(self, text: str) -> list[float]:
        vector = self._try_embed_single(self.primary, text)
        if vector:
            return vector
        return self._try_embed_single(self.secondary, text)

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    @staticmethod
    def _try_embed_single(embedder: object, text: str) -> list[float]:
        try:
            vector = list(embedder.embed(text))
        except Exception:
            return []
        return vector
