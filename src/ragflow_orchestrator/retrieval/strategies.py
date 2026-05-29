from __future__ import annotations

from ragflow_orchestrator.adapters.common import cosine_similarity
from ragflow_orchestrator.models import RetrievalQuery, RetrievalResult
from ragflow_orchestrator.protocols import Embedder, RAGProvider


class SemanticRetriever:
    def __init__(self, provider: RAGProvider, embedder: Embedder) -> None:
        self.provider = provider
        self.embedder = embedder

    def search(self, query_text: str, top_k: int = 3, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        retrieval_query = RetrievalQuery(
            text_query=query_text,
            top_k=top_k,
            dense_vector=self.embedder.embed(query_text),
            filters=[{"key": key, "value": value} for key, value in (filters or {}).items()],
        )
        return self.provider.retrieve(retrieval_query)


class HybridRetriever:
    """Simple hybrid strategy: dense similarity + lexical overlap score."""

    def __init__(self, provider: RAGProvider, embedder: Embedder, lexical_weight: float = 0.15) -> None:
        self.provider = provider
        self.embedder = embedder
        self.lexical_weight = max(0.0, min(1.0, lexical_weight))

    def search(self, query_text: str, top_k: int = 3, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        dense = self.embedder.embed(query_text)
        retrieval_query = RetrievalQuery(
            text_query=query_text,
            top_k=max(top_k * 3, top_k),
            dense_vector=dense,
            filters=[{"key": key, "value": value} for key, value in (filters or {}).items()],
        )
        candidates = self.provider.retrieve(retrieval_query)
        query_tokens = set(query_text.lower().split())

        rescored: list[RetrievalResult] = []
        for item in candidates:
            text_tokens = set(item.chunk.text.lower().split())
            lexical = 0.0 if not query_tokens else len(query_tokens & text_tokens) / len(query_tokens)
            dense_score = cosine_similarity(dense, item.chunk.vector) if item.chunk.vector else item.score
            score = (1 - self.lexical_weight) * dense_score + self.lexical_weight * lexical
            rescored.append(RetrievalResult(chunk=item.chunk, score=score, provider=item.provider))

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]


class RerankedRetriever:
    def __init__(self, base_strategy: object, reranker: object) -> None:
        self.base_strategy = base_strategy
        self.reranker = reranker

    def search(self, query_text: str, top_k: int = 3, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        candidates = self.base_strategy.search(query_text=query_text, top_k=max(top_k * 3, top_k), filters=filters)
        return self.reranker.rerank(query_text=query_text, candidates=candidates, top_k=top_k)
