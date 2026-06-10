from __future__ import annotations

from ragflow_orchestrator.adapters.common import cosine_similarity
from ragflow_orchestrator.models import RetrievalQuery, RetrievalResult
from ragflow_orchestrator.protocols import Embedder, RAGProvider
from ragflow_orchestrator.retrieval.strategies import HybridRetriever, SemanticRetriever


class MetadataAwareHybridRetriever:
    """Hybrid retriever with additional metadata-aware score boosts."""

    def __init__(self, provider: RAGProvider, embedder: Embedder, lexical_weight: float = 0.15) -> None:
        self.provider = provider
        self.embedder = embedder
        self.lexical_weight = max(0.0, min(1.0, lexical_weight))

    def search(self, query_text: str, top_k: int = 3, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        dense = self.embedder.embed(query_text)
        retrieval_query = RetrievalQuery(
            text_query=query_text,
            top_k=max(top_k * 4, top_k),
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

            metadata = item.chunk.metadata
            quality = float(metadata.get("quality_score", item.chunk.quality_score))
            risk = float(metadata.get("risk_score", item.chunk.risk_score))
            source_type = str(metadata.get("source_type", item.chunk.source_type))

            source_boost = 0.05 if source_type and source_type in query_text.lower() else 0.0
            score = (1 - self.lexical_weight) * dense_score + self.lexical_weight * lexical
            score += 0.2 * quality
            score -= 0.1 * risk
            score += source_boost
            rescored.append(RetrievalResult(chunk=item.chunk, score=score, provider=f"{item.provider}+metadata_hybrid"))

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]


class AdaptiveRetriever:
    """Chooses retrieval strategy based on query characteristics and applies weighted rerank."""

    def __init__(self, provider: RAGProvider, embedder: Embedder, reranker: object | None = None) -> None:
        self.semantic = SemanticRetriever(provider=provider, embedder=embedder)
        self.hybrid = HybridRetriever(provider=provider, embedder=embedder)
        self.metadata_hybrid = MetadataAwareHybridRetriever(provider=provider, embedder=embedder)
        self.reranker = reranker

    def search(self, query_text: str, top_k: int = 3, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        strategy = self._pick_strategy(query_text)
        base = strategy.search(query_text=query_text, top_k=top_k, filters=filters)
        if self.reranker is None:
            return base
        return self.reranker.rerank(query_text=query_text, candidates=base, top_k=top_k)

    def _pick_strategy(self, query_text: str) -> object:
        lowered = query_text.lower()
        code_markers = ("def ", "class ", "stack trace", "exception", "sql", "endpoint")
        if any(marker in lowered for marker in code_markers):
            return self.metadata_hybrid
        if len(query_text.split()) <= 3:
            return self.semantic
        return self.hybrid
