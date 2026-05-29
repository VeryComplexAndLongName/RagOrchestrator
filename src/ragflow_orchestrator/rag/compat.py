from __future__ import annotations

from ragflow_orchestrator.context import DocChunk, to_doc_chunk
from ragflow_orchestrator.models import RetrievalQuery


class PromptStyleRAGProviderAdapter:
    """Adapter to expose prompt_orchestrator-style retrieve(query, limit)."""

    def __init__(self, provider: object, embedder: object) -> None:
        self.provider = provider
        self.embedder = embedder

    def retrieve(self, query: str, limit: int) -> list[DocChunk]:
        payload = RetrievalQuery(
            text_query=query,
            top_k=limit,
            dense_vector=self.embedder.embed(query),
        )
        rows = self.provider.retrieve(payload)
        return [to_doc_chunk(row.chunk, score=row.score) for row in rows]
