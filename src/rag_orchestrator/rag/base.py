from __future__ import annotations

from abc import ABC, abstractmethod

from rag_orchestrator.context import DocChunk


class RAGProvider(ABC):
    @abstractmethod
    def retrieve(self, query: str, limit: int) -> list[DocChunk]:
        raise NotImplementedError
