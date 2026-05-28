from __future__ import annotations

from typing import Iterable, Protocol

from rag_orchestrator.models import BaseChunk, RetrievalQuery, RetrievalResult


class Chunker(Protocol):
    def chunk(self, source_id: str, text: str, metadata: dict[str, object] | None = None) -> list[BaseChunk]:
        ...


class Cleaner(Protocol):
    def clean(self, text: str) -> str:
        ...


class Embedder(Protocol):
    @property
    def dimensions(self) -> int:
        ...

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        ...


class RAGProvider(Protocol):
    name: str

    def ensure_schema(self, vector_dim: int) -> None:
        ...

    def upsert_chunks(self, chunks: list[BaseChunk]) -> None:
        ...

    def delete_chunks(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        ...

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        ...

    def healthcheck(self) -> bool:
        ...


class MigrationStep(Protocol):
    version: int
    description: str

    def up(self) -> None:
        ...

    def down(self) -> None:
        ...
