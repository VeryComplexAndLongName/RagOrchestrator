from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ragflow_orchestrator.dedup import DedupStore
from ragflow_orchestrator.models import BaseChunk, RetrievalQuery, RetrievalResult
from ragflow_orchestrator.protocols import Chunker, Cleaner, Embedder, RAGProvider
from ragflow_orchestrator.telemetry import init_telemetry, telemetry


@dataclass(slots=True)
class IngestSummary:
    source_id: str
    total_chunks: int
    average_length: float
    duplicate_chunks_skipped: int = 0


class RAGOrchestrator:
    def __init__(
        self,
        provider: RAGProvider,
        embedder: Embedder,
        chunker: Chunker,
        cleaner: Cleaner,
        dedup_store: DedupStore | None = None,
    ) -> None:
        init_telemetry(service_name="ragflow-orchestrator")
        self.provider = provider
        self.embedder = embedder
        self.chunker = chunker
        self.cleaner = cleaner
        self.dedup_store = dedup_store or self._build_default_dedup_store(provider)
        self.provider.ensure_schema(embedder.dimensions)

    def ingest(self, source_id: str, raw_text: str, metadata: dict[str, object] | None = None) -> IngestSummary:
        started = perf_counter()
        with telemetry.span("rag.ingest", {"source.id": source_id}):
            try:
                cleaned = self.cleaner.clean(raw_text)
                chunks = self.chunker.chunk(source_id=source_id, text=cleaned, metadata=metadata or {})
                vectors = self.embedder.embed_many([chunk.text for chunk in chunks])

                final_chunks: list[BaseChunk] = []
                duplicates = 0
                for chunk, vector in zip(chunks, vectors, strict=False):
                    fingerprint = DedupStore.fingerprint(chunk.text)
                    if self.dedup_store.is_known(fingerprint):
                        duplicates += 1
                        continue
                    chunk.vector = vector
                    chunk.metadata = dict(chunk.metadata)
                    chunk.metadata["dedup_fingerprint"] = fingerprint
                    final_chunks.append(chunk)
                    self.dedup_store.add(fingerprint=fingerprint, source_id=source_id, chunk_id=chunk.id)

                self.provider.upsert_chunks(final_chunks)

                total_len = sum(len(chunk.text) for chunk in final_chunks)
                avg_len = total_len / len(final_chunks) if final_chunks else 0.0
                summary = IngestSummary(
                    source_id=source_id,
                    total_chunks=len(final_chunks),
                    average_length=avg_len,
                    duplicate_chunks_skipped=duplicates,
                )
                telemetry.record_ingest(
                    duration_ms=(perf_counter() - started) * 1000.0,
                    chunks=summary.total_chunks,
                    duplicates=summary.duplicate_chunks_skipped,
                    status="ok",
                )
                return summary
            except Exception as exc:
                telemetry.record_error("ingest", type(exc).__name__)
                telemetry.record_ingest(
                    duration_ms=(perf_counter() - started) * 1000.0,
                    chunks=0,
                    duplicates=0,
                    status="error",
                )
                raise

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        filters: dict[str, object] | None = None,
        include_deleted: bool = False,
    ) -> list[RetrievalResult]:
        started = perf_counter()
        with telemetry.span("rag.search", {"retrieval.top_k": top_k}):
            try:
                vector = self.embedder.embed(query_text)
                query = RetrievalQuery(
                    text_query=query_text,
                    top_k=top_k,
                    include_deleted=include_deleted,
                    dense_vector=vector,
                    filters=[{"key": key, "value": value} for key, value in (filters or {}).items()],
                )
                results = self.provider.retrieve(query)
                telemetry.record_search(
                    duration_ms=(perf_counter() - started) * 1000.0,
                    top_k=top_k,
                    results=len(results),
                    status="ok",
                )
                return results
            except Exception as exc:
                telemetry.record_error("search", type(exc).__name__)
                telemetry.record_search(
                    duration_ms=(perf_counter() - started) * 1000.0,
                    top_k=top_k,
                    results=0,
                    status="error",
                )
                raise

    def delete(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        started = perf_counter()
        with telemetry.span("rag.delete", {"chunks.count": len(chunk_ids), "soft_delete": soft_delete}):
            try:
                self.provider.delete_chunks(chunk_ids=chunk_ids, soft_delete=soft_delete)
                telemetry.record_delete(
                    duration_ms=(perf_counter() - started) * 1000.0,
                    deleted_count=len(chunk_ids),
                    status="ok",
                )
            except Exception as exc:
                telemetry.record_error("delete", type(exc).__name__)
                telemetry.record_delete(
                    duration_ms=(perf_counter() - started) * 1000.0,
                    deleted_count=len(chunk_ids),
                    status="error",
                )
                raise

    @staticmethod
    def _build_default_dedup_store(provider: RAGProvider) -> DedupStore:
        db_path = getattr(provider, "db_path", None)
        if db_path:
            base = Path(str(db_path))
            dedup_path = base.with_suffix(base.suffix + ".dedup.sqlite")
            return DedupStore(str(dedup_path))
        return DedupStore()
