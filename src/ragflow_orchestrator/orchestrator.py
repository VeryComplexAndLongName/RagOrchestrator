from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

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
                    chunk.metadata = self._enrich_metadata(
                        source_id=source_id,
                        chunk=chunk,
                        metadata=chunk.metadata,
                        fingerprint=fingerprint,
                    )
                    chunk.semantic_type = str(chunk.metadata.get("semantic_type", chunk.semantic_type))
                    chunk.quality_score = float(chunk.metadata.get("quality_score", chunk.quality_score))
                    chunk.token_count = int(chunk.metadata.get("token_count", chunk.token_count))
                    chunk.source_type = str(chunk.metadata.get("source_type", chunk.source_type))
                    chunk.domain = str(chunk.metadata.get("domain", chunk.domain))
                    chunk.risk_score = float(chunk.metadata.get("risk_score", chunk.risk_score))
                    chunk.embedding_model = str(chunk.metadata.get("embedding_model", chunk.embedding_model))
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

    def _enrich_metadata(
        self,
        source_id: str,
        chunk: BaseChunk,
        metadata: dict[str, object],
        fingerprint: str,
    ) -> dict[str, object]:
        out = dict(metadata)
        text = chunk.text

        source_url = str(out.get("source_url") or out.get("source_origin") or "")
        inferred_source_type = self._infer_source_type(source_id=source_id, metadata=out)
        inferred_document_type = self._infer_document_type(metadata=out)
        inferred_domain = self._infer_domain(source_url)
        inferred_semantic_type = self._infer_semantic_type(chunk=chunk, text=text)
        token_count = self._count_tokens(text)
        quality_score = self._estimate_quality_score(text=text, metadata=out)
        risk_score = self._estimate_risk_score(text=text, metadata=out)
        embedding_model = self._embedding_model_name()

        out.setdefault("source_id", source_id)
        out["dedup_fingerprint"] = fingerprint
        out.setdefault("source_type", inferred_source_type)
        out.setdefault("document_type", inferred_document_type)
        out.setdefault("domain", inferred_domain)
        out.setdefault("semantic_type", inferred_semantic_type)
        out["token_count"] = token_count
        out["quality_score"] = quality_score
        out["risk_score"] = risk_score
        out["embedding_model"] = embedding_model
        return out

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len([token for token in text.split() if token])

    @staticmethod
    def _infer_source_type(source_id: str, metadata: dict[str, object]) -> str:
        explicit = metadata.get("source_type")
        if explicit:
            return str(explicit)
        if ":" in source_id:
            prefix = source_id.split(":", maxsplit=1)[0].strip()
            if prefix:
                return prefix
        return "unknown"

    @staticmethod
    def _infer_document_type(metadata: dict[str, object]) -> str:
        explicit = metadata.get("document_type") or metadata.get("doctype")
        if explicit:
            return str(explicit)
        file_ext = str(metadata.get("file_ext") or "").lower().strip()
        if file_ext in {".md", ".markdown"}:
            return "markdown"
        if file_ext in {".html", ".htm"}:
            return "html"
        if file_ext == ".docx":
            return "docx"
        if file_ext == ".xlsx":
            return "xlsx"
        if file_ext == ".pdf":
            return "pdf"
        if file_ext == ".json" or file_ext == ".jsonl":
            return "json"
        if file_ext == ".csv":
            return "csv"
        if file_ext == ".xml":
            return "xml"
        if file_ext == ".txt":
            return "txt"
        return "unsupported"

    @staticmethod
    def _infer_domain(source_url: str) -> str:
        if not source_url:
            return ""
        parsed = urlparse(source_url)
        return parsed.netloc.lower()

    @staticmethod
    def _infer_semantic_type(chunk: BaseChunk, text: str) -> str:
        if chunk.kind.value == "code":
            return "code"
        lowered = text.lower()
        if "|" in text and "\n" in text:
            return "table"
        if any(token in lowered for token in ("error", "exception", "traceback", "failed")):
            return "log"
        if any(token in lowered for token in ("endpoint", "request", "response", "api")):
            return "api"
        if "?" in text and any(token in lowered for token in ("q:", "question", "faq")):
            return "faq"
        return "narrative"

    @staticmethod
    def _estimate_quality_score(text: str, metadata: dict[str, object]) -> float:
        length = len(text)
        score = 0.8

        if length < 60:
            score -= 0.35
        elif length < 120:
            score -= 0.15
        elif length > 6000:
            score -= 0.2

        lowered = text.lower()
        noise_markers = ("cookie", "all rights reserved", "subscribe", "privacy policy", "javascript")
        score -= 0.08 * sum(1 for marker in noise_markers if marker in lowered)

        if str(metadata.get("language") or "").strip() == "":
            score -= 0.05

        return max(0.0, min(1.0, score))

    @staticmethod
    def _estimate_risk_score(text: str, metadata: dict[str, object]) -> float:
        lowered = text.lower()
        score = 0.0
        risk_terms = (
            "password",
            "token",
            "secret",
            "private key",
            "ssn",
            "credit card",
            "passport",
        )
        score += 0.1 * sum(1 for term in risk_terms if term in lowered)

        source_type = str(metadata.get("source_type") or "")
        if source_type in {"email_ticket", "jira", "confluence"}:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _embedding_model_name(self) -> str:
        for attr in ("model_name", "model"):
            value = getattr(self.embedder, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return type(self.embedder).__name__
