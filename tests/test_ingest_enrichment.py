from __future__ import annotations

from pathlib import Path

from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset


def test_ingest_enriches_chunk_metadata_and_fields(tmp_path: Path) -> None:
    provider = create_provider("postgres+qdrant", dsn="postgresql://rag_user:rag_password@localhost:5432/rag_db", qdrant_url="http://localhost:6333", qdrant_collection="enriched_chunks")
    preset = document_preset()
    orchestrator = RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=64),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )

    orchestrator.ingest(
        source_id="gitlab:team/repo",
        raw_text="API endpoint for deployment status and incident logs.",
        metadata={"source_url": "https://gitlab.com/team/repo"},
    )

    hits = orchestrator.search("deployment endpoint", top_k=1)
    assert hits

    top = hits[0].chunk
    assert top.source_type == "gitlab"
    assert top.semantic_type in {"api", "log", "narrative"}
    assert top.token_count > 0
    assert 0.0 <= top.quality_score <= 1.0
    assert top.domain == "gitlab.com"
    assert top.embedding_model

    assert top.metadata.get("source_type") == top.source_type
    assert top.metadata.get("domain") == top.domain
    assert top.metadata.get("token_count") == top.token_count


