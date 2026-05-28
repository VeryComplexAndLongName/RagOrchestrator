from __future__ import annotations

from pathlib import Path

from rag_orchestrator.embedding import HashEmbedder
from rag_orchestrator.evaluation import RerankProfileConfig, evaluate_rerank_profiles
from rag_orchestrator.factory import create_provider
from rag_orchestrator.orchestrator import RAGOrchestrator
from rag_orchestrator.presets import document_preset


def test_evaluate_rerank_profiles_with_cosine(tmp_path: Path) -> None:
    provider = create_provider("sqlite+vec", db_path=str(tmp_path / "profiles.db"), table_name="profile_chunks")
    embedder = HashEmbedder(dimensions=64)
    preset = document_preset()

    orchestrator = RAGOrchestrator(
        provider=provider,
        embedder=embedder,
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )

    orchestrator.ingest(
        source_id="math_utils.py",
        raw_text="def add(a,b): return a+b\ndef sub(a,b): return a-b",
        metadata={"language": "python", "doctype": "code"},
    )

    reports = evaluate_rerank_profiles(
        provider=provider,
        embedder=embedder,
        dataset_path="datasets/retrieval_eval.jsonl",
        profiles=[
            RerankProfileConfig(name="cosine_profile", reranker_provider="cosine"),
        ],
        top_k=2,
    )

    assert "cosine_profile" in reports
    assert reports["cosine_profile"]
