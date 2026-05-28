from __future__ import annotations

from pathlib import Path

from rag_orchestrator.embedding import HashEmbedder
from rag_orchestrator.evaluation import evaluate_strategies
from rag_orchestrator.factory import create_provider
from rag_orchestrator.orchestrator import RAGOrchestrator
from rag_orchestrator.presets import document_preset
from rag_orchestrator.retrieval import CosineReranker, HybridRetriever, RerankedRetriever, SemanticRetriever


def test_strategy_evaluation_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    provider = create_provider("sqlite+vec", db_path=str(db_path), table_name="eval_chunks")
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
    orchestrator.ingest(
        source_id="rag_doc",
        raw_text="RAG orchestration unifies ingestion and retrieval.",
        metadata={"language": "en", "doctype": "note"},
    )

    semantic = SemanticRetriever(provider=provider, embedder=embedder)
    hybrid = HybridRetriever(provider=provider, embedder=embedder)
    reranked = RerankedRetriever(base_strategy=semantic, reranker=CosineReranker(embedder=embedder))

    report = evaluate_strategies(
        strategies={
            "semantic": semantic,
            "hybrid": hybrid,
            "semantic_cosine_rerank": reranked,
        },
        dataset_path="datasets/retrieval_eval.jsonl",
        top_k=2,
    )

    assert report
    assert report[0].strategy_name in {"semantic", "hybrid", "semantic_cosine_rerank"}
