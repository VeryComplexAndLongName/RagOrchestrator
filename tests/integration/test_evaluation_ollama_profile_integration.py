from __future__ import annotations

import pytest

from ragflow_orchestrator.embedding import HashEmbedder, OllamaEmbedder
from ragflow_orchestrator.evaluation import RerankProfileConfig, evaluate_rerank_profiles
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset


@pytest.mark.integration
def test_evaluate_rerank_profiles_with_ollama(tmp_path) -> None:
    provider = create_provider("sqlite+vec", db_path=str(tmp_path / "ollama_profiles.db"), table_name="ollama_profile_chunks")
    embedder = HashEmbedder(dimensions=64)
    preset = document_preset()

    orchestrator = RAGOrchestrator(
        provider=provider,
        embedder=embedder,
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )

    orchestrator.ingest(
        source_id="rag_doc",
        raw_text="RAG orchestration helps retrieval quality evaluation.",
        metadata={"language": "en", "doctype": "note"},
    )

    try:
        models = OllamaEmbedder.list_models()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Ollama is unavailable: {exc}")

    llm_candidates = [model for model in models if "embed" not in model.lower()]
    if not llm_candidates:
        pytest.skip("No Ollama LLM model available for reranking")

    reports = evaluate_rerank_profiles(
        provider=provider,
        embedder=embedder,
        dataset_path="datasets/retrieval_eval.jsonl",
        profiles=[
            RerankProfileConfig(name="cosine_profile", reranker_provider="cosine"),
            RerankProfileConfig(
                name="ollama_profile",
                reranker_provider="ollama",
                reranker_model=llm_candidates[0],
            ),
        ],
        top_k=2,
    )

    assert "cosine_profile" in reports
    assert "ollama_profile" in reports
    assert reports["ollama_profile"]
