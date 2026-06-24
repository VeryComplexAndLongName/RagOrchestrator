import os

from ragflow_orchestrator.embedding import HashEmbedder, OllamaEmbedder
from ragflow_orchestrator.evaluation import RerankProfileConfig, evaluate_rerank_profiles
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset


def build_demo_index() -> tuple[object, object]:
    provider = create_provider(
        "postgres+qdrant",
        dsn=os.getenv("RAG_POSTGRES_DSN", "postgresql://rag_user:rag_password@localhost:5432/rag_db"),
        qdrant_url=os.getenv("RAG_QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("RAG_QDRANT_COLLECTION", "eval_chunks"),
    )
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
        raw_text="RAG orchestration unifies ingestion, retrieval and evaluation.",
        metadata={"language": "en", "doctype": "note"},
    )

    return provider, embedder


def main() -> None:
    provider, embedder = build_demo_index()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("RAG_RERANK_MODEL")

    if not ollama_model:
        try:
            models = OllamaEmbedder.list_models(base_url=ollama_base_url)
            candidates = [model for model in models if "embed" not in model.lower()]
            if candidates:
                ollama_model = candidates[0]
        except Exception:
            ollama_model = None

    profiles: list[RerankProfileConfig] = [
        RerankProfileConfig(
            name="cosine_profile",
            reranker_provider="cosine",
        ),
    ]

    if ollama_model:
        profiles.append(
            RerankProfileConfig(
                name="ollama_profile",
                reranker_provider="ollama",
                reranker_model=ollama_model,
                reranker_options={"base_url": ollama_base_url, "timeout_seconds": 60},
            )
        )
    else:
        print("ollama_profile skipped: set RAG_RERANK_MODEL or run local Ollama with available LLM model")

    profile_reports = evaluate_rerank_profiles(
        provider=provider,
        embedder=embedder,
        dataset_path="datasets/retrieval_eval.jsonl",
        profiles=profiles,
        top_k=2,
    )

    for profile_name, reports in profile_reports.items():
        print(f"\n[{profile_name}]")
        for item in reports:
            print(
                f"{item.strategy_name}: "
                f"precision@k={item.precision_at_k:.3f}, "
                f"recall@k={item.recall_at_k:.3f}, "
                f"mrr={item.mrr:.3f}"
            )


if __name__ == "__main__":
    main()
