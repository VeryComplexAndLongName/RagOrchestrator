import os

from ragflow_orchestrator import HashEmbedder, RAGOrchestrator, create_provider, document_preset
from ragflow_orchestrator.query_engine import RAGQueryEngine


def main() -> None:
    provider = create_provider(
        "postgres+qdrant",
        dsn=os.getenv("RAG_POSTGRES_DSN", "postgresql://rag_user:rag_password@localhost:5432/rag_db"),
        qdrant_url=os.getenv("RAG_QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("RAG_QDRANT_COLLECTION", "query_chunks"),
    )
    preset = document_preset()

    orchestrator = RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=128),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )

    orchestrator.ingest(
        source_id="repo:demo",
        raw_text="This repository implements Telegram bots and webhook handlers.",
        metadata={"source_type": "github_repo", "full_name": "demo/repo", "stars": 42},
    )

    engine = RAGQueryEngine(orchestrator)
    result = engine.answer("Найди репозитории, которые реализуют Telegram-ботов", top_k=3)

    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Context chunks:", len(result.context))


if __name__ == "__main__":
    main()
