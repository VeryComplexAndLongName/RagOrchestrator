from ragflow_orchestrator import HashEmbedder, RAGOrchestrator, create_provider, document_preset
from ragflow_orchestrator.query_engine import RAGQueryEngine


def main() -> None:
    provider = create_provider("sqlite+vec", db_path="query_demo.db", table_name="query_chunks")
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
