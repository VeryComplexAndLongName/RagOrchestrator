from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import code_preset


def main() -> None:
    provider = create_provider("sqlite+vec", db_path="example_rag.db")
    preset = code_preset()

    orchestrator = RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=256),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )

    source_code = """
    def add(a, b):
        return a + b

    def sub(a, b):
        return a - b
    """

    summary = orchestrator.ingest(
        source_id="math_utils.py",
        raw_text=source_code,
        metadata={"language": "python", "repo": "demo"},
    )

    print("ingested:", summary)

    results = orchestrator.search("function that adds numbers", top_k=2)
    for item in results:
        print(f"score={item.score:.4f} id={item.chunk.id} text={item.chunk.text!r}")


if __name__ == "__main__":
    main()
