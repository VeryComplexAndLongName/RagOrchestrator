from ragflow_orchestrator import (
    DocumentFolderConfig,
    DocumentFolderTemplate,
    HashEmbedder,
    LanguageMode,
    RAGOrchestrator,
    WebCrawlConfig,
    WebCrawlTemplate,
    create_provider,
    document_preset,
)


def build_orchestrator() -> RAGOrchestrator:
    provider = create_provider("sqlite+vec", db_path="template_demo.db", table_name="template_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=128),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def run_web_template(orchestrator: RAGOrchestrator) -> None:
    template = WebCrawlTemplate(orchestrator)
    report = template.run(
        WebCrawlConfig(
            urls=["https://example.com"],
            max_depth=1,
            same_domain_only=True,
            language_mode=LanguageMode.AUTO,
        )
    )
    print("web ingested:", len(report.ingested), "failed:", len(report.failed), "skipped:", len(report.skipped))


def run_folder_template(orchestrator: RAGOrchestrator) -> None:
    template = DocumentFolderTemplate(orchestrator)
    report = template.run(
        DocumentFolderConfig(
            folders=["datasets"],
            recursive=True,
            extensions=[".txt", ".md", ".html", ".docx", ".pdf", ".xlsx"],
            language_mode=LanguageMode.MIXED,
        )
    )
    print("files ingested:", len(report.ingested), "failed:", len(report.failed), "skipped:", len(report.skipped))


def main() -> None:
    orchestrator = build_orchestrator()
    run_web_template(orchestrator)
    run_folder_template(orchestrator)


if __name__ == "__main__":
    main()
