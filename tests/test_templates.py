from __future__ import annotations

from pathlib import Path

from rag_orchestrator.embedding import HashEmbedder
from rag_orchestrator.factory import create_provider
from rag_orchestrator.orchestrator import RAGOrchestrator
from rag_orchestrator.presets import document_preset
from rag_orchestrator.templates import (
    DocumentFolderConfig,
    DocumentFolderTemplate,
    EmailTicketConfig,
    EmailTicketTemplate,
    IncrementalSyncConfig,
    IncrementalSyncTemplate,
    LanguageMode,
    RepoCodeConfig,
    RepoCodeTemplate,
    WebCrawlConfig,
    WebCrawlTemplate,
)


def _build_orchestrator(tmp_path: Path) -> RAGOrchestrator:
    provider = create_provider("sqlite+vec", db_path=str(tmp_path / "tpl.db"), table_name="tpl_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=64),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def test_document_folder_template_ingests_txt(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text("Привет hello mixed document", encoding="utf-8")

    orchestrator = _build_orchestrator(tmp_path)
    template = DocumentFolderTemplate(orchestrator)

    report = template.run(
        DocumentFolderConfig(
            folders=[str(docs)],
            recursive=True,
            extensions=[".txt"],
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_web_template_with_stub_html(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = WebCrawlTemplate(orchestrator)

    def fake_fetch(url: str, timeout: int = 15) -> str:
        del timeout
        if url.endswith("/start"):
            return "<html><body><a href='https://site.local/page'>next</a><h1>Hello world</h1></body></html>"
        return "<html><body><p>Second page</p></body></html>"

    template._fetch_html = fake_fetch  # type: ignore[method-assign]

    report = template.run(
        WebCrawlConfig(
            urls=["https://site.local/start"],
            max_depth=1,
            same_domain_only=True,
            max_pages=5,
            language_mode=LanguageMode.FORCE_EN,
        )
    )

    assert len(report.ingested) == 2
    assert not report.failed


def test_repo_code_template_ingests_python_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def add(a,b): return a+b", encoding="utf-8")

    orchestrator = _build_orchestrator(tmp_path)
    template = RepoCodeTemplate(orchestrator)

    report = template.run(
        RepoCodeConfig(
            repos=[str(repo)],
            recursive=True,
            extensions=[".py"],
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_email_ticket_template_ingests_jsonl(tmp_path: Path) -> None:
    tickets = tmp_path / "tickets"
    tickets.mkdir()
    (tickets / "tickets.jsonl").write_text(
        "{\"ticket_id\":\"t1\",\"subject\":\"Issue\",\"body\":\"Need help with login\"}\n",
        encoding="utf-8",
    )

    orchestrator = _build_orchestrator(tmp_path)
    template = EmailTicketTemplate(orchestrator)

    report = template.run(
        EmailTicketConfig(
            sources=[str(tickets)],
            recursive=True,
            extensions=[".jsonl"],
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_incremental_sync_only_ingests_changed_files(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "note.txt"
    file_path.write_text("version one", encoding="utf-8")

    orchestrator = _build_orchestrator(tmp_path)
    template = IncrementalSyncTemplate(orchestrator)
    cfg = IncrementalSyncConfig(
        folders=[str(docs)],
        recursive=True,
        extensions=[".txt"],
        state_file=str(tmp_path / ".sync_state.json"),
        language_mode=LanguageMode.AUTO,
    )

    first = template.run(cfg)
    second = template.run(cfg)
    file_path.write_text("version two", encoding="utf-8")
    third = template.run(cfg)

    assert len(first.ingested) == 1
    assert len(second.ingested) == 0
    assert len(third.ingested) == 1


def test_orchestrator_skips_duplicate_content(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    first = orchestrator.ingest(source_id="a", raw_text="same same same", metadata={"doctype": "note"})
    second = orchestrator.ingest(source_id="b", raw_text="same same same", metadata={"doctype": "note"})

    assert first.total_chunks > 0
    assert second.total_chunks == 0
    assert second.duplicate_chunks_skipped > 0
