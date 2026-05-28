from __future__ import annotations

from pathlib import Path

from rag_orchestrator.templates.base import BaseIngestionTemplate
from rag_orchestrator.templates.extractors import extract_text
from rag_orchestrator.templates.models import DocumentFolderConfig, IngestionError, TemplateRunReport
from rag_orchestrator.templates.utils import extract_text_from_html


class DocumentFolderTemplate(BaseIngestionTemplate):
    def run(self, config: DocumentFolderConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        ext_set = {item.lower() for item in config.extensions}

        for folder in config.folders:
            root = Path(folder)
            if not root.exists():
                report.failed.append(IngestionError(source=folder, reason="folder does not exist"))
                continue

            iterator = root.rglob("*") if config.recursive else root.glob("*")
            for path in iterator:
                if not path.is_file():
                    continue
                if path.suffix.lower() not in ext_set:
                    continue

                try:
                    raw_text = extract_text(path)
                    if path.suffix.lower() == ".html":
                        raw_text = extract_text_from_html(raw_text)
                    if not raw_text.strip():
                        report.skipped.append(IngestionError(source=str(path), reason="empty extracted text"))
                        continue

                    language = self._language_tag(text=raw_text, mode=config.language_mode)
                    summary = self.orchestrator.ingest(
                        source_id=str(path),
                        raw_text=raw_text,
                        metadata={
                            "source_type": "file",
                            "file_path": str(path),
                            "file_ext": path.suffix.lower(),
                            "language": language,
                        },
                    )
                    report.ingested.append(summary)
                except Exception as exc:  # pragma: no cover
                    report.failed.append(IngestionError(source=str(path), reason=str(exc)))

        return report
