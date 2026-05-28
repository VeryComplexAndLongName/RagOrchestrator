from __future__ import annotations

from pathlib import Path

from rag_orchestrator.templates.base import BaseIngestionTemplate
from rag_orchestrator.templates.models import IngestionError, RepoCodeConfig, TemplateRunReport


class RepoCodeTemplate(BaseIngestionTemplate):
    def run(self, config: RepoCodeConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        ext_set = {item.lower() for item in config.extensions}

        for repo in config.repos:
            root = Path(repo)
            if not root.exists():
                report.failed.append(IngestionError(source=repo, reason="repo path does not exist"))
                continue

            iterator = root.rglob("*") if config.recursive else root.glob("*")
            for path in iterator:
                if not path.is_file():
                    continue
                if not config.include_hidden and any(part.startswith(".") for part in path.parts):
                    continue
                if path.suffix.lower() not in ext_set:
                    continue

                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if not text.strip():
                        report.skipped.append(IngestionError(source=str(path), reason="empty file"))
                        continue

                    language = self._language_tag(text=text, mode=config.language_mode)
                    summary = self.orchestrator.ingest(
                        source_id=str(path),
                        raw_text=text,
                        metadata={
                            "source_type": "repo_code",
                            "repo_root": str(root),
                            "file_path": str(path),
                            "file_ext": path.suffix.lower(),
                            "language": language,
                        },
                    )
                    if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                        report.skipped.append(IngestionError(source=str(path), reason="all chunks are duplicates"))
                        continue
                    report.ingested.append(summary)
                except Exception as exc:  # pragma: no cover
                    report.failed.append(IngestionError(source=str(path), reason=str(exc)))

        return report
