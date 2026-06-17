from __future__ import annotations

import json
from pathlib import Path

from ragflow_orchestrator.document_pipeline import detect_document_type
from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.extractors import extract_text
from ragflow_orchestrator.templates.models import IncrementalSyncConfig, IngestionError, TemplateRunReport
from ragflow_orchestrator.templates.utils import extract_text_from_html


class IncrementalSyncTemplate(BaseIngestionTemplate):
    template_name = "incremental_sync"
    description = "Ingests only changed files from folders using persisted file fingerprints."

    def run(self, config: IncrementalSyncConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        ext_set = {item.lower() for item in config.extensions}
        state_path = Path(config.state_file)
        state = self._load_state(state_path)
        next_state: dict[str, str] = {}

        for folder in config.folders:
            root = Path(folder)
            if not root.exists():
                report.failed.append(IngestionError(source=folder, reason="folder does not exist"))
                continue

            iterator = root.rglob("*") if config.recursive else root.glob("*")
            for path in iterator:
                if not path.is_file() or path.suffix.lower() not in ext_set:
                    continue

                fingerprint = self._file_fingerprint(path)
                key = str(path.resolve())
                next_state[key] = fingerprint
                if state.get(key) == fingerprint:
                    report.skipped.append(IngestionError(source=key, reason="unchanged"))
                    continue

                try:
                    raw_text = extract_text(path)
                    if path.suffix.lower() == ".html":
                        raw_text = extract_text_from_html(raw_text)
                    if not raw_text.strip():
                        report.skipped.append(IngestionError(source=key, reason="empty extracted text"))
                        continue

                    language = self._language_tag(text=raw_text, mode=config.language_mode)
                    document_type = detect_document_type(path=path, text=raw_text).document_type.value
                    summary = self.orchestrator.ingest(
                        source_id=key,
                        raw_text=raw_text,
                        metadata=self._metadata_for_document_source(
                            "incremental_file",
                            {
                                "file_path": key,
                                "file_ext": path.suffix.lower(),
                                "language": language,
                            },
                            document_type=document_type,
                        ),
                    )
                    if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                        report.skipped.append(IngestionError(source=key, reason="all chunks are duplicates"))
                        continue
                    report.ingested.append(summary)
                except Exception as exc:  # pragma: no cover
                    report.failed.append(IngestionError(source=key, reason=str(exc)))

        self._save_state(state_path, next_state)
        return report

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    @staticmethod
    def _load_state(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_state(path: Path, state: dict[str, str]) -> None:
        path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
