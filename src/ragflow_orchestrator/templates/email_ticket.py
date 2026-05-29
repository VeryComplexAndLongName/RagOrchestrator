from __future__ import annotations

import csv
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path

from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import EmailTicketConfig, IngestionError, TemplateRunReport


class EmailTicketTemplate(BaseIngestionTemplate):
    def run(self, config: EmailTicketConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        ext_set = {item.lower() for item in config.extensions}

        for source in config.sources:
            path = Path(source)
            if not path.exists():
                report.failed.append(IngestionError(source=source, reason="source path does not exist"))
                continue

            targets = path.rglob("*") if path.is_dir() and config.recursive else path.glob("*") if path.is_dir() else [path]
            for item in targets:
                if not item.is_file():
                    continue
                if item.suffix.lower() not in ext_set:
                    continue

                try:
                    messages = self._read_tickets(item)
                    for idx, payload in enumerate(messages):
                        text = payload.get("text", "")
                        if not text.strip():
                            report.skipped.append(IngestionError(source=f"{item}#{idx}", reason="empty ticket body"))
                            continue

                        language = self._language_tag(text=text, mode=config.language_mode)
                        metadata = {
                            "source_type": "email_ticket",
                            "file_path": str(item),
                            "language": language,
                            "ticket_id": payload.get("ticket_id", f"{item}:{idx}"),
                            "subject": payload.get("subject", ""),
                            "sender": payload.get("sender", ""),
                        }
                        summary = self.orchestrator.ingest(
                            source_id=f"{item}:{idx}",
                            raw_text=text,
                            metadata=metadata,
                        )
                        if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                            report.skipped.append(IngestionError(source=f"{item}#{idx}", reason="duplicate ticket"))
                            continue
                        report.ingested.append(summary)
                except Exception as exc:  # pragma: no cover
                    report.failed.append(IngestionError(source=str(item), reason=str(exc)))

        return report

    @staticmethod
    def _read_tickets(path: Path) -> list[dict[str, str]]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [{"text": text, "ticket_id": path.stem}]

        if suffix == ".jsonl":
            rows: list[dict[str, str]] = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                rows.append(
                    {
                        "ticket_id": str(payload.get("ticket_id") or payload.get("id") or ""),
                        "subject": str(payload.get("subject") or ""),
                        "sender": str(payload.get("sender") or payload.get("from") or ""),
                        "text": str(payload.get("body") or payload.get("text") or ""),
                    }
                )
            return rows

        if suffix == ".csv":
            rows = []
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(
                        {
                            "ticket_id": str(row.get("ticket_id") or row.get("id") or ""),
                            "subject": str(row.get("subject") or ""),
                            "sender": str(row.get("sender") or row.get("from") or ""),
                            "text": str(row.get("body") or row.get("text") or ""),
                        }
                    )
            return rows

        if suffix == ".eml":
            with path.open("rb") as handle:
                msg = BytesParser(policy=policy.default).parse(handle)
            body = msg.get_body(preferencelist=("plain",))
            payload = body.get_content() if body else msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                text = payload.decode("utf-8", errors="replace")
            else:
                text = str(payload or "")
            return [
                {
                    "ticket_id": path.stem,
                    "subject": str(msg.get("subject") or ""),
                    "sender": str(msg.get("from") or ""),
                    "text": str(text or ""),
                }
            ]

        return []
