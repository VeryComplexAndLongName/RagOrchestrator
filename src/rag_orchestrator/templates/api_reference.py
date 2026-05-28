from __future__ import annotations

import json
from pathlib import Path
from urllib import request

from rag_orchestrator.templates.base import BaseIngestionTemplate
from rag_orchestrator.templates.models import APIReferenceConfig, IngestionError, TemplateRunReport


class APIReferenceTemplate(BaseIngestionTemplate):
    def run(self, config: APIReferenceConfig) -> TemplateRunReport:
        report = TemplateRunReport()

        for source in config.sources:
            try:
                spec = self._load_spec(source)
                chunks = self._build_chunks(spec, include_operations=config.include_operations, include_schemas=config.include_schemas)
                if not chunks:
                    report.skipped.append(IngestionError(source=source, reason="empty API spec"))
                    continue

                for idx, chunk in enumerate(chunks):
                    language = self._language_tag(text=chunk, mode=config.language_mode)
                    source_url = source if source.startswith(("http://", "https://")) else None
                    summary = self.orchestrator.ingest(
                        source_id=f"api:{source}:{idx}",
                        raw_text=chunk,
                        metadata=self._metadata_for_url_source(
                            "api_reference",
                            {"source": source, "language": language},
                            source_url,
                        ),
                    )
                    if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                        report.skipped.append(IngestionError(source=f"{source}#{idx}", reason="duplicate chunk"))
                        continue
                    report.ingested.append(summary)
            except Exception as exc:  # pragma: no cover
                report.failed.append(IngestionError(source=source, reason=str(exc)))

        return report

    @staticmethod
    def _load_spec(source: str) -> dict:
        if source.startswith("http://") or source.startswith("https://"):
            req = request.Request(url=source, method="GET")
            req.add_header("Accept", "application/json, application/yaml, text/yaml, text/plain")
            with request.urlopen(req, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
        else:
            body = Path(source).read_text(encoding="utf-8", errors="replace")

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("YAML API specs require PyYAML (pip install pyyaml)") from exc
            parsed = yaml.safe_load(body)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid API spec structure")
            return parsed

    @staticmethod
    def _build_chunks(spec: dict, include_operations: bool, include_schemas: bool) -> list[str]:
        out: list[str] = []
        info = spec.get("info") or {}
        title = str(info.get("title") or "API")
        version = str(info.get("version") or "")
        description = str(info.get("description") or "")
        out.append(f"API: {title}\nVersion: {version}\nDescription:\n{description}".strip())

        if include_operations:
            paths = spec.get("paths") or {}
            for path, methods in paths.items():
                if not isinstance(methods, dict):
                    continue
                for method, details in methods.items():
                    if not isinstance(details, dict):
                        continue
                    summary = str(details.get("summary") or "")
                    desc = str(details.get("description") or "")
                    operation_id = str(details.get("operationId") or "")
                    out.append(
                        f"Operation: {method.upper()} {path}\nOperationId: {operation_id}\nSummary: {summary}\nDescription:\n{desc}".strip()
                    )

        if include_schemas:
            schemas = (((spec.get("components") or {}).get("schemas")) or {})
            for name, schema in schemas.items():
                out.append(f"Schema: {name}\n{json.dumps(schema, ensure_ascii=True, indent=2)}")

        return [chunk for chunk in out if chunk.strip()]
