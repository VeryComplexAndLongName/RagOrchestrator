from __future__ import annotations

import json
from pathlib import Path
from urllib import request

from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import APIReferenceConfig, IngestionError, TemplateRunReport


class APIReferenceTemplate(BaseIngestionTemplate):
    template_name = "api_reference"
    description = "Ingests OpenAPI/Swagger API specifications from local files or URLs."

    def run(self, config: APIReferenceConfig) -> TemplateRunReport:
        report = TemplateRunReport()

        for source in config.sources:
            try:
                spec = self._load_spec(source)
                chunks = self._build_chunks(
                    spec,
                    include_operations=config.include_operations,
                    include_schemas=config.include_schemas,
                    max_items=config.max_items,
                )
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
    def _load_spec(source: str) -> object:
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
            if parsed is None:
                raise ValueError("Invalid API payload: empty document")
            return parsed

    @staticmethod
    def _build_chunks(spec: object, include_operations: bool, include_schemas: bool, max_items: int | None = None) -> list[str]:
        def _as_dict(value: object) -> dict[str, object]:
            return value if isinstance(value, dict) else {}

        if isinstance(spec, list):
            total_items = len(spec)
            selected = spec[:max_items] if max_items is not None else spec
            if max_items is not None and max_items < total_items:
                out = [f"JSON response array with {total_items} item(s); ingesting first {len(selected)} item(s)"]
            else:
                out = [f"JSON response array with {total_items} item(s)"]
            for idx, item in enumerate(selected, start=1):
                out.append(f"Item #{idx}\n{json.dumps(item, ensure_ascii=True, indent=2)}")
            return [chunk for chunk in out if chunk.strip()]

        if not isinstance(spec, dict):
            return [f"API response value:\n{json.dumps(spec, ensure_ascii=True, indent=2)}"]

        is_openapi = bool(spec.get("openapi") or spec.get("swagger") or spec.get("paths") or spec.get("components"))
        if not is_openapi:
            return [f"JSON object response\n{json.dumps(spec, ensure_ascii=True, indent=2)}"]

        out: list[str] = []
        info = _as_dict(spec.get("info"))
        title = str(info.get("title") or "API")
        version = str(info.get("version") or "")
        description = str(info.get("description") or "")
        out.append(f"API: {title}\nVersion: {version}\nDescription:\n{description}".strip())

        if include_operations:
            paths = _as_dict(spec.get("paths"))
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
            components = _as_dict(spec.get("components"))
            schemas = _as_dict(components.get("schemas"))
            for name, schema in schemas.items():
                out.append(f"Schema: {name}\n{json.dumps(schema, ensure_ascii=True, indent=2)}")

        return [chunk for chunk in out if chunk.strip()]
