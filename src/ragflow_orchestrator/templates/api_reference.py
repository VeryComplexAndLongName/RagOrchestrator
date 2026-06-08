from __future__ import annotations

import json
import re
from pathlib import Path
from urllib import request
from xml.etree import ElementTree as ET

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

        if APIReferenceTemplate._looks_like_xml(body):
            return APIReferenceTemplate._xml_to_text(body)

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
                plain_text = APIReferenceTemplate._to_plain_text(item)
                out.append(f"Item #{idx}\n{plain_text}")
            return [chunk for chunk in out if chunk.strip()]

        if not isinstance(spec, dict):
            if isinstance(spec, str):
                return [f"API response text\n{spec}"]
            return [f"API response value:\n{APIReferenceTemplate._to_plain_text(spec)}"]

        is_openapi = bool(spec.get("openapi") or spec.get("swagger") or spec.get("paths") or spec.get("components"))
        if not is_openapi:
            return [f"JSON object response\n{APIReferenceTemplate._to_plain_text(spec)}"]

        out = []
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
                out.append(f"Schema: {name}\n{APIReferenceTemplate._to_plain_text(schema)}")

        return [chunk for chunk in out if chunk.strip()]

    @staticmethod
    def _looks_like_xml(text: str) -> bool:
        return bool(re.match(r"^\s*(<\?xml\b|<[A-Za-z_][\w:.-]*[\s>/])", text or ""))

    @staticmethod
    def _xml_to_text(xml_text: str) -> str:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return xml_text

        lines: list[str] = []

        def _local_name(tag: str) -> str:
            if "}" in tag:
                return tag.rsplit("}", 1)[1]
            return tag

        def _walk(node: ET.Element, path: str) -> None:
            for attr, value in node.attrib.items():
                if value:
                    lines.append(f"{path}@{attr}: {value}")
            text = (node.text or "").strip()
            if text:
                lines.append(f"{path}: {text}")
            children = list(node)
            for idx, child in enumerate(children, start=1):
                child_path = f"{path}/{_local_name(child.tag)}[{idx}]"
                _walk(child, child_path)

        root_name = _local_name(root.tag)
        _walk(root, root_name)
        if lines:
            return "\n".join(lines)
        return f"XML document: {root_name}"

    @staticmethod
    def _to_plain_text(value: object, prefix: str = "") -> str:
        lines: list[str] = []

        def _emit(current: object, key: str) -> None:
            if isinstance(current, dict):
                if not current and key:
                    lines.append(f"{key}: {{}}")
                    return
                for child_key, child_value in current.items():
                    child_path = f"{key}.{child_key}" if key else str(child_key)
                    _emit(child_value, child_path)
                return

            if isinstance(current, list):
                if not current and key:
                    lines.append(f"{key}: []")
                    return
                for idx, child_value in enumerate(current, start=1):
                    child_path = f"{key}[{idx}]" if key else f"[{idx}]"
                    _emit(child_value, child_path)
                return

            if isinstance(current, str):
                text = current.strip()
                if not text:
                    return
                lines.append(f"{key}: {text}" if key else text)
                return

            if current is None:
                lines.append(f"{key}: null" if key else "null")
                return

            rendered = str(current)
            lines.append(f"{key}: {rendered}" if key else rendered)

        _emit(value, prefix)
        if not lines:
            return ""
        return "\n".join(lines)
