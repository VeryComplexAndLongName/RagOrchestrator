from __future__ import annotations

import base64
import json
from urllib import parse, request

from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import ConfluenceWikiConfig, IngestionError, TemplateRunReport
from ragflow_orchestrator.templates.utils import extract_text_from_html


class ConfluenceWikiTemplate(BaseIngestionTemplate):
    template_name = "confluence_wiki"
    description = "Ingests Confluence pages by space key or explicit page IDs via REST API."

    def run(self, config: ConfluenceWikiConfig) -> TemplateRunReport:
        report = TemplateRunReport()

        page_ids = set(config.page_ids)
        for space_key in config.space_keys:
            try:
                for page_id in self._list_space_page_ids(config, space_key):
                    if len(page_ids) >= config.max_pages:
                        break
                    page_ids.add(page_id)
            except Exception as exc:  # pragma: no cover
                report.failed.append(IngestionError(source=f"space:{space_key}", reason=str(exc)))

        for page_id in list(page_ids)[: config.max_pages]:
            try:
                page = self._get_page(config, page_id)
                title = str(page.get("title") or f"page:{page_id}")
                raw_html = str(((page.get("body") or {}).get("storage") or {}).get("value") or "")
                text = extract_text_from_html(raw_html)
                if not text:
                    report.skipped.append(IngestionError(source=title, reason="empty extracted text"))
                    continue

                language = self._language_tag(text=text, mode=config.language_mode)
                source_url = f"{config.base_url.rstrip('/')}/rest/api/content/{page_id}?expand=body.storage,title"
                summary = self.orchestrator.ingest(
                    source_id=f"confluence:{page_id}",
                    raw_text=text,
                    metadata=self._metadata_for_url_source(
                        "confluence",
                        {
                            "page_id": str(page_id),
                            "title": title,
                            "language": language,
                        },
                        source_url,
                    ),
                )
                if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                    report.skipped.append(IngestionError(source=title, reason="all chunks are duplicates"))
                    continue
                report.ingested.append(summary)
            except Exception as exc:  # pragma: no cover
                report.failed.append(IngestionError(source=f"page:{page_id}", reason=str(exc)))

        return report

    def _list_space_page_ids(self, config: ConfluenceWikiConfig, space_key: str) -> list[str]:
        out: list[str] = []
        start = 0
        while len(out) < config.max_pages:
            query = parse.urlencode({"spaceKey": space_key, "type": "page", "limit": 50, "start": start})
            url = f"{config.base_url.rstrip('/')}/rest/api/content?{query}"
            payload = self._request_json(config, url)
            rows = payload.get("results") or []
            for row in rows:
                page_id = str(row.get("id") or "")
                if page_id:
                    out.append(page_id)
                    if len(out) >= config.max_pages:
                        break
            if not rows:
                break
            start += len(rows)
        return out

    def _get_page(self, config: ConfluenceWikiConfig, page_id: str) -> dict:
        query = parse.urlencode({"expand": "body.storage,title"})
        url = f"{config.base_url.rstrip('/')}/rest/api/content/{page_id}?{query}"
        payload = self._request_json(config, url)
        return dict(payload)

    @staticmethod
    def _request_json(config: ConfluenceWikiConfig, url: str) -> dict:
        req = request.Request(url=url, method="GET")
        req.add_header("Accept", "application/json")

        if config.auth_mode == "bearer" and config.token:
            req.add_header("Authorization", f"Bearer {config.token}")
        elif config.auth_mode == "basic" and config.username and config.password:
            token = base64.b64encode(f"{config.username}:{config.password}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")

        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
