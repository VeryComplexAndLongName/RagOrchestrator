from __future__ import annotations

from collections import deque
from urllib import request

from rag_orchestrator.templates.base import BaseIngestionTemplate
from rag_orchestrator.templates.models import IngestionError, TemplateRunReport, WebCrawlConfig
from rag_orchestrator.templates.utils import extract_links, extract_text_from_html, is_same_domain


class WebCrawlTemplate(BaseIngestionTemplate):
    def run(self, config: WebCrawlConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        queue: deque[tuple[str, int, str]] = deque()
        visited: set[str] = set()

        for url in config.urls:
            queue.append((url, 0, url))

        while queue and len(visited) < config.max_pages:
            url, depth, root_url = queue.popleft()
            if url in visited:
                continue
            if depth > config.max_depth:
                continue
            visited.add(url)

            try:
                html = self._fetch_html(url)
                text = extract_text_from_html(html)
                if not text:
                    report.skipped.append(IngestionError(source=url, reason="empty text after extraction"))
                    continue

                language = self._language_tag(text=text, mode=config.language_mode)
                summary = self.orchestrator.ingest(
                    source_id=url,
                    raw_text=text,
                    metadata=self._metadata_for_url_source(
                        "web",
                        {"url": url, "depth": depth, "language": language},
                        url,
                    ),
                )
                report.ingested.append(summary)

                if depth < config.max_depth:
                    for next_url in extract_links(url, html):
                        if config.same_domain_only and not is_same_domain(root_url, next_url):
                            continue
                        if next_url not in visited:
                            queue.append((next_url, depth + 1, root_url))
            except Exception as exc:  # pragma: no cover
                report.failed.append(IngestionError(source=url, reason=str(exc)))

        return report

    @staticmethod
    def _fetch_html(url: str, timeout: int = 15) -> str:
        req = request.Request(url=url, headers={"User-Agent": "rag-orchestrator/0.1"})
        with request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                raise ValueError(f"Unsupported content type: {content_type}")
            return response.read().decode("utf-8", errors="replace")
