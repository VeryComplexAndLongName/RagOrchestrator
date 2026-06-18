from __future__ import annotations

from collections import deque
from urllib import request

from ragflow_orchestrator.document_pipeline import detect_document_type
from ragflow_orchestrator.retry import retry
from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import IngestionError, TemplateRunReport, WebCrawlConfig
from ragflow_orchestrator.templates.utils import extract_links, extract_text_from_html, is_same_domain


class WebCrawlTemplate(BaseIngestionTemplate):
    template_name = "web_crawl"
    description = "Ingests website pages by crawling seed URLs with depth and domain controls."

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
                fetched = self._fetch_html(url)
                if isinstance(fetched, tuple):
                    body, content_type = fetched
                else:
                    body, content_type = fetched, "text/html"
                if "html" in content_type:
                    text = extract_text_from_html(body)
                else:
                    text = body.strip()
                if not text:
                    report.skipped.append(IngestionError(source=url, reason="empty text after extraction"))
                    continue

                language = self._language_tag(text=text, mode=config.language_mode)
                document_type = detect_document_type(text=text, content_type=content_type, source_name=url).document_type.value
                summary = self.orchestrator.ingest(
                    source_id=url,
                    raw_text=text,
                    metadata=self._metadata_for_document_source(
                        "web",
                        {"url": url, "depth": depth, "language": language},
                        document_type=document_type,
                        source_url=url,
                        content_type=content_type,
                    ),
                )
                report.ingested.append(summary)

                if depth < config.max_depth:
                    for next_url in extract_links(url, body):
                        if config.same_domain_only and not is_same_domain(root_url, next_url):
                            continue
                        if next_url not in visited:
                            queue.append((next_url, depth + 1, root_url))
            except Exception as exc:  # pragma: no cover
                report.failed.append(IngestionError(source=url, reason=str(exc)))

        return report

    @staticmethod
    def _fetch_html(url: str, timeout: int = 15) -> tuple[str, str] | str:
        """Fetch HTML from URL with retry logic."""
        return WebCrawlTemplate._fetch_html_with_retry(url, timeout)

    @staticmethod
    @retry(max_retries=3, initial_delay=1.0, max_delay=10.0, backoff_factor=2.0)
    def _fetch_html_with_retry(url: str, timeout: int = 15) -> tuple[str, str] | str:
        req = request.Request(url=url, headers={"User-Agent": "rag-orchestrator/0.1"})
        with request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read().decode("utf-8", errors="replace")
            return body, content_type.split(";", 1)[0].strip().lower()
