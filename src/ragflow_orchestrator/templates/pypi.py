from __future__ import annotations

import json
import re
from urllib import parse, request

from ragflow_orchestrator.retry import retry
from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import IngestionError, PyPIConfig, TemplateRunReport
from ragflow_orchestrator.templates.utils import html_to_text


def _markdown_to_plain_text(value: str) -> str:
    text = value
    # Convert markdown links to "label: url" so URLs remain retrievable.
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1: \2", text)
    # Strip emphasis markers while preserving content.
    text = re.sub(r"(\*\*|__|\*|_)", "", text)
    # Collapse markdown heading markers.
    text = re.sub(r"^[#>\-\s]+", "", text, flags=re.MULTILINE)
    return text


class PyPITemplate(BaseIngestionTemplate):
    template_name = "pypi"
    description = "Ingests package metadata, descriptions, and release facts from PyPI."

    def run(self, config: PyPIConfig) -> TemplateRunReport:
        report = TemplateRunReport()

        for package in config.packages:
            try:
                payload = self._fetch_package_payload(package)
                chunks = self._build_chunks(
                    package=package,
                    payload=payload,
                    include_release_history=config.include_release_history,
                    max_releases_per_package=config.max_releases_per_package,
                    include_project_urls=config.include_project_urls,
                )
                if not chunks:
                    report.skipped.append(IngestionError(source=package, reason="empty package metadata"))
                    continue

                for idx, chunk in enumerate(chunks):
                    language = self._language_tag(text=chunk, mode=config.language_mode)
                    source_url = f"https://pypi.org/pypi/{parse.quote(package, safe='')}/json"
                    summary = self.orchestrator.ingest(
                        source_id=f"pypi:{package}:{idx}",
                        raw_text=chunk,
                        metadata=self._metadata_for_url_source(
                            "pypi_package",
                            {
                                "package": package,
                                "language": language,
                            },
                            source_url,
                        ),
                    )
                    if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                        report.skipped.append(IngestionError(source=f"{package}#{idx}", reason="duplicate chunk"))
                        continue
                    report.ingested.append(summary)
            except Exception as exc:  # pragma: no cover
                report.failed.append(IngestionError(source=package, reason=str(exc)))

        return report

    @staticmethod
    def _fetch_package_payload(package: str) -> dict:
        """Fetch package JSON from PyPI with retry logic."""
        return PyPITemplate._fetch_package_payload_with_retry(package)

    @staticmethod
    @retry(max_retries=3, initial_delay=1.0, max_delay=10.0, backoff_factor=2.0)
    def _fetch_package_payload_with_retry(package: str) -> dict:
        safe_package = parse.quote(package, safe="")
        url = f"https://pypi.org/pypi/{safe_package}/json"
        req = request.Request(url=url, method="GET")
        req.add_header("Accept", "application/json")
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
        return json.loads(body)

    @staticmethod
    def _build_chunks(
        package: str,
        payload: dict,
        include_release_history: bool,
        max_releases_per_package: int,
        include_project_urls: bool,
    ) -> list[str]:
        info = payload.get("info") or {}
        name = str(info.get("name") or package)
        version = str(info.get("version") or "")
        summary = str(info.get("summary") or "")
        description = str(info.get("description") or "")
        description = html_to_text(description)
        description = _markdown_to_plain_text(description)
        home_page = str(info.get("home_page") or "")
        package_url = str(info.get("package_url") or "")
        project_url = str(info.get("project_url") or "")
        docs_url = str(info.get("docs_url") or "")
        download_url = str(info.get("download_url") or "")
        bugtrack_url = str(info.get("bugtrack_url") or "")
        license_name = str(info.get("license") or "")
        requires_python = str(info.get("requires_python") or "")
        author = str(info.get("author") or "")
        author_email = str(info.get("author_email") or "")
        maintainer = str(info.get("maintainer") or "")
        maintainer_email = str(info.get("maintainer_email") or "")
        keywords = str(info.get("keywords") or "")
        classifiers = info.get("classifiers") or []
        requires_dist = info.get("requires_dist") or []
        provides_extra = info.get("provides_extra") or []
        project_urls = info.get("project_urls") or {}

        short_url_lines: list[str] = []
        for label, value in (
            ("Homepage", home_page),
            ("Package URL", package_url),
            ("Project URL", project_url),
            ("Repository", str(project_urls.get("Source") or "")),
            ("Repository", str(project_urls.get("Code") or "")),
            ("Documentation", docs_url),
            ("Download", download_url),
            ("Bugtracker", bugtrack_url),
        ):
            cleaned = value.strip()
            if cleaned and cleaned.startswith(("http://", "https://")):
                short_url_lines.append(f"- {label}: {cleaned}")
        if short_url_lines:
            deduped_lines = list(dict.fromkeys(short_url_lines))
            chunks = [
                "\n".join(
                    [
                        f"Package: {name}",
                        f"Version: {version}",
                        "Key URLs:",
                        "\n".join(deduped_lines),
                    ]
                ).strip()
            ]
        else:
            chunks = []

        chunks.append(
            "\n".join(
                [
                    f"Package: {name}",
                    f"Version: {version}",
                    f"Summary: {summary}",
                    f"Home page: {home_page}",
                    f"Package URL: {package_url}",
                    f"Project URL: {project_url}",
                    f"Docs URL: {docs_url}",
                    f"Download URL: {download_url}",
                    f"Bugtrack URL: {bugtrack_url}",
                    f"License: {license_name}",
                    f"Requires-Python: {requires_python}",
                    f"Author: {author}",
                    f"Author email: {author_email}",
                    f"Maintainer: {maintainer}",
                    f"Maintainer email: {maintainer_email}",
                    f"Keywords: {keywords}",
                    "Classifiers:",
                    "\n".join(f"- {item}" for item in classifiers if item),
                    "Description:",
                    description,
                ]
            ).strip()
        )

        dep_lines = [str(item).strip() for item in requires_dist if str(item).strip()]
        extra_lines = [str(item).strip() for item in provides_extra if str(item).strip()]
        if dep_lines or extra_lines:
            chunks.append(
                "\n".join(
                    [
                        f"Package: {name}",
                        f"Version: {version}",
                        "Dependencies (requires_dist):",
                        "\n".join(f"- {item}" for item in dep_lines) if dep_lines else "- none",
                        "Extras (provides_extra):",
                        "\n".join(f"- {item}" for item in extra_lines) if extra_lines else "- none",
                    ]
                ).strip()
            )

        if include_project_urls:
            if isinstance(project_urls, dict) and project_urls:
                chunks.append(
                    "\n".join(
                        [
                            f"Package: {name}",
                            "Project URLs:",
                            "\n".join(f"- {key}: {value}" for key, value in project_urls.items()),
                        ]
                    ).strip()
                )

        if include_release_history:
            releases = payload.get("releases") or {}
            release_versions = sorted(releases.keys(), reverse=True)[:max_releases_per_package]
            for release in release_versions:
                files = releases.get(release) or []
                if not isinstance(files, list):
                    continue
                file_lines: list[str] = []
                for file_item in files:
                    if not isinstance(file_item, dict):
                        continue
                    file_lines.append(
                        " | ".join(
                            [
                                f"filename={file_item.get('filename', '')}",
                                f"packagetype={file_item.get('packagetype', '')}",
                                f"python_version={file_item.get('python_version', '')}",
                                f"size={file_item.get('size', 0)}",
                                f"upload_time={file_item.get('upload_time', '')}",
                            ]
                        )
                    )

                chunks.append(
                    "\n".join(
                        [
                            f"Package: {name}",
                            f"Release: {release}",
                            "Files:",
                            "\n".join(file_lines) if file_lines else "No files",
                        ]
                    ).strip()
                )

        return [chunk for chunk in chunks if chunk.strip()]
