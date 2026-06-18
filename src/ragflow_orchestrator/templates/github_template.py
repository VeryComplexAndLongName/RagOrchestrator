from __future__ import annotations

import base64
import json
from urllib import parse, request

from ragflow_orchestrator.graph import SqlGraphStore
from ragflow_orchestrator.retry import retry
from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import GitHubConfig, IngestionError, TemplateRunReport


class GitHubTemplate(BaseIngestionTemplate):
    template_name = "github"
    description = "Ingests GitHub repositories, README data, and optional contributor graph metadata."

    def __init__(self, orchestrator, graph_store: SqlGraphStore | None = None) -> None:
        super().__init__(orchestrator)
        self.graph_store = graph_store or SqlGraphStore()

    def run(self, config: GitHubConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        processed = 0

        for owner in config.owners:
            repos = self._list_owner_repos(config, owner)
            for repo in repos[: config.max_repos_per_owner]:
                if processed >= config.max_projects:
                    break
                processed += 1

                try:
                    repo_id = str(repo.get("id") or "")
                    name = str(repo.get("name") or "")
                    full_name = str(repo.get("full_name") or f"{owner}/{name}")
                    html_url = str(repo.get("html_url") or "")
                    description = str(repo.get("description") or "")
                    stars = int(repo.get("stargazers_count") or 0)
                    forks = int(repo.get("forks_count") or 0)

                    self.graph_store.upsert_repository(
                        repo_id=repo_id,
                        platform="github",
                        name=name,
                        full_name=full_name,
                        url=html_url,
                        description=description,
                        stars=stars,
                        forks=forks,
                    )

                    if config.include_contributors:
                        contributors = self._list_contributors(config, full_name)
                        for contributor in contributors:
                            contributor_id = str(contributor.get("id") or "")
                            login = str(contributor.get("login") or "")
                            c_url = str(contributor.get("html_url") or "")
                            contributions = int(contributor.get("contributions") or 0)
                            if contributor_id and login:
                                self.graph_store.upsert_contributor(contributor_id, login, c_url)
                                self.graph_store.upsert_contribution_edge(repo_id, contributor_id, contributions)

                    text_parts = [
                        f"Repository: {full_name}",
                        f"Description: {description}",
                        f"Stars: {stars}",
                        f"Forks: {forks}",
                        f"URL: {html_url}",
                    ]

                    if config.include_readme:
                        readme = self._get_readme(config, full_name)
                        if readme.strip():
                            text_parts.append("README:\n" + readme)

                    raw_text = "\n".join(text_parts).strip()
                    if not raw_text:
                        report.skipped.append(IngestionError(source=full_name, reason="empty repository payload"))
                        continue

                    language = self._language_tag(raw_text, config.language_mode)
                    summary = self.orchestrator.ingest(
                        source_id=f"github:{full_name}",
                        raw_text=raw_text,
                        metadata=self._metadata_for_url_source(
                            "github_repo",
                            {
                                "platform": "github",
                                "owner": owner,
                                "full_name": full_name,
                                "stars": stars,
                                "forks": forks,
                                "language": language,
                            },
                            str(repo.get("url") or html_url or None),
                        ),
                    )
                    if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                        report.skipped.append(IngestionError(source=full_name, reason="all chunks are duplicates"))
                        continue
                    report.ingested.append(summary)
                except Exception as exc:  # pragma: no cover
                    report.failed.append(IngestionError(source=str(repo.get("full_name") or owner), reason=str(exc)))

        return report

    def _list_owner_repos(self, config: GitHubConfig, owner: str) -> list[dict]:
        query = parse.urlencode({"per_page": config.max_repos_per_owner, "sort": "updated"})
        url = f"https://api.github.com/users/{owner}/repos?{query}"
        payload = self._request_json(config, url)
        return payload if isinstance(payload, list) else []

    def _list_contributors(self, config: GitHubConfig, full_name: str) -> list[dict]:
        query = parse.urlencode({"per_page": 100})
        url = f"https://api.github.com/repos/{full_name}/contributors?{query}"
        payload = self._request_json(config, url)
        return payload if isinstance(payload, list) else []

    def _get_readme(self, config: GitHubConfig, full_name: str) -> str:
        url = f"https://api.github.com/repos/{full_name}/readme"
        payload = self._request_json(config, url)
        content = str(payload.get("content") or "")
        encoding = str(payload.get("encoding") or "")
        if not content:
            return ""
        if encoding == "base64":
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return content

    @staticmethod
    def _request_json(config: GitHubConfig, url: str) -> object:
        """Fetch JSON from URL with retry logic."""
        return GitHubTemplate._fetch_json_with_retry(config, url)

    @staticmethod
    @retry(max_retries=3, initial_delay=1.0, max_delay=10.0, backoff_factor=2.0)
    def _fetch_json_with_retry(config: GitHubConfig, url: str) -> object:
        req = request.Request(url=url, method="GET")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if config.auth_mode == "bearer" and config.token:
            req.add_header("Authorization", f"Bearer {config.token}")
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
        return json.loads(body)
