from __future__ import annotations

import base64
import json
from urllib import parse, request

from ragflow_orchestrator.graph import SqlGraphStore
from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import GitLabConfig, IngestionError, TemplateRunReport


class GitLabTemplate(BaseIngestionTemplate):
    template_name = "gitlab"
    description = "Ingests GitLab projects, README data, and optional contributor graph metadata."

    def __init__(self, orchestrator, graph_store: SqlGraphStore | None = None) -> None:
        super().__init__(orchestrator)
        self.graph_store = graph_store or SqlGraphStore()

    def run(self, config: GitLabConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        processed = 0

        for owner in config.groups_or_users:
            projects = self._list_projects(config, owner)
            for project in projects[: config.max_repos_per_owner]:
                if processed >= config.max_projects:
                    break
                processed += 1

                try:
                    project_id = str(project.get("id") or "")
                    name = str(project.get("name") or "")
                    full_name = str(project.get("path_with_namespace") or name)
                    web_url = str(project.get("web_url") or "")
                    description = str(project.get("description") or "")
                    stars = int(project.get("star_count") or 0)
                    forks = int(project.get("forks_count") or 0)

                    self.graph_store.upsert_repository(
                        repo_id=project_id,
                        platform="gitlab",
                        name=name,
                        full_name=full_name,
                        url=web_url,
                        description=description,
                        stars=stars,
                        forks=forks,
                    )

                    if config.include_contributors:
                        members = self._list_contributors(config, project_id)
                        for member in members:
                            contributor_id = str(member.get("id") or "")
                            username = str(member.get("username") or member.get("name") or "")
                            c_url = str(member.get("web_url") or "")
                            contributions = int(member.get("commits") or 1)
                            if contributor_id and username:
                                self.graph_store.upsert_contributor(contributor_id, username, c_url)
                                self.graph_store.upsert_contribution_edge(project_id, contributor_id, contributions)

                    text_parts = [
                        f"Repository: {full_name}",
                        f"Description: {description}",
                        f"Stars: {stars}",
                        f"Forks: {forks}",
                        f"URL: {web_url}",
                    ]

                    if config.include_readme:
                        readme = self._get_readme(config, project_id)
                        if readme.strip():
                            text_parts.append("README:\n" + readme)

                    raw_text = "\n".join(text_parts).strip()
                    if not raw_text:
                        report.skipped.append(IngestionError(source=full_name, reason="empty repository payload"))
                        continue

                    language = self._language_tag(raw_text, config.language_mode)
                    summary = self.orchestrator.ingest(
                        source_id=f"gitlab:{full_name}",
                        raw_text=raw_text,
                        metadata=self._metadata_for_url_source(
                            "gitlab_repo",
                            {
                                "platform": "gitlab",
                                "owner": owner,
                                "full_name": full_name,
                                "stars": stars,
                                "forks": forks,
                                "language": language,
                            },
                            web_url or None,
                        ),
                    )
                    if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                        report.skipped.append(IngestionError(source=full_name, reason="all chunks are duplicates"))
                        continue
                    report.ingested.append(summary)
                except Exception as exc:  # pragma: no cover
                    report.failed.append(IngestionError(source=str(project.get("path_with_namespace") or owner), reason=str(exc)))

        return report

    def _list_projects(self, config: GitLabConfig, owner: str) -> list[dict]:
        query = parse.urlencode({"search": owner, "simple": "true", "per_page": config.max_repos_per_owner})
        url = f"{config.base_url.rstrip('/')}/api/v4/projects?{query}"
        payload = self._request_json(config, url)
        projects = payload if isinstance(payload, list) else []
        owner_lower = owner.lower()
        return [
            p
            for p in projects
            if owner_lower in str(p.get("path_with_namespace") or "").lower()
        ]

    def _list_contributors(self, config: GitLabConfig, project_id: str) -> list[dict]:
        url = f"{config.base_url.rstrip('/')}/api/v4/projects/{parse.quote(project_id, safe='')}/repository/contributors"
        payload = self._request_json(config, url)
        return payload if isinstance(payload, list) else []

    def _get_readme(self, config: GitLabConfig, project_id: str) -> str:
        file_path = parse.quote("README.md", safe="")
        url = f"{config.base_url.rstrip('/')}/api/v4/projects/{parse.quote(project_id, safe='')}/repository/files/{file_path}?ref=HEAD"
        payload = self._request_json(config, url)
        content = str(payload.get("content") or "")
        encoding = str(payload.get("encoding") or "")
        if not content:
            return ""
        if encoding == "base64":
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return content

    @staticmethod
    def _request_json(config: GitLabConfig, url: str) -> object:
        req = request.Request(url=url, method="GET")
        req.add_header("Accept", "application/json")
        if config.auth_mode == "bearer" and config.token:
            req.add_header("Authorization", f"Bearer {config.token}")
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
        return json.loads(body)
