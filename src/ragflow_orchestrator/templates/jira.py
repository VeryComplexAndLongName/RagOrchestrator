from __future__ import annotations

import base64
import json
from urllib import parse, request

from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import IngestionError, JiraConfig, TemplateRunReport


class JiraTemplate(BaseIngestionTemplate):
    template_name = "jira"
    description = "Ingests Jira issues with optional comments based on JQL queries."

    def run(self, config: JiraConfig) -> TemplateRunReport:
        report = TemplateRunReport()
        issues = self._search_issues(config)

        for issue in issues[: config.max_issues]:
            try:
                key = str(issue.get("key") or "unknown")
                fields = issue.get("fields") or {}
                text = self._issue_text(fields, include_comments=config.include_comments)
                if not text.strip():
                    report.skipped.append(IngestionError(source=key, reason="empty issue text"))
                    continue

                language = self._language_tag(text=text, mode=config.language_mode)
                source_url = str(issue.get("self") or f"{config.base_url.rstrip('/')}/browse/{key}")
                summary = self.orchestrator.ingest(
                    source_id=f"jira:{key}",
                    raw_text=text,
                    metadata=self._metadata_for_url_source(
                        "jira",
                        {
                            "issue_key": key,
                            "project": str(((fields.get("project") or {}).get("key") or "")),
                            "issue_type": str(((fields.get("issuetype") or {}).get("name") or "")),
                            "updated": str(fields.get("updated") or ""),
                            "language": language,
                        },
                        source_url,
                    ),
                )
                if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                    report.skipped.append(IngestionError(source=key, reason="all chunks are duplicates"))
                    continue
                report.ingested.append(summary)
            except Exception as exc:  # pragma: no cover
                report.failed.append(IngestionError(source=str(issue.get("key") or "issue"), reason=str(exc)))

        return report

    def _search_issues(self, config: JiraConfig) -> list[dict]:
        out: list[dict] = []
        start_at = 0
        while len(out) < config.max_issues:
            fields = "summary,description,comment,updated,project,issuetype"
            query = parse.urlencode(
                {
                    "jql": config.jql,
                    "startAt": start_at,
                    "maxResults": min(50, config.max_issues - len(out)),
                    "fields": fields,
                }
            )
            url = f"{config.base_url.rstrip('/')}/rest/api/2/search?{query}"
            payload = self._request_json(config, url)
            issues = payload.get("issues") or []
            out.extend(issues)
            if not issues:
                break
            start_at += len(issues)
        return out

    @staticmethod
    def _issue_text(fields: dict, include_comments: bool) -> str:
        summary = str(fields.get("summary") or "")
        description = fields.get("description")
        desc_text = JiraTemplate._flatten_jira_doc(description)

        parts = [f"Summary: {summary}", f"Description:\n{desc_text}"]
        if include_comments:
            comments = (((fields.get("comment") or {}).get("comments") or []))
            for idx, comment in enumerate(comments):
                body = JiraTemplate._flatten_jira_doc(comment.get("body"))
                if body.strip():
                    parts.append(f"Comment {idx + 1}:\n{body}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _flatten_jira_doc(node: object) -> str:
        if node is None:
            return ""
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            text = str(node.get("text") or "")
            children = node.get("content") or []
            child_text = " ".join(JiraTemplate._flatten_jira_doc(child) for child in children)
            return " ".join(part for part in [text, child_text] if part).strip()
        if isinstance(node, list):
            return " ".join(JiraTemplate._flatten_jira_doc(item) for item in node)
        return str(node)

    @staticmethod
    def _request_json(config: JiraConfig, url: str) -> dict:
        req = request.Request(url=url, method="GET")
        req.add_header("Accept", "application/json")

        if config.auth_mode == "bearer" and config.token:
            req.add_header("Authorization", f"Bearer {config.token}")
        elif config.auth_mode == "basic" and config.username and config.password:
            token = base64.b64encode(f"{config.username}:{config.password}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")

        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
