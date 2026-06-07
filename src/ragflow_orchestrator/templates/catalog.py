from __future__ import annotations

from importlib import import_module

from ragflow_orchestrator.templates.base import BaseIngestionTemplate

_TEMPLATE_IMPORTS: tuple[tuple[str, str, str], ...] = (
    ("api_reference", "ragflow_orchestrator.templates.api_reference", "APIReferenceTemplate"),
    ("bitrix", "ragflow_orchestrator.templates.bitrix", "BitrixTemplate"),
    ("confluence_wiki", "ragflow_orchestrator.templates.confluence_wiki", "ConfluenceWikiTemplate"),
    ("document_folder", "ragflow_orchestrator.templates.document_folder", "DocumentFolderTemplate"),
    ("email_ticket", "ragflow_orchestrator.templates.email_ticket", "EmailTicketTemplate"),
    ("github", "ragflow_orchestrator.templates.github_template", "GitHubTemplate"),
    ("gitlab", "ragflow_orchestrator.templates.gitlab_template", "GitLabTemplate"),
    ("incremental_sync", "ragflow_orchestrator.templates.incremental_sync", "IncrementalSyncTemplate"),
    ("jira", "ragflow_orchestrator.templates.jira", "JiraTemplate"),
    ("pypi", "ragflow_orchestrator.templates.pypi", "PyPITemplate"),
    ("repo_code", "ragflow_orchestrator.templates.repo_code", "RepoCodeTemplate"),
    ("web_crawl", "ragflow_orchestrator.templates.web_crawl", "WebCrawlTemplate"),
)


def list_installed_templates() -> list[dict[str, str]]:
    """Return installed ingestion templates with their names and descriptions.

    A template is considered installed if its module can be imported and the expected
    template class is present.
    """

    items: list[dict[str, str]] = []
    for fallback_name, module_path, class_name in _TEMPLATE_IMPORTS:
        try:
            module = import_module(module_path)
        except Exception:
            continue

        template_cls = getattr(module, class_name, None)
        if not isinstance(template_cls, type):
            continue
        if not issubclass(template_cls, BaseIngestionTemplate):
            continue

        name = str(getattr(template_cls, "template_name", fallback_name) or fallback_name)
        description = str(getattr(template_cls, "description", "")).strip() or f"{name} template."
        items.append({"name": name, "description": description})

    return sorted(items, key=lambda item: item["name"])


def resolve_template_class(name: str) -> type[BaseIngestionTemplate]:
    """Return the template class registered for a scenario name."""

    for fallback_name, module_path, class_name in _TEMPLATE_IMPORTS:
        try:
            module = import_module(module_path)
        except Exception as exc:
            if name == fallback_name:
                raise ValueError(f"Template '{name}' is registered but could not be imported") from exc
            continue

        template_cls = getattr(module, class_name, None)
        if not isinstance(template_cls, type) or not issubclass(template_cls, BaseIngestionTemplate):
            if name == fallback_name:
                raise ValueError(f"Template '{name}' is registered but its class is missing")
            continue

        registered_name = str(getattr(template_cls, "template_name", fallback_name) or fallback_name)
        if name in {fallback_name, registered_name}:
            return template_cls

    raise ValueError(f"Unknown scenario: {name}")
