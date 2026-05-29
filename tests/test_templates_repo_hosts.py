from __future__ import annotations

import json
from pathlib import Path

from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.graph import SqlGraphStore
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset
from ragflow_orchestrator.templates import GitHubConfig, GitHubTemplate, GitLabConfig, GitLabTemplate, LanguageMode
from ragflow_orchestrator.templates.runner import run_template_from_json


def _build_orchestrator(tmp_path: Path) -> RAGOrchestrator:
    provider = create_provider("sqlite+vec", db_path=str(tmp_path / "repo_hosts.db"), table_name="repo_host_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=64),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def test_github_template_with_stubs(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = GitHubTemplate(orchestrator, graph_store=SqlGraphStore(str(tmp_path / "gh_graph.db")))

    template._list_owner_repos = lambda cfg, owner: [  # type: ignore[method-assign]
        {
            "id": 1,
            "name": "tg-bot",
            "full_name": f"{owner}/tg-bot",
            "html_url": "https://github.com/demo/tg-bot",
            "description": "Telegram bot",
            "stargazers_count": 10,
            "forks_count": 2,
        }
    ]
    template._list_contributors = lambda cfg, full_name: [  # type: ignore[method-assign]
        {"id": 100, "login": "alice", "html_url": "https://github.com/alice", "contributions": 3}
    ]
    template._get_readme = lambda cfg, full_name: "README telegram bot"  # type: ignore[method-assign]

    report = template.run(
        GitHubConfig(
            owners=["demo"],
            max_projects=5,
            max_repos_per_owner=5,
            include_contributors=True,
            include_readme=True,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_gitlab_template_with_stubs(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = GitLabTemplate(orchestrator, graph_store=SqlGraphStore(str(tmp_path / "gl_graph.db")))

    template._list_projects = lambda cfg, owner: [  # type: ignore[method-assign]
        {
            "id": 9,
            "name": "tg-bot",
            "path_with_namespace": f"{owner}/tg-bot",
            "web_url": "https://gitlab.com/demo/tg-bot",
            "description": "Telegram bot",
            "star_count": 5,
            "forks_count": 1,
        }
    ]
    template._list_contributors = lambda cfg, project_id: [  # type: ignore[method-assign]
        {"id": 200, "username": "bob", "web_url": "https://gitlab.com/bob", "commits": 7}
    ]
    template._get_readme = lambda cfg, project_id: "README gitlab telegram bot"  # type: ignore[method-assign]

    report = template.run(
        GitLabConfig(
            base_url="https://gitlab.com",
            groups_or_users=["demo"],
            max_projects=5,
            max_repos_per_owner=5,
            include_contributors=True,
            include_readme=True,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert len(report.ingested) == 1
    assert not report.failed


def test_run_template_from_json_with_github(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "ragflow_orchestrator.templates.github_template.GitHubTemplate._list_owner_repos",
        lambda self, cfg, owner: [
            {
                "id": 3,
                "name": "repo",
                "full_name": f"{owner}/repo",
                "html_url": "https://github.com/demo/repo",
                "description": "desc",
                "stargazers_count": 1,
                "forks_count": 0,
            }
        ],
    )
    monkeypatch.setattr(
        "ragflow_orchestrator.templates.github_template.GitHubTemplate._list_contributors",
        lambda self, cfg, full_name: [],
    )
    monkeypatch.setattr(
        "ragflow_orchestrator.templates.github_template.GitHubTemplate._get_readme",
        lambda self, cfg, full_name: "README",
    )

    cfg = {
        "orchestrator": {
            "provider": {
                "kind": "sqlite+vec",
                "params": {
                    "db_path": str(tmp_path / "runner_gh.db"),
                    "table_name": "runner_gh_chunks",
                },
            },
            "embedding": {
                "provider": "hash",
                "options": {"dimensions": 64},
            },
            "pipeline": {"preset": "document"},
        },
        "graph_store": {"db_path": str(tmp_path / "graph_runner.db")},
        "active_scenario": "github",
        "scenarios": {
            "github": {
                "owners": ["demo"],
                "max_projects": 3,
                "max_repos_per_owner": 3,
                "include_readme": True,
                "include_contributors": False,
                "auth_mode": "none",
                "language_mode": "auto",
            }
        },
    }

    cfg_path = tmp_path / "templates.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")

    report = run_template_from_json(str(cfg_path))
    assert len(report.ingested) == 1
    assert not report.failed
