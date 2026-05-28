from __future__ import annotations

from pathlib import Path

from rag_orchestrator.graph import SqlGraphStore


def test_graph_store_queries(tmp_path: Path) -> None:
    store = SqlGraphStore(str(tmp_path / "graph.db"))

    store.upsert_repository(
        repo_id="1",
        platform="github",
        name="tg-bot",
        full_name="demo/tg-bot",
        url="https://github.com/demo/tg-bot",
        description="Telegram bot framework",
        stars=100,
        forks=10,
    )
    store.upsert_repository(
        repo_id="2",
        platform="github",
        name="other",
        full_name="demo/other",
        url="https://github.com/demo/other",
        description="Other project",
        stars=5,
        forks=1,
    )

    store.upsert_contributor("u1", "alice", "https://github.com/alice")
    store.upsert_contributor("u2", "bob", "https://github.com/bob")
    store.upsert_contribution_edge("1", "u1", 30)
    store.upsert_contribution_edge("1", "u2", 12)

    repos = store.find_repositories_by_keyword("telegram")
    assert repos
    assert repos[0]["full_name"] == "demo/tg-bot"

    contributors = store.get_contributor_count("demo/tg-bot")
    assert contributors == 2

    popular = store.get_most_popular_repository()
    assert popular is not None
    assert popular["full_name"] == "demo/tg-bot"
