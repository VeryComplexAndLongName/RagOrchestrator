from __future__ import annotations

from typing import Protocol


class GraphStorageProtocol(Protocol):
    def upsert_repository(
        self,
        repo_id: str,
        platform: str,
        name: str,
        full_name: str,
        url: str,
        description: str,
        stars: int,
        forks: int,
    ) -> None:
        ...

    def upsert_contributor(self, contributor_id: str, login: str, url: str) -> None:
        ...

    def upsert_contribution_edge(self, repo_id: str, contributor_id: str, contributions: int) -> None:
        ...

    def find_repositories_by_keyword(self, keyword: str, limit: int = 10) -> list[dict]:
        ...

    def get_contributor_count(self, repo_full_name: str) -> int:
        ...

    def get_most_popular_repository(self) -> dict | None:
        ...
