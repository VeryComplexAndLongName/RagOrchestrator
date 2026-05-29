from __future__ import annotations

import sqlite3
from pathlib import Path


class SqlGraphStore:
    def __init__(self, db_path: str = ".rag_graph.sqlite") -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repositories (
                    repo_id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    name TEXT NOT NULL,
                    full_name TEXT NOT NULL UNIQUE,
                    url TEXT NOT NULL,
                    description TEXT NOT NULL,
                    stars INTEGER NOT NULL,
                    forks INTEGER NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contributors (
                    contributor_id TEXT PRIMARY KEY,
                    login TEXT NOT NULL,
                    url TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_contributors (
                    repo_id TEXT NOT NULL,
                    contributor_id TEXT NOT NULL,
                    contributions INTEGER NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (repo_id, contributor_id),
                    FOREIGN KEY (repo_id) REFERENCES repositories(repo_id),
                    FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_repos_full_name ON repositories(full_name)")
            conn.commit()

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repositories (repo_id, platform, name, full_name, url, description, stars, forks)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    platform=excluded.platform,
                    name=excluded.name,
                    full_name=excluded.full_name,
                    url=excluded.url,
                    description=excluded.description,
                    stars=excluded.stars,
                    forks=excluded.forks,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (repo_id, platform, name, full_name, url, description, stars, forks),
            )
            conn.commit()

    def upsert_contributor(self, contributor_id: str, login: str, url: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO contributors (contributor_id, login, url)
                VALUES (?, ?, ?)
                ON CONFLICT(contributor_id) DO UPDATE SET
                    login=excluded.login,
                    url=excluded.url,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (contributor_id, login, url),
            )
            conn.commit()

    def upsert_contribution_edge(self, repo_id: str, contributor_id: str, contributions: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repo_contributors (repo_id, contributor_id, contributions)
                VALUES (?, ?, ?)
                ON CONFLICT(repo_id, contributor_id) DO UPDATE SET
                    contributions=excluded.contributions,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (repo_id, contributor_id, contributions),
            )
            conn.commit()

    def find_repositories_by_keyword(self, keyword: str, limit: int = 10) -> list[dict]:
        pattern = f"%{keyword.lower()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT full_name, description, stars, forks, url
                FROM repositories
                WHERE lower(full_name) LIKE ? OR lower(description) LIKE ?
                ORDER BY stars DESC, forks DESC
                LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_contributor_count(self, repo_full_name: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(rc.contributor_id) AS cnt
                FROM repositories r
                LEFT JOIN repo_contributors rc ON rc.repo_id = r.repo_id
                WHERE r.full_name = ?
                """,
                (repo_full_name,),
            ).fetchone()
        return int((row or {"cnt": 0})["cnt"])

    def get_most_popular_repository(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT full_name, description, stars, forks, url
                FROM repositories
                ORDER BY stars DESC, forks DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None
