from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class DedupStore:
    def __init__(self, db_path: str = ".rag_dedup.sqlite") -> None:
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
                CREATE TABLE IF NOT EXISTS dedup_fingerprints (
                    fingerprint TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def fingerprint(text: str) -> str:
        normalized = " ".join(text.split()).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_known(self, fingerprint: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM dedup_fingerprints WHERE fingerprint = ? LIMIT 1",
                (fingerprint,),
            ).fetchone()
        return row is not None

    def add(self, fingerprint: str, source_id: str, chunk_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO dedup_fingerprints (fingerprint, source_id, chunk_id)
                VALUES (?, ?, ?)
                """,
                (fingerprint, source_id, chunk_id),
            )
            conn.commit()
