from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rag_orchestrator.adapters.common import cosine_similarity, from_iso, to_iso
from rag_orchestrator.models import BaseChunk, RetrievalFilter, RetrievalQuery, RetrievalResult


class SQLiteVecProvider:
    """SQLite provider with JSON metadata and local cosine search fallback."""

    name = "sqlite+vec"

    def __init__(self, db_path: str = "rag.db", table_name: str = "rag_chunks") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._vector_dim = 0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self, vector_dim: int) -> None:
        self._vector_dim = vector_dim
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    is_deleted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_id ON {self.table_name} (source_id)"
            )
            conn.commit()

    def upsert_chunks(self, chunks: list[BaseChunk]) -> None:
        if not chunks:
            return
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute(
                    f"""
                    INSERT INTO {self.table_name}
                    (id, text, vector, metadata, source_id, chunk_index, created_at, kind, version, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text = excluded.text,
                        vector = excluded.vector,
                        metadata = excluded.metadata,
                        source_id = excluded.source_id,
                        chunk_index = excluded.chunk_index,
                        created_at = excluded.created_at,
                        kind = excluded.kind,
                        version = excluded.version,
                        is_deleted = excluded.is_deleted
                    """,
                    (
                        chunk.id,
                        chunk.text,
                        json.dumps(chunk.vector),
                        json.dumps(chunk.metadata),
                        chunk.source_id,
                        chunk.chunk_index,
                        to_iso(chunk.created_at),
                        chunk.kind.value,
                        chunk.version,
                        1 if chunk.is_deleted else 0,
                    ),
                )
            conn.commit()

    def delete_chunks(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            if soft_delete:
                conn.execute(
                    f"UPDATE {self.table_name} SET is_deleted = 1 WHERE id IN ({placeholders})",
                    tuple(chunk_ids),
                )
            else:
                conn.execute(f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})", tuple(chunk_ids))
            conn.commit()

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM {self.table_name}").fetchall()

        results: list[RetrievalResult] = []
        for row in rows:
            if not query.include_deleted and bool(row["is_deleted"]):
                continue
            metadata = json.loads(row["metadata"])
            if not self._matches_filters(metadata, query.filters):
                continue

            vector = json.loads(row["vector"])
            score = cosine_similarity(query.dense_vector, vector) if query.dense_vector else 0.0

            if not query.dense_vector and query.text_query:
                hay = f"{row['text']} {json.dumps(metadata)}".lower()
                score = 1.0 if query.text_query.lower() in hay else 0.0

            chunk = BaseChunk(
                id=row["id"],
                text=row["text"],
                vector=vector,
                metadata=metadata,
                source_id=row["source_id"],
                chunk_index=row["chunk_index"],
                created_at=from_iso(row["created_at"]),
                kind=row["kind"],
                version=row["version"],
                is_deleted=bool(row["is_deleted"]),
            )
            results.append(RetrievalResult(chunk=chunk, score=score, provider=self.name))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: query.top_k]

    def healthcheck(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def _matches_filters(metadata: dict[str, object], filters: list[RetrievalFilter]) -> bool:
        for flt in filters:
            value = metadata.get(flt.key)
            if flt.op == "eq" and value != flt.value:
                return False
            if flt.op == "in" and value not in flt.value:
                return False
        return True
