from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ragflow_orchestrator.adapters.common import cosine_similarity, from_iso, to_iso
from ragflow_orchestrator.models import BaseChunk, RetrievalFilter, RetrievalQuery, RetrievalResult


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
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    semantic_type TEXT NOT NULL DEFAULT 'generic',
                    quality_score REAL NOT NULL DEFAULT 0.5,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    source_type TEXT NOT NULL DEFAULT 'unknown',
                    domain TEXT NOT NULL DEFAULT '',
                    risk_score REAL NOT NULL DEFAULT 0.0,
                    embedding_model TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._ensure_columns(conn)
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_id ON {self.table_name} (source_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_type ON {self.table_name} (source_type)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_domain ON {self.table_name} (domain)"
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
                    (
                        id, text, vector, metadata, source_id, chunk_index, created_at, kind, version, is_deleted,
                        semantic_type, quality_score, token_count, source_type, domain, risk_score, embedding_model
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text = excluded.text,
                        vector = excluded.vector,
                        metadata = excluded.metadata,
                        source_id = excluded.source_id,
                        chunk_index = excluded.chunk_index,
                        created_at = excluded.created_at,
                        kind = excluded.kind,
                        version = excluded.version,
                        is_deleted = excluded.is_deleted,
                        semantic_type = excluded.semantic_type,
                        quality_score = excluded.quality_score,
                        token_count = excluded.token_count,
                        source_type = excluded.source_type,
                        domain = excluded.domain,
                        risk_score = excluded.risk_score,
                        embedding_model = excluded.embedding_model
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
                        chunk.semantic_type,
                        chunk.quality_score,
                        chunk.token_count,
                        chunk.source_type,
                        chunk.domain,
                        chunk.risk_score,
                        chunk.embedding_model,
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
                semantic_type=row["semantic_type"] if "semantic_type" in row.keys() else "generic",
                quality_score=float(row["quality_score"]) if "quality_score" in row.keys() else 0.5,
                token_count=int(row["token_count"]) if "token_count" in row.keys() else 0,
                source_type=row["source_type"] if "source_type" in row.keys() else "unknown",
                domain=row["domain"] if "domain" in row.keys() else "",
                risk_score=float(row["risk_score"]) if "risk_score" in row.keys() else 0.0,
                embedding_model=row["embedding_model"] if "embedding_model" in row.keys() else "",
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

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(f"PRAGMA table_info({self.table_name})").fetchall()
        existing = {row[1] for row in rows}
        required = {
            "semantic_type": "TEXT NOT NULL DEFAULT 'generic'",
            "quality_score": "REAL NOT NULL DEFAULT 0.5",
            "token_count": "INTEGER NOT NULL DEFAULT 0",
            "source_type": "TEXT NOT NULL DEFAULT 'unknown'",
            "domain": "TEXT NOT NULL DEFAULT ''",
            "risk_score": "REAL NOT NULL DEFAULT 0.0",
            "embedding_model": "TEXT NOT NULL DEFAULT ''",
        }
        for column, ddl in required.items():
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {column} {ddl}")
