from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ragflow_orchestrator.errors import ProviderDependencyError
from ragflow_orchestrator.models import BaseChunk, RetrievalQuery, RetrievalResult

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except ImportError:  # pragma: no cover - optional dependency
    create_engine = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[misc, assignment]


class PGVectorProvider:
    name = "pgvector"

    def __init__(self, connection_string: str, table_name: str = "rag_chunks") -> None:
        if create_engine is None:
            raise ProviderDependencyError("Install pgvector deps: pip install rag-orchestrator[pgvector]")
        self.engine: Engine = create_engine(connection_string)
        self.table_name = table_name
        self._vector_dim = 0

    def ensure_schema(self, vector_dim: int) -> None:
        self._vector_dim = vector_dim
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        vector vector({vector_dim}) NOT NULL,
                        metadata JSONB NOT NULL,
                        source_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        kind TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                        semantic_type TEXT NOT NULL DEFAULT 'generic',
                        quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                        token_count INTEGER NOT NULL DEFAULT 0,
                        source_type TEXT NOT NULL DEFAULT 'unknown',
                        domain TEXT NOT NULL DEFAULT '',
                        risk_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        embedding_model TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
            )
            conn.execute(
                text(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS semantic_type TEXT NOT NULL DEFAULT 'generic'")
            )
            conn.execute(
                text(
                    f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.5"
                )
            )
            conn.execute(
                text(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS token_count INTEGER NOT NULL DEFAULT 0")
            )
            conn.execute(
                text(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'unknown'")
            )
            conn.execute(text(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS domain TEXT NOT NULL DEFAULT ''"))
            conn.execute(
                text(
                    f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS risk_score DOUBLE PRECISION NOT NULL DEFAULT 0.0"
                )
            )
            conn.execute(
                text(f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT ''")
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_id ON {self.table_name} (source_id)"
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_type ON {self.table_name} (source_type)"
                )
            )
            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_domain ON {self.table_name} (domain)")
            )

    def upsert_chunks(self, chunks: list[BaseChunk]) -> None:
        if not chunks:
            return
        sql = text(
            f"""
            INSERT INTO {self.table_name}
                        (
                                id, text, vector, metadata, source_id, chunk_index, created_at, kind, version, is_deleted,
                                semantic_type, quality_score, token_count, source_type, domain, risk_score, embedding_model
                        )
            VALUES (
              :id,
              :text,
              CAST(:vector AS vector),
              CAST(:metadata AS jsonb),
              :source_id,
              :chunk_index,
              :created_at,
              :kind,
              :version,
                            :is_deleted,
                            :semantic_type,
                            :quality_score,
                            :token_count,
                            :source_type,
                            :domain,
                            :risk_score,
                            :embedding_model
            )
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                vector = EXCLUDED.vector,
                metadata = EXCLUDED.metadata,
                source_id = EXCLUDED.source_id,
                chunk_index = EXCLUDED.chunk_index,
                created_at = EXCLUDED.created_at,
                kind = EXCLUDED.kind,
                version = EXCLUDED.version,
                is_deleted = EXCLUDED.is_deleted,
                semantic_type = EXCLUDED.semantic_type,
                quality_score = EXCLUDED.quality_score,
                token_count = EXCLUDED.token_count,
                source_type = EXCLUDED.source_type,
                domain = EXCLUDED.domain,
                risk_score = EXCLUDED.risk_score,
                embedding_model = EXCLUDED.embedding_model
            """
        )
        with self.engine.begin() as conn:
            for chunk in chunks:
                conn.execute(
                    sql,
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "vector": self._as_pg_vector(chunk.vector),
                        "metadata": json.dumps(chunk.metadata),
                        "source_id": chunk.source_id,
                        "chunk_index": chunk.chunk_index,
                        "created_at": chunk.created_at,
                        "kind": chunk.kind.value,
                        "version": chunk.version,
                        "is_deleted": chunk.is_deleted,
                        "semantic_type": chunk.semantic_type,
                        "quality_score": chunk.quality_score,
                        "token_count": chunk.token_count,
                        "source_type": chunk.source_type,
                        "domain": chunk.domain,
                        "risk_score": chunk.risk_score,
                        "embedding_model": chunk.embedding_model,
                    },
                )

    def delete_chunks(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        if not chunk_ids:
            return
        with self.engine.begin() as conn:
            if soft_delete:
                conn.execute(
                    text(f"UPDATE {self.table_name} SET is_deleted = TRUE WHERE id = ANY(:ids)"),
                    {"ids": chunk_ids},
                )
            else:
                conn.execute(text(f"DELETE FROM {self.table_name} WHERE id = ANY(:ids)"), {"ids": chunk_ids})

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        if not query.dense_vector:
            return []

        where_parts = ["TRUE"]
        params: dict[str, Any] = {
            "qvec": self._as_pg_vector(query.dense_vector),
            "limit": query.top_k,
        }

        if not query.include_deleted:
            where_parts.append("is_deleted = FALSE")

        for i, flt in enumerate(query.filters):
            if flt.op != "eq":
                continue
            key = f"fkey{i}"
            val = f"fval{i}"
            where_parts.append(f"metadata ->> :{key} = :{val}")
            params[key] = flt.key
            params[val] = str(flt.value)

        sql = text(
            f"""
            SELECT
                id,
                text,
                vector::text AS vector,
                metadata,
                source_id,
                chunk_index,
                created_at,
                kind,
                version,
                is_deleted,
                semantic_type,
                quality_score,
                token_count,
                source_type,
                domain,
                risk_score,
                embedding_model,
                1 - (vector <=> CAST(:qvec AS vector)) AS score
            FROM {self.table_name}
            WHERE {' AND '.join(where_parts)}
            ORDER BY vector <=> CAST(:qvec AS vector)
            LIMIT :limit
            """
        )

        with self.engine.begin() as conn:
            rows = conn.execute(sql, params).mappings().all()

        results: list[RetrievalResult] = []
        for row in rows:
            created_at = row["created_at"]
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            chunk = BaseChunk(
                id=row["id"],
                text=row["text"],
                vector=self._parse_pg_vector(row["vector"]),
                metadata=dict(row["metadata"]),
                source_id=row["source_id"],
                chunk_index=row["chunk_index"],
                created_at=created_at,
                kind=row["kind"],
                version=row["version"],
                is_deleted=row["is_deleted"],
                semantic_type=row.get("semantic_type", "generic"),
                quality_score=float(row.get("quality_score", 0.5)),
                token_count=int(row.get("token_count", 0)),
                source_type=row.get("source_type", "unknown"),
                domain=row.get("domain", ""),
                risk_score=float(row.get("risk_score", 0.0)),
                embedding_model=row.get("embedding_model", ""),
            )
            results.append(RetrievalResult(chunk=chunk, score=float(row["score"]), provider=self.name))

        return results

    def healthcheck(self) -> bool:
        try:
            with self.engine.begin() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @staticmethod
    def _as_pg_vector(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"

    @staticmethod
    def _parse_pg_vector(vector_text: str) -> list[float]:
        cleaned = vector_text.strip("[]")
        if not cleaned:
            return []
        return [float(item) for item in cleaned.split(",")]
