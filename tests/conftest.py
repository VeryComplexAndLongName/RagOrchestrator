from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.models import BaseChunk


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "qdrant: requires local qdrant service")
    config.addinivalue_line("markers", "pgvector: requires local postgres+pgvector service")


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333")


@pytest.fixture(scope="session")
def pgvector_dsn() -> str:
    explicit = os.getenv("PGVECTOR_DSN")
    if explicit:
        return explicit

    pg_user = os.getenv("PGUSER", "postgres")
    pg_password = os.getenv("PGPASSWORD", "")
    pg_host = os.getenv("PGHOST", "localhost")
    pg_port = os.getenv("PGPORT", "5432")
    pg_db = os.getenv("PGDATABASE", "app")

    auth = pg_user
    if pg_password:
        auth = f"{pg_user}:{pg_password}"
    return f"postgresql+psycopg://{auth}@{pg_host}:{pg_port}/{pg_db}"


@pytest.fixture()
def sample_chunks() -> list[BaseChunk]:
    embedder = HashEmbedder(dimensions=64)
    now = datetime.now(timezone.utc)

    rows = [
        (
            "math:add",
            "def add(a, b): return a + b",
            {"language": "python", "doctype": "code", "tenant_id": "t1"},
            "math_utils.py",
            0,
        ),
        (
            "math:sub",
            "def sub(a, b): return a - b",
            {"language": "python", "doctype": "code", "tenant_id": "t1"},
            "math_utils.py",
            1,
        ),
        (
            "rag:overview",
            "RAG orchestration unifies ingestion, retrieval and quality metrics.",
            {"language": "en", "doctype": "note", "tenant_id": "t2"},
            "rag_doc",
            0,
        ),
    ]

    chunks: list[BaseChunk] = []
    for chunk_id, text, metadata, source_id, chunk_index in rows:
        chunks.append(
            BaseChunk(
                id=chunk_id,
                text=text,
                vector=embedder.embed(text),
                metadata=metadata,
                source_id=source_id,
                chunk_index=chunk_index,
                created_at=now,
            )
        )
    return chunks


@pytest.fixture(scope="session")
def query_embedder() -> HashEmbedder:
    return HashEmbedder(dimensions=64)
