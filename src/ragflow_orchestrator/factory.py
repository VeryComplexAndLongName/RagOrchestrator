from __future__ import annotations

from ragflow_orchestrator.adapters import PGVectorProvider, QdrantProvider, SQLiteVecProvider
from ragflow_orchestrator.errors import ConfigurationError
from ragflow_orchestrator.protocols import RAGProvider


def create_provider(kind: str, **kwargs: object) -> RAGProvider:
    """Create a RAG provider based on configuration.

    Supported kinds:
    - qdrant: Qdrant only (legacy, now deprecated)
    - pgvector, postgres, postgresql: PostgreSQL with pgvector (legacy)
    - postgres+qdrant, postgresql+qdrant: PostgreSQL metadata + Qdrant vectors (NEW)

    For postgres+qdrant, pass dsn (PostgreSQL connection string) and qdrant_url.
    """
    kind_lower = kind.lower()

    if kind_lower in {"sqlite", "sqlite+vec", "sqlite_vec"}:
        return SQLiteVecProvider(**kwargs)

    # Legacy providers (deprecated but kept for compatibility)
    if kind_lower in {"qdrant"}:
        return QdrantProvider(**kwargs)
    if kind_lower in {"pgvector", "postgres", "postgresql"}:
        return PGVectorProvider(**kwargs)
    
    # New: PostgreSQL + Qdrant (recommended configuration)
    if kind_lower in {"postgres+qdrant", "postgresql+qdrant", "postgres_qdrant"}:
        from ragflow_orchestrator.adapters.postgres_qdrant_provider import PostgresQdrantProvider
        return PostgresQdrantProvider(**kwargs)
    
    raise ConfigurationError(f"Unknown provider kind: {kind}")

