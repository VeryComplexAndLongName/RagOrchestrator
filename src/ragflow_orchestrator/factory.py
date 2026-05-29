from __future__ import annotations

from ragflow_orchestrator.adapters import PGVectorProvider, QdrantProvider, SQLiteVecProvider
from ragflow_orchestrator.errors import ConfigurationError
from ragflow_orchestrator.protocols import RAGProvider


def create_provider(kind: str, **kwargs: object) -> RAGProvider:
    kind_lower = kind.lower()
    if kind_lower in {"sqlite", "sqlite+vec", "sqlite_vec"}:
        return SQLiteVecProvider(**kwargs)
    if kind_lower in {"qdrant"}:
        return QdrantProvider(**kwargs)
    if kind_lower in {"pgvector", "postgres", "postgresql"}:
        return PGVectorProvider(**kwargs)
    raise ConfigurationError(f"Unknown provider kind: {kind}")
