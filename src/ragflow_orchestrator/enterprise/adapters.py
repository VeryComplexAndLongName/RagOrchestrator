from __future__ import annotations

from ragflow_orchestrator.models import RetrievalQuery, RetrievalResult
from ragflow_orchestrator.protocols import Embedder, RAGProvider


class ACLAwareRetrieverAdapter:
    """Query retriever adapter with explicit ACL propagation into RetrievalQuery."""

    def __init__(
        self,
        *,
        provider: RAGProvider,
        embedder: Embedder,
        include_deleted: bool = False,
    ) -> None:
        self.provider = provider
        self.embedder = embedder
        self.include_deleted = include_deleted

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        payload = dict(filters or {})
        acl_principals_raw = payload.pop("acl_principals", None)
        include_deleted_raw = payload.pop("include_deleted", self.include_deleted)

        acl_principals: list[str] | None = None
        if isinstance(acl_principals_raw, list):
            acl_principals = [str(item) for item in acl_principals_raw]

        include_deleted = bool(include_deleted_raw)

        query = RetrievalQuery(
            text_query=question,
            top_k=top_k,
            dense_vector=self.embedder.embed(question),
            filters=[{"key": key, "value": value} for key, value in payload.items()],
            include_deleted=include_deleted,
            acl_principals=acl_principals,
        )
        return self.provider.retrieve(query)


class StaticRetrieverAdapter:
    """Small adapter useful for tests and offline simulations."""

    def __init__(self, rows: list[RetrievalResult]) -> None:
        self._rows = rows

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        _ = question, filters
        return self._rows[:top_k]
