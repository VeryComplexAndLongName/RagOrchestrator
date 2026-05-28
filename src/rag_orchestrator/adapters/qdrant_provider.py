from __future__ import annotations

import uuid

from rag_orchestrator.errors import ProviderDependencyError
from rag_orchestrator.models import BaseChunk, RetrievalQuery, RetrievalResult

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]


class QdrantProvider:
    name = "qdrant"

    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "rag_chunks") -> None:
        if QdrantClient is None:
            raise ProviderDependencyError("Install qdrant dependencies: pip install rag-orchestrator[qdrant]")
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def ensure_schema(self, vector_dim: int) -> None:
        collections = self.client.get_collections().collections
        exists = any(item.name == self.collection_name for item in collections)
        if exists:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[BaseChunk]) -> None:
        if not chunks:
            return
        points = []
        for chunk in chunks:
            point_id = self._to_qdrant_point_id(chunk.id)
            payload = {
                "chunk_id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "source_id": chunk.source_id,
                "chunk_index": chunk.chunk_index,
                "created_at": chunk.created_at.isoformat(),
                "kind": chunk.kind.value,
                "version": chunk.version,
                "is_deleted": chunk.is_deleted,
            }
            points.append(PointStruct(id=point_id, vector=chunk.vector, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_chunks(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        if not chunk_ids:
            return
        point_ids = [self._to_qdrant_point_id(chunk_id) for chunk_id in chunk_ids]
        if soft_delete:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_deleted": True},
                points=point_ids,
            )
            return
        self.client.delete(collection_name=self.collection_name, points_selector=point_ids)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        if not query.dense_vector:
            return []

        conditions = []
        for flt in query.filters:
            if flt.op != "eq":
                continue
            conditions.append(FieldCondition(key=f"metadata.{flt.key}", match=MatchValue(value=flt.value)))

        if not query.include_deleted:
            conditions.append(FieldCondition(key="is_deleted", match=MatchValue(value=False)))

        q_filter = Filter(must=conditions) if conditions else None
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query.dense_vector,
                query_filter=q_filter,
                limit=query.top_k,
                with_payload=True,
                with_vectors=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query.dense_vector,
                query_filter=q_filter,
                limit=query.top_k,
                with_payload=True,
                with_vectors=True,
            )
            hits = getattr(response, "points", response)

        results: list[RetrievalResult] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = BaseChunk(
                id=str(payload.get("chunk_id") or hit.id),
                text=payload.get("text", ""),
                vector=list(hit.vector or []),
                metadata=payload.get("metadata", {}),
                source_id=payload.get("source_id", ""),
                chunk_index=payload.get("chunk_index", 0),
                created_at=payload.get("created_at"),
                kind=payload.get("kind", "generic"),
                version=payload.get("version", 1),
                is_deleted=payload.get("is_deleted", False),
            )
            results.append(RetrievalResult(chunk=chunk, score=float(hit.score), provider=self.name))

        return results

    def healthcheck(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    @staticmethod
    def _to_qdrant_point_id(chunk_id: str) -> int | str:
        if chunk_id.isdigit():
            return int(chunk_id)
        try:
            parsed = uuid.UUID(chunk_id)
            return str(parsed)
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
