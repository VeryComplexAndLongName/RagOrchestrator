from __future__ import annotations

import uuid
from typing import Any, Iterable, Iterator

from ragflow_orchestrator.errors import ProviderDependencyError
from ragflow_orchestrator.models import BaseChunk, RetrievalQuery, RetrievalResult

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        PayloadSchemaType,
        PointStruct,
        Range,
        VectorParams,
    )
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]


# Payload-поля, по которым осмысленно строить фильтры -> создаём индексы.
_INDEXED_KEYWORD_FIELDS = (
    "source_id",
    "kind",
    "semantic_type",
    "source_type",
    "domain",
    "embedding_model",
    # структура нормативки
    "metadata.clause_path",
    "metadata.standard_ref",
    "metadata.section",
    "metadata.edition_label",
    "metadata.document_id",
    "metadata.version_id",
    # ACL-проекция
    "metadata.acl_principals",
)
_INDEXED_BOOL_FIELDS = (
    "is_deleted",
    "metadata.is_restricted",
)
_INDEXED_INT_FIELDS = (
    "chunk_index",
    "metadata.page",
)


class QdrantProvider:
    name = "qdrant"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "rag_chunks",
        upsert_batch_size: int = 256,
        api_key: str | None = None,
    ) -> None:
        if QdrantClient is None:
            raise ProviderDependencyError(
                "Install qdrant dependencies: pip install rag-orchestrator[qdrant]"
            )
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.upsert_batch_size = max(1, int(upsert_batch_size))

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------
    def ensure_schema(self, vector_dim: int) -> None:
        collections = self.client.get_collections().collections
        exists = any(item.name == self.collection_name for item in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Payload-индексы критичны для скорости фильтрации на больших коллекциях."""
        def _try(field: str, schema: Any) -> None:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                # индекс уже существует или версия клиента иначе валидирует — не критично
                pass

        for field in _INDEXED_KEYWORD_FIELDS:
            _try(field, PayloadSchemaType.KEYWORD)
        for field in _INDEXED_BOOL_FIELDS:
            _try(field, PayloadSchemaType.BOOL)
        for field in _INDEXED_INT_FIELDS:
            _try(field, PayloadSchemaType.INTEGER)

    # --------------------------------------------------------
    # Upsert (с батчингом)
    # --------------------------------------------------------
    def upsert_chunks(self, chunks: list[BaseChunk]) -> None:
        if not chunks:
            return
        points = [self._to_point(chunk) for chunk in chunks]
        for batch in self._batched(points, self.upsert_batch_size):
            self.client.upsert(collection_name=self.collection_name, points=list(batch))

    def _to_point(self, chunk: BaseChunk) -> "PointStruct":
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
            "semantic_type": chunk.semantic_type,
            "quality_score": chunk.quality_score,
            "token_count": chunk.token_count,
            "source_type": chunk.source_type,
            "domain": chunk.domain,
            "risk_score": chunk.risk_score,
            "embedding_model": chunk.embedding_model,
        }
        return PointStruct(id=point_id, vector=chunk.vector, payload=payload)

    # --------------------------------------------------------
    # Delete
    # --------------------------------------------------------
    def delete_chunks(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        if not chunk_ids:
            return
        point_ids = [self._to_qdrant_point_id(cid) for cid in chunk_ids]
        if soft_delete:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_deleted": True},
                points=point_ids,
            )
            return
        self.client.delete(collection_name=self.collection_name, points_selector=point_ids)

    # --------------------------------------------------------
    # ACL: массовое обновление прав по документу (без реингеста)
    # --------------------------------------------------------
    def update_acl_by_document(
        self,
        document_id: str,
        is_restricted: bool,
        principals: list[str] | None = None,
    ) -> None:
        """
        Обновить ACL-проекцию у ВСЕХ чанков документа, не трогая векторы.

        Меняются только два payload-поля:
          metadata.is_restricted  — ограничен ли документ
          metadata.acl_principals — список разрешённых ролей/групп/пользователей

        Фильтрация идёт по metadata.document_id, поэтому при ingest это поле
        обязательно должно попадать в metadata каждого чанка. Обновляются все
        версии документа разом (права у вас общие на документ).

        set_payload в Qdrant выполняет частичное обновление (merge): остальные
        ключи payload и сами векторы остаются нетронутыми.
        """
        flt = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={
                "metadata": {
                    "is_restricted": bool(is_restricted),
                    "acl_principals": list(principals or []),
                }
            },
            points=flt,
        )

    # --------------------------------------------------------
    # Retrieve  (с расширенными фильтрами + ACL)
    # --------------------------------------------------------
    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        if not query.dense_vector:
            return []

        q_filter = self._build_filter(query)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query.dense_vector,
                query_filter=q_filter,
                limit=query.top_k,
                with_payload=True,
                with_vectors=True,
            )
            hits = getattr(response, "points", response)
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query.dense_vector,
                query_filter=q_filter,
                limit=query.top_k,
                with_payload=True,
                with_vectors=True,
            )

        return [self._hit_to_result(hit) for hit in hits]

    def _build_filter(self, query: RetrievalQuery) -> "Filter | None":
        must: list[Any] = []
        must_not: list[Any] = []
        should: list[Any] = []

        # 1. Пользовательские фильтры с операторами
        for flt in query.filters:
            op = getattr(flt, "op", "eq")
            cond = self._condition_from_filter(flt)
            if cond is None:
                continue
            if op == "neq":
                must_not.append(cond)
            else:
                must.append(cond)

        # 2. Мягкое удаление
        if not getattr(query, "include_deleted", False):
            must.append(FieldCondition(key="is_deleted", match=MatchValue(value=False)))

        # 3. ACL: политика "нет записей = доступно всем".
        #    Показать чанк, если он НЕ ограничен, ИЛИ его acl_principals
        #    пересекается с ролями/идентификаторами пользователя.
        principals = getattr(query, "acl_principals", None)
        if principals:
            should.append(
                FieldCondition(key="metadata.is_restricted", match=MatchValue(value=False))
            )
            should.append(
                FieldCondition(
                    key="metadata.acl_principals",
                    match=MatchAny(any=list(principals)),
                )
            )

        if not must and not must_not and not should:
            return None

        if should and _SUPPORTS_MIN_SHOULD:
            return Filter(
                must=must or None,
                must_not=must_not or None,
                should=should,
                min_should=_MinShould(conditions=should, min_count=1),
            )
        if should:
            # Фолбэк для старых клиентов: ACL-блок заворачиваем во вложенный
            # Filter(should=...), что даёт ту же семантику "хотя бы одно".
            must = list(must) + [Filter(should=should)]
        return Filter(
            must=must or None,
            must_not=must_not or None,
        )

    def _condition_from_filter(self, flt: Any) -> "FieldCondition | None":
        """Поддержка операторов: eq, neq, any, in, gte, lte."""
        key = flt.key
        full_key = key if key.startswith(
            ("metadata.", "is_", "source_", "chunk_", "kind",
             "semantic_type", "domain", "embedding_model")
        ) else f"metadata.{key}"
        op = getattr(flt, "op", "eq")
        value = flt.value

        if op in ("eq", "neq"):
            return FieldCondition(key=full_key, match=MatchValue(value=value))
        if op in ("any", "in"):
            values = value if isinstance(value, (list, tuple)) else [value]
            return FieldCondition(key=full_key, match=MatchAny(any=list(values)))
        if op == "gte":
            return FieldCondition(key=full_key, range=Range(gte=value))
        if op == "lte":
            return FieldCondition(key=full_key, range=Range(lte=value))
        return None

    def _hit_to_result(self, hit: Any) -> RetrievalResult:
        payload = hit.payload or {}
        chunk = BaseChunk(
            id=str(payload.get("chunk_id") or hit.id),
            text=payload.get("text", ""),
            vector=list(getattr(hit, "vector", None) or []),
            metadata=payload.get("metadata", {}) or {},
            source_id=payload.get("source_id", ""),
            chunk_index=int(payload.get("chunk_index", 0) or 0),
            kind=payload.get("kind", "generic"),
            version=int(payload.get("version", 1) or 1),
            is_deleted=bool(payload.get("is_deleted", False)),
            semantic_type=payload.get("semantic_type", "generic"),
            quality_score=float(payload.get("quality_score", 0.5) or 0.5),
            token_count=int(payload.get("token_count", 0) or 0),
            source_type=payload.get("source_type", "unknown"),
            domain=payload.get("domain", ""),
            risk_score=float(payload.get("risk_score", 0.0) or 0.0),
            embedding_model=payload.get("embedding_model", ""),
        )
        # created_at оставляем дефолтным (now). Реальная дата — в payload и Postgres.
        return RetrievalResult(chunk=chunk, score=float(hit.score), provider=self.name)

    # --------------------------------------------------------
    # Админ / обслуживание
    # --------------------------------------------------------
    def count(self, include_deleted: bool = False) -> int:
        flt = None
        if not include_deleted:
            flt = Filter(must=[FieldCondition(key="is_deleted", match=MatchValue(value=False))])
        return self.client.count(
            collection_name=self.collection_name, count_filter=flt, exact=True
        ).count

    def scroll_all(
        self, batch_size: int = 256, with_vectors: bool = False
    ) -> Iterator[BaseChunk]:
        """Постраничный обход всей коллекции (для реиндекса/аудита)."""
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for p in points:
                payload = p.payload or {}
                yield BaseChunk(
                    id=str(payload.get("chunk_id") or p.id),
                    text=payload.get("text", ""),
                    vector=list(getattr(p, "vector", None) or []),
                    metadata=payload.get("metadata", {}) or {},
                    source_id=payload.get("source_id", ""),
                    chunk_index=int(payload.get("chunk_index", 0) or 0),
                )
            if offset is None:
                break

    def delete_by_source(self, source_id: str, soft_delete: bool = True) -> None:
        """Удалить/пометить все чанки одной версии (source_id == version_id)."""
        flt = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))])
        if soft_delete:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_deleted": True},
                points=flt,
            )
        else:
            self.client.delete(collection_name=self.collection_name, points_selector=flt)

    def delete_by_document(self, document_id: str, soft_delete: bool = True) -> None:
        """Удалить/пометить все чанки документа (все версии) по metadata.document_id."""
        flt = Filter(
            must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
        )
        if soft_delete:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_deleted": True},
                points=flt,
            )
        else:
            self.client.delete(collection_name=self.collection_name, points_selector=flt)

    def healthcheck(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    @staticmethod
    def _batched(items: list[Any], size: int) -> Iterable[list[Any]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    @staticmethod
    def _to_qdrant_point_id(chunk_id: str) -> int | str:
        if chunk_id.isdigit():
            return int(chunk_id)
        try:
            return str(uuid.UUID(chunk_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


# Совместимость по версиям qdrant-client: min_should есть не везде.
try:
    from qdrant_client.models import MinShould as _MinShould  # type: ignore
    _SUPPORTS_MIN_SHOULD = True
except Exception:  # pragma: no cover
    _MinShould = None  # type: ignore
    _SUPPORTS_MIN_SHOULD = False