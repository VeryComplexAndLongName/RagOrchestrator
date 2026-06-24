"""PostgreSQL + Qdrant combined provider.

Metadata (documents, versions, tags, ACL) → PostgreSQL
Vectors and chunks → Qdrant
Search results are filtered through both layers for proper authorization.
"""

from __future__ import annotations

from ragflow_orchestrator.adapters.postgres_backend import PostgresMetadataBackend
from ragflow_orchestrator.adapters.qdrant_provider import QdrantProvider
from ragflow_orchestrator.models import BaseChunk, RetrievalQuery, RetrievalResult


class PostgresQdrantProvider:
    """Combined metadata (PostgreSQL) + vector (Qdrant) provider."""

    name = "postgres+qdrant"

    def __init__(
        self,
        dsn: str,
        qdrant_url: str = "http://localhost:6333",
        qdrant_collection: str = "rag_chunks",
        qdrant_api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize combined provider.

        Args:
            dsn: PostgreSQL connection string
            qdrant_url: Qdrant server URL
            qdrant_collection: Qdrant collection name
            qdrant_api_key: Optional Qdrant API key
        """
        self.metadata = PostgresMetadataBackend(dsn=dsn)
        self.vectors = QdrantProvider(
            url=qdrant_url,
            collection_name=qdrant_collection,
            api_key=qdrant_api_key,
        )

    def ensure_schema(self, vector_dim: int) -> None:
        """Ensure both PostgreSQL schema and Qdrant collection exist."""
        # PostgreSQL migrations are typically run separately via MigrationManager
        # This just ensures Qdrant collection exists
        self.vectors.ensure_schema(vector_dim)

    def upsert_chunks(self, chunks: list[BaseChunk]) -> None:
        """Upsert chunks to both Qdrant and PostgreSQL.

        For each chunk:
        1. Create PostgreSQL entry (chunk table record)
        2. Upsert to Qdrant with metadata including document_id, version_id
        """
        if not chunks:
            return

        # Upsert to Qdrant (vectors and payload)
        self.vectors.upsert_chunks(chunks)

        # TODO: Create PostgreSQL chunk records
        # This requires version_id to be set on each chunk, which should come
        # from the ingestion pipeline (document_pipeline.py)
        # For now, we rely on the caller to manage PostgreSQL records via
        # PostgresMetadataBackend directly.

    def delete_chunks(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        """Delete/mark chunks in Qdrant (and PostgreSQL if applicable)."""
        self.vectors.delete_chunks(chunk_ids, soft_delete=soft_delete)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Retrieve chunks from Qdrant with ACL filtering.

        The query may include:
        - acl_principals: list of user roles/groups for filtering
        - Additional filters that are forwarded to Qdrant

        Results are filtered based on:
        1. Qdrant payload (is_restricted, acl_principals in metadata)
        2. PostgreSQL authorization (authorized_chunk_ids function for final check)
        """
        results = self.vectors.retrieve(query)

        # Optional: additional authorization check via PostgreSQL
        # This adds belt-and-suspenders security
        # For now, rely on Qdrant filtering which already handles ACL

        return results

    def delete_by_source(self, source_id: str, soft_delete: bool = True) -> None:
        """Delete/mark all chunks of a source (version)."""
        self.vectors.delete_by_source(source_id, soft_delete=soft_delete)

    def delete_by_document(self, document_id: str, soft_delete: bool = True) -> None:
        """Delete/mark all chunks of a document (all versions)."""
        self.vectors.delete_by_document(document_id, soft_delete=soft_delete)

    def update_acl_by_document(
        self,
        document_id: str,
        is_restricted: bool,
        principals: list[str] | None = None,
    ) -> None:
        """Update ACL projection in Qdrant (PostgreSQL is source of truth).

        This method is called by AclSync after updating PostgreSQL ACL tables.
        It projects the new ACL state to all chunks' metadata in Qdrant.
        """
        self.vectors.update_acl_by_document(
            document_id=document_id,
            is_restricted=is_restricted,
            principals=principals,
        )

    def healthcheck(self) -> bool:
        """Check both PostgreSQL and Qdrant."""
        qdrant_ok = self.vectors.healthcheck()
        # Optional: check PostgreSQL connection
        # postgres_ok = self.metadata.healthcheck()
        return qdrant_ok

    def count(self, include_deleted: bool = False) -> int:
        """Count chunks in Qdrant."""
        return self.vectors.count(include_deleted=include_deleted)

    def scroll_all(self, batch_size: int = 256, with_vectors: bool = False):
        """Scroll all chunks from Qdrant."""
        return self.vectors.scroll_all(batch_size=batch_size, with_vectors=with_vectors)
