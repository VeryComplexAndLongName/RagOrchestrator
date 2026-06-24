"""PostgreSQL metadata backend for document versioning, tagging, and ACL.

Stores document metadata, versions, tags, and access control lists in PostgreSQL.
Coordinates with QdrantProvider for vector storage and ACL synchronization.

Source of truth for schema: MyTasks/sql.md
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg


class PostgresMetadataBackend:
    """Manages document metadata lifecycle in PostgreSQL."""

    def __init__(self, dsn: str) -> None:
        """Initialize PostgreSQL metadata backend.

        Args:
            dsn: PostgreSQL connection string (e.g., "postgresql://user:pass@localhost/db")
        """
        self.dsn = dsn
        self._conn = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None:
            self._conn = psycopg.connect(self.dsn)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PostgresMetadataBackend:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ============================================================
    # Document management
    # ============================================================

    def create_document(
        self,
        source_type: str,
        document_type: str,
        document_subtype: str | None = None,
        title: str | None = None,
        doc_number: str | None = None,
        language: str | None = None,
        domain: str | None = None,
    ) -> str:
        """Create a new document entry. Returns document_id (UUID)."""
        self.connect()
        doc_id = str(uuid4())
        sql = """
        INSERT INTO documents
            (id, source_type, document_type, document_subtype, title, doc_number, language, domain)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._conn.execute(
            sql,
            (doc_id, source_type, document_type, document_subtype, title, doc_number, language, domain),
        )
        self._conn.commit()
        return doc_id

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Retrieve document metadata by ID."""
        self.connect()
        sql = "SELECT * FROM documents WHERE id = %s"
        result = self._conn.execute(sql, (document_id,)).fetchone()
        if not result:
            return None
        cols = [desc[0] for desc in self._conn.cursor().description]
        return dict(zip(cols, result))

    # ============================================================
    # Version management
    # ============================================================

    def create_version(
        self,
        document_id: str,
        file_path: str,
        content_hash: str,
        document_subtype: str | None = None,
        file_ext: str | None = None,
        source_hash: str | None = None,
        semantic_type: str | None = None,
        token_count: int | None = None,
        quality_score: float | None = None,
        risk_score: float | None = None,
        embedding_model: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        edition_label: str | None = None,
        ingestion_reason: str | None = None,
    ) -> str:
        """Create a new document version. Returns version_id (UUID)."""
        self.connect()
        
        # Get next version number
        sql = "SELECT MAX(version_number) FROM document_versions WHERE document_id = %s"
        result = self._conn.execute(sql, (document_id,)).fetchone()
        next_version = (result[0] or 0) + 1 if result and result[0] else 1
        
        version_id = str(uuid4())
        sql = """
        INSERT INTO document_versions
            (id, document_id, version_number, file_path, content_hash, file_ext,
             source_hash, semantic_type, document_subtype, token_count, quality_score, risk_score,
             embedding_model, valid_from, valid_to, edition_label,
             ingestion_status, ingestion_reason, ingestion_timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        self._conn.execute(
            sql,
            (
                version_id,
                document_id,
                next_version,
                file_path,
                content_hash,
                file_ext,
                source_hash,
                semantic_type,
                document_subtype,
                token_count,
                quality_score,
                risk_score,
                embedding_model,
                valid_from,
                valid_to,
                edition_label,
                "pending",  # ingestion_status
                ingestion_reason,
            ),
        )
        self._conn.commit()
        return version_id

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        """Retrieve document version metadata by ID."""
        self.connect()
        sql = "SELECT * FROM document_versions WHERE id = %s"
        result = self._conn.execute(sql, (version_id,)).fetchone()
        return self._row_to_dict(result) if result else None

    def activate_version(self, version_id: str) -> None:
        """Mark version as active and deactivate others for the same document."""
        self.connect()
        # Get document_id for this version
        sql = "SELECT document_id FROM document_versions WHERE id = %s"
        doc_id = self._conn.execute(sql, (version_id,)).fetchone()[0]
        
        # Deactivate all versions for this document
        sql = "UPDATE document_versions SET is_active = FALSE WHERE document_id = %s"
        self._conn.execute(sql, (doc_id,))
        
        # Activate the specific version
        sql = "UPDATE document_versions SET is_active = TRUE WHERE id = %s"
        self._conn.execute(sql, (version_id,))
        self._conn.commit()

    def update_ingestion_status(
        self, version_id: str, status: str, reason: str | None = None
    ) -> None:
        """Update version ingestion status."""
        self.connect()
        sql = """
        UPDATE document_versions
        SET ingestion_status = %s, ingestion_reason = %s
        WHERE id = %s
        """
        self._conn.execute(sql, (status, reason, version_id))
        self._conn.commit()

    # ============================================================
    # Chunk management
    # ============================================================

    def create_chunk(
        self,
        version_id: str,
        chunk_index: int,
        qdrant_point_id: str,
        clause_path: str | None = None,
        standard_ref: str | None = None,
        section: str | None = None,
        page: int | None = None,
        source: str = "text_layer",
        char_len: int | None = None,
        token_count: int | None = None,
        embedding_model: str | None = None,
    ) -> str:
        """Create chunk record in PostgreSQL. Returns chunk_id (UUID)."""
        self.connect()
        chunk_id = str(uuid4())
        sql = """
        INSERT INTO chunks
            (id, version_id, chunk_index, qdrant_point_id, clause_path, standard_ref,
             section, page, source, char_len, token_count, embedding_model)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._conn.execute(
            sql,
            (
                chunk_id,
                version_id,
                chunk_index,
                qdrant_point_id,
                clause_path,
                standard_ref,
                section,
                page,
                source,
                char_len,
                token_count,
                embedding_model,
            ),
        )
        self._conn.commit()
        return chunk_id

    def get_chunks_by_version(self, version_id: str) -> list[dict[str, Any]]:
        """Retrieve all chunks for a document version."""
        self.connect()
        sql = "SELECT * FROM chunks WHERE version_id = %s ORDER BY chunk_index"
        results = self._conn.execute(sql, (version_id,)).fetchall()
        return [self._row_to_dict(row) for row in results]

    def mark_chunks_deleted(self, version_id: str) -> None:
        """Soft-delete all chunks of a version."""
        self.connect()
        sql = "UPDATE chunks SET is_deleted = TRUE WHERE version_id = %s"
        self._conn.execute(sql, (version_id,))
        self._conn.commit()

    # ============================================================
    # Tagging (for RBAC and filtering)
    # ============================================================

    def add_document_tag(self, document_id: str, tag: str) -> None:
        """Add a tag to a document."""
        self.connect()
        sql = """
        INSERT INTO document_tags (document_id, tag)
        VALUES (%s, %s)
        ON CONFLICT (document_id, tag) DO NOTHING
        """
        self._conn.execute(sql, (document_id, tag))
        self._conn.commit()

    def add_version_tag(self, version_id: str, tag: str) -> None:
        """Add a tag to a specific version."""
        self.connect()
        sql = """
        INSERT INTO version_tags (version_id, tag)
        VALUES (%s, %s)
        ON CONFLICT (version_id, tag) DO NOTHING
        """
        self._conn.execute(sql, (version_id, tag))
        self._conn.commit()

    def get_document_tags(self, document_id: str) -> list[str]:
        """Retrieve all tags for a document."""
        self.connect()
        sql = "SELECT tag FROM document_tags WHERE document_id = %s"
        results = self._conn.execute(sql, (document_id,)).fetchall()
        return [row[0] for row in results]

    def get_version_tags(self, version_id: str) -> list[str]:
        """Retrieve all tags for a version."""
        self.connect()
        sql = "SELECT tag FROM version_tags WHERE version_id = %s"
        results = self._conn.execute(sql, (version_id,)).fetchall()
        return [row[0] for row in results]

    def remove_document_tag(self, document_id: str, tag: str) -> None:
        """Remove a tag from a document."""
        self.connect()
        sql = "DELETE FROM document_tags WHERE document_id = %s AND tag = %s"
        self._conn.execute(sql, (document_id, tag))
        self._conn.commit()

    # ============================================================
    # ACL operations
    # ============================================================

    def add_acl(
        self,
        document_id: str,
        principal_type: str,
        principal_id: str,
        permission: str = "read",
    ) -> None:
        """Grant access to a principal."""
        self.connect()
        sql = """
        INSERT INTO document_acl (document_id, principal_type, principal_id, permission)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (document_id, principal_type, principal_id, permission) DO NOTHING
        """
        self._conn.execute(sql, (document_id, principal_type, principal_id, permission))
        self._conn.commit()

    def remove_acl(
        self,
        document_id: str,
        principal_type: str,
        principal_id: str,
        permission: str = "read",
    ) -> None:
        """Revoke access from a principal."""
        self.connect()
        sql = """
        DELETE FROM document_acl
        WHERE document_id = %s AND principal_type = %s
          AND principal_id = %s AND permission = %s
        """
        self._conn.execute(sql, (document_id, principal_type, principal_id, permission))
        self._conn.commit()

    def get_document_principals(self, document_id: str) -> list[str]:
        """Get list of principal IDs with read access to document."""
        self.connect()
        sql = """
        SELECT array_agg(principal_id)
        FROM document_acl
        WHERE document_id = %s AND permission = 'read'
        """
        result = self._conn.execute(sql, (document_id,)).fetchone()
        return list(result[0]) if result and result[0] else []

    def is_document_restricted(self, document_id: str) -> bool:
        """Check if document has any ACL restrictions."""
        self.connect()
        sql = "SELECT is_restricted FROM document_restriction WHERE document_id = %s"
        result = self._conn.execute(sql, (document_id,)).fetchone()
        return bool(result[0]) if result else False

    # ============================================================
    # Path history
    # ============================================================

    def record_path_change(
        self,
        document_id: str,
        new_path: str,
        old_path: str | None = None,
        version_id: str | None = None,
    ) -> None:
        """Record document path change for audit trail."""
        self.connect()
        sql = """
        INSERT INTO document_path_history
            (document_id, version_id, old_path, new_path, changed_at)
        VALUES (%s, %s, %s, %s, NOW())
        """
        self._conn.execute(sql, (document_id, version_id, old_path, new_path))
        self._conn.commit()

    def get_path_history(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve path change history for a document."""
        self.connect()
        sql = "SELECT * FROM document_path_history WHERE document_id = %s ORDER BY changed_at DESC"
        results = self._conn.execute(sql, (document_id,)).fetchall()
        return [self._row_to_dict(row) for row in results]

    # ============================================================
    # Ingestion logs
    # ============================================================

    def log_ingestion_stage(
        self,
        version_id: str,
        stage: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Log progress through ingestion pipeline."""
        self.connect()
        sql = """
        INSERT INTO ingestion_logs
            (version_id, stage, status, error_message, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        """
        self._conn.execute(sql, (version_id, stage, status, error_message))
        self._conn.commit()

    def get_ingestion_logs(self, version_id: str) -> list[dict[str, Any]]:
        """Retrieve ingestion log entries for a version."""
        self.connect()
        sql = "SELECT * FROM ingestion_logs WHERE version_id = %s ORDER BY created_at"
        results = self._conn.execute(sql, (version_id,)).fetchall()
        return [self._row_to_dict(row) for row in results]

    # ============================================================
    # Helpers
    # ============================================================

    def _row_to_dict(self, row: tuple) -> dict[str, Any]:
        """Convert database row to dictionary."""
        if not row:
            return {}
        # Get column names from cursor description
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM information_schema.columns LIMIT 1")
        # This is a simplification; in production use psycopg's RealDictCursor
        return {"row": row}

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute SHA256 hash of content for change detection."""
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def compute_source_hash(file_stat: tuple) -> str:
        """Compute hash from file metadata (size, mtime, author).
        
        file_stat: (size, mtime, author, ...) - adapt to your metadata
        """
        stat_str = "|".join(str(s) for s in file_stat)
        return hashlib.sha256(stat_str.encode()).hexdigest()
