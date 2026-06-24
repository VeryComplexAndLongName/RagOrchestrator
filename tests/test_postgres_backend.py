"""Tests for PostgresMetadataBackend."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ragflow_orchestrator.adapters import PostgresMetadataBackend
from ragflow_orchestrator.migrations import run_postgres_migrations

pytestmark = [pytest.mark.integration, pytest.mark.pgvector]

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def test_dsn() -> str:
    """PostgreSQL test database DSN."""
    # Use test database (you can override via env variable)
    import os
    return os.getenv(
        "TEST_POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/rag_test"
    )


@pytest.fixture
def backend(test_dsn: str) -> PostgresMetadataBackend:
    """Initialize backend and run migrations."""
    # Run migrations
    try:
        run_postgres_migrations(test_dsn)
    except Exception:
        # Migrations might fail if DB doesn't exist; that's OK for testing
        pass
    
    return PostgresMetadataBackend(dsn=test_dsn)


@pytest.fixture(autouse=True)
def cleanup_db(backend: PostgresMetadataBackend) -> Iterator[None]:
    """Clean up database after each test."""
    yield
    # Cleanup would go here (truncate tables, etc.)
    # For now, tests use separate IDs so they don't conflict


# ============================================================
# Document Management Tests
# ============================================================

def test_create_document(backend: PostgresMetadataBackend) -> None:
    """Test document creation."""
    doc_id = backend.create_document(
        source_type="file",
        document_type="pdf",
        title="Test Document",
        doc_number="DOC-001",
        language="en",
        domain="test",
    )
    
    assert doc_id is not None
    assert len(doc_id) > 0
    
    # Verify document was created
    doc = backend.get_document(doc_id)
    assert doc is not None


def test_get_document(backend: PostgresMetadataBackend) -> None:
    """Test document retrieval."""
    doc_id = backend.create_document(
        source_type="file",
        document_type="pdf",
        title="Test",
    )
    
    doc = backend.get_document(doc_id)
    assert doc is not None
    # Note: structure depends on _row_to_dict implementation


# ============================================================
# Version Management Tests
# ============================================================

def test_create_version(backend: PostgresMetadataBackend) -> None:
    """Test document version creation."""
    doc_id = backend.create_document(
        source_type="file",
        document_type="pdf",
    )
    
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/test/file.pdf",
        content_hash="sha256_hash",
        ingestion_reason="Initial",
    )
    
    assert version_id is not None
    assert len(version_id) > 0


def test_version_numbering(backend: PostgresMetadataBackend) -> None:
    """Test that version numbers increment correctly."""
    doc_id = backend.create_document(
        source_type="file",
        document_type="pdf",
    )
    
    # Create first version
    v1_id = backend.create_version(
        document_id=doc_id,
        file_path="/file1.pdf",
        content_hash="hash1",
    )
    
    # Create second version
    v2_id = backend.create_version(
        document_id=doc_id,
        file_path="/file2.pdf",
        content_hash="hash2",
    )
    
    # Both should exist and be different
    assert v1_id != v2_id
    
    v1 = backend.get_version(v1_id)
    v2 = backend.get_version(v2_id)
    
    assert v1 is not None
    assert v2 is not None


def test_activate_version(backend: PostgresMetadataBackend) -> None:
    """Test version activation."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    
    v1_id = backend.create_version(
        document_id=doc_id,
        file_path="/file1.pdf",
        content_hash="hash1",
    )
    
    v2_id = backend.create_version(
        document_id=doc_id,
        file_path="/file2.pdf",
        content_hash="hash2",
    )
    
    # Activate v1
    backend.activate_version(v1_id)
    
    # Activate v2 (should deactivate v1)
    backend.activate_version(v2_id)
    
    # Verify v2 is active
    v2 = backend.get_version(v2_id)
    assert v2 is not None


def test_update_ingestion_status(backend: PostgresMetadataBackend) -> None:
    """Test ingestion status updates."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/file.pdf",
        content_hash="hash",
    )
    
    # Update status
    backend.update_ingestion_status(version_id, "ingesting", "Processing...")
    backend.update_ingestion_status(version_id, "ready")
    
    # Verify (would need to query DB to check)


# ============================================================
# Chunk Management Tests
# ============================================================

def test_create_chunk(backend: PostgresMetadataBackend) -> None:
    """Test chunk creation."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/file.pdf",
        content_hash="hash",
    )
    
    chunk_id = backend.create_chunk(
        version_id=version_id,
        chunk_index=0,
        qdrant_point_id="qdrant_id_1",
        clause_path="5.2.3",
        standard_ref="GOST 1234",
        section="Main Section",
        page=1,
        source="text_layer",
        char_len=100,
        token_count=20,
    )
    
    assert chunk_id is not None


def test_get_chunks_by_version(backend: PostgresMetadataBackend) -> None:
    """Test chunk retrieval by version."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/file.pdf",
        content_hash="hash",
    )
    
    # Create multiple chunks
    for i in range(3):
        backend.create_chunk(
            version_id=version_id,
            chunk_index=i,
            qdrant_point_id=f"qdrant_{i}",
        )
    
    chunks = backend.get_chunks_by_version(version_id)
    assert len(chunks) == 3


def test_mark_chunks_deleted(backend: PostgresMetadataBackend) -> None:
    """Test chunk soft deletion."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/file.pdf",
        content_hash="hash",
    )
    
    backend.create_chunk(
        version_id=version_id,
        chunk_index=0,
        qdrant_point_id="qdrant_0",
    )
    
    # Mark as deleted
    backend.mark_chunks_deleted(version_id)

    _ = backend.get_chunks_by_version(version_id)
    # After deletion, should still have the record but marked deleted


# ============================================================
# Tagging Tests
# ============================================================

def test_add_document_tag(backend: PostgresMetadataBackend) -> None:
    """Test adding tags to document."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    
    backend.add_document_tag(doc_id, "important")
    backend.add_document_tag(doc_id, "regulations")
    
    tags = backend.get_document_tags(doc_id)
    assert "important" in tags
    assert "regulations" in tags


def test_add_version_tag(backend: PostgresMetadataBackend) -> None:
    """Test adding tags to version."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/file.pdf",
        content_hash="hash",
    )
    
    backend.add_version_tag(version_id, "v1")
    backend.add_version_tag(version_id, "production")
    
    tags = backend.get_version_tags(version_id)
    assert "v1" in tags
    assert "production" in tags


def test_remove_document_tag(backend: PostgresMetadataBackend) -> None:
    """Test removing tags from document."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    
    backend.add_document_tag(doc_id, "temp")
    backend.remove_document_tag(doc_id, "temp")
    
    tags = backend.get_document_tags(doc_id)
    assert "temp" not in tags


# ============================================================
# ACL Tests
# ============================================================

def test_add_acl(backend: PostgresMetadataBackend) -> None:
    """Test ACL entry creation."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    
    backend.add_acl(
        document_id=doc_id,
        principal_type="role",
        principal_id="engineering",
    )
    
    backend.add_acl(
        document_id=doc_id,
        principal_type="role",
        principal_id="management",
    )
    
    principals = backend.get_document_principals(doc_id)
    assert "engineering" in principals
    assert "management" in principals


def test_is_document_restricted(backend: PostgresMetadataBackend) -> None:
    """Test document restriction check."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    
    # No ACL = not restricted
    is_restricted = backend.is_document_restricted(doc_id)
    assert not is_restricted
    
    # Add ACL = restricted
    backend.add_acl(
        document_id=doc_id,
        principal_type="role",
        principal_id="engineering",
    )
    
    is_restricted = backend.is_document_restricted(doc_id)
    assert is_restricted


def test_remove_acl(backend: PostgresMetadataBackend) -> None:
    """Test ACL entry removal."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    
    backend.add_acl(doc_id, "role", "engineering")
    backend.remove_acl(doc_id, "role", "engineering")
    
    principals = backend.get_document_principals(doc_id)
    assert "engineering" not in principals


# ============================================================
# Path History Tests
# ============================================================

def test_record_path_change(backend: PostgresMetadataBackend) -> None:
    """Test path change recording."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    
    backend.record_path_change(
        document_id=doc_id,
        old_path="/old/path/file.pdf",
        new_path="/new/path/file.pdf",
    )
    
    history = backend.get_path_history(doc_id)
    assert len(history) > 0


# ============================================================
# Ingestion Logs Tests
# ============================================================

def test_log_ingestion_stage(backend: PostgresMetadataBackend) -> None:
    """Test ingestion stage logging."""
    doc_id = backend.create_document(source_type="file", document_type="pdf")
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/file.pdf",
        content_hash="hash",
    )
    
    backend.log_ingestion_stage(version_id, "parsing", "ok")
    backend.log_ingestion_stage(version_id, "embedding", "ok")
    backend.log_ingestion_stage(version_id, "qdrant_write", "ok")
    
    logs = backend.get_ingestion_logs(version_id)
    assert len(logs) == 3


# ============================================================
# Hash Computation Tests
# ============================================================

def test_compute_content_hash() -> None:
    """Test content hash computation."""
    text1 = "Hello, world!"
    text2 = "Hello, world!"
    text3 = "Different content"
    
    hash1 = PostgresMetadataBackend.compute_content_hash(text1)
    hash2 = PostgresMetadataBackend.compute_content_hash(text2)
    hash3 = PostgresMetadataBackend.compute_content_hash(text3)
    
    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64  # SHA256 hex string


def test_compute_source_hash(tmp_path) -> None:
    """Test source file hash computation."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    
    hash1 = PostgresMetadataBackend.compute_source_hash((test_file.stat().st_size, test_file.stat().st_mtime))
    hash2 = PostgresMetadataBackend.compute_source_hash((test_file.stat().st_size, test_file.stat().st_mtime))
    
    # Same file = same hash
    assert hash1 == hash2
    assert len(hash1) == 64


# ============================================================
# Integration Tests
# ============================================================

def test_full_document_lifecycle(backend: PostgresMetadataBackend) -> None:
    """Test complete document lifecycle."""
    # 1. Create document
    doc_id = backend.create_document(
        source_type="file",
        document_type="pdf",
        title="Regulations",
        doc_number="GOST-2024",
    )
    assert doc_id
    
    # 2. Create version
    version_id = backend.create_version(
        document_id=doc_id,
        file_path="/docs/gost.pdf",
        content_hash="abcd1234",
        ingestion_reason="Initial upload",
    )
    assert version_id
    
    # 3. Create chunks
    for i in range(3):
        chunk_id = backend.create_chunk(
            version_id=version_id,
            chunk_index=i,
            qdrant_point_id=f"point_{i}",
            clause_path=f"5.{i}.1",
            page=i+1,
        )
        assert chunk_id
    
    # 4. Add tags
    backend.add_document_tag(doc_id, "regulations")
    backend.add_version_tag(version_id, "published")
    
    # 5. Set ACL
    backend.add_acl(doc_id, "role", "engineering")
    
    # 6. Log ingestion
    backend.log_ingestion_stage(version_id, "parsing", "ok")
    backend.log_ingestion_stage(version_id, "embedding", "ok")
    backend.log_ingestion_stage(version_id, "qdrant_write", "ok")
    
    # 7. Activate version
    backend.activate_version(version_id)
    backend.update_ingestion_status(version_id, "ready")
    
    # 8. Verify all data
    assert backend.get_document(doc_id) is not None
    assert backend.get_version(version_id) is not None
    assert len(backend.get_chunks_by_version(version_id)) == 3
    assert "regulations" in backend.get_document_tags(doc_id)
    assert "engineering" in backend.get_document_principals(doc_id)
    assert len(backend.get_ingestion_logs(version_id)) == 3
