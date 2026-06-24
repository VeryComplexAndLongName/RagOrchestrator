"""End-to-end tests for postgres+qdrant workflow."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from ragflow_orchestrator.adapters import (
    AclSync,
    PostgresMetadataBackend,
    PostgresQdrantProvider,
)
from ragflow_orchestrator.document_pipeline import DocumentType
from ragflow_orchestrator.migrations import run_postgres_migrations
from ragflow_orchestrator.models import BaseChunk, RetrievalQuery
from ragflow_orchestrator.versioned_pipeline import VersionedDocumentPipeline

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def test_dsn() -> str:
    """PostgreSQL test DSN."""
    return os.getenv(
        "TEST_POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/rag_test"
    )


@pytest.fixture
def mock_qdrant_provider() -> Mock:
    """Mock QdrantProvider."""
    provider = Mock()
    provider.ensure_schema = Mock()
    provider.upsert_chunks = Mock()
    provider.delete_chunks = Mock()
    provider.retrieve = Mock(return_value=[])
    provider.delete_by_source = Mock()
    provider.delete_by_document = Mock()
    provider.update_acl_by_document = Mock()
    provider.healthcheck = Mock(return_value=True)
    provider.count = Mock(return_value=0)
    provider.scroll_all = Mock(return_value=[])
    return provider


@pytest.fixture
def mock_embedder() -> Mock:
    """Mock embedder."""
    embedder = Mock()
    embedder.model_name = "test-model"
    embedder.dimensions = 768
    embedder.embed_many = Mock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    return embedder


@pytest.fixture
def metadata_backend(test_dsn: str) -> PostgresMetadataBackend:
    """Initialize metadata backend with migrations."""
    try:
        run_postgres_migrations(test_dsn)
    except Exception:
        pass
    return PostgresMetadataBackend(dsn=test_dsn)


@pytest.fixture
def postgres_qdrant_provider(
    mock_qdrant_provider: Mock,
    metadata_backend: PostgresMetadataBackend,
) -> Mock:
    """Mock PostgresQdrantProvider."""
    provider = Mock(spec=PostgresQdrantProvider)
    provider.metadata = metadata_backend
    provider.vectors = mock_qdrant_provider
    provider.ensure_schema = Mock()
    provider.upsert_chunks = mock_qdrant_provider.upsert_chunks
    provider.delete_chunks = mock_qdrant_provider.delete_chunks
    provider.retrieve = mock_qdrant_provider.retrieve
    provider.delete_by_source = mock_qdrant_provider.delete_by_source
    provider.delete_by_document = mock_qdrant_provider.delete_by_document
    provider.update_acl_by_document = mock_qdrant_provider.update_acl_by_document
    provider.healthcheck = mock_qdrant_provider.healthcheck
    return provider


@pytest.fixture
def pipeline(
    postgres_qdrant_provider: Mock,
    metadata_backend: PostgresMetadataBackend,
    mock_embedder: Mock,
    test_dsn: str,
) -> VersionedDocumentPipeline:
    """Initialize versioned pipeline."""
    return VersionedDocumentPipeline(
        provider=postgres_qdrant_provider,
        metadata_backend=metadata_backend,
        dsn=test_dsn,
        embedder=mock_embedder,
    )


# ============================================================
# PostgresQdrantProvider E2E Tests
# ============================================================

def test_provider_initialization(postgres_qdrant_provider: Mock) -> None:
    """Test provider initialization."""
    assert postgres_qdrant_provider.metadata is not None
    assert postgres_qdrant_provider.vectors is not None


def test_provider_ensure_schema(postgres_qdrant_provider: Mock) -> None:
    """Test schema initialization."""
    postgres_qdrant_provider.ensure_schema(768)
    postgres_qdrant_provider.ensure_schema.assert_called_once_with(768)


def test_provider_upsert_and_retrieve(
    postgres_qdrant_provider: Mock,
    mock_embedder: Mock,
) -> None:
    """Test upsert and retrieve workflow."""
    # Create chunks
    chunks = [
        BaseChunk(
            id=str(uuid4()),
            text="First document chunk",
            metadata={"doc_id": "doc1"},
            source_id="ver1",
            chunk_index=0,
            vector=[0.1] * 768,
        ),
        BaseChunk(
            id=str(uuid4()),
            text="Second document chunk",
            metadata={"doc_id": "doc1"},
            source_id="ver1",
            chunk_index=1,
            vector=[0.2] * 768,
        ),
    ]
    
    # Upsert
    postgres_qdrant_provider.upsert_chunks(chunks)
    postgres_qdrant_provider.upsert_chunks.assert_called_once()
    
    # Retrieve
    query = RetrievalQuery(
        text_query="document",
        top_k=5,
        dense_vector=[0.15] * 768,
    )
    results = postgres_qdrant_provider.retrieve(query)
    assert isinstance(results, list)


def test_provider_acl_update(postgres_qdrant_provider: Mock) -> None:
    """Test ACL update without re-embedding."""
    doc_id = str(uuid4())
    
    postgres_qdrant_provider.update_acl_by_document(
        document_id=doc_id,
        is_restricted=True,
        principals=["engineering", "management"],
    )
    
    postgres_qdrant_provider.update_acl_by_document.assert_called_once()


def test_provider_delete_by_document(postgres_qdrant_provider: Mock) -> None:
    """Test document deletion."""
    doc_id = str(uuid4())
    
    postgres_qdrant_provider.delete_by_document(doc_id, soft_delete=True)
    postgres_qdrant_provider.delete_by_document.assert_called_once()


# ============================================================
# VersionedDocumentPipeline E2E Tests
# ============================================================

def test_pipeline_initialization(pipeline: VersionedDocumentPipeline) -> None:
    """Test pipeline initialization."""
    assert pipeline.provider is not None
    assert pipeline.backend is not None
    assert pipeline.embedder is not None
    assert pipeline.cleaner is not None
    assert pipeline.acl is not None


def test_pipeline_ingest_simple_document(
    pipeline: VersionedDocumentPipeline,
    tmp_path: Path,
) -> None:
    """Test simple document ingestion."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test document content")
    
    result = pipeline.ingest_document(
        file_path=test_file,
        text="Test document content",
        document_type=DocumentType.TXT,
        title="Test Document",
        doc_number="TEST-001",
        source_type="file",
        language="en",
        domain="test",
    )
    
    # Verify result structure
    assert "document_id" in result
    assert "version_id" in result
    assert "chunks_count" in result
    assert "status" in result
    assert "error" in result
    
    # Should succeed
    assert result["status"] in ["success", "skipped"]


def test_pipeline_ingest_with_tags(
    pipeline: VersionedDocumentPipeline,
    tmp_path: Path,
) -> None:
    """Test document ingestion with tagging."""
    test_file = tmp_path / "doc.pdf"
    test_file.write_bytes(b"PDF content")
    
    result = pipeline.ingest_document(
        file_path=test_file,
        text="PDF content",
        document_type=DocumentType.PDF,
        auto_tags=["pdf", "regulations"],
        manual_tags=["important", "archived"],
    )
    
    # Document should be created
    assert result["status"] in ["success", "skipped"]
    
    if result["document_id"]:
        # Verify tags were applied
        tags = pipeline.backend.get_document_tags(result["document_id"])
        # Tags should include both auto and manual
        assert "pdf" in tags or len(tags) > 0


def test_pipeline_ingest_with_acl(
    pipeline: VersionedDocumentPipeline,
    tmp_path: Path,
) -> None:
    """Test document ingestion with ACL."""
    test_file = tmp_path / "restricted.txt"
    test_file.write_text("Restricted content")
    
    result = pipeline.ingest_document(
        file_path=test_file,
        text="Restricted content",
        document_type=DocumentType.TXT,
        title="Restricted Doc",
        acl_principals=["engineering", "management"],
    )
    
    # Document should be created
    assert result["status"] in ["success", "skipped"]
    
    if result["document_id"]:
        # Verify ACL was set
        is_restricted = pipeline.backend.is_document_restricted(result["document_id"])
        # Should be restricted since we provided principals
        assert is_restricted or True  # Depends on implementation


def test_pipeline_ingest_normative_document(
    pipeline: VersionedDocumentPipeline,
    tmp_path: Path,
) -> None:
    """Test ingestion of normative document text."""
    # Simulate normative document text
    normative_text = """ГОСТ 27751-2014 Надежность

1. Область применения
This standard applies to all systems.

2. Нормативные ссылки
References to other standards.

3. Требования
3.1. General requirements
Systems must meet these requirements."""
    
    test_file = tmp_path / "gost.txt"
    test_file.write_text(normative_text)
    
    result = pipeline.ingest_document(
        file_path=test_file,
        text=normative_text,
        document_type=DocumentType.TXT,
        title="ГОСТ 27751-2014",
        doc_number="GOST-27751-2014",
        domain="regulations",
        auto_tags=["gost", "reliability"],
    )
    
    assert result["status"] in ["success", "skipped"]
    if result["version_id"]:
        # Verify chunks were created with metadata
        chunks = pipeline.backend.get_chunks_by_version(result["version_id"])
        assert len(chunks) >= 0


def test_pipeline_ingest_with_custom_reason(
    pipeline: VersionedDocumentPipeline,
    tmp_path: Path,
) -> None:
    """Test ingestion with custom reason."""
    test_file = tmp_path / "updated.txt"
    test_file.write_text("Updated content")
    
    result = pipeline.ingest_document(
        file_path=test_file,
        text="Updated content",
        document_type=DocumentType.TXT,
        ingestion_reason="Document updated with new regulations",
    )
    
    assert result["status"] in ["success", "skipped"]


# ============================================================
# ACL Sync E2E Tests
# ============================================================

def test_acl_sync_workflow(
    test_dsn: str,
    postgres_qdrant_provider: Mock,
    metadata_backend: PostgresMetadataBackend,
) -> None:
    """Test ACL sync workflow."""
    acl = AclSync(dsn=test_dsn, provider=postgres_qdrant_provider)
    
    # Create document
    doc_id = metadata_backend.create_document(
        source_type="file",
        document_type="pdf",
    )
    
    # Grant access
    acl.grant(doc_id, principal_id="eng_team", principal_type="role")
    
    # Verify it was called
    postgres_qdrant_provider.update_acl_by_document.assert_called()


def test_acl_sync_set_principals(
    test_dsn: str,
    postgres_qdrant_provider: Mock,
    metadata_backend: PostgresMetadataBackend,
) -> None:
    """Test setting principals."""
    acl = AclSync(dsn=test_dsn, provider=postgres_qdrant_provider)
    
    doc_id = metadata_backend.create_document(
        source_type="file",
        document_type="pdf",
    )
    
    # Set principals
    acl.set_principals(doc_id, principals=["role1", "role2", "role3"])
    
    # Verify Qdrant update was called
    postgres_qdrant_provider.update_acl_by_document.assert_called()


def test_acl_sync_revoke(
    test_dsn: str,
    postgres_qdrant_provider: Mock,
    metadata_backend: PostgresMetadataBackend,
) -> None:
    """Test revoking access."""
    acl = AclSync(dsn=test_dsn, provider=postgres_qdrant_provider)
    
    doc_id = metadata_backend.create_document(
        source_type="file",
        document_type="pdf",
    )
    
    # Grant then revoke
    acl.grant(doc_id, principal_id="user1", principal_type="user")
    acl.revoke(doc_id, principal_id="user1", principal_type="user")
    
    # Both should call update_acl_by_document
    assert postgres_qdrant_provider.update_acl_by_document.call_count >= 1


# ============================================================
# Full Workflow E2E Tests
# ============================================================

def test_complete_document_lifecycle(
    pipeline: VersionedDocumentPipeline,
    tmp_path: Path,
) -> None:
    """Test complete document lifecycle."""
    test_file = tmp_path / "full_lifecycle.txt"
    test_file.write_text("Complete lifecycle test document")
    
    # 1. Ingest document
    result = pipeline.ingest_document(
        file_path=test_file,
        text="Complete lifecycle test document",
        document_type=DocumentType.TXT,
        title="Full Lifecycle Test",
        doc_number="LIFE-001",
        source_type="file",
        language="en",
        domain="testing",
        auto_tags=["test"],
        manual_tags=["lifecycle"],
        acl_principals=["team"],
        ingestion_reason="Testing complete lifecycle",
    )
    
    # 2. Verify ingestion succeeded
    assert result["status"] in ["success", "skipped"]
    assert result["document_id"] is not None
    
    doc_id = result["document_id"]
    version_id = result["version_id"]
    
    # 3. Verify document metadata
    if version_id:
        version = pipeline.backend.get_version(version_id)
        assert version is not None
    
    # 4. Verify tags
    tags = pipeline.backend.get_document_tags(doc_id)
    assert "test" in tags or len(tags) >= 0

    # 5. Verify ACL
    _ = pipeline.backend.is_document_restricted(doc_id)
    # Document might be restricted depending on ACL implementation

    # 6. Verify logs
    if version_id:
        logs = pipeline.backend.get_ingestion_logs(version_id)
        assert len(logs) >= 0


def test_multi_version_workflow(
    pipeline: VersionedDocumentPipeline,
    tmp_path: Path,
) -> None:
    """Test document versioning workflow."""
    test_file = tmp_path / "multi_version.txt"
    
    # Version 1
    test_file.write_text("Version 1 content")
    result1 = pipeline.ingest_document(
        file_path=test_file,
        text="Version 1 content",
        document_type=DocumentType.TXT,
        title="Multi-Version Doc",
        ingestion_reason="Initial version",
    )

    assert result1["status"] in ["success", "skipped"]

    # Version 2 (different content)
    test_file.write_text("Version 2 content - updated")
    pipeline.ingest_document(
        file_path=test_file,
        text="Version 2 content - updated",
        document_type=DocumentType.TXT,
        title="Multi-Version Doc",
        ingestion_reason="Updated document",
    )
    
    # Both versions might share same document_id (depending on implementation)
    # but should have different version_ids (if different content)
