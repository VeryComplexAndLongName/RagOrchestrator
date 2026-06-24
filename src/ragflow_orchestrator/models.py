from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkKind(str, Enum):
    GENERIC = "generic"
    CODE = "code"
    CONTRACT = "contract"
    TABLE = "table"
    PDF = "pdf"
    HTML = "html"
    WORD = "word"
    MIXED = "mixed"


class IngestionStatus(str, Enum):
    """Status of document version ingestion process."""
    PENDING = "pending"
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"


class DocumentSubtype(str, Enum):
    NORMATIVE = "normative"
    DESCRIPTION = "description"
    SPECIFICATION = "specification"
    INSTRUCTION = "instruction"
    POLICY = "policy"
    CONTRACT_LEGAL = "contract_legal"
    REPORT = "report"
    FAQ = "faq"
    REFERENCE = "reference"
    CODE_DOC = "code_doc"
    AGREEMENT = "agreement"
    UNKNOWN = "unknown"


class BaseChunk(BaseModel):
    """Base chunk model with extended metadata for versioning and ACL."""
    model_config = ConfigDict(extra="allow")

    id: str
    text: str
    vector: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_id: str  # deprecated: will be version_id; kept for backward compatibility
    chunk_index: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: ChunkKind = ChunkKind.GENERIC
    version: int = 1
    is_deleted: bool = False
    semantic_type: str = "generic"
    quality_score: float = 0.5
    token_count: int = 0
    source_type: str = "unknown"
    domain: str = ""
    risk_score: float = 0.0
    embedding_model: str = ""
    
    # New fields for versioning and ACL
    document_id: str | None = None  # UUID of parent document
    version_id: str | None = None   # UUID of document version
    
    # Normative document structure (from additional.md)
    clause_path: str | None = None  # "5.2.3" or "п.3.1.1"
    document_subtype: str | None = None


class CodeChunk(BaseChunk):
    kind: ChunkKind = ChunkKind.CODE
    language: str
    file_path: str
    function_name: str | None = None


class ContractChunk(BaseChunk):
    kind: ChunkKind = ChunkKind.CONTRACT
    clause_type: str
    parties: list[str] = Field(default_factory=list)


# ============================================================
# New models for PostgreSQL metadata management
# ============================================================

class DocumentMetadata(BaseModel):
    """Metadata for a document (base entity)."""
    id: str = Field(description="Document UUID")
    source_type: str  # file, confluence, bitrix, ...
    document_type: str  # txt, pdf, docx, ...
    title: str | None = None
    doc_number: str | None = None  # "ПП РФ N 815"
    language: str | None = None
    domain: str | None = None
    document_subtype: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentVersionMetadata(BaseModel):
    """Metadata for a specific version of a document."""
    id: str = Field(description="Version UUID")
    document_id: str
    version_number: int
    source_hash: str | None = None  # file hash for change detection
    content_hash: str  # semantic hash of content
    file_path: str
    file_ext: str | None = None
    semantic_type: str | None = None
    token_count: int | None = None
    quality_score: float | None = None
    risk_score: float | None = None
    embedding_model: str | None = None
    document_subtype: str | None = None
    valid_from: str | None = None  # ISO date
    valid_to: str | None = None
    edition_label: str | None = None  # "ред. до 20.05.2022"
    ingestion_status: IngestionStatus
    ingestion_reason: str | None = None  # why version was created
    ingestion_timestamp: datetime
    created_at: datetime
    updated_at: datetime
    is_active: bool


class DocumentTag(BaseModel):
    """Tag applied to a document (for RBAC and filtering)."""
    id: int
    document_id: str
    tag: str


class VersionTag(BaseModel):
    """Tag applied to a specific version."""
    id: int
    version_id: str
    tag: str


class DocumentAclPrincipal(BaseModel):
    """ACL entry granting access to a principal."""
    id: int
    document_id: str
    principal_type: str  # role, user, group
    principal_id: str
    permission: str = "read"
    created_at: datetime


class DocumentPathHistory(BaseModel):
    """History of path changes for a document version."""
    id: int
    document_id: str
    version_id: str | None
    old_path: str | None
    new_path: str
    changed_at: datetime


class IngestionLog(BaseModel):
    """Log entry for document version ingestion process."""
    id: int
    version_id: str
    stage: str  # parsing, ocr, chunking, embedding, qdrant_write
    status: str  # ok, failed
    error_message: str | None = None
    created_at: datetime


# ============================================================
# Query and retrieval models
# ============================================================

class RetrievalFilter(BaseModel):
    key: str
    op: str = "eq"
    value: Any


class RetrievalQuery(BaseModel):
    text_query: str
    top_k: int = 3
    filters: list[RetrievalFilter] = Field(default_factory=list)
    include_deleted: bool = False
    dense_vector: list[float] = Field(default_factory=list)
    sparse_vector: dict[str, float] = Field(default_factory=dict)
    acl_principals: list[str] | None = None  # user roles/groups for ACL filtering


class RetrievalResult(BaseModel):
    chunk: BaseChunk
    score: float
    provider: str
