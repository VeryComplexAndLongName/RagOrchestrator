"""Integrated document pipeline with versioning, ACL, and normative document support.

This module extends the basic document pipeline to:
1. Manage document versions and metadata in PostgreSQL
2. Apply automatic and manual tags for classification
3. Log ingestion progress
4. Synchronize ACL between PostgreSQL and Qdrant
5. Support specialized parsers (normative documents, code, contracts)
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ragflow_orchestrator.adapters import (
    PostgresMetadataBackend,
    PostgresQdrantProvider,
    AclSync,
)
from ragflow_orchestrator.cleaning.normative_parser import (
    parse_normative_pdf,
    parse_normative_text,
)
from ragflow_orchestrator.models import BaseChunk, IngestionStatus
from ragflow_orchestrator.config import SubtypeClassificationConfig
from ragflow_orchestrator.document_pipeline import (
    DocumentType,
    DocumentDetection,
    DocumentAwareCleaner,
)
from ragflow_orchestrator.subtype_classifier import DocumentSubtypeClassifier


class VersionedDocumentPipeline:
    """Pipeline with versioning, ACL, and intelligent document type routing."""

    def __init__(
        self,
        provider: PostgresQdrantProvider,
        metadata_backend: PostgresMetadataBackend,
        dsn: str,
        embedder: Any,
        cleaner: DocumentAwareCleaner | None = None,
        subtype_config: SubtypeClassificationConfig | None = None,
        subtype_classifier: DocumentSubtypeClassifier | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            provider: PostgresQdrantProvider (combined metadata + vector storage)
            metadata_backend: PostgresMetadataBackend
            dsn: PostgreSQL connection string (for AclSync)
            embedder: Embedder instance
            cleaner: Optional custom cleaner
        """
        self.provider = provider
        self.backend = metadata_backend
        self.embedder = embedder
        self.cleaner = cleaner or DocumentAwareCleaner()
        self.acl = AclSync(dsn=dsn, provider=provider)
        self.subtype_classifier = subtype_classifier or DocumentSubtypeClassifier(
            config=subtype_config or SubtypeClassificationConfig()
        )

    def ingest_document(
        self,
        file_path: str | Path,
        text: str,
        document_type: DocumentType | None = None,
        title: str | None = None,
        doc_number: str | None = None,
        source_type: str = "file",
        language: str | None = None,
        domain: str | None = None,
        auto_tags: list[str] | None = None,
        manual_tags: list[str] | None = None,
        acl_principals: list[str] | None = None,
        ingestion_reason: str = "Initial ingestion",
    ) -> dict[str, Any]:
        """Ingest a document with full versioning and metadata support.

        Args:
            file_path: Path to source file
            text: Raw document text
            document_type: Auto-detected if None
            title: Document title
            doc_number: Document number (e.g., "ПП РФ N 815")
            source_type: "file", "confluence", "api", etc.
            language: Document language (e.g., "ru", "en")
            domain: Domain/category (e.g., "regulations", "code")
            auto_tags: Automatic tags (e.g., by file type)
            manual_tags: Manual tags (e.g., by user or policy)
            acl_principals: List of principal IDs with read access
            ingestion_reason: Reason for this version

        Returns:
            dict with ingestion result: {
                "document_id": str,
                "version_id": str,
                "chunks_count": int,
                "status": str,
                "error": str | None,
            }
        """
        file_path = Path(file_path)
        result = {
            "document_id": None,
            "version_id": None,
            "chunks_count": 0,
            "document_subtype": None,
            "subtype_confidence": 0.0,
            "status": "failed",
            "error": None,
        }

        try:
            # 1. Detect document type
            if document_type is None:
                detected = self._detect_document_type(file_path, text)
                document_type = detected.document_type

            # 2. Clean text
            cleaned_text = self.cleaner.clean(text)

            # 2.1 Detect document subtype with hybrid classifier
            subtype_prediction = self.subtype_classifier.predict(
                text=cleaned_text,
                title=title,
                document_type=document_type.value,
            )
            result["document_subtype"] = subtype_prediction.subtype
            result["subtype_confidence"] = subtype_prediction.confidence

            # 3. Create document (or use existing)
            doc_id = self._get_or_create_document(
                source_type=source_type,
                document_type=document_type,
                document_subtype=subtype_prediction.subtype,
                title=title,
                doc_number=doc_number,
                language=language,
                domain=domain,
            )
            result["document_id"] = doc_id

            # 4. Compute hashes for change detection
            content_hash = self._compute_content_hash(cleaned_text)
            source_hash = self._compute_source_hash(file_path) if file_path.exists() else None

            # 5. Check if version already exists
            existing_version = self._find_existing_version(doc_id, content_hash)
            if existing_version:
                result["version_id"] = existing_version
                result["status"] = "skipped"
                result["chunks_count"] = self._count_chunks_for_version(existing_version)
                return result

            # 6. Create new version
            version_id = self.backend.create_version(
                document_id=doc_id,
                file_path=str(file_path),
                content_hash=content_hash,
                document_subtype=subtype_prediction.subtype,
                source_hash=source_hash,
                ingestion_reason=ingestion_reason,
                file_ext=file_path.suffix or None,
                embedding_model=getattr(self.embedder, "model_name", "unknown"),
            )
            result["version_id"] = version_id
            self.backend.update_ingestion_status(version_id, "ingesting")
            self.backend.log_ingestion_stage(version_id, "parsing", "ok")

            # 7. Route to specialized parser or generic chunker
            chunks = self._chunk_document(
                text=cleaned_text,
                file_path=file_path,
                document_type=document_type,
                doc_id=doc_id,
                version_id=version_id,
                title=title,
            )

            if not chunks:
                raise RuntimeError(f"Failed to chunk document: {file_path}")

            # 8. Embed chunks
            chunk_texts = [c.text for c in chunks]
            vectors = self.embedder.embed_many(chunk_texts)
            self.backend.log_ingestion_stage(version_id, "embedding", "ok")

            # 9. Enrich chunks with metadata
            for chunk, vector in zip(chunks, vectors):
                chunk.document_id = doc_id
                chunk.version_id = version_id
                chunk.document_subtype = subtype_prediction.subtype
                chunk.vector = vector
                if isinstance(chunk.metadata, dict):
                    chunk.metadata.setdefault("document_subtype", subtype_prediction.subtype)

            # 10. Upsert to provider (PostgreSQL + Qdrant)
            self.provider.upsert_chunks(chunks)
            self.backend.log_ingestion_stage(version_id, "qdrant_write", "ok")

            # 11. Create chunk records in PostgreSQL
            for chunk in chunks:
                self.backend.create_chunk(
                    version_id=version_id,
                    chunk_index=chunk.chunk_index,
                    qdrant_point_id=chunk.id,
                    clause_path=chunk.metadata.get("clause_path") if isinstance(chunk.metadata, dict) else None,
                    standard_ref=chunk.metadata.get("standard_ref") if isinstance(chunk.metadata, dict) else None,
                    section=chunk.metadata.get("section") if isinstance(chunk.metadata, dict) else None,
                    page=chunk.metadata.get("page", 1) if isinstance(chunk.metadata, dict) else 1,
                    source=chunk.metadata.get("source", "text_layer") if isinstance(chunk.metadata, dict) else "text_layer",
                    token_count=chunk.token_count,
                    char_len=len(chunk.text),
                    embedding_model=getattr(self.embedder, "model_name", "unknown"),
                )

            # 12. Apply automatic tags
            all_tags = auto_tags or []
            all_tags.extend(self._generate_auto_tags(file_path, document_type))
            all_tags.append(f"subtype:{subtype_prediction.subtype}")
            for tag in all_tags:
                self.backend.add_document_tag(doc_id, tag)
                self.backend.add_version_tag(version_id, tag)

            # 13. Apply manual tags
            for tag in manual_tags or []:
                self.backend.add_document_tag(doc_id, tag)
                self.backend.add_version_tag(version_id, tag)

            # 14. Activate version
            self.backend.activate_version(version_id)
            self.backend.update_ingestion_status(version_id, "ready")

            # 15. Set ACL (if specified)
            if acl_principals:
                self.acl.set_principals(doc_id, principals=acl_principals)

            result["status"] = "success"
            result["chunks_count"] = len(chunks)

        except Exception as e:
            result["error"] = str(e)
            if result.get("version_id"):
                self.backend.update_ingestion_status(
                    result["version_id"],
                    "failed",
                    reason=str(e),
                )
                self.backend.log_ingestion_stage(
                    result["version_id"],
                    "ingestion",
                    "failed",
                    error_message=str(e),
                )

        return result

    # ============================================================
    # Private helpers
    # ============================================================

    def _detect_document_type(self, file_path: Path, text: str) -> DocumentDetection:
        """Detect document type from file extension and content."""
        # Try extension first
        ext = file_path.suffix.lower()
        if ext in {".pdf", ".docx", ".xlsx"}:
            doc_type = {
                ".pdf": DocumentType.PDF,
                ".docx": DocumentType.DOCX,
                ".xlsx": DocumentType.XLSX,
            }[ext]
            return DocumentDetection(doc_type, source="extension")

        # Heuristics for text-based formats
        if text.startswith("<?xml"):
            return DocumentDetection(DocumentType.XML, source="content")
        if text.startswith("<!DOCTYPE") or text.startswith("<html"):
            return DocumentDetection(DocumentType.HTML, source="content")
        if text.startswith("#") or "\n# " in text:
            return DocumentDetection(DocumentType.MARKDOWN, source="content")
        if text.startswith("[") or text.startswith("{"):
            return DocumentDetection(DocumentType.JSON, source="content")

        # Default
        return DocumentDetection(DocumentType.TXT, source="default")

    def _get_or_create_document(
        self,
        source_type: str,
        document_type: DocumentType,
        document_subtype: str | None,
        title: str | None,
        doc_number: str | None,
        language: str | None,
        domain: str | None,
    ) -> str:
        """Get existing document by number, or create new."""
        # TODO: Implement lookup by doc_number if needed
        # For now, always create new
        return self.backend.create_document(
            source_type=source_type,
            document_type=document_type.value,
            document_subtype=document_subtype,
            title=title,
            doc_number=doc_number,
            language=language,
            domain=domain,
        )

    def _find_existing_version(self, document_id: str, content_hash: str) -> str | None:
        """Check if version with same content_hash already exists."""
        # TODO: Query PostgreSQL for existing version with this content_hash
        # For now, return None (always create new)
        return None

    def _count_chunks_for_version(self, version_id: str) -> int:
        """Count chunks for a version."""
        chunks = self.backend.get_chunks_by_version(version_id)
        return len(chunks)

    def _chunk_document(
        self,
        text: str,
        file_path: Path,
        document_type: DocumentType,
        doc_id: str,
        version_id: str,
        title: str | None,
    ) -> list[BaseChunk]:
        """Route to appropriate chunker based on document type."""
        if document_type == DocumentType.PDF and file_path.suffix.lower() == ".pdf":
            # For actual PDF files, use normative parser
            # (for text extracted from PDF, route to text chunker)
            try:
                norm_chunks = parse_normative_pdf(
                    pdf_path=str(file_path),
                    doc_id=doc_id,
                    doc_title=title,
                )
                # Convert NormativeChunk to BaseChunk
                chunks = []
                for i, nc in enumerate(norm_chunks):
                    chunk = BaseChunk(
                        id=nc.id,
                        text=nc.text,
                        metadata={
                            "clause_path": nc.clause_path,
                            "standard_ref": nc.standard_ref,
                            "section": nc.section,
                            "page": nc.page,
                            "source": nc.source,
                            "semantic_type": nc.semantic_type,
                        },
                        source_id=version_id,
                        chunk_index=i,
                    )
                    chunks.append(chunk)
                return chunks
            except Exception:
                # Fallback to text chunking
                pass

        # Text-based chunking (default)
        # TODO: Import and use appropriate chunker based on document_type
        # For now, simple fixed-size chunking
        from ragflow_orchestrator.chunking.fixed import FixedWindowChunker
        chunker = FixedWindowChunker(chunk_size=900, chunk_overlap=120)
        return chunker.chunk(
            source_id=version_id,
            text=text,
            metadata={"document_type": document_type.value},
        )

    def _generate_auto_tags(self, file_path: Path, document_type: DocumentType) -> list[str]:
        """Generate automatic tags based on file type and content."""
        tags = []
        
        # File extension tags
        ext = file_path.suffix.lower().lstrip(".")
        if ext:
            tags.append(ext)
        
        # Document type tags
        tags.append(document_type.value)
        
        # For normative documents, infer from title if present
        # (This could be enhanced with ML classification)
        
        return tags

    @staticmethod
    def _compute_content_hash(text: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _compute_source_hash(file_path: Path) -> str | None:
        """Compute hash from file metadata (size, mtime)."""
        if not file_path.exists():
            return None
        try:
            stat = file_path.stat()
            stat_str = f"{stat.st_size}|{stat.st_mtime}"
            return hashlib.sha256(stat_str.encode()).hexdigest()
        except Exception:
            return None
