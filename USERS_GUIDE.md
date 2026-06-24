# RAG Orchestrator - Complete User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Installation and Configuration](#installation-and-configuration)
5. [Core Components](#core-components)
6. [Usage Examples](#usage-examples)
7. [Document Management](#document-management)
8. [Versioning and ACL](#versioning-and-acl)
9. [Normative Document Parsing](#normative-document-parsing)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

RAG Orchestrator is a document management system with support for:

- **Versioning**: Track document changes while preserving full history.
- **Access control (ACL)**: Manage permissions at document level.
- **Classification**: Apply automatic and manual tags.
- **Normative document parsing**: Parse GOST/SP/SNiP-style documents while preserving hierarchy.
- **Full-text retrieval**: Vector and BM25 retrieval with Qdrant.
- **Scalability**: PostgreSQL for metadata and Qdrant for vectors.

---

## Quick Start

### 1. Install

```bash
# Install core dependencies
pip install rag-orchestrator[postgres,qdrant]

# Optional extras for normative document parsing
pip install rag-orchestrator[pdf,ocr]
```

### 2. Initialize

```python
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.migrations import run_postgres_migrations
from ragflow_orchestrator.adapters import PostgresMetadataBackend
from ragflow_orchestrator.versioned_pipeline import VersionedDocumentPipeline

# 1. Configure PostgreSQL
dsn = "postgresql://user:password@localhost/rag_db"

# 2. Run migrations
run_postgres_migrations(dsn)

# 3. Create provider
provider = create_provider(
    kind="postgres+qdrant",
    dsn=dsn,
    qdrant_url="http://localhost:6333",
    qdrant_collection="documents",
)

# 4. Create metadata backend
backend = PostgresMetadataBackend(dsn=dsn)

# 5. Create pipeline (requires an embedder)
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("intfloat/multilingual-e5-large")

pipeline = VersionedDocumentPipeline(
    provider=provider,
    metadata_backend=backend,
    dsn=dsn,
    embedder=embedder,
)
```

### 3. Ingest a Document

```python
from ragflow_orchestrator.document_pipeline import DocumentType

result = pipeline.ingest_document(
    file_path="/path/to/document.pdf",
    text=open("/path/to/document.pdf").read(),
    document_type=DocumentType.PDF,
    title="GOST 27751-2014",
    doc_number="GOST-27751-2014",
    source_type="file",
    language="ru",
    domain="regulations",
    auto_tags=["gost", "reliability"],
    manual_tags=["important"],
    acl_principals=["engineering", "management"],
)

print(f"Document ID: {result['document_id']}")
print(f"Version ID: {result['version_id']}")
print(f"Chunks: {result['chunks_count']}")
print(f"Status: {result['status']}")
```

### 4. Search Documents

```python
from ragflow_orchestrator.models import RetrievalQuery

query = RetrievalQuery(
    text_query="system reliability",
    top_k=10,
    acl_principals=["engineering"],
    dense_vector=embedder.encode("system reliability").tolist(),
)

results = provider.retrieve(query)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.chunk.text}")
    print(f"Document: {result.chunk.document_id}")
    print()
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│           VersionedDocumentPipeline                     │
│  (Orchestration, Versioning, ACL, Ingestion)           │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
      ┌────────▼─────────┐      ┌─────────▼────────────┐
      │  PostgreSQL      │      │    Qdrant            │
      │  (Metadata)      │      │    (Vectors)         │
      ├──────────────────┤      ├──────────────────────┤
      │ documents        │      │ embeddings           │
      │ versions         │      │ payload metadata     │
      │ chunks           │      │ - document_id        │
      │ tags             │      │ - version_id         │
      │ acl              │      │ - is_restricted      │
      │ history          │      │ - acl_principals     │
      │ logs             │      │ - clause_path        │
      └──────────────────┘      └──────────────────────┘
```

### Data Flow

```
1. Document Input
   ├─ File path
   ├─ Raw text
   └─ Metadata
       ↓
2. Type Detection
   ├─ File extension
   ├─ Content analysis
   └─ Routing to appropriate parser
       ↓
3. Parsing & Chunking
   ├─ NormativeChunker (for regulations)
   ├─ TextChunker (for generic text)
   └─ Extract: clause_path, standard_ref, section
       ↓
4. Embedding
   ├─ Embed chunk texts
   └─ Get vectors
       ↓
5. Storage
   ├─ PostgreSQL: metadata, tags, ACL, logs
   └─ Qdrant: vectors + payload
       ↓
6. Search
   ├─ Vector similarity search
   ├─ ACL filtering
   └─ Return results
```

---

## Installation and Configuration

### PostgreSQL

```bash
# 1. Install PostgreSQL 14+
sudo apt-get install postgresql postgresql-contrib

# 2. Create database
createdb rag_db

# 3. Configure connection via .env or config
export POSTGRES_DSN="postgresql://postgres:password@localhost:5432/rag_db"
```

### Qdrant

```bash
# 1. Pull image
docker pull qdrant/qdrant

# 2. Run container
docker run -d -p 6333:6333 \
  -e QDRANT_API_KEY=your_api_key \
  qdrant/qdrant

# 3. Check health
curl http://localhost:6333/health
```

### Embedder (optional)

```bash
# Recommended for Russian and multilingual corpora:
# intfloat/multilingual-e5-large
# ai-forever/ru-en-RoSBERTa

pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('intfloat/multilingual-e5-large')"
```

---

## Core Components

### PostgresMetadataBackend

PostgreSQL metadata lifecycle management.

```python
from ragflow_orchestrator.adapters import PostgresMetadataBackend

backend = PostgresMetadataBackend(dsn="postgresql://...")

# Documents
doc_id = backend.create_document(
    source_type="file",
    document_type="pdf",
    title="Document Title",
    doc_number="DOC-001",
    language="ru",
    domain="regulations",
)

# Versions
version_id = backend.create_version(
    document_id=doc_id,
    file_path="/path/to/file.pdf",
    content_hash="sha256...",
    ingestion_reason="Initial upload",
)

# Tags
backend.add_document_tag(doc_id, "important")
backend.add_version_tag(version_id, "v1.0")

# ACL
backend.add_acl(doc_id, principal_type="role", principal_id="engineering")

# Logs
backend.log_ingestion_stage(version_id, "parsing", "ok")
```

### PostgresQdrantProvider

Combined PostgreSQL + Qdrant provider.

```python
from ragflow_orchestrator.factory import create_provider

provider = create_provider(
    kind="postgres+qdrant",
    dsn="postgresql://...",
    qdrant_url="http://localhost:6333",
    qdrant_collection="documents",
)

provider.ensure_schema(vector_dim=768)
provider.upsert_chunks(chunks)
results = provider.retrieve(query)
provider.update_acl_by_document(doc_id, is_restricted=True, principals=[...])
```

### AclSync

ACL synchronization between PostgreSQL and Qdrant.

```python
from ragflow_orchestrator.adapters import AclSync

acl = AclSync(dsn="postgresql://...", provider=provider)

# Grant access
acl.grant(doc_id, principal_id="user1", principal_type="user")

# Replace full principal set
acl.set_principals(doc_id, principals=["role1", "role2"])

# Open access for everyone
acl.set_principals(doc_id, principals=[])

# Force resync
acl.resync(doc_id)
```

### NormativeChunker

Specialized parser for regulatory/normative documents.

```python
from ragflow_orchestrator.cleaning.normative_parser import (
    parse_normative_pdf,
    parse_normative_text,
)

# Parse from PDF
chunks = parse_normative_pdf(
    pdf_path="/path/to/gost.pdf",
    doc_id="doc_uuid",
    doc_title="GOST 27751-2014",
)

# Parse from plain text
chunks = parse_normative_text(
    text="1. Scope...",
    doc_id="doc_uuid",
)

for chunk in chunks:
    print(f"Clause: {chunk.clause_path}")
    print(f"Standard: {chunk.standard_ref}")
    print(f"Section: {chunk.section}")
    print(f"Page: {chunk.page}")
    print(f"Source: {chunk.source}")
```

---

## Usage Examples

### Example 1: Ingest and Index Documents

```python
from pathlib import Path
from ragflow_orchestrator.document_pipeline import DocumentType

docs_dir = Path("/data/regulations")

for pdf_file in docs_dir.glob("*.pdf"):
    with open(pdf_file, "rb") as f:
        text = extract_text_from_pdf(f)  # Your extractor

    result = pipeline.ingest_document(
        file_path=pdf_file,
        text=text,
        document_type=DocumentType.PDF,
        title=pdf_file.stem,
        auto_tags=["pdf", "regulations"],
    )

    if result["status"] == "success":
        print(f"Loaded: {pdf_file.name}")
        print(f"  Document ID: {result['document_id']}")
        print(f"  Chunks: {result['chunks_count']}")
    else:
        print(f"Error: {result['error']}")
```

### Example 2: Search with ACL

```python
from ragflow_orchestrator.models import RetrievalQuery

user_roles = ["engineering", "management"]

results = provider.retrieve(
    RetrievalQuery(
        text_query="system reliability",
        top_k=10,
        dense_vector=embedder.encode("system reliability").tolist(),
        acl_principals=user_roles,
    )
)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result.score:.3f}")
    print(f"   Document: {result.chunk.document_id}")
    print(f"   Section: {result.chunk.metadata.get('section', 'N/A')}")
    print(f"   Text: {result.chunk.text[:200]}...")
```

### Example 3: Version Management

```python
with open("updated_regulations.pdf", "rb") as f:
    new_text = extract_text(f)

result = pipeline.ingest_document(
    file_path="updated_regulations.pdf",
    text=new_text,
    title="GOST 27751-2014",
    ingestion_reason="Updated with latest amendments",
    auto_tags=["amendment", "v2.1"],
)

if result["status"] == "success":
    version_id = result["version_id"]
    print(f"New version: {version_id}")
```

### Example 4: Access Control Management

```python
from ragflow_orchestrator.adapters import AclSync

acl = AclSync(dsn=dsn, provider=provider)

# Restrict access
doc_id = "..."
acl.set_principals(doc_id, principals=["engineering", "management"])

# Open for everyone
acl.set_principals(doc_id, principals=[])

# Grant access to a new role
acl.grant(doc_id, principal_id="audit", principal_type="role")
```

---

## Document Management

### Document Lifecycle

```
1. CREATION
   └─ create_document()
      └─ document_id (UUID)

2. VERSIONING
   ├─ create_version()
   ├─ content_hash (change detection)
   └─ version_id (UUID)

3. INGESTION
   ├─ Parse text
   ├─ Chunk document
   ├─ Embed chunks
   ├─ Store in PostgreSQL and Qdrant
   └─ Log progress

4. CLASSIFICATION
   ├─ Auto tags (file/type based)
   └─ Manual tags (policy/user based)

5. AUTHORIZATION
   ├─ ACL configuration
   └─ ACL projection to Qdrant

6. SEARCH & RETRIEVAL
   ├─ Vector search in Qdrant
   ├─ ACL filtering
   └─ Return authorized results

7. UPDATES
   ├─ Detect changes (content_hash)
   ├─ Create new version
   └─ Keep history for audit
```

### Change Detection

```python
# RAG Orchestrator automatically:
# 1) computes content hash (SHA256)
# 2) compares with existing versions
# 3) skips duplicates
# 4) creates a new version only if content changed

content_hash = PostgresMetadataBackend.compute_content_hash(text)
source_hash = PostgresMetadataBackend.compute_source_hash((size, mtime))
```

---

## Versioning and ACL

### Versioning

```python
# Each version stores:
# - version_number: incrementing integer
# - content_hash: SHA256 of content
# - source_hash: hash from file metadata (size/mtime)
# - ingestion_status: pending | ingesting | ready | failed
# - ingestion_reason: why this version was created
# - is_active: only one active version per document

# Old versions remain in DB for:
# - audit/history
# - rollback
# - comparison/diff
```

### ACL (Access Control)

```python
# Policy: if there are no ACL entries, document is public.

# ACL in PostgreSQL:
# - document_id: UUID
# - principal_type: role | user | group
# - principal_id: principal identifier
# - permission: read (extensible)

# ACL projection in Qdrant payload:
# - metadata.is_restricted: bool
# - metadata.acl_principals: list[str]

# During retrieval:
# 1) Qdrant applies is_restricted/acl_principals filtering
# 2) PostgreSQL can be used for final verification
```

---

## Normative Document Parsing

### Features

- Automatic PDF subtype detection (text-layer vs scanned).
- OCR via PaddleOCR.
- Hierarchy preservation (section -> clause -> sub-clause).
- Standard extraction (GOST, SP, SNiP references).
- Context enrichment for better embeddings.

### Usage

```python
from ragflow_orchestrator.cleaning.normative_parser import parse_normative_pdf

chunks = parse_normative_pdf(
    pdf_path="/path/to/gost.pdf",
    doc_id="doc_uuid",
    doc_title="GOST 27751-2014",
    text_threshold=100,
    min_chunk_len=120,
    max_chunk_len=1200,
)

for chunk in chunks:
    print(f"ID: {chunk.id}")
    print(f"Clause: {chunk.clause_path}")
    print(f"Standard: {chunk.standard_ref}")
    print(f"Section: {chunk.section}")
    print(f"Page: {chunk.page}")
    print(f"Source: {chunk.source}")
    print(f"Text: {chunk.text}")
```

### Regex Patterns

```python
# Clause numbers
CLAUSE_RE = r'^\s*((?:раздел\s+|п[ункт]?\s+)?(\d+(?:\.\d+){0,3}))[\.:]?\s+'

# Standard references
STD_RE = r'((?:ГОСТ\s+(?:Р\s+)?[\d\-\.]+|СП\s+[\d\.]+|СНиП\s+[\dIVX\-\.\*]+))'

# Section headers
SECTION_RE = r'^(\d+(?:\.\d+)?)\.\s+([А-ЯЁ][а-яё\s\-,\.]+)$'
```

---

## API Reference

### VersionedDocumentPipeline

```python
pipeline.ingest_document(
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
) -> dict[str, Any]
```

Returns:

```python
{
    "document_id": str | None,
    "version_id": str | None,
    "chunks_count": int,
    "status": str,  # "success" | "skipped" | "failed"
    "error": str | None,
}
```

### PostgresMetadataBackend

```python
# Documents
backend.create_document(...)
backend.get_document(doc_id)

# Versions
backend.create_version(...)
backend.get_version(version_id)
backend.activate_version(version_id)
backend.update_ingestion_status(version_id, status, reason)

# Chunks
backend.create_chunk(...)
backend.get_chunks_by_version(version_id)
backend.mark_chunks_deleted(version_id)

# Tags
backend.add_document_tag(doc_id, tag)
backend.add_version_tag(version_id, tag)
backend.get_document_tags(doc_id)
backend.get_version_tags(version_id)
backend.remove_document_tag(doc_id, tag)

# ACL
backend.add_acl(doc_id, principal_type, principal_id)
backend.remove_acl(doc_id, principal_type, principal_id)
backend.get_document_principals(doc_id)
backend.is_document_restricted(doc_id)

# History
backend.record_path_change(doc_id, new_path, old_path, version_id)
backend.get_path_history(doc_id)

# Logs
backend.log_ingestion_stage(version_id, stage, status, error_message)
backend.get_ingestion_logs(version_id)
```

---

## Troubleshooting

### Problem: PostgreSQL connection refused

```python
import psycopg

try:
    conn = psycopg.connect("postgresql://localhost/rag_db")
    print("Connected!")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

# Ensure service is running
# sudo systemctl status postgresql
```

### Problem: Qdrant collection not found

```python
# Ensure collection is initialized
provider.ensure_schema(vector_dim=768)

# Or create collection manually
import requests
requests.post("http://localhost:6333/collections", json={
    "name": "documents",
    "vectors": {
        "size": 768,
        "distance": "Cosine"
    }
})
```

### Problem: OCR not working

```bash
pip install paddleocr
python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ru')"
```

### Problem: Migrations fail

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://localhost/rag_db")
engine.execute("SELECT 1")

from ragflow_orchestrator.migrations import run_postgres_migrations
try:
    run_postgres_migrations(dsn)
except Exception as e:
    print(f"Migration error: {e}")
    import traceback
    traceback.print_exc()
```

### Ingestion Debugging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ragflow_orchestrator")
logger.setLevel(logging.DEBUG)

result = pipeline.ingest_document(...)
if result["status"] == "failed":
    print(f"Error: {result['error']}")

logs = backend.get_ingestion_logs(version_id)
for log in logs:
    print(f"{log['stage']}: {log['status']}")
    if log['error_message']:
        print(f"  Error: {log['error_message']}")
```

---

## Support and Documentation

- **GitHub Issues**: report bugs and request features.
- **Documentation**: https://github.com/ragflow/orchestrator/docs
- **Examples**: https://github.com/ragflow/orchestrator/examples
- **API Docs**: autogenerated Pydantic model documentation.
