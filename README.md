# ragflow_orchestrator

Universal and extensible RAG module with standardized interfaces and adapters.

![Ragflow Orchestrator](RagflowOrchestrator.png)

## Authors

- Alexander Ivanov
- email: VeryComplexAndLongName@gamil.com
- Telegram: @alexander_ivan0v

## Key Updates

- Storage architecture is PostgreSQL + Qdrant only.
- New combined provider: `postgres+qdrant` (PostgreSQL metadata + Qdrant vectors).
- Adaptive document pipeline supports semantic subtype classification with hybrid scoring:
  - rules-based scoring
  - optional LLM scoring (`ollama` or `openai_compat`)
  - weighted merge and confidence threshold fallback
- Document subtype is persisted in PostgreSQL metadata (`documents`, `document_versions`) and propagated into chunk metadata/tags.
- Added infrastructure compose files for local development:
  - `docker-compose.postgres.yml`
  - `docker-compose.qdrant.yml`
  - `docker-compose.infra.yml`

## Goals

- One internal chunk contract across vector stores.
- Standardized ingestion pipeline: cleaning -> chunking -> embedding -> upsert.
- Adaptive document pipeline: detection -> normalization -> strategy-aware chunking.
- Standard retrieval APIs with semantic/hybrid strategies.
- First-class interoperability with PromptOrchestrator pipelines.
- Extensible migration framework and quality evaluation utilities.

## Providers

Supported provider kinds in `create_provider`:

- `postgres+qdrant` (recommended)
- `postgresql+qdrant`
- `postgres_qdrant`
- `pgvector` / `postgres` / `postgresql` (legacy)
- `qdrant` (legacy)

## Quick Start (PostgreSQL + Qdrant)

1. Start infrastructure:

```bash
docker compose -f docker-compose.infra.yml up -d
```

2. Configure environment:

```bash
export RAG_POSTGRES_DSN="postgresql://rag_user:rag_password@localhost:5432/rag_db"
export RAG_QDRANT_URL="http://localhost:6333"
export RAG_QDRANT_COLLECTION="rag_chunks"
```

PowerShell:

```powershell
$env:RAG_POSTGRES_DSN = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
$env:RAG_QDRANT_URL = "http://localhost:6333"
$env:RAG_QDRANT_COLLECTION = "rag_chunks"
```

3. Ingest and search:

```python
from ragflow_orchestrator.factory import create_provider
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.embedding import HashEmbedder
from ragflow_orchestrator.presets import document_preset

provider = create_provider(
    "postgres+qdrant",
    dsn="postgresql://rag_user:rag_password@localhost:5432/rag_db",
    qdrant_url="http://localhost:6333",
    qdrant_collection="rag_chunks",
)
preset = document_preset()

orchestrator = RAGOrchestrator(
    provider=provider,
    embedder=HashEmbedder(dimensions=256),
    chunker=preset.chunker,
    cleaner=preset.cleaner,
)

orchestrator.ingest(
    source_id="doc-1",
    raw_text="RAG orchestration standardizes ingestion and retrieval.",
    metadata={"tenant_id": "t1", "language": "en", "doctype": "note"},
)

hits = orchestrator.search("How does orchestration help?", top_k=3)
for hit in hits:
    print(hit.score, hit.chunk.id, hit.chunk.text)
```

## ConfigStore Example

```python
from ragflow_orchestrator import (
    ConfigStore,
    EmbeddingConfig,
    ModuleConfig,
    PipelineConfig,
    ProviderConfig,
    RAGOrchestratorFactory,
)

store = ConfigStore(
    ModuleConfig(
        provider=ProviderConfig(
            kind="postgres+qdrant",
            params={
                "dsn": "postgresql://rag_user:rag_password@localhost:5432/rag_db",
                "qdrant_url": "http://localhost:6333",
                "qdrant_collection": "rag_chunks",
            },
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text:latest",
            options={"base_url": "http://localhost:11434", "timeout_seconds": 60},
        ),
        pipeline=PipelineConfig(preset="document"),
    )
)

orchestrator = RAGOrchestratorFactory.from_config_store(store)
```

## Document Subtype Classification

Subtype classification is integrated into ingestion and versioned metadata pipeline.

Main behavior:

- If subtype is missing, classifier predicts it from content and metadata.
- Final subtype and confidence are attached to:
  - document metadata
  - version metadata
  - chunk metadata/tags
- Fallback subtype is applied when confidence is below threshold.

Config fields in `ModuleConfig.subtype_classification`:

- `enabled`
- `fallback_subtype`
- `confidence_threshold`
- `rules_weight`
- `llm_weight`
- `allowed_subtypes`
- `llm` settings:
  - `provider`: `none | ollama | openai_compat`
  - `model`
  - `base_url`
  - `api_key_env`
  - `timeout_seconds`
  - `temperature`

## Document Pipeline

Default document preset routes content by detected type.

Supported types include:

- `pdf`
- `docx`
- `xlsx`
- `html`
- `markdown`
- `json`
- `xml`
- `csv`
- `txt`
- `code`
- `unsupported`

Detection can use extension hints, magic bytes, and content-type metadata.

## Examples

Examples in `examples/` now use PostgreSQL + Qdrant only:

- `examples/basic_usage.py`
- `examples/query_rag.py`
- `examples/template_ingestion.py`
- `examples/evaluate_retrieval.py`

Each example reads environment variables:

- `RAG_POSTGRES_DSN`
- `RAG_QDRANT_URL`
- `RAG_QDRANT_COLLECTION`

## Docker Compose Files

### 1) PostgreSQL + pgAdmin only

```bash
docker compose -f docker-compose.postgres.yml up -d
```

Services:

- `postgres` on `localhost:5432`
- `pgadmin` on `localhost:5050`

### 2) Qdrant only

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

Service:

- `qdrant` on `localhost:6333` (HTTP), `localhost:6334` (gRPC)

### 3) Full infrastructure (recommended)

```bash
docker compose -f docker-compose.infra.yml up -d
```

Services:

- `postgres`
- `pgadmin`
- `qdrant`

## OpenTelemetry (Optional)

Enable local collector:

```bash
docker compose -f docker-compose.otel.yml up -d
```

Files:

- `docker-compose.otel.yml`
- `observability/otel-collector-config.yaml`
- `observability/signoz-dashboard-ragflow.yaml`

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[qdrant]
pip install -e .[pgvector]
pip install -e .[hf]
pip install -e .[all]
```

## Testing and Quality

```bash
ruff check .
mypy src tests scripts
pytest -q
```

## Integration Tests

Defaults:

- `QDRANT_URL=http://localhost:6333`
- `PGVECTOR_DSN=postgresql+psycopg://postgres:N0th1ing@localhost:5432/app`

Run preflight:

```bash
python scripts/preflight_check.py
```

Run preflight + integration tests:

```bash
python scripts/run_preflight_and_integration.py
```

## PromptOrchestrator Interoperability

Use `PromptStyleRAGProviderAdapter` to expose `retrieve(query, limit)` style retrieval for PromptOrchestrator flows while keeping ragflow_orchestrator ingestion/storage responsibilities.

## Repository Structure

- `src/ragflow_orchestrator/`: package source
- `examples/`: usage samples
- `scripts/`: runnable demos and utility scripts
- `tests/`: test suite
- `datasets/`: evaluation datasets

## License

See `LICENSE`.
