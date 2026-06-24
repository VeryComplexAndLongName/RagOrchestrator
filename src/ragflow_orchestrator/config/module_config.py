from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    kind: str = "postgres+qdrant"
    params: dict[str, object] = Field(
        default_factory=lambda: {
            "dsn": "postgresql://rag_user:rag_password@localhost:5432/rag_db",
            "qdrant_url": "http://localhost:6333",
            "qdrant_collection": "rag_chunks",
        }
    )


class EmbeddingConfig(BaseModel):
    provider: str = "hash"
    model: str | None = None
    options: dict[str, object] = Field(default_factory=dict)
    dimensions: int | None = 256
    base_url: str | None = "http://localhost:11434"

    def as_provider_options(self) -> dict[str, object]:
        merged = dict(self.options)
        if self.dimensions is not None and "dimensions" not in merged:
            merged["dimensions"] = self.dimensions
        if self.base_url is not None and "base_url" not in merged:
            merged["base_url"] = self.base_url
        return merged


class PipelineConfig(BaseModel):
    preset: str = "document"


class SubtypeLLMConfig(BaseModel):
    enabled: bool = False
    provider: str = "none"  # none | ollama | openai_compat
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 20.0
    temperature: float = 0.0


class SubtypeClassificationConfig(BaseModel):
    enabled: bool = True
    fallback_subtype: str = "description"
    confidence_threshold: float = 0.65
    rules_weight: float = 0.7
    llm_weight: float = 0.3
    allowed_subtypes: list[str] = Field(
        default_factory=lambda: [
            "normative",
            "description",
            "specification",
            "instruction",
            "policy",
            "contract_legal",
            "report",
            "faq",
            "reference",
            "code_doc",
            "agreement",
            "unknown",
        ]
    )
    llm: SubtypeLLMConfig = Field(default_factory=SubtypeLLMConfig)


class ModuleConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    subtype_classification: SubtypeClassificationConfig = Field(default_factory=SubtypeClassificationConfig)
