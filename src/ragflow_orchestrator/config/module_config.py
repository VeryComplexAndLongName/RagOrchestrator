from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    kind: str = "sqlite+vec"
    params: dict[str, object] = Field(default_factory=lambda: {"db_path": "rag.db", "table_name": "rag_chunks"})


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


class ModuleConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
