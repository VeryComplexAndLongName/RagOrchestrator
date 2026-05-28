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


class BaseChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    text: str
    vector: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_id: str
    chunk_index: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: ChunkKind = ChunkKind.GENERIC
    version: int = 1
    is_deleted: bool = False


class CodeChunk(BaseChunk):
    kind: ChunkKind = ChunkKind.CODE
    language: str
    file_path: str
    function_name: str | None = None


class ContractChunk(BaseChunk):
    kind: ChunkKind = ChunkKind.CONTRACT
    clause_type: str
    parties: list[str] = Field(default_factory=list)


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


class RetrievalResult(BaseModel):
    chunk: BaseChunk
    score: float
    provider: str
