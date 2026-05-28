from __future__ import annotations

from pydantic import BaseModel, Field

from rag_orchestrator.models import BaseChunk


class DocChunk(BaseModel):
    id: str
    content: str
    score: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


def to_doc_chunk(chunk: BaseChunk, score: float | None = None) -> DocChunk:
    return DocChunk(
        id=chunk.id,
        content=chunk.text,
        score=score,
        metadata={str(key): str(value) for key, value in chunk.metadata.items()},
    )
