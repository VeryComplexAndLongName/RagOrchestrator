from __future__ import annotations

from datetime import datetime, timezone

from ragflow_orchestrator.models import BaseChunk


class FixedWindowChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, source_id: str, text: str, metadata: dict[str, object] | None = None) -> list[BaseChunk]:
        metadata = metadata or {}
        chunks: list[BaseChunk] = []
        step = self.chunk_size - self.chunk_overlap
        now = datetime.now(timezone.utc)

        cursor = 0
        index = 0
        while cursor < len(text):
            part = text[cursor : cursor + self.chunk_size]
            if part.strip():
                chunks.append(
                    BaseChunk(
                        id=f"{source_id}:{index}",
                        text=part,
                        metadata=dict(metadata),
                        source_id=source_id,
                        chunk_index=index,
                        created_at=now,
                    )
                )
            index += 1
            cursor += step
        return chunks
