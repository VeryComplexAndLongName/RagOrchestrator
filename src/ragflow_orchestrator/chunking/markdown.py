from __future__ import annotations

from datetime import datetime, timezone

from ragflow_orchestrator.models import BaseChunk


class MarkdownHeadingChunker:
    def chunk(self, source_id: str, text: str, metadata: dict[str, object] | None = None) -> list[BaseChunk]:
        metadata = metadata or {}
        chunks: list[BaseChunk] = []
        lines = text.splitlines()
        current_heading = "root"
        buffer: list[str] = []
        index = 0
        now = datetime.now(timezone.utc)

        def flush() -> None:
            nonlocal index
            if not buffer:
                return
            body = "\n".join(buffer).strip()
            buffer.clear()
            if not body:
                return
            chunk_meta = dict(metadata)
            chunk_meta["heading"] = current_heading
            chunks.append(
                BaseChunk(
                    id=f"{source_id}:{index}",
                    text=body,
                    metadata=chunk_meta,
                    source_id=source_id,
                    chunk_index=index,
                    created_at=now,
                )
            )
            index += 1

        for line in lines:
            if line.lstrip().startswith("#"):
                flush()
                current_heading = line.strip("# ").strip() or "untitled"
                continue
            buffer.append(line)

        flush()
        return chunks
