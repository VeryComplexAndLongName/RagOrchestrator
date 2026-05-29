from __future__ import annotations

import ast
from datetime import datetime, timezone

from ragflow_orchestrator.models import BaseChunk, ChunkKind


class PythonCodeChunker:
    """Chunks Python code by top-level functions/classes and falls back to file chunk."""

    def chunk(self, source_id: str, text: str, metadata: dict[str, object] | None = None) -> list[BaseChunk]:
        metadata = metadata or {}
        chunks: list[BaseChunk] = []
        now = datetime.now(timezone.utc)

        try:
            module = ast.parse(text)
        except SyntaxError:
            return [
                BaseChunk(
                    id=f"{source_id}:0",
                    text=text,
                    metadata=dict(metadata),
                    source_id=source_id,
                    chunk_index=0,
                    created_at=now,
                    kind=ChunkKind.CODE,
                )
            ]

        lines = text.splitlines()
        index = 0
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = max(node.lineno - 1, 0)
            end = getattr(node, "end_lineno", node.lineno)
            snippet = "\n".join(lines[start:end]).strip()
            if not snippet:
                continue
            chunk_meta = dict(metadata)
            chunk_meta["symbol_type"] = type(node).__name__
            chunk_meta["symbol_name"] = node.name
            chunks.append(
                BaseChunk(
                    id=f"{source_id}:{index}",
                    text=snippet,
                    metadata=chunk_meta,
                    source_id=source_id,
                    chunk_index=index,
                    created_at=now,
                    kind=ChunkKind.CODE,
                )
            )
            index += 1

        if chunks:
            return chunks

        return [
            BaseChunk(
                id=f"{source_id}:0",
                text=text,
                metadata=dict(metadata),
                source_id=source_id,
                chunk_index=0,
                created_at=now,
                kind=ChunkKind.CODE,
            )
        ]
