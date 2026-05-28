from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from rag_orchestrator.models import BaseChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def merge_metadata(metadata: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(metadata)
    if extra:
        merged.update(extra)
    return merged


def make_chunk(**kwargs: Any) -> BaseChunk:
    return BaseChunk(**kwargs)
