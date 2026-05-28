from __future__ import annotations

import math
from dataclasses import dataclass

from rag_orchestrator.models import BaseChunk


@dataclass(slots=True)
class ChunkQualityReport:
    total_chunks: int
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    avg_vector_norm: float


@dataclass(slots=True)
class RetrievalEvalCase:
    expected_chunk_ids: set[str]
    retrieved_chunk_ids: list[str]


@dataclass(slots=True)
class RetrievalQualityReport:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def evaluate_chunks(chunks: list[BaseChunk]) -> ChunkQualityReport:
    if not chunks:
        return ChunkQualityReport(0, 0.0, 0, 0, 0.0)

    lengths = [len(chunk.text) for chunk in chunks]
    norms = []
    for chunk in chunks:
        norm = math.sqrt(sum(value * value for value in chunk.vector)) if chunk.vector else 0.0
        norms.append(norm)

    return ChunkQualityReport(
        total_chunks=len(chunks),
        avg_chunk_length=sum(lengths) / len(lengths),
        min_chunk_length=min(lengths),
        max_chunk_length=max(lengths),
        avg_vector_norm=sum(norms) / len(norms),
    )


def evaluate_retrieval(cases: list[RetrievalEvalCase], k: int) -> RetrievalQualityReport:
    if not cases:
        return RetrievalQualityReport(0.0, 0.0, 0.0, 0.0)

    precision_sum = 0.0
    recall_sum = 0.0
    rr_sum = 0.0
    ndcg_sum = 0.0

    for case in cases:
        top_k = case.retrieved_chunk_ids[:k]
        hits = [chunk_id for chunk_id in top_k if chunk_id in case.expected_chunk_ids]
        precision_sum += len(hits) / max(k, 1)
        recall_sum += len(hits) / max(len(case.expected_chunk_ids), 1)

        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(top_k, start=1):
            if chunk_id in case.expected_chunk_ids:
                reciprocal_rank = 1.0 / rank
                break
        rr_sum += reciprocal_rank

        dcg = 0.0
        for rank, chunk_id in enumerate(top_k, start=1):
            if chunk_id in case.expected_chunk_ids:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(k, len(case.expected_chunk_ids))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

    total = len(cases)
    return RetrievalQualityReport(
        precision_at_k=precision_sum / total,
        recall_at_k=recall_sum / total,
        mrr=rr_sum / total,
        ndcg_at_k=ndcg_sum / total,
    )
