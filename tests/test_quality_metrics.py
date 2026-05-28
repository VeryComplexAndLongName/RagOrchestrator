from __future__ import annotations

from rag_orchestrator.quality import RetrievalEvalCase, evaluate_retrieval


def test_evaluate_retrieval_includes_ndcg() -> None:
    report = evaluate_retrieval(
        cases=[
            RetrievalEvalCase(
                expected_chunk_ids={"a", "b"},
                retrieved_chunk_ids=["a", "x", "b"],
            )
        ],
        k=3,
    )

    assert report.precision_at_k >= 0.0
    assert report.recall_at_k >= 0.0
    assert report.mrr >= 0.0
    assert 0.0 <= report.ndcg_at_k <= 1.0
